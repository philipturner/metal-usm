# Metal Unified Shared Memory

Draft of the USM implementation for the hipSYCL Metal backend.

> Note: This document needs to be rewritten. Recent discoveries (explained in the last section) warrant major changes to this approach.

Low performance route with higher capabilities:
- Subdivide all of physical memory into ~8 MB chunks. Create a one-step lookup table that maps chunks of `malloc`'d CPU memory to GPU memory. `malloc`'d addresses typically fall into a range with the same magnitude as physical RAM, offset by 4 GB.
- Virtual memory addresses, which typically exceed RAM size, fall into a two-step lookup table.
- A GPU-side function call is invoked during any page fault, stalling until the CPU fills in the missing page.

High-performance route with limited capabilities (default):
- Allocate a large Metal heap before program initialization, place all shared allocations in there. At runtime, the shader tests the upper 16 bits of any GPU pointer. If empty, it adds the difference between the `MTLHeap`'s GPU and CPU base address. Otherwise, it proceeds as if the address is a typical GPU address.
- There is no way to mutate the upper 16 bits of a bound buffer's address. In SPIR-V or AIR, annotate bound buffers (residing in uniform registers) as not USM pointers, but fix the upper 16 bits before copying by value. This only happens for SYCL accessors (which can use buffer bindings) and not for USM pointers (which can't use buffer bindings). Wrong ... all captured arguments can be force-converted into Metal buffers or offsets within buffers, then placed in uniform registers. Address translation happens when encoding, becoming zero-cost inside the shader!
- Even indirectly captured SYCL USM pointers are recognized by the compiler. This allows you to detect them, translate their CPU address to a `MTLBuffer` during encoding, and place them in the buffer argument table.
- Apple GPU cores contain 31 slots for binding Metal buffers. Allocate an extra buffer binding for when you exceed 31 buffers. An indirect buffer will store the extra arguments. All indirectly stored arguments must be annotated as USM pointers, because you first load them into GPU registers, copying by value. Their address will still be translated to the GPU version during encoding.
- If you have exactly 31 arguments, locate the last one in an indirect buffer binding. That way, there's never more than 30 genuine buffers bound. This simplifies some compiler logic and reduces the chance of bugs regarding the 31st argument.

In the injected assembly, branch everything so that it favors either always device or always shared pointers. This approach avoids needing to copy stuff between registers when translating pointers. However, it bloats the code slightly more and reduces performance when simultaneously fetching device + shared pointers from different threads in a SIMD. Setting that aside, there are compile-time options to choose which USM mode to favor:
- Disable shared memory - fastest performance and no injected assembly. Does not tag bound buffer pointers when copying by value. `aspect::usm_shared_allocations` returns false.
- Default - assumes most computations use device memory. 1 cycle penalty for every memory access, except from the first 30 pointers captured as lambda arguments in SYCL C++ source.
- Optimize shared memory\* - assume most computations use shared memory. Up to 5 cycle penalty for every memory access. Translates the address before checking its upper 16 bits, allowing certain compiler optimizations.
- System allocations\* - enable the "low performance route" idea described above. >10 cycle penalty for every memory access. Reserves a few GPU registers for address translation, to avoid invoking a function call. `aspect::usm_system_allocations` returns true.

> \*No guarantee these modes will ever be implemented.

Here are compiler options for controlling the heap for shared memory. These are only valid when "system allocations" is not specified at compile time.
- Default: some significant fraction of maximum working set size, such as 1/8.
- Custom by fraction: a proportion of maximum working set size.
- Custom by absolute size: specify the heap's size in gigabytes.
- Adaptive\*: start with either default or custom size, but dynamically expand until reaching a pre-determined upper bound (e.g. 3/4 working set size, can be customized). Afterward, throw an out-of-memory error. The heap may automatically contract if enough memory is released, for enough time. While reallocating the heap during expansion, flush all queues and halt GPU work to ensure memory safety.

> \*No guarantee this mode will ever be implemented.

## The Problem of Changing Addresses

Imagine that you pass a CPU pointer into a GPU kernel, which will then pass it into a CPU-accessible data structure. During encoding, the captured pointer argument was translated into the GPU address space. The CPU will segfault when reading from the pointer. There are also other issues. For example, what if you simply cast an integer to a pointer, then capture it by the kernel? The backend would change its value by a constant offset, which you don't want. It may not be valid to process threadgroup memory pointers this way, but it seems quite useful for RAM memory pointers.

Given that USM shared memory is enabled, the compiler will have to examine the pointer in IR. Based on how it's used, we may have to translate from the GPU address space back to the CPU address space. Here's how that works.

1. At compile time, you don't know whether a pointers was stored in the CPU or GPU address space. The user could have fetched either a "shared" pointer from the small `MTLHeap` or a "device"-only pointer that can utilize much greater memory. We can't restrict pointers to being "shared"-only in order to fix this problem, because that drastically limits the maximum allocatable memory.
2. At runtime, we'll need a table that tells whether bound buffers were originally from the CPU address space. At the end of the 31st buffer binding, append a list of all pointers in their original forms. This buffer could also store the offset between CPU and GPU buffers. The offset changes at runtime as the heap re-allocates, so this is probably necessary anyway (as opposed to a static `function_constant` entered at kernel load time).
3. Instead of storing an array of booleans telling whether each pointer was originally from CPU, we store the entire original address. This reduces the amount of injected assembly necessary for translation. Also, the pointers will likely fall into L1 cache, so there's little benefit to reducing memory bandwidth from 8 to 2 or 1 bytes/access.
4. DO NOT treat buffers after the 30th buffer binding like USM pointers in shader code. Treat them as accessors. We need more complex rules than copy-by-value anyways. The original IR never treated these arguments as something fetched from device memory through indexing. Therefore, this is possible to pull off. Store their original addresses alongside their translated addresses, within the 31st buffer.
5. When reading/writing from a pointer's memory value, fetch that from uniform registers (API buffer bindings) when possible. When operating on the pointer itself, even through bit casting, fetch the original address from device memory. The pointer will now be treated as a regular USM pointer, with a penalty for reading from it.
6. In some instances, you might want to add an offset to the pointer's original value, then retain that modified pointer for use later inside a loop. Not sure the best way to optimize this, but for now don't.

## The Problem of `useResource(_:usage:)`

<details>
<summary>Thoughts on the problem</summary>

While debugging emulated 64-bit atomics, I learned something important about `MTLComputeCommandEncoder.useResource(_:usage:)`. If you don't call this function on an indirectly encoded resource, the GPU will freeze at runtime and force you to restart your computer. Or worse, it will keep running in the background while consuming 1/2 the TDP. One possible reason is, Metal shares CPU virtual memory that can be paged to the disk. The GPU maps that memory just before dispatching certain commands, then potentially unmaps it afterward. With USM pointers, we cannot know which memory to map beforehand.

One solution is, limit all USM allocations (even device ones) to a single ~2 GB `MTLHeap`. This limits the maximum allocatable memory and prevents more than ~10 SYCL applications from ever running simultaneously. However, it's the easiest and the fastest. Another approach is the idea for sharing `malloc` system allocations. I recall that it worked once without calling `useResource(_:usage:)`. Perhaps that's because it was already mapped from disk to CPU? Depending on the answer, this drastically alters viability to the second approach. In one case, a CPU thread could simply listen for page faults and respond by filling in GPU virtual addresses to a soft-TLB. In the other case, it would have to shuffle data between a fixed amount of `useResource'd` memory and any random address the GPU decided to read from. It would be no different from Intel, AMD, and NVIDIA's method of virtualizing shared memory, defeating the advantage brought by Apple silicon's hardware unified memory.

https://tallendev.github.io/assets/papers/sc21.pdf

While experimenting, I found a solution to the problem of limited heap memory. I allocated several terabytes of virtual memory, then chose a spot in the middle of the allocation as a "base CPU address". I allocated a Metal buffer there, then noted the "base GPU VA". For the next allocation, I try allocating slightly off from the CPU "base address". Then, check the GPU VA. The delta from this new allocation's CPU address and the CPU base address should match that between GPU addresses. Otherwise, adjust the CPU base address according to the difference in deltas. This often requires only one reallocation to get right. However, it works. If you deallocate the first allocation's `MTLBuffer`, Metal will allocate the next buffer at the deallocated one's VA.

I should update the rest of this draft to account for no limits on memory. The only limit is device RAM size. Devices can't be paged to disk because of `useResource(_:usage:)`. If I want SYCL accessors to be [transformable into USM pointers](https://hipsycl.github.io/hipsycl/extension/hipsycl-091-buffer-usm-interop/), there's no way to notify the `MTLComputeCommandEncoder` that they're being "used". Therefore, all memory in any form must originate from the ~32 GB of tracked memory. This brings up another problem. How do I divide such a massive address space into few enough `MTLHeap`s that I can call `useResource(_:usage:)` on every one during command encoding? I can take inspiration from PyTorch's MPS allocator, which allocates small Metal buffers from large Metal heaps. Alternatively, statically divide the device's RAM into fixed fractions. Some chunks are larger to accomodate large allocations, while smaller chunks lower the memory footprint of a small program.

I did some more investigation into overhead of using heaps. The `useHeaps(_:)` function has overhead scaling linearly with number of heaps, not the amount of memory passed in. That means it doesn't look at each virtual memory page separately. 128 heaps increased overhead by ~50%, so we should set that as the default maximum (users should be able to adjust it). This will determine the size of heaps for small allocations in a PyTorch-like allocator. On my machine, `recommendedMaxWorkingSetSize / 128` equals 171 MB. We should round up the size to a power of 2, in this case 256 MB. On my iPhone with 4 GB of RAM accessible to any one app, it could be 32 MB. On an M1/M2 Ultra with 128 GB of working set, it would be 1 GB.

</details>

> TL;DR - Both shared and device USM pointers will occur in the same address space. The CPU and GPU addresses will always be off by a constant integer, allowing easy translation. Furthermore, we don't need to statically allocate a large `MTLHeap` beforehand. We can scale from almost nothing to almost all device RAM. 
> 
> We can also assign properties to certain partitions of the USM memory range, e.g. whether they're host-accessible. Thus, we can check whether a pointer is `shared` or `device` without tagging its upper 16 bits. Furthermore, all stray pointers can/must exist in their CPU form. We will no longer check a pointer's address space inside the GPU kernel.
