# Metal Unified Shared Memory

Draft of the USM implementation for the hipSYCL Metal backend.

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
- Optimize shared memory - assume most computations use shared memory. Up to 5 cycle penalty for every memory access. Translates the address before checking its upper 16 bits, allowing certain compiler optimizations.
- System allocations\* - enable the "low performance route" idea described above. >10 cycle penalty for every memory access. Reserves a few GPU registers for address translation, to avoid invoking a function call. `aspect::usm_system_allocations` returns true.

> \*No guarantee this mode will ever be implemented.

Here are compiler options for controlling the heap for shared memory. These are only valid when "system allocations" is not specified at compile time.
- Default: some significant fraction of maximum working set size, such as 1/8.
- Custom by fraction: a proportion of maximum working set size.
- Custom by absolute size: specify the heap's size in gigabytes.
- Adaptive\*: start with either default or custom size, but dynamically expand until reaching a pre-determined upper bound (e.g. 3/4 working set size, can be customized). Afterward, throw an out-of-memory error. The heap may automatically contract if enough memory is released, for enough time. While reallocating the heap during expansion, flush all queues and halt GPU work to ensure memory safety.

> \*No guarantee this mode will ever be implemented.

## The Problem of Changing Addresses

Imagine that you pass a CPU pointer into a GPU kernel, which will then pass it into a CPU-accessible data structure. During encoding, the captured pointer argument was translated into the GPU address space. The CPU will segfault when reading from the pointer. There are also other issues. For example, what if you simply cast an integer to a pointer, then capture it by the kernel? The backend would change its value by a constant offset, which you don't want. It may not be valid to process threadgroup memory pointers this way, but it seems quite useful for RAM memory pointers.

Given that USM shared memory is enabled, the compiler will have to examine the pointer in IR. Based on how it's used, it may have to translate from the GPU address space back to the CPU address space. Here's how that works.

1. At compile time, you don't know whether a pointers was stored in the CPU or GPU address space. The user could have fetched either a "shared" pointer from the small `MTLHeap` or a "device"-only pointer that can utilize much greater memory. We can't restrict pointers to being "shared"-only in order to fix this problem, because that drastically limits the maximum allocatable memory.
