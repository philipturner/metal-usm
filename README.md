# Metal Unified Shared Memory

Access CPU pointers from Metal shaders

Low performance route with higher capabilities:
- Subdivide all of physical memory into ~8 MB chunks. Create a one-step lookup table that maps chunks of `malloc`'d CPU memory to GPU memory. `malloc`'d addresses typically fall into a range with the same magnitude as physical RAM, offset by 4 GB.
- Virtual memory addresses, which typically exceed RAM size, fall into a two-step lookup table.
- A GPU-side function call is invoked during any page fault, stalling until the CPU fills in the missing page.

High-performance route with limited capabilities (default):
- Allocate a large Metal heap before program initialization, place all shared allocations in there. At runtime, the shader tests the upper 16 bits of any GPU pointer. If empty, it adds the difference between the `MTLHeap`'s GPU and CPU base address. Otherwise, it proceeds as if the address is a typical GPU address.
- annotate bound buffers (residing in uniform registers) as not USM pointers, but fix the upper 16 bits before copying by value. This only happens for SYCL accessors (which can use buffer bindings) and not for USM pointers (which can't use buffer bindings). Wrong ... all captured arguments can be force-converted into Metal buffers or offsets within buffers, then placed in uniform registers. Address translation happens when encoding, becoming zero-cost inside the shader!
- Even indirectly captured SYCL USM pointers are recognized by the compiler. This allows you to detect them, translate the CPU address to a GPU address during encoding, and place them in the buffer argument table.
- Allocate an extra buffer binding for when you exceed 31 buffers. An indirect buffer will store the extra arguments. Pointers here must be treated like USM pointers, because you first load them into shader memory, copying by value.
- If you have exactly 31 arguments, locate the last one in an indirect buffer binding. That way, there's never more than 30 genuine buffers bound. This simplifies some compiler logic and reduces the chance of bugs regarding the 31st argument.

In the injected assembly, branch everything so that it favors either always device or always shared pointers. This approach avoids needing to copy stuff between registers when translating pointers. However, it bloats the code slightly more and reduces performance when simultaneously fetching device + shared pointers from different threads in the SIMD. Create a compile-time option to choose which USM mode to favor.
- Disable shared memory - fastest performance and no injected assembly. Does not tag bound buffer pointers when copying by value. `aspect::usm_shared_allocations` returns false.
- Default - assumes most computations use device memory. 0.5 cycle penalty for every memory access.
- Optimize shared memory - assume most computations use shared memory. Up to 4.0 cycle penalty for every memory access. Translates the address before checking its upper 16 bits, allowing certain compiler optimizations.
- System allocations\* - enable the "low performance route" idea described above. Reserves a few GPU registers for address translation, to avoid invoking a function call. `aspect::usm_system_allocations` returns true.

> \*No guarantee this mode will ever be implemented.

Compiler options for controlling heap for shared memory:
- Default: some significant fraction of maximum working set size, such as 1/8.
- Custom by fraction: a proportion of maximum working set size.
- Custom by absolute magnitude: specify the heap's size in gigabytes.
- Adaptive\*: start with either default or custom size, but expand until reaching a pre-determined fraction of working set size (e.g. 3/4, can be customized). Afterward, throws an out-of-memory error. When expanding the heap, flushes all queues and halts GPU work to ensure memory safety.

> \*No guarantee this mode will ever be implemented.
