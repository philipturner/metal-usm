# Metal Unified Shared Memory

Access CPU pointers from Metal shaders

Ideas:
- High performance route with limited capabilities: allocate a large Metal heap before program initialization, place all shared allocations in there. At runtime, the shader tests the upper 16 bits of any GPU pointer. If empty, it adds the difference between the `MTLHeap`'s GPU and CPU base address. Otherwise, it proceeds as if the address is a typical GPU address.
- Low performance route with higher capabilities: subdivide all of system memory into ~8 MB chunks. Create a lookup table that maps chunks of CPU memory to GPU memory, which is accessed every time the GPU accesses memory. Virtual memory addresses, which typically exceed RAM size, fall into a two-step lookup table.
