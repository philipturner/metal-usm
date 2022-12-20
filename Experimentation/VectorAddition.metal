//
//  VectorAddition.metal
//  USMBenchmarks
//
//  Created by Philip Turner on 12/19/22.
//

#include <metal_stdlib>
using namespace metal;

// Simultaneously accesses 4 memory allocations.
// If may allocation size is 16 GB and number of threads is 2G, each thread
// would read 8 bytes. However, each thread reads 16 bytes so we only need
// 1G threads.
kernel void vectorAddition(device ulong2* bufferA [[buffer(0)]],
                           device ulong2* bufferB [[buffer(1)]],
                           device ulong2* bufferC [[buffer(2)]],
                           device ulong2* bufferD [[buffer(3)]],
                           uint tid [[thread_position_in_grid]])
{
  auto valueA = bufferA[tid];
  auto valueB = bufferB[tid];
  auto valueC = bufferC[tid];
  auto valueD = valueA + valueB + valueC;
  bufferD[tid] = valueD;
}
