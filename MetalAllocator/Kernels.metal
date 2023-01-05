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

struct Pointers1 {
  ulong bufferA;
  ulong bufferB;
  ulong bufferC;
  ulong bufferD;
};

struct Pointers2 {
  device ulong2* bufferA;
  device ulong2* bufferB;
  device ulong2* bufferC;
  device ulong2* bufferD;
};

kernel void vectorAddition(constant Pointers1 *arguments [[buffer(0)]],
                           uint tid [[thread_position_in_grid]])
{
  auto valueA = ((constant Pointers2*)arguments)->bufferA[tid];
  auto valueB = ((constant Pointers2*)arguments)->bufferB[tid];
  auto valueC = ((constant Pointers2*)arguments)->bufferC[tid];
  auto valueD = valueA + valueB + valueC;
  ((constant Pointers2*)arguments)->bufferD[tid] = valueD;
}
