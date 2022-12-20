//
//  HostMessaging.metal
//  USMBenchmarks
//
//  Created by Philip Turner on 12/19/22.
//

#include <metal_stdlib>
using namespace metal;

// This file is compiled with -Os, minimizing the chance that the compiler
// inlines something it's not supposed to.

void atomicWrite(device uint3* returnCodes, uint3 desired) {
  auto lower = (device atomic_uint*)returnCodes;
  auto upper = lower + 1;
  atomic_store_explicit(lower, desired[0], memory_order_relaxed);
  atomic_store_explicit(upper, desired[1], memory_order_relaxed);
  atomic_store_explicit(upper + 1, desired[2], memory_order_relaxed);
}

// Returns the number of read messages and i.
// This bizarrely structured kernel has the best performance when interfacing
// with the CPU.
//
// Average message time: consistently 4000 nanoseconds
// Maximum message time: consistently 1000-2000 microseconds
__attribute__((__noinline__))
uint2 kernelBody(device atomic_uint *pipe,
                 constant uint &totalMessages,
                 constant uint &timeout)
{
  volatile uint sentMessages = 0;
  volatile uint i = 0;
  for (; i < timeout; ++i) {
    // starts at 0
    // flips 0 -> 1
    // flips 2 -> 3
    // flips 4 -> 5
    uint prefetched = atomic_load_explicit(pipe, memory_order_relaxed);
    if (prefetched == __UINT32_MAX__) {
      // This will never happen.
      return {0, 0};
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    uint expected = sentMessages;
    uint desired = sentMessages + 1;
    auto result = atomic_compare_exchange_weak_explicit(
      pipe, &expected, desired, memory_order_relaxed, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device);
    if (result) {
      sentMessages += 2;
    } else if (expected < sentMessages) {
      // Not sure why, but this empty conditional block needs to exist here.
      // Real-world worst-case performance changes drastically otherwise.
      sentMessages = max(expected, sentMessages);
    }
    
    if (sentMessages >= totalMessages) {
      break;
    }
  }
  
  return {sentMessages, i};
}

kernel void hostMessaging(device atomic_uint *pipe [[buffer(0)]],
                          constant uint &totalMessages [[buffer(2)]],
                          constant uint &timeout [[buffer(3)]],
                          device uint3 *returnCodes [[buffer(4)]],
                          uint tid [[thread_position_in_grid]])
{
  bool invalidArgs = false;
  if (atomic_load_explicit(pipe, memory_order_relaxed) != 0) {
    invalidArgs = true;
  }
  if (atomic_load_explicit((device atomic_uint*)returnCodes, memory_order_relaxed) != 0) {
    invalidArgs = true;
  }
  if (atomic_load_explicit(((device atomic_uint*)returnCodes) + 1, memory_order_relaxed) != 0) {
    invalidArgs = true;
  }
  if (invalidArgs) {
    // 1 = invalid arguments
    atomicWrite(returnCodes, {1, 0, 1});
    return;
  }

  uint2 sentMessagesAndI = kernelBody(pipe, totalMessages, timeout);
  
  if (sentMessagesAndI[0] >= totalMessages) {
    // 2 = success
    atomicWrite(returnCodes, {2, sentMessagesAndI[1], 0});
  } else {
    // 3 = timed out
    atomicWrite(returnCodes, {3, sentMessagesAndI[0], atomic_load_explicit(pipe, memory_order_relaxed)});
  }
}
