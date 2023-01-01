//
//  Kernels.metal
//  BlitEncoderAlternative
//
//  Created by Philip Turner on 12/30/22.
//

#include <metal_stdlib>
using namespace metal;

// Scans multi-GB chunks of memory at once.
// Loads with 32-bit resolution (first/last words treated specially).
// Unaligned data transferred through threadgroup memory and/or SIMD shuffles.
// After realignment, start writing to RAM.
struct Arguments {
  uint src_start;
  uint src_end;
  uint dst_start;
  uint dst_end;
  
  // Offset in 32-bit words, rounded down.
  // TODO: Reassociated variable accesses to the new names.
  
  // Real offset, in bytes after word offset.
  ushort src_start_distance;
  ushort src_end_distance;
  ushort dst_start_distance;
  ushort dst_end_distance;
};

// First try with just SIMD shuffles, then enhance with threadgroup memory.
// Simds per threadgroup: 8 + 1 + 1
//
// Middle simds handle contiguous chunks (cheap).
// Outer simds handle unaligned addresses (expensive).
// Still 8-9 threadgroups/grid, 64-72 active simds.
//
// For outer simds:
// - First, aligned-write the contiguous words (masked by thread).
// - Then, subdivide bytes from the remaining word across threads.
//
// This scheme decreases average arithmetic intensity by ~80%.
// - Allowed cycles/unaligned byte: ~13
// - Allowed cycles/32-bit word: ~52

// `src_base` and `dst_base` rounded to 4096-byte pages.
// TODO: Properly align the grid, so you don't read out-of-bounds and cause a
// page fault.
kernel void copyBufferEdgeCases
 (
  device uchar4 *src_base [[buffer(0)]],
  device uchar4 *dst_base [[buffer(1)]],
  constant Arguments &args [[buffer(2)]],
  
  // Threadgroup size assumed to be 256.
  // Simdgroup size assumed to be 32.
  uint tid [[thread_position_in_grid]],
  uint tgid [[threadgroup_position_in_grid]],
  ushort simd_index [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  // TODO: When using threadgroups, realign not just on word boundaries. Also
  // realign on memory transaction boundaries (128 bytes). In copyBufferAligned,
  // check whether odd multiples of 4 tank performance.
  //
  // TODO: Shuffle as much data as possible through simds.
  // TODO: Rewrite this. Only the first/last threadgroups of the entire grid
  // should read misaligned data. All can write misaligned data. This removes
  // the need for "helper" simdgroups.
  //
  // Every single threadgroup goes through the same execution path. First/last
  // threadgroups just write X many fewer bytes to the destination.
  
  // Notes:
  // - `next_data` not needed if `src` is aligned.
  // - `prev_data` not needed if `dst` is aligned.
  //
  // We're optimizing for edge cases here. It takes some cycles to check whether
  // you're in a certain edge case. A future version could use function
  // constants and optimize everything away, but that's 3x the device code.
  //
  // On another note, a "helper" simdgroup will be the one prefetching
  // everything. That simdgroup can run independently, transferring its contents
  // to threadgroup memory asynchronously. Then it will rematerialize during the
  // threadgroup barrier. It can perform the conditional inside its execution
  // context; it's the only simdgroup wasting a thread-cycle. Finally, the
  // helper simdgroup will return early.
  uchar4 curr_data;
  if (true) {
    curr_data = src_base[args.src_start + tid];
  } else {
    // TODO: Move helper thread's prefetch here.
  }
  
  uchar4 prefetched_next_data;
  if (true) {
    // Prefetch `next_data` (helper simdgroup only).
    uint sid = tgid * 8 + simd_index;
    prefetched_next_data = src_base[args.src_start + sid * 32 + 32];
    
    // Transfer to threadgroup memory for future threads.
  }
  
  // Transfer last lane's `curr_data` through threadgroup memory.
  if (lane_id == 31) {
    
  }
  
  // For the first lane, we don't care about previous bytes yet.
  uchar4 prev = simd_shuffle_up(curr_data, 1);
  
  // Fetch first lane's `prev` from threadgroup memory.
  if (lane_id == 0) {
    // For the first simdgroup, this will be undefined.
    // threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  
  // This approach makes the compiler fully optimize the swizzles.
  uchar4 lo, hi;
#define LO_HI_BLOCK(GET_DATA) \
  switch (args.dst_start_distance) { \
    case 1: { \
      uchar4 data = GET_DATA; \
      lo = uchar4(prev[3], data[0], data[1], data[2]); \
      hi = uchar4(data[3], 0, 0, 0); \
      break; \
    } \
    case 2: { \
      uchar4 data = GET_DATA; \
      lo = uchar4(prev[2], prev[3], data[0], data[1]); \
      hi = uchar4(data[2], data[3], 0, 0); \
      break; \
    } \
    case 3: { \
      uchar4 data = GET_DATA; \
      lo = uchar4(prev[1], prev[2], data[3], data[0]); \
      hi = uchar4(data[1], data[2], data[3], 0); \
      break; \
    } \
    default: { \
      uchar4 data = GET_DATA; \
      lo = data; \
      /* hi = not needed */ \
      break; \
    } \
  } \
  
  if (args.src_start_distance == 0) {
    LO_HI_BLOCK(curr_data);
  } else {
    // Transfer first lane's `curr_data` through threadgroup memory.
    if (lane_id == 0) {
      
    }
    
    // Must initialize `next_data` outside of the loop, otherwise results of
    // `simd_shuffle_down` are officially undefined.
    uchar4 next_data = simd_shuffle_down(curr_data, 1);
    
    // Fetch last lane's `next_data` from threadgroup memory.
    if (lane_id == 31) {
      // threadgroup_barrier(mem_flags::mem_threadgroup);
      next_data = prefetched_next_data;
    }
    
    switch (args.src_start_distance) {
      case 1: {
        LO_HI_BLOCK(uchar4(curr_data[1], curr_data[2], curr_data[3], next_data[0]));
        break;
      }
      case 2: {
        LO_HI_BLOCK(uchar4(curr_data[2], curr_data[3], next_data[0], next_data[1]));
        break;
      }
      default: /*3*/ {
        LO_HI_BLOCK(uchar4(curr_data[3], next_data[0], next_data[1], next_data[2]));
        break;
      }
    }
  }
  
  // It doesn't matter whether we use "helper" simdgroups here. Writes are
  // non-blocking, and either way something issues instructions every clock.
  uint dst_offset = args.dst_start + tid;
  if (dst_offset < args.dst_end) {
    // Mask the first lane. (first simdgroup only)
    if (true && lane_id == 0) {
      // We can treat 1-byte words as "packed formats". The compiler won't
      // secretly read and bitmask the final destination (a 32-bit write).
      switch (args.src_start_distance) {
        case 1: {
          ((device uchar*)dst_base)[dst_offset * 4 + 1] = lo[1];
          // fallthrough
        }
        case 2: {
          ((device uchar2*)dst_base)[dst_offset * 2 + 1] = lo.zw;
          return;
        }
        case 3: {
          ((device uchar*)dst_base)[dst_offset * 4 + 3] = lo[3];
          return;
        }
        default: {
          // Continue to the next control block; coalesce this write with other
          // threads.
          break;
        }
      }
    }
    
    dst_base[dst_offset] = lo;
    
    // Write `hi`. (last simdgroup only)
    if (true && lane_id == 31) {
      // TODO: Finish writing the kernel here.
    }
  } else {
    
  }
}

kernel void copyBufferAligned
(
 device uchar4 *src_base [[buffer(0)]],
 device uchar4 *dst_base [[buffer(1)]],
 uint tid [[thread_position_in_grid]])
{
  // TODO: Only use this shader when 128-byte aligned.
  dst_base[tid] = src_base[tid];
}
