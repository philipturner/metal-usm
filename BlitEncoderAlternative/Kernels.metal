//
//  Kernels.metal
//  BlitEncoderAlternative
//
//  Created by Philip Turner on 12/30/22.
//

#include <metal_stdlib>
using namespace metal;

// Only use this when 64-byte aligned.
kernel void copyBufferAligned
(
 device uchar4 *src_base [[buffer(0)]],
 device uchar4 *dst_base [[buffer(1)]],
 uint tid [[thread_position_in_grid]])
{
  dst_base[tid] = src_base[tid];
}

// Scans multi-GB chunks of memory, reordering memory transactions to fit 128 B
// boundaries. Unaligned data transfers are shuffled through threadgroup memory.
// The middle 6 simdgroups of every threadgroup have less arithmetic intensity,
// and almost always incur full 128-byte transactions.
struct Arguments {
  // Offset in 32-bit words, rounded down.
  uint src_start;
  uint dst_start;
  
  // Offset in 32-bit words, rounded down.
  uint src_end;
  uint dst_end;
  
  // Real offset, in absolute bytes after word offset.
  ushort src_start_distance;
  ushort dst_start_distance;
  
  // Real offset, in absolute bytes after word offset.
  ushort src_end_distance;
  ushort dst_end_distance;
  
  // Absolute realignment is the delta:
  //   dst_absolute % 128 - src_absolute % 128
  // Word realignment is the delta:
  //   floor_divide(src_absolute % 128 - dst_absolute % 128, 4)
  // Bytes after is the delta:
  //   absolute_realignment - 4 * word_realignment
  short word_realignment;
  ushort bytes_after_word_realignment;
};

__attribute__((__always_inline__))
uchar4 combine_data(uchar4 lo, uchar4 hi, ushort byte_realignment) {
  switch (byte_realignment) {
    case 0: {
      return lo;
    }
    case 1: {
      return uchar4(lo[1], lo[2], lo[3], hi[0]);
    }
    case 2: {
      return uchar4(lo[2], lo[3], hi[0], hi[1]);
    }
    default: /*3*/ {
      return uchar4(lo[3], hi[0], hi[1], hi[2]);
    }
  }
}

constant bool using_threadgroups = false;
constant ushort num_active_threads = 32 * 8;
constant ushort transaction_bytes = 128;
constant ushort transaction_words = transaction_bytes / 4;

// `src_base` and `dst_base` are rounded to 1024-bit chunks.
// Allowed arithmetic intensity: 5308/(408/4/2) = 104 cycles/word copied.
// Dispatch threads so that reads are unaligned, writes are aligned. Reads might
// happen from the cache anyway.
kernel void copyBufferEdgeCases
 (
  device uchar4 *src_base [[buffer(0)]],
  device uchar4 *dst_base [[buffer(1)]],
  constant Arguments &args [[buffer(2)]],
  
  // Threadgroup size assumed to be 256 + 32.
  // Simdgroup size assumed to be 32.
  uint tgid [[threadgroup_position_in_grid]],
  ushort thread_index [[thread_position_in_threadgroup]],
  ushort simd_index [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  uchar4 lo, hi;
  uint dst_index;
//  if (using_threadgroups) {
//    int absolute_start = tgid * num_active_threads;
//    int read_start = absolute_start + args.word_realignment;
//    int read_group_base = read_start & ~(transaction_words - 1);
//    int src_index = read_group_base + thread_index;
//
//    // Don't read out of bounds, you may cause a soft fault or harm bandwidth.
//    int ram_index = src_index;
//    ram_index = max(ram_index, int(args.src_start));
//    ram_index = min(ram_index, int(args.src_end));
//    ram_index = max(ram_index, read_start);
//    ram_index = min(ram_index, read_start + num_active_threads);
//    uchar4 src_data = src_base[ram_index];
//
//    // Transfer contiguous chunks to threadgroup memory.
//    threadgroup uchar4 transferred_words[transaction_words + num_active_threads + transaction_words];
//    transferred_words[src_index - absolute_start + transaction_words] = src_data;
//    if (simd_index == num_active_threads / 32) {
//      // We don't need the last simd anymore.
//      return;
//    }
//
//    // Realign the data.
//    short tg_base_index = thread_index + args.word_realignment;
//    threadgroup_barrier(mem_flags::mem_threadgroup);
//    lo = transferred_words[tg_base_index + transaction_words];
//    hi = transferred_words[tg_base_index + transaction_words + 1];
//    dst_index = absolute_start + thread_index;
//  } else {
    int absolute_start = tgid * 256 + simd_index * 32;
    int read_start = absolute_start + args.word_realignment;
    int read_group_base = read_start & ~(transaction_words - 1);
    int src_index = read_group_base + lane_id;
    
    // Do not read out-of-bounds. That could cause a soft fault or create
    // errors in Metal Shader Validation.
    int ram_index_lo = src_index;
    int ram_index_hi = src_index + 32;
    
    // TODO: Wrap this in an indirect command buffer. The first and last
    // X threadgroups run a different set of shader code.
    // Or, load this with a function constant, checking MTL_SHADER_VALIDATION
    // in the environment variables.
//    if (ram_index_hi >= int(args.src_end)) {
//      int upper_bound =
//        int(args.src_end) - select(1, 0, args.src_end_distance > 0);
//      ram_index_lo = min(ram_index_lo, upper_bound);
//      ram_index_hi = min(ram_index_hi, upper_bound);
//    }
//    if (ram_index_lo < int(args.src_start)) {
//      ram_index_lo = max(ram_index_lo, int(args.src_start));
//      ram_index_hi = max(ram_index_hi, int(args.src_start));
//    }
    
    uchar4 src_data_lo = src_base[ram_index_lo];
    uchar4 src_data_hi = src_base[ram_index_hi];
    
    ushort pivot_lane = ushort(args.word_realignment) % 32;
    uchar4 served_lo = (lane_id < pivot_lane)
      ? src_data_hi : src_data_lo;
    uchar4 served_hi = (lane_id <= pivot_lane)
      ? src_data_hi : src_data_lo;
    lo = simd_shuffle_rotate_down(served_lo, args.word_realignment);
    hi = simd_shuffle_rotate_down(served_hi, args.word_realignment + 1);
    dst_index = absolute_start + lane_id;
//  }
  uchar4 data = combine_data(lo, hi, args.bytes_after_word_realignment);
  
  // Mask the write, checking both upper and lower bounds.
  if (dst_index > args.dst_start && dst_index < args.dst_end) {
    dst_base[dst_index] = data;
  } else {
    ushort start = (dst_index < args.dst_start) ? 4 : 0;
    ushort end = (dst_index > args.dst_end) ? 0 : 4;
    if (dst_index == args.dst_start) {
      start = args.dst_start_distance;
    }
    if (dst_index == args.dst_end) {
      end = args.dst_end_distance;
    }
    for (ushort i = start; i < end; ++i) {
      ((device uchar*)dst_base)[dst_index * 4 + i] = data[i];
    }
  }
}
