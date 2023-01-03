//
//  Kernels.metal
//  BlitEncoderAlternative
//
//  Created by Philip Turner on 12/30/22.
//

#include <metal_stdlib>
using namespace metal;

// Use when input is 4-byte aligned and output is 64-byte aligned.
kernel void copyBufferAligned
(
 device uchar4 *src_base [[buffer(0)]],
 device uchar4 *dst_base [[buffer(1)]],
 uint tid [[thread_position_in_grid]])
{
  dst_base[tid] = src_base[tid];
}

// Scans multi-GB chunks of memory, reordering memory transactions to fit 64 B
// boundaries. Unaligned data transfers are shuffled through threadgroup memory.
// The middle 6 simdgroups of every threadgroup have less arithmetic intensity,
// and almost always incur full 64-byte transactions.
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
  //   dst_absolute % 64 - src_absolute % 64
  // Word realignment is the delta:
  //   floor_divide(src_absolute % 64 - dst_absolute % 64, 4)
  // Bytes after is the delta:
  //   absolute_realignment - 4 * word_realignment
  short word_realignment;
  
  // word_realignment & ~(16 - 1)
  short word_rounded_realignment;
  
  // `byte_realignment` * 8
  ushort byte_realignment_lo_shift;
  
  // 32 - `byte_realignment` * 8
  ushort byte_realignment_hi_shift;
};

// This function routinely reads out-of-bounds, causing superfluous errors in
// Metal Shader Validation. This constant activates bounds checking in shader
// code, with a slight performance penalty.
constant bool use_shader_validation [[function_constant(0)]];

// Whether real realignment, in bytes after word realignment, is zero.
constant bool byte_realignment_is_zero [[function_constant(1)]];

// This part is often *faster* than blit encoders!
//
// `src_base` and `dst_base` are rounded to 1024-bit chunks.
// Allowed arithmetic intensity: 5308/(408/4/2) = 104 cycles/word copied.
// Dispatch threads so that reads are unaligned, writes are aligned. Reads might
// happen from the cache anyway.
kernel void copyBufferEdgeCases
 (
  device uchar4 *src_base [[buffer(0)]],
  device uchar4 *dst_base [[buffer(1)]],
  constant Arguments &args [[buffer(2)]],
  
  // Threadgroup size assumed to be 256.
  // Simdgroup size assumed to be 32.
  uint tgid [[threadgroup_position_in_grid]],
  ushort thread_id [[thread_position_in_threadgroup]])
{
  int index_offset = tgid * 224;
  int src_index = index_offset + args.word_rounded_realignment + thread_id;

  // Don't read out of bounds, you may cause a soft fault.
  if (use_shader_validation) {
    src_index = max(src_index, int(args.src_start));
    src_index = min(src_index, int(args.src_end));
  }
  uchar4 src_data = src_base[src_index];
  
  // Transfer contiguous chunks to threadgroup memory.
  threadgroup uchar4 transferred_words[16 + 240 + 16];
  short tg_write_index = args.word_rounded_realignment + thread_id;
  transferred_words[16 + tg_write_index] = src_data;
  if (thread_id >= 224) {
    // We don't need the last simd anymore.
    return;
  }
  
  // Realign the data.
  short tg_read_index = args.word_realignment + thread_id;
  uint dst_index = index_offset + thread_id;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uchar4 lo = transferred_words[16 + tg_read_index];
  
  uchar4 data;
  if (byte_realignment_is_zero) {
    data = lo;
  } else {
    uchar4 hi = transferred_words[16 + tg_read_index + 1];
    uint lo_bits = as_type<uint>(lo) >> args.byte_realignment_lo_shift;
    uint hi_bits = as_type<uint>(hi) << args.byte_realignment_hi_shift;
    data = as_type<uchar4>(lo_bits | hi_bits);
  }
  
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
