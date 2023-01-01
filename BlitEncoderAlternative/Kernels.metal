//
//  Kernels.metal
//  BlitEncoderAlternative
//
//  Created by Philip Turner on 12/30/22.
//

#include <metal_stdlib>
using namespace metal;

// Only use this when 128-byte aligned.
// TODO: Is it also faster when 64-byte aligned?
kernel void copyBufferAligned
(
 device uchar4 *src_base [[buffer(0)]],
 device uchar4 *dst_base [[buffer(1)]],
 uint tid [[thread_position_in_grid]])
{
  dst_base[tid] = src_base[tid];
}

// Scans 4-GB chunks of memory, reordering memory transactions to fit 128-byte
// boundaries. Unaligned data transfers are shuffled through threadgroup memory.
// The middle 6 simdgroups of every threadgroup have less arithmetic intensity,
// and almost always incur full 128-byte transactions.
struct Arguments {
  // Offset in 32-bit words, rounded down.
  uint src_start;
  uint src_end;
  uint dst_start;
  uint dst_end;
  
  // Absolute realignment is the delta:
  //   dst_absolute % 128 - src_absolute % 128
  // Word realignment is the delta:
  //   floor_divide(dst_absolute % 128 - src_absolute % 128, 4)
  // Bytes after is the delta:
  //   absolute_realignment - 4 * word_realignment
  short word_realignment;
  ushort bytes_after_word_realignment;
  
  // Real offset, in bytes after word offset.
  ushort dst_start_distance;
  ushort dst_end_distance;
};

__attribute__((__always_inline__))
uchar4 combine_data(uchar4 lo, uchar4 hi, ushort byte_realignment) {
  switch (byte_realignment) {
    case 0: {
      return hi;
    }
    case 1: {
      return uchar4(lo[3], hi[0], hi[1], hi[2]);
    }
    case 2: {
      return uchar4(lo[2], lo[3], hi[0], hi[1]);
    }
    default: /*3*/ {
      return uchar4(lo[1], lo[2], lo[3], hi[0]);
    }
  }
}

__attribute__((__always_inline__))
void write_end(uchar4 data, uint dst_offset, device uchar4 *dst_base, ushort num_bytes) {
  if (num_bytes >= 2) {
    ((device uchar2*)dst_base)[dst_offset * 2] = data.xy;
    if (num_bytes == 3) {
      ((device uchar*)dst_base)[dst_offset * 4 + 2] = data[2];
    }
  } else if (num_bytes == 1) {
    ((device uchar*)dst_base)[dst_offset * 4 + 0] = data[0];
  }
}

__attribute__((__always_inline__))
void write_middle(uchar4 data, uint dst_offset, device uchar4 *dst_base, constant Arguments &args, ushort thread_index) {
  // Watch for the edge case where `dst_end % 128` conflicts with
  // `dst_start % 128`. For more information, read the large comment in the
  // section starting with (simd_index == 7).
  if (thread_index <= args.dst_start || dst_offset >= args.dst_end) {
    if (dst_offset > args.dst_end) {
      // This thread is too far right. We don't want it.
      return;
    }
    if (thread_index < args.dst_start) {
      // This thread is too far left. We don't want it.
      return;
    }
    
    if (thread_index == args.dst_start) {
      if (dst_offset == args.dst_end) {
        // Edge case where `dst_start % 128` overlaps with `dst_end % 128`.
        ushort start = args.dst_start_distance;
        ushort end = args.dst_end_distance;
        for (ushort i = start; i < end; ++i) {
          ((device uchar*)dst_base)[dst_offset * 4 + i] = data[i];
        }
        return;
      } else {
        ushort num_bytes = args.dst_start_distance;
        if (num_bytes == 3) {
          ((device uchar*)dst_base)[dst_offset * 4 + 3] = data[3];
          return;
        } else if (num_bytes >= 1) {
          if (num_bytes == 1) {
            ((device uchar*)dst_base)[dst_offset * 4 + 1] = data[1];
          }
          ((device uchar2*)dst_base)[dst_offset * 2 + 1] = data.zw;
          return;
        } else {
          // Continue to the final statement of the entire function; coalesce
          // this write with other threads.
        }
      }
    } else if (dst_offset == args.dst_end) {
      // Edge case where `dst_end` occurs before the last simdgroup.
      write_end(data, dst_offset, dst_base, args.dst_end_distance);
      return;
    }
  }
  
  // Write aligned chunk to memory.
  dst_base[dst_offset] = data;
}

// `src_base` and `dst_base` are rounded to 1024-bit chunks.
// Allowed arithmetic intensity: 5308/(408/4/2) = 104 cycles/word copied.
kernel void copyBufferEdgeCases
 (
  device uchar4 *src_base [[buffer(0)]],
  device uchar4 *dst_base [[buffer(1)]],
  constant Arguments &args [[buffer(2)]],
  
  // Threadgroup size assumed to be 256.
  // Simdgroup size assumed to be 32.
  uint tid [[thread_position_in_grid]],
  uint tgid [[threadgroup_position_in_grid]],
  ushort thread_index [[thread_position_in_threadgroup]],
  ushort simd_index [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  uint src_index;
  if (tid <= args.src_end) {
    // Read through the rounded-down end, inclusive.
    src_index = tid;
  } else {
    // Don't read outside page boundaries; you'll cause a soft fault.
    src_index = 0;
  }
  uchar4 src_data = src_base[src_index];
  
  // Transfer data to threadgroup memory, then perform some computations before
  // synchronizing (reduces simd latency).
  threadgroup uchar4 transferred_words[256];
  transferred_words[thread_index] = src_data;
  
  short hi_reach_back = short(thread_index) - args.word_realignment;
  if (args.word_realignment < 0) {
    // Word realignment is negative, so we would reach forward.
    // We don't want that, so we correct it by looking another block backward.
    hi_reach_back -= 32;
    
    // What would happen:
    // First simd doesn't read garbage (0+X to 32+X).
    // Second simd doesn't read garbage (32+X to 64+X).
    // Last simd reads garbage (224+X to 256+X).
    // Later, the last simd reads ultimate garbage: (256+X to 288+X).
    //
    // To fix this, all simds are tempered back by 32.
    // First simd reads garbage (-32+X to 0+X).
    // Second simd doesn't read garbage (0+X to 32+X).
    // Last simd doesn't read garbage (192+X to 224+X).
    // Later, the last simd reads normal garbage: (224+X to 256+X).
  } else {
    // With positive realignment, you want to shift up. New values come from an
    // even lower address. There's a good chance the value is garbage. `hi` is a
    // conservative reach backward, `lo` is a farther stretch.
    // hi_reach_back not changed
    
    // First simd reads garbage: (0-X to 32-X).
    // Second simd doesn't read garbage (32-X to 64-X).
    // Last simd doesn't read garbage (224-X to 256-X).
    // Later, the last simd reads garbage: (256-X to 288-X).
  }
  
  short lo_reach_back = hi_reach_back - 1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uchar4 lo = transferred_words[lo_reach_back];
  uchar4 hi = transferred_words[hi_reach_back];
  uchar4 data = combine_data(lo, hi, args.bytes_after_word_realignment);
  
  // We now have <=256 words of realigned memory, ready to write masked.
  // We cannot return early from the `copyBufferEdgeCases` function. A thread
  // might exist in SIMD 7, and equal the starting word. Later on, it might
  // have to write the 32nd word. Therefore we keep the `return` keyword inside
  // the body of `write_start`.
  write_middle(data, tid, dst_base, args, thread_index);
  
  // Realignment creates another chunk; only the last simd handles this.
  if (simd_index == 7) {
    uchar4 lo, hi;
    if (args.word_realignment >= 0) {
      // Threadgroup shuffle up -> SIMD shuffle down
      // If realignment is 0-0.75 words, `hi` is garbage and `lo` is not, for
      // the first thread.
      lo = simd_shuffle_down(src_data, 31 - args.word_realignment); // 31 - 0
      hi = simd_shuffle_down(src_data, 32 - args.word_realignment); // 32 - 0
    } else {
      // Threadgroup shuffle down -> SIMD shuffle up
      // If realignment is -32-31.25 words, `hi` is garbage and `lo` is not, for
      // the first thread.
      lo = simd_shuffle_down(src_data, -1 - args.word_realignment); // -1 + 32
      hi = simd_shuffle_down(src_data,  0 - args.word_realignment); //  0 + 32
    }
    uchar4 data = combine_data(lo, hi, args.bytes_after_word_realignment);
    
    // If dst_end % 128 and dst_start % 128 overlap, what happens?
    // A = dst_end = (dst_end_bytes) / 4
    // B = dst_start = (dst_start_bytes % 128) / 4
    // B = lane_id
    // tid % 32 = lane_id
    // tid = (dst_start_bytes / 4) + 256 * X
    // X = 1 just for simplicity, can be any integer
    //
    // C = dst_end_distance = dst_end_bytes % 4
    // D = dst_start_distance = dst_start_bytes % 4
    // span 1 = absolute byte range dictated by `dst_end(_words)`
    // span 2 = absolute byte range dictated by `dst_start(_words)`
    // A = B -> overlap
    //
    // 1: C = 0, D = 0, span 1 = (4A..<4A), span 2 = (4B..<4B)
    // 2: C = 0, D = 1, span 1 = (4A..<4A), span 2 = (4B..<4B+1)
    // 3: C = 0, D = 2, span 1 = (4A..<4A), span 2 = (4B..<4B+2)
    // 4: C = 0, D = 3, span 1 = (4A..<4A), span 2 = (4B..<4B+3)
    // 5: C = 1, D = 0, span 1 = (4A..<4A+1), span 2 = (4B..<4B)
    // 6: C = 1, D = 1, span 1 = (4A..<4A+1), span 2 = (4B..<4B+1)
    // 7: C = 1, D = 2, span 1 = (4A..<4A+1), span 2 = (4B..<4B+2)
    // 8: C = 1, D = 3, span 1 = (4A..<4A+1), span 2 = (4B..<4B+3)
    // At case 5, C incorrectly dominates.
    //
    // We didn't read enough bytes to properly fill the additional bytes of C.
    // The next threadgroup will write them; their first thread from realignment
    // should handle the edge case.
    
    uint virtual_tid = tid + 32;
    if (virtual_tid >= args.dst_end) {
      if (virtual_tid == args.dst_end) {
        // Test for the edge case explained above.
        ushort num_bytes = args.dst_end_distance;
        if (thread_index == args.dst_start) {
          num_bytes = min(num_bytes, args.dst_start_distance);
        }
        
        // Masked-write the entire blit's final bytes.
        write_end(data, virtual_tid, dst_base, num_bytes);
      }
    } else {
      // This cannot be <= args.dst_start, because of how we dispatch threads.
      if (thread_index == args.dst_start) {
        // Masked-write this threadgroup's final bytes.
        write_end(data, virtual_tid, dst_base, args.dst_end_distance);
      } else {
        // Write aligned chunk to memory.
        dst_base[virtual_tid] = data;
      }
    }
  }
}
