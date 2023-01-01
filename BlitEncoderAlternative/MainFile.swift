//
//  MainFile.swift
//  BlitEncoderAlternative
//
//  Created by Philip Turner on 12/30/22.
//

import Metal
import QuartzCore

// Bandwidth variation with different parameters.
// - Reference implementation 1: blit encoder, aligned
// - Reference implementation 2: blit encoder, src_offset=2, dst_offset=289
// - Custom implementation: trivial compute shader, aligned
//
// Methodology:
// - Taking the mode, not absolute maximum, of bandwidth.
// - Time recorded for aligned blit encoder.
// - Blit encoders have an unfair advantage, taking 2 microseconds less overall.
// - Metal API validation and frame capture turned off.
//
//   Size | Time | RefA | RefU |  1 B |  2 B |  4 B |  8 B | 16 B | 32 B |
//   1 GB | 5712 |  375 |  330 |  342 |  376 |  378 |  377 |  375 |  375 |
// 256 MB | 1446 |  375 |  330 |  341 |  375 |  374 |  376 |  374 |  375 |
//  64 MB |  424 |  316 |  186 |  178 |  295 |  324 |  319 |  311 |  312 |
//  16 MB |  114 |  293 |  182 |  167 |  273 |  298 |  290 |  285 |  287 |
//   4 MB |   38 |  218 |  154 |  137 |  205 |  216 |  209 |  200 |  198 |
//   1 MB |   16 |  125 |   96 |   83 |  106 |  106 |  103 |  103 |  100 |
// 256 KB |   10 |   48 |   53 |   39 |   43 |   44 |   43 |   43 |   42 |
//  64 KB |    8 |   15 |   20 |   13 |   13 |   13 |   13 |   12 |   11 |
//  16 KB |    7 |    4 |    5 |    3 |    3 |    3 |    3 |    3 |    3 |

func mainFunc() {
  // TODO: Validate that this copies correctly and without bugs.
  // TODO: Gather wide range of data about SIMD-only blit.
  // TODO: Make threadgroup enhancement reversible (compiler macro).
  
  // Constants to change program execution.
  let usingBlitEncoder = true
  let usingAlignedBlit = true
  let bufferSize = 16 * 1024 * 1024
  let numTrials = 15
  let byteOffset1 = 12
  let byteOffset2 = 17
  let byteOffsetMax = max(byteOffset1, byteOffset2)
  
  // Initialize basic resources.
  let device = MTLCreateSystemDefaultDevice()!
  let commandQueue = device.makeCommandQueue()!
  let library = device.makeDefaultLibrary()!
  
  let computePipelineDesc = MTLComputePipelineDescriptor()
  computePipelineDesc.computeFunction =
    library.makeFunction(name: "copyBufferAligned")!
  computePipelineDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
  let pipeline = try! device.makeComputePipelineState(
    descriptor: computePipelineDesc, options: [], reflection: nil)
  
  // Initialize buffers for copying. These will be swapped, so they are "var"
  // variables in Swift.
  var buffer1 = device.makeBuffer(length: bufferSize)!
  var buffer2 = device.makeBuffer(length: bufferSize)!
  print(buffer1.gpuAddress % 128)
  print(buffer2.gpuAddress % 128)
  
  // Currently measured in seconds.
  var minCopyTime: Double = 1_000
  
  for _ in 0..<numTrials {
    defer {
      swap(&buffer1, &buffer2)
    }
    
    let commandBuffer = commandQueue.makeCommandBuffer()!
    if usingBlitEncoder {
      let encoder = commandBuffer.makeBlitCommandEncoder()!
      encoder.copy(
        from: buffer1,
        sourceOffset: byteOffset1 + (usingAlignedBlit ? 0 : 2),
        to: buffer2,
        destinationOffset: byteOffset2 + (usingAlignedBlit ? 0 : 289),
        size: bufferSize - byteOffsetMax - (usingAlignedBlit ? 0 : 289))
      encoder.endEncoding()
    } else {
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(buffer1, offset: byteOffset1, index: 0)
      encoder.setBuffer(buffer2, offset: byteOffset2, index: 1)
      
      encoder.dispatchThreads(
        MTLSizeMake((bufferSize - byteOffsetMax) / 4, 1, 1),
        threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
      encoder.endEncoding()
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let copyTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    minCopyTime = min(minCopyTime, copyTime)
  }
  
  // Print copying time (us) and bandwidth.
  let bytesTransferred = 2 * bufferSize
  let bandwidth = Double(bytesTransferred) / minCopyTime
  print("Copy Time: \(Int(minCopyTime * 1e6)) us")
  print("Bandwidth: \(Int(bandwidth / 1e9)) GB/s")
}
