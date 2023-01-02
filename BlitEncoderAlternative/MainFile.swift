//
//  MainFile.swift
//  BlitEncoderAlternative
//
//  Created by Philip Turner on 12/30/22.
//

import Metal
import QuartzCore

func mainFunc() {
  validationTest()
//  originalTest()
}

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

// Thoroughly validates a wide range of edge cases, and times them. Compares to
// the blit encoder. It can also disable the accelerated "copyBufferAligned"
// fast-path.
func validationTest() {
  // Note that Metal Frame Capture and API validation are probably enabled.
  let testingPerformance = true
  let bufferSize = 64 * 1024 * 1024
  let numTrials = 15
  let numActiveThreads = 32 * 7
  let transactionBytes = 128
  let transactionWords = transactionBytes / 4
  let usingThreadgroups = true
  
  let usingAlignedFastPath = false
  let usingBlitEncoder = false
  let forceUseCustomOffsets = false
  let generateRepeatingPattern = false
  let printArguments = false
  
  let isAligned = false
  let srcOffset = isAligned ? 128 : 2
  let dstOffset = isAligned ? 128 : 289
  var customBytesCopied: Int? = nil
  
  // 156 - 195
  
  // Size = 256 KB A
  // Blit:            48 GB/s
  // Fast path:       44 GB/s
  // 224/256 threads: 32 GB/s
  // 256/288 threads: 32 GB/s
  // 224/256 64 B:    27 GB/s
  // 256/288 64 B:    27 GB/s
  
  // Size = 256 KB U
  // Blit:            53 GB/s
  // 224/256 threads: 32 GB/s
  // 256/288 threads: 32 GB/s
  // 224/256 64 B:    27 GB/s
  // 256/288 64 B:    28 GB/s
  
  // Size = 64 MB A
  // Blit:                  324 GB/s
  // Fast Path:             325 GB/s
  // 224/256 threads: 165 - 206 GB/s ???
  // 256/288 threads: 159 - 198 GB/s ???
  // 224/256 64 B:    163 - 193 GB/s ???
  // 256/288 64 B:    157 - 188 GB/s ???
  
  // Size = 64 MB U
  // Blit:            186 - 219 GB/s
  // 224/256 threads: 156 - 195 GB/s
  // 256/288 threads: 155 - 194 GB/s
  // 224/256 64 B:    159 - 190 GB/s ???
  // 256/288 64 B:    153 - 188 GB/s ???
  
  // Only used in performance testing mode.
  #if true
  
  
  #else
  let srcOffset = Int.random(in: 0..<16384)
  let dstOffset = Int.random(in: 0..<16384)
  var customBytesCopied: Int? = bufferSize - max(srcOffset, dstOffset)
  customBytesCopied! -= Int.random(in: 0..<16384)
  if testingPerformance {
    print("Source offset: \(srcOffset)")
    print("Destination offset: \(dstOffset)")
    print("Bytes copied: \(customBytesCopied!)")
  }
  #endif
  
  let testStart = CACurrentMediaTime()
  let device = MTLCreateSystemDefaultDevice()!
  let commandQueue = device.makeCommandQueue()!
  let library = device.makeDefaultLibrary()!
  
  var pipelines: [String: MTLComputePipelineState] = [:]
  for name in ["copyBufferAligned", "copyBufferEdgeCases"] {
    let constants = MTLFunctionConstantValues()
    var use_shader_validation: Bool = false
    constants.setConstantValue(&use_shader_validation, type: .bool, index: 0)
    
    let desc = MTLComputePipelineDescriptor()
    desc.computeFunction = try! library.makeFunction(
      name: name, constantValues: constants)
    desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
    
    let pipeline = try! device.makeComputePipelineState(
      descriptor: desc, options: [], reflection: nil)
    precondition(pipeline.maxTotalThreadsPerThreadgroup == 1024)
    pipelines[name] = pipeline
  }
  
  var bufferSrc = device.makeBuffer(length: bufferSize)!
  var bufferDst = device.makeBuffer(length: bufferSize)!
  var minCopyTime: Double = 1_000
  var maxBandwidth: Double = 0
  
  let referenceSrc = malloc(bufferSize)!
  let referenceDst = malloc(bufferSize)!
  defer {
    free(referenceSrc)
    free(referenceDst)
  }
  if !testingPerformance {
    // Only generate random numbers once, minimizing time wasted on the CPU.
    if generateRepeatingPattern {
      let referenceSrcCasted = referenceSrc.assumingMemoryBound(to: UInt8.self)
      for i in 0..<bufferSize {
        referenceSrcCasted[i] = UInt8(truncatingIfNeeded: i % 256)
      }
    } else {
      let referenceSrcCasted = referenceSrc.assumingMemoryBound(to: Int.self)
      for i in 0..<bufferSize / 8 {
        // Logical or this with a mask, so that no single byte equals zero.
        let randomInteger = Int.random(in: 0..<Int.max)
        let mask = 0x0101010101010101
        referenceSrcCasted[i] = randomInteger | mask
      }
    }
  }
  
outer:
  for _ in 0..<numTrials {
    defer {
      swap(&bufferSrc, &bufferDst)
    }
    
    var thisSrcOffset: Int
    var thisDstOffset: Int
    var thisBytesCopied: Int
    if testingPerformance || forceUseCustomOffsets {
      thisSrcOffset = srcOffset
      thisDstOffset = dstOffset
      if let customBytesCopied = customBytesCopied {
        thisBytesCopied = customBytesCopied
      } else {
        thisBytesCopied = bufferSize - max(thisSrcOffset, thisDstOffset)
      }
    } else {
      thisSrcOffset = Int.random(in: 0..<16384)
      thisDstOffset = Int.random(in: 0..<16384)
      thisBytesCopied = bufferSize - max(thisSrcOffset, thisDstOffset)
      thisBytesCopied -= Int.random(in: 0..<16384)
    }
    precondition(thisSrcOffset >= 0)
    precondition(thisDstOffset >= 0)
    precondition(thisBytesCopied > 0)
    
    if !testingPerformance {
      // Clear the destination buffer and reference destination.
      memset(referenceDst, 0, bufferSize)
      memset(bufferDst.contents(), 0, bufferSize)
      
      // Initialize the source buffer and simulate a correct memcpy.
      memcpy(bufferSrc.contents(), referenceSrc, bufferSize)
      memcpy(
        referenceDst + thisDstOffset, referenceSrc + thisSrcOffset,
        thisBytesCopied)
    }
    
    // Encode GPU work.
    let commandBuffer = commandQueue.makeCommandBuffer()!
    if usingBlitEncoder {
      let encoder = commandBuffer.makeBlitCommandEncoder()!
      defer { encoder.endEncoding() }
      encoder.copy(
        from: bufferSrc, sourceOffset: thisSrcOffset, to: bufferDst,
        destinationOffset: thisDstOffset, size: thisBytesCopied)
    } else {
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      defer { encoder.endEncoding() }
      
      var useFastPath = false
      if usingAlignedFastPath {
        if thisSrcOffset % 64 == 0 && thisDstOffset % 64 == 0 {
          if thisBytesCopied % 4 == 0 {
            useFastPath = true
          }
        }
      }
      
      if useFastPath {
        let pipeline = pipelines["copyBufferAligned"]!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferSrc, offset: thisSrcOffset, index: 0)
        encoder.setBuffer(bufferDst, offset: thisDstOffset, index: 1)
        encoder.dispatchThreads(
          MTLSizeMake(thisBytesCopied / 4, 1, 1),
          threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
      } else {
        let pipeline = pipelines["copyBufferEdgeCases"]!
        encoder.setComputePipelineState(pipeline)
        
        // Encode buffer bindings.
        let srcTrueVA = bufferSrc.gpuAddress + UInt64(thisSrcOffset)
        let dstTrueVA = bufferDst.gpuAddress + UInt64(thisDstOffset)
        let src_base = srcTrueVA & ~UInt64(transactionBytes - 1)
        let dst_base = dstTrueVA & ~UInt64(transactionBytes - 1)
        let srcBaseOffset = Int(src_base - bufferSrc.gpuAddress)
        let dstBaseOffset = Int(dst_base - bufferDst.gpuAddress)
        encoder.setBuffer(bufferSrc, offset: srcBaseOffset, index: 0)
        encoder.setBuffer(bufferDst, offset: dstBaseOffset, index: 1)
        
        precondition(bufferSrc.gpuAddress % UInt64(transactionBytes) == 0)
        precondition(bufferDst.gpuAddress % UInt64(transactionBytes) == 0)
        precondition(src_base % UInt64(transactionBytes) == 0)
        precondition(dst_base % UInt64(transactionBytes) == 0)
        
        // Encode dispatch arguments.
        struct Arguments {
          var src_start: UInt32 = 0
          var dst_start: UInt32 = 0
          var src_end: UInt32 = 0
          var dst_end: UInt32 = 0
          
          var src_start_distance: UInt16 = 0
          var dst_start_distance: UInt16 = 0
          var src_end_distance: UInt16 = 0
          var dst_end_distance: UInt16 = 0
          
          var word_realignment: Int16 = 0
          var bytes_after_word_realignment: UInt16 = 0
          var word_rounded_realignment: Int16 = 0
        }
        var arguments = Arguments()
        let srcEndVA = srcTrueVA + UInt64(thisBytesCopied)
        let dstEndVA = dstTrueVA + UInt64(thisBytesCopied)
        
        // This is rounded down.
        let srcStartDelta = Int(srcTrueVA) - Int(src_base)
        let dstStartDelta = Int(dstTrueVA) - Int(dst_base)
        arguments.src_start = .init(srcStartDelta / 4)
        arguments.dst_start = .init(dstStartDelta / 4)
        
        // This is rounded up.
        let srcEndDelta = Int(srcEndVA) - Int(src_base)
        let dstEndDelta = Int(dstEndVA) - Int(dst_base)
        arguments.src_end = .init(srcEndDelta / 4)
        arguments.dst_end = .init(dstEndDelta / 4)
        
        arguments.src_start_distance =
          .init(srcStartDelta - 4 * Int(arguments.src_start))
        arguments.dst_start_distance =
          .init(dstStartDelta - 4 * Int(arguments.dst_start))
        arguments.src_end_distance =
          .init(srcEndDelta - 4 * Int(arguments.src_end))
        arguments.dst_end_distance =
          .init(dstEndDelta - 4 * Int(arguments.dst_end))
        
        let absolute_realignment = Int(srcTrueVA % UInt64(transactionBytes)) - Int(dstTrueVA % UInt64(transactionBytes))
        let word_realignment = (absolute_realignment + transactionBytes) / 4 - transactionBytes / 4
        arguments.word_realignment = Int16(word_realignment)
        arguments.bytes_after_word_realignment =
          .init(absolute_realignment - 4 * word_realignment)
        arguments.word_rounded_realignment =
          arguments.word_realignment & ~Int16(transactionWords - 1)
        precondition(word_realignment * 4 <= absolute_realignment)
        
        let argumentsSize = MemoryLayout<Arguments>.stride
        precondition(argumentsSize == 32)
        encoder.setBytes(&arguments, length: argumentsSize, index: 2)
        if printArguments {
          print(arguments)
        }
        
        // Dispatch correct amount of threads.
        let _thisBytesCopied = UInt64(thisBytesCopied)
        var srcUpperChunkBoundary = srcTrueVA + _thisBytesCopied - 1
        var dstUpperChunkBoundary = dstTrueVA + _thisBytesCopied - 1
        srcUpperChunkBoundary = srcUpperChunkBoundary & ~(UInt64(transactionBytes) - 1) + UInt64(transactionBytes)
        dstUpperChunkBoundary = dstUpperChunkBoundary & ~(UInt64(transactionBytes) - 1) + UInt64(transactionBytes)
        
        let dstScannedBytes = Int(dstUpperChunkBoundary - dst_base)
        let numWords = dstScannedBytes / 4
        let numChunks = numWords / transactionWords
        if usingThreadgroups {
          let numActiveSimds = numActiveThreads / 32
          let numChunksRoundedUp =
            (numChunks + numActiveSimds - 1) / numActiveSimds * numActiveSimds
          let numThreadgroups = numChunksRoundedUp / numActiveSimds
          encoder.dispatchThreadgroups(
            MTLSizeMake(numThreadgroups, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(numActiveThreads + 32, 1, 1))
        } else {
          let numThreadgroups = (numChunks + 8 - 1) / 8
          encoder.dispatchThreadgroups(
            MTLSizeMake(numThreadgroups, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
        }
        
      }
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    if testingPerformance {
      // Update `minCopyTime` and `maxBandwidth`.
      // Bandwidth should report the actual bytes transferred.
      let time = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
      let bandwidth = Double(2 * thisBytesCopied) / time
      if bandwidth > maxBandwidth {
        maxBandwidth = bandwidth
        minCopyTime = time
      }
    } else {
      // Validate that results are correct. Otherwise, print how it failed.
      let error = memcmp(referenceDst, bufferDst.contents(), bufferSize)
      if error != 0 {
        print("Failed with:")
        print("Source offset: \(thisSrcOffset)")
        print("Destination offset: \(thisDstOffset)")
        print("Bytes copied: \(thisBytesCopied)")
        
        let referenceDstCasted = referenceDst
          .assumingMemoryBound(to: SIMD4<UInt8>.self)
        let bufferDstCasted = bufferDst
          .contents().assumingMemoryBound(to: SIMD4<UInt8>.self)
        var numFailures = 0
        
        for i in 0..<bufferSize / 4 {
          let referenceElement = referenceDstCasted[i]
          let bufferElement = bufferDstCasted[i]
          if any(referenceElement .!= bufferElement) {
            let dataString = "ref [\(referenceElement[0]), \(referenceElement[1]), \(referenceElement[2]), \(referenceElement[3])] != buf [\(bufferElement[0]), \(bufferElement[1]), \(bufferElement[2]), \(bufferElement[3])]"
            print("Word \(i) (bytes \(i * 4)...\(i * 4 + 3)): \(dataString)")
            numFailures += 1
          }
          if numFailures >= 5 {
            print("Too many failures, quitting now.")
            break outer
          }
        }
      }
    }
  }
  
  if testingPerformance {
    // Print copying time (us) and bandwidth.
    print("Copy Time: \(Int(minCopyTime * 1e6)) us")
    print("Bandwidth: \(Int(maxBandwidth / 1e9)) GB/s")
  }
  
  let testEnd = CACurrentMediaTime()
  let seconds = String(format: "%.3f", testEnd - testStart)
  print("Elapsed time: \(seconds) s")
}

// Original testing function.
func originalTest() {
  // Constants to change program execution.
  let usingBlitEncoder = false
  let usingAlignedBlit = true
  let bufferSize = 256 * 1024
  let numTrials = 15
  let byteOffset1 = 0
  let byteOffset2 = 0
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
