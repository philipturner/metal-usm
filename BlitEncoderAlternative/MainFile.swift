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

// NOTE: hipSYCL must break up large memcpy/memset calls into 2 GB chunks. This
// removes worries about integer overflows and improves responsiveness - max 10
// milliseconds on macOS, 80 milliseconds on iOS.

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

// New statistics, after creating the optimized blit encoder alternative.
// The overhead of it being a compute pipeline (rather than a presumably
// built-in function) may harm device-side sequential throughput (+3 us). This
// is amortized with multiple commands/encoder, and the custom version becomes
// faster at 4 commands/encoder.
//
// A = (128, 128)-byte offset
// B = ( 16,  64)-byte offset
// C = (  4,  32)-byte offset
// D = (  2, 289)-byte offset
// E = ( 55, 289)-byte offset
//
//   Size | RefA : CusA | RefB : CusB | RefC : CusC | RefD : CusD | RefE : CusE
// 256 MB |  375 :  375 |  376 :  376 |  376 :  376 |  329 :  376 |  376 :  376
//  64 MB |  325 :  325 |  321 :  322 |  315 :  313 |  219 :  310 |  302 :  309
//  16 MB |  300 :  299 |  298 :  297 |  299 :  284 |  182 :  279 |  294 :  278
//   4 MB |  223 :  217 |  221 :  216 |  230 :  206 |  154 :  193 |  229 :  193
//   1 MB |  126 :  105 |  125 :  106 |  125 :  103 |   96 :   89 |  125 :   89
// 256 KB |   48 :   44 |   48 :   44 |   62 :   44 |   53 :   35 |   59 :   35
//
// Reference reuses the same blit encoder for every command. Custom reuses the
// same compute encoder, with the custom kernel. Although 4 KB statistics are
// rounded off, CusA takes consistently less microseconds of copy time (with
// several repeats).
//
//           2 Repeats     4 Repeats     8 Repeats    32 Repeats    128 Repeats
//   Size | RefA : CusA | RefA : CusA | RefA : CusA | RefA : CusA | RefA : CusA
//   4 MB |  228 :  231 |  232 :  240 |  233 :  241 |  312 :  281 |  365 :  366
//   1 MB |  132 :  120 |  136 :  130 |  137 :  136 |  139 :  136 |  229 :  217
// 256 KB |   52 :   55 |   54 :   64 |   54 :   69 |   55 :   73 |   65 :   67
//  64 KB |   17 :   17 |   18 :   21 |   18 :   24 |   18 :   26 |   18 :   27
//  16 KB |    5 :    4 |    5 :    5 |    5 :    6 |    5 :    7 |    5 :    7
//   4 KB |    1 :    1 |    1 :    1 |    1 :    1 |    1 :    1 |    1 :    1

// Thoroughly validates a wide range of edge cases, and times them. Compares to
// the blit encoder. It can also disable the "copyBufferAligned" fast-path.
func validationTest() {
  let testingPerformance = true
  let bufferSize = testingPerformance ? 128 * 1024 * 1024 : 256 * 1024
  let numTrials = 15
  let numRepetitions = 1
  let doingMemset = true
  var pattern: [UInt8] = [234, 77, 90, 99]
  
  let usingBlitEncoder = false
  let usingAlignedFastPath = true
  let forceUseCustomOffsets = false
  let generateRepeatingPattern = false
  let printArguments = true
  
  // Only used in performance testing mode.
  #if true
  let isAligned = true
  let srcOffset = isAligned ? 0 : 128
  let dstOffset = isAligned ? 1 : 128
  var customBytesCopied: Int? = nil
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
  
  var copyAlignedPipeline: MTLComputePipelineState
  var fillAlignedPipeline: MTLComputePipelineState
  var copyEdge0Pipeline: MTLComputePipelineState
  var copyEdge1Pipeline: MTLComputePipelineState
  var fillEdge0Pipeline: MTLComputePipelineState
  var fillEdge1Pipeline: MTLComputePipelineState
  do {
    let copyAlignedFunction = library.makeFunction(name: "copyBufferAligned")!
    let fillAlignedFunction = library.makeFunction(name: "fillBufferAligned")!
    copyAlignedPipeline = try! device.makeComputePipelineState(
      function: copyAlignedFunction)
    fillAlignedPipeline = try! device.makeComputePipelineState(
      function: fillAlignedFunction)
    
    // Copy buffer edge 0
    let constants = MTLFunctionConstantValues()
    var use_shader_validation: Bool = false
    constants.setConstantValue(&use_shader_validation, type: .bool, index: 0)
    
    var byte_realignment_is_zero: Bool = true
    constants.setConstantValue(&byte_realignment_is_zero, type: .bool, index: 1)
    let copyEdge0Function = try! library.makeFunction(
      name: "copyBufferEdgeCases", constantValues: constants)
    copyEdge0Pipeline = try! device.makeComputePipelineState(
      function: copyEdge0Function)
    
    // Copy buffer edge 1
    byte_realignment_is_zero = false
    constants.setConstantValue(&byte_realignment_is_zero, type: .bool, index: 1)
    let copyEdge1Function = try! library.makeFunction(
      name: "copyBufferEdgeCases", constantValues: constants)
    copyEdge1Pipeline = try! device.makeComputePipelineState(
      function: copyEdge1Function)
    
    // Fill buffer edge 0
    var pattern_small_and_divisible_4: Bool = true
    constants.setConstantValue(
      &pattern_small_and_divisible_4, type: .bool, index: 2)
    let fillEdge0Function = try! library.makeFunction(
      name: "fillBufferEdgeCases", constantValues: constants)
    fillEdge0Pipeline = try! device.makeComputePipelineState(
      function: fillEdge0Function)
    
    // Fill buffer edge 1
    pattern_small_and_divisible_4 = false
    constants.setConstantValue(
      &pattern_small_and_divisible_4, type: .bool, index: 2)
    let fillEdge1Function = try! library.makeFunction(
      name: "fillBufferEdgeCases", constantValues: constants)
    fillEdge1Pipeline = try! device.makeComputePipelineState(
      function: fillEdge1Function)
  }
  
  func makeBuffer(_ index: Int) -> MTLBuffer {
    return device.makeBuffer(length: bufferSize)!
  }
  
  var buffersSrc = (0..<numRepetitions).map(makeBuffer)
  var buffersDst = (0..<numRepetitions).map(makeBuffer)
  var minCopyTime: Double = 1_000
  var maxBandwidth: Double = 0
  
  let referenceSrc = malloc(bufferSize)!
  let referenceDst = malloc(bufferSize)!
  defer {
    free(referenceSrc)
    free(referenceDst)
  }
  if !testingPerformance {
    if doingMemset {
      if pattern.count == 1 {
        let value = Int32(pattern[0])
        memset(referenceSrc, value, bufferSize)
      } else if pattern.count == 2 {
        var value = SIMD4<UInt8>(
          pattern[0], pattern[1], pattern[0], pattern[1])
        memset_pattern4(referenceSrc, &value, bufferSize)
      } else {
        for i in 0..<bufferSize / pattern.count {
          let offset = i * pattern.count
          memcpy(referenceSrc + offset, &pattern, pattern.count)
        }
        for i in 0..<bufferSize % pattern.count {
          let offset = i + (bufferSize / pattern.count) * pattern.count
          memcpy(referenceSrc + offset, &pattern[i], 1)
        }
      }
    } else if generateRepeatingPattern {
      // Only generate random numbers once, minimizing time wasted on the CPU.
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
      swap(&buffersSrc, &buffersDst)
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
      if doingMemset {
        thisSrcOffset = 0
      } else {
        thisSrcOffset = Int.random(in: 0..<16384)
      }
      thisDstOffset = Int.random(in: 0..<16384)
      thisBytesCopied = bufferSize - max(thisSrcOffset, thisDstOffset)
      thisBytesCopied -= Int.random(in: 0..<16384)
    }
    if doingMemset {
      precondition(
        thisSrcOffset == 0, "Memset doesn't support nonzero source offsets.")
    }
    precondition(thisSrcOffset >= 0)
    precondition(thisDstOffset >= 0)
    precondition(thisBytesCopied > 0)
    
    if !testingPerformance {
      // Clear the destination buffer and reference destination.
      memset(referenceDst, 0, bufferSize)
      memset(buffersDst[0].contents(), 0, bufferSize)
      
      // Initialize the source buffer and simulate a correct memcpy.
      memcpy(buffersSrc[0].contents(), referenceSrc, bufferSize)
      memcpy(
        referenceDst + thisDstOffset, referenceSrc + thisSrcOffset,
        thisBytesCopied)
    }
    
    // Encode GPU work.
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let amountOfWork = testingPerformance ? numRepetitions : 1
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    if usingBlitEncoder {
      encoder.endEncoding()
    }
    var _blitEncoder: MTLBlitCommandEncoder?
    for bufferID in 0..<amountOfWork {
      let bufferSrc = buffersSrc[bufferID]
      let bufferDst = buffersDst[bufferID]
      if usingBlitEncoder {
        if _blitEncoder == nil {
          _blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        }
        let encoder = _blitEncoder!
        if doingMemset {
          guard pattern.count == 1 else {
            fatalError("Blit encoder doesn't support pattern size > 1.")
          }
          let range = thisDstOffset..<thisDstOffset + thisBytesCopied
          encoder.fill(buffer: bufferDst, range: range, value: pattern[0])
        } else {
          encoder.copy(
            from: bufferSrc, sourceOffset: thisSrcOffset, to: bufferDst,
            destinationOffset: thisDstOffset, size: thisBytesCopied)
        }
      } else if doingMemset {
        let dstAddress = bufferDst.gpuAddress + UInt64(thisDstOffset)
        if usingAlignedFastPath,
           pattern.count.nonzeroBitCount == 1,
           pattern.count <= 32,
           dstAddress % 4 == 0,
           thisBytesCopied % 4 == 0 {
          var _pattern = pattern
          while _pattern.count < 32 {
            _pattern += _pattern
          }
          encoder.setComputePipelineState(fillAlignedPipeline)
          encoder.setBytes(&_pattern, length: _pattern.count, index: 0)
          encoder.setBuffer(bufferDst, offset: thisDstOffset, index: 1)
          encoder.dispatchThreads(
            MTLSizeMake(thisBytesCopied / 4, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
        } else {
          var _pattern = pattern
          while true {
            if _pattern.count * 2 > 256 * 4 {
              break
            } else {
              _pattern += _pattern
            }
          }
          
          let dstTrueVA = bufferDst.gpuAddress + UInt64(thisDstOffset)
          let dst_base = dstTrueVA & ~(64 - 1)
          let dstBaseOffset = Int(dst_base - bufferDst.gpuAddress)
          precondition(bufferDst.gpuAddress % 64 == 0)
          precondition(dst_base % 64 == 0)
          
          struct FillArguments {
            var dst_start: UInt32 = 0
            var dst_end: UInt32 = 0
            var dst_start_distance: UInt16 = 0
            var dst_end_distance: UInt16 = 0
            
            var pattern_size_words: UInt32 = 0
            var pattern_size_bytes: UInt32 = 0
            var active_threads: UInt16 = 0
            
            var pattern_small_and_divisible_4: Bool = false
            var pattern_divisible_4: Bool = false
          }
          var arguments = FillArguments()
          let dstEndVA = dstTrueVA + UInt64(thisBytesCopied)
          
          let dstStartDelta = Int(dstTrueVA) - Int(dst_base)
          let dstEndDelta = Int(dstEndVA) - Int(dst_base)
          arguments.dst_start = .init(dstStartDelta / 4)
          arguments.dst_end = .init(dstEndDelta / 4)
          arguments.dst_start_distance =
            .init(dstStartDelta - 4 * Int(arguments.dst_start))
          arguments.dst_end_distance =
            .init(dstEndDelta - 4 * Int(arguments.dst_end))
          
          let pattern_size = _pattern.count
          arguments.pattern_size_words = .init(pattern_size / 4)
          arguments.pattern_size_bytes = .init(pattern_size)
          arguments.active_threads = 256
          
          if pattern_size % 4 == 0 {
            if pattern_size <= 1024 {
              arguments.pattern_small_and_divisible_4 = true;
              arguments.active_threads = .init(arguments.pattern_size_words)
            } else {
              arguments.pattern_divisible_4 = true;
            }
          }
          
          var is_fast_path = arguments.pattern_small_and_divisible_4
          is_fast_path = is_fast_path && (pattern_size == 1024)
          
          // Pad the pattern's start and end.
          let endPadding = (_pattern[0], _pattern[1], _pattern[2])
          let startPadding = _pattern[(pattern_size - dstStartDelta)...]
          _pattern = startPadding + _pattern
          if !is_fast_path {
            _pattern.append(endPadding.0)
            _pattern.append(endPadding.1)
            _pattern.append(endPadding.2)
          }
          
          if is_fast_path {
            encoder.setComputePipelineState(fillEdge0Pipeline)
          } else {
            encoder.setComputePipelineState(fillEdge1Pipeline)
          }
          
          let argumentsLength = MemoryLayout<FillArguments>.stride
          encoder.setBytes(&_pattern, length: _pattern.count, index: 0)
          encoder.setBuffer(bufferDst, offset: dstBaseOffset, index: 1)
          encoder.setBytes(&arguments, length: argumentsLength, index: 2)
          
          // Dispatch correct amount of threads.
          let _thisBytesCopied = UInt64(thisBytesCopied)
          var dstUpperWordBoundary = dstTrueVA + _thisBytesCopied - 1
          dstUpperWordBoundary = dstUpperWordBoundary & ~(4 - 1) + 4
          
          let dstScannedWords = Int(dstUpperWordBoundary - dst_base) / 4
          let numThreads = Int(arguments.active_threads)
          encoder.dispatchThreadgroups(
            MTLSizeMake((dstScannedWords + numThreads - 1) / numThreads, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
        }
      } else {
        var useFastPath = false
        if usingAlignedFastPath {
          var minDstOffset: Int
          if thisBytesCopied > 12 * 1024 * 1024 {
            minDstOffset = 64
          } else {
            // The edge-cases shader becomes slower between 8-16 MB.
            // This is 16x the magnitude of the L2 cache on M1, but something
            // plausable on the M1 Max. Coincidence?
            minDstOffset = 4
          }
          if thisSrcOffset % 4 == 0 && thisDstOffset % minDstOffset == 0 {
            if thisBytesCopied % 4 == 0 {
              useFastPath = true
            }
          }
        }
        
        if useFastPath {
          encoder.setComputePipelineState(copyAlignedPipeline)
          encoder.setBuffer(bufferSrc, offset: thisSrcOffset, index: 0)
          encoder.setBuffer(bufferDst, offset: thisDstOffset, index: 1)
          encoder.dispatchThreads(
            MTLSizeMake(thisBytesCopied / 4, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
        } else {
          let srcTrueVA = bufferSrc.gpuAddress + UInt64(thisSrcOffset)
          let dstTrueVA = bufferDst.gpuAddress + UInt64(thisDstOffset)
          let src_base = srcTrueVA & ~(64 - 1)
          let dst_base = dstTrueVA & ~(64 - 1)
          let srcBaseOffset = Int(src_base - bufferSrc.gpuAddress)
          let dstBaseOffset = Int(dst_base - bufferDst.gpuAddress)
          precondition(bufferSrc.gpuAddress % 64 == 0)
          precondition(bufferDst.gpuAddress % 64 == 0)
          precondition(src_base % 64 == 0)
          precondition(dst_base % 64 == 0)
          
          struct CopyArguments {
            var src_start: UInt32 = 0
            var dst_start: UInt32 = 0
            var src_end: UInt32 = 0
            var dst_end: UInt32 = 0
            
            var src_start_distance: UInt16 = 0
            var dst_start_distance: UInt16 = 0
            var src_end_distance: UInt16 = 0
            var dst_end_distance: UInt16 = 0
            
            var word_realignment: Int16 = 0
            var word_rounded_realignment: Int16 = 0
            var byte_realignment_lo_shift: UInt16 = 0
            var byte_realignment_hi_shift: UInt16 = 0
          }
          var arguments = CopyArguments()
          let srcEndVA = srcTrueVA + UInt64(thisBytesCopied)
          let dstEndVA = dstTrueVA + UInt64(thisBytesCopied)
          
          let srcStartDelta = Int(srcTrueVA) - Int(src_base)
          let dstStartDelta = Int(dstTrueVA) - Int(dst_base)
          arguments.src_start = .init(srcStartDelta / 4)
          arguments.dst_start = .init(dstStartDelta / 4)
          
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
          
          let absolute_realignment = Int(srcTrueVA % 64) - Int(dstTrueVA % 64)
          let word_realignment = (absolute_realignment + 64) / 4 - 16
          arguments.word_realignment = Int16(word_realignment)
          arguments.word_rounded_realignment =
          arguments.word_realignment & ~(16 - 1)
          precondition(word_realignment * 4 <= absolute_realignment)
          
          let byte_realignment = absolute_realignment - 4 * word_realignment
          arguments.byte_realignment_lo_shift = UInt16(byte_realignment * 8)
          arguments.byte_realignment_hi_shift = UInt16(32 - byte_realignment * 8)
          
          if byte_realignment == 0 {
            encoder.setComputePipelineState(copyEdge0Pipeline)
          } else {
            encoder.setComputePipelineState(copyEdge1Pipeline)
          }
          encoder.setBuffer(bufferSrc, offset: srcBaseOffset, index: 0)
          encoder.setBuffer(bufferDst, offset: dstBaseOffset, index: 1)
          
          let argumentsSize = MemoryLayout<CopyArguments>.stride
          precondition(argumentsSize == 32)
          encoder.setBytes(&arguments, length: argumentsSize, index: 2)
          if printArguments {
            print(arguments)
            print("byte_realignment: \(byte_realignment)")
          }
          
          // Dispatch correct amount of threads.
          let _thisBytesCopied = UInt64(thisBytesCopied)
          var srcUpperChunkBoundary = srcTrueVA + _thisBytesCopied - 1
          var dstUpperChunkBoundary = dstTrueVA + _thisBytesCopied - 1
          srcUpperChunkBoundary = srcUpperChunkBoundary & ~(64 - 1) + 64
          dstUpperChunkBoundary = dstUpperChunkBoundary & ~(64 - 1) + 64
          
          let dstScannedBytes = Int(dstUpperChunkBoundary - dst_base)
          let numWords = dstScannedBytes / 4
          let helperThreads = Int(arguments.word_realignment + 16) % 16 + 1
          encoder.dispatchThreadgroups(
            MTLSizeMake((numWords + 224 - 16) / 224, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(224 + helperThreads, 1, 1))
        }
      }
    }
    if !usingBlitEncoder {
      encoder.endEncoding()
    }
    _blitEncoder?.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    if testingPerformance {
      // Update `minCopyTime` and `maxBandwidth`.
      // Bandwidth should report the actual bytes transferred.
      let time = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
      var bytesCopied = numRepetitions * thisBytesCopied
      if !doingMemset {
        bytesCopied *= 2
      }
      let bandwidth = Double(bytesCopied) / time
      if bandwidth > maxBandwidth {
        maxBandwidth = bandwidth
        minCopyTime = time
      }
    } else {
      // Validate that results are correct. Otherwise, print how it failed.
      let error = memcmp(referenceDst, buffersDst[0].contents(), bufferSize)
      if error != 0 {
        print("Failed with:")
        print("Source offset: \(thisSrcOffset)")
        print("Destination offset: \(thisDstOffset)")
        print("Bytes copied: \(thisBytesCopied)")
        
        let referenceDstCasted = referenceDst
          .assumingMemoryBound(to: SIMD4<UInt8>.self)
        let bufferDstCasted = buffersDst[0]
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
