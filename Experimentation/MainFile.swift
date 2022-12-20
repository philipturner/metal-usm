//
//  MainFile.swift
//  USMBenchmarks
//
//  Created by Philip Turner on 12/19/22.
//

import Metal
import Atomics
import MetalKit

let device = MTLCreateSystemDefaultDevice()!
var rangeCache: [Range<Int>] = []
var bufferCache: [Int: MTLBuffer] = [:]
var nextBufferID: Int = 0

// Return the GPU VA so you can check whether they're contiguous.
func allocBuffer(address: UnsafeMutableRawPointer, offset: Int, length: Int) -> (Int, UInt64) {
  precondition(length > 2)
  precondition(!rangeCache.contains(where: { $0.contains(offset) }))
  precondition(!rangeCache.contains(where: { $0.contains(offset + length / 2) }))
  precondition(!rangeCache.contains(where: { $0.contains(offset + length - 1) }))
  
  let buffer = device.makeBuffer(bytesNoCopy: address, length: length)!
  let returnedBufferID = nextBufferID
  nextBufferID += 1
  bufferCache[returnedBufferID] = buffer
  rangeCache.append(offset..<offset + length)
  return (returnedBufferID, buffer.gpuAddress)
}

func deallocBuffer(bufferID: Int, offset: Int, length: Int) {
  let firstIndex = rangeCache.firstIndex(of: offset..<offset + length)!
  rangeCache.remove(at: firstIndex)
  _ = bufferCache[bufferID]!
  bufferCache[bufferID] = nil
}

func mainFunc() {
  defer { print("Execution finished.") }
  //  MAP_FIXED | MAP_ANONYMOUS
  //  PROT_READ | PROT_WRITE
  
  print("EACCESS", EACCES)
  print("EAGAIN", EAGAIN)
  print("EBADF", EBADF)
  print("EINVAL", EINVAL)
  print("ENFILE", ENFILE)
  print("ENODEV", ENODEV)
  print("ENOMEM", ENOMEM)
  print("EOVERFLOW", EOVERFLOW)
  print("EPERM", EPERM)
  print("ETXTBSY", ETXTBSY)
  print("SIGSEGV", SIGSEGV)
  print("SIGBUS", SIGBUS)
  
  //  let temporary_file = tmpfile()!
  //  print(fileno(temporary_file))
  //  let fd = fileno(temporary_file)
  //  ftruncate(fd, 1024 * 1024);
  
  // 4 * 16384
  // 4 TB of virtual memory is enough to span all addresses we need.
  // Aim to set GPU VA
  let length = 1024 * 1024 * 1024 * 1024 * 4
  
  // 0x0000001500018000
  let bitPattern = 0x0000001500018000 * 0b10000
  let desiredPointer = UnsafeMutableRawPointer(bitPattern: bitPattern)!
  let actualPointer = mmap(desiredPointer, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, Int32(-1), 0);
  precondition(actualPointer == desiredPointer, "\(actualPointer) \(desiredPointer) \(errno)")
  
  let chunkSize = 32768
  let paddingSize = 32768
  let baseCPUOffset = 2 * 1024 * 1024 * 1024
  var maxCPUOffset = baseCPUOffset // What to allocate onto by default.
  
  // GPU virtual memory pointers are 40 bits (1 TB)
  // Set the very first allocation to [-2 TB, 2 TB] within the middle of a
  // massive virtual memory range.
  // Possibly allocate large MTLHeap's within this memory range, then
  // `useResource(_:usage:)` each of the heaps before dispatching a command.
  // Use the PyTorch allocator as a reference for this, and find how many heaps
  // are present on an average basis.
  
  // Try to keep allocating until you align it right.
  struct Allocation {
    var bufferID: Int
    var gpuAddress: UInt64
    var length: Int
    var cpuOffset: Int
  }
  var successes: [Allocation] = []
  let (baseBufferID, baseVA) = allocBuffer(address: desiredPointer, offset: maxCPUOffset, length: 32768)
  successes.append(Allocation(bufferID: baseBufferID, gpuAddress: baseVA, length: 32768, cpuOffset: maxCPUOffset))
  maxCPUOffset += 32768
  
  func allocate(length: Int) -> Allocation {
    // On Apple GPUs, Metal allocations are typically off by 32768 bytes.
    var currentOffset = maxCPUOffset + 32768
    var (currentBufferID, currentVA) = allocBuffer(address: desiredPointer, offset: currentOffset, length: length)
    
    var numAttempts = 1
    var deltaCPU = currentOffset - baseCPUOffset
    var deltaGPU = Int(currentVA) - Int(baseVA)
    while deltaCPU != deltaGPU {
      print("Failed \(numAttempts)-th attempt, off by \(deltaCPU - deltaGPU).")
      deallocBuffer(bufferID: currentBufferID, offset: currentOffset, length: length)
      
      currentOffset = (deltaGPU - deltaCPU) + currentOffset
      (currentBufferID, currentVA) = allocBuffer(address: desiredPointer, offset: currentOffset, length: length)
      deltaCPU = currentOffset - baseCPUOffset
      deltaGPU = Int(currentVA) - Int(baseVA)
      
      numAttempts += 1
      if numAttempts >= 3 {
        // debug breakpoint here
      }
    }
    
    // Adjust the max CPU offset so another allocation can know where to start.
    // We don't know whether the current max is too high or too low. Should we
    // worry about running out of GPU address space?
    maxCPUOffset = max(maxCPUOffset, currentOffset + length)
    print("It finally worked, after \(numAttempts) attempts. \(OpaquePointer(bitPattern: deltaGPU) as Any)")
    
    return Allocation(bufferID: currentBufferID, gpuAddress: currentVA, length: length, cpuOffset: currentOffset)
  }
  
  let scaleFactor = 256 // 8-16 GB
  successes.append(allocate(length: 32768 * 1024 * scaleFactor))
  successes.append(allocate(length: 65536 * 1024 * scaleFactor))
  successes.append(allocate(length: 65536 * 1024 * scaleFactor))
  successes.append(allocate(length: 32768 * 1024 * scaleFactor))
  // [32768, 65536, 65536, 32768]
  
  func remove(at index: Int) {
    let alloc = successes.remove(at: index)
    deallocBuffer(bufferID: alloc.bufferID, offset: alloc.cpuOffset, length: alloc.length)
  }
  remove(at: 4)
  remove(at: 3)
  remove(at: 2)
  remove(at: 1)
  successes.append(allocate(length: 32768 * 1024 * scaleFactor))
  successes.append(allocate(length: 65536 * 1024 * scaleFactor))
  successes.append(allocate(length: 65536 * 1024 * scaleFactor))
  successes.append(allocate(length: 32768 * 1024 * scaleFactor))
  // [32768, 65536, 65536, 32768]
}

func oldMainFunc2() {
  let device = MTLCreateSystemDefaultDevice()!
  let library = device.makeDefaultLibrary()!
  let function = library.makeFunction(name: "hostMessaging")!
  let pipeline = try! device.makeComputePipelineState(function: function)
  
  let pipe_buf = device.makeBuffer(length: 4)!
  let totalMessages_buf = device.makeBuffer(length: 4)!
  let timeout_buf = device.makeBuffer(length: 4)!
  let returnCodes_buf = device.makeBuffer(length: 16)!
  
  let numLoops = 30
  var numCompletedLoops: Double = 0
  var largestTimeSum: Double = 0
  for _ in 0..<numLoops {
    print()
    let totalMessages = 10_000
    let timeout = 300_000_0
    readBuffer(pipe_buf)[0] = 0
    readBuffer(totalMessages_buf)[0] = UInt32(totalMessages)
    readBuffer(timeout_buf)[0] = UInt32(timeout)
    readBuffer(returnCodes_buf)[0] = 0
    readBuffer(returnCodes_buf)[1] = 0
    
    let queue = device.makeCommandQueue()!
    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(pipe_buf, offset: 0, index: 0)
    encoder.setBuffer(totalMessages_buf, offset: 0, index: 2)
    encoder.setBuffer(timeout_buf, offset: 0, index: 3)
    encoder.setBuffer(returnCodes_buf, offset: 0, index: 4)
    
    encoder.dispatchThreads(
      MTLSizeMake(1, 1, 1),
      threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
    encoder.endEncoding()
    commandBuffer.commit()
    
    // Try to distinguish latency between atomics feedback and driver feedback.
    // Why is round-trip latency 100-200 us?
    var returnTime: Double?
    var largestMessageTime: Double = 0
    var largestMessageIters = 0
    var largestMessageID = -1
    do {
      precondition(MemoryLayout<UnsafeAtomic<UInt32>.Storage>.stride == 4)
      precondition(MemoryLayout<UInt32.AtomicRepresentation>.stride == 4)
      let returnAtomic = readBuffer2(returnCodes_buf)
      let pipeAtomic = readBuffer2(pipe_buf)
      
      // 1 isn't actually the initial number of sent messages
      var sentMessages: UInt32 = 1
      var currentMessageStart = CACurrentMediaTime()
      var currentMessageIters = 0
      var maxSentMessages: UInt32 = 1
      var allNumIters = 0
      while true {
        // starts at 0
        // flips 1 -> 2
        // flips 3 -> 4
        // flips 5 -> 6
        let result = pipeAtomic.weakCompareExchange(expected: sentMessages, desired: sentMessages + 1, successOrdering: .relaxed, failureOrdering: .relaxed)
        if result.exchanged {
          sentMessages += 2
          
          if sentMessages > maxSentMessages {
            maxSentMessages = sentMessages
            let nowTime = CACurrentMediaTime()
            let currentMessageTime = nowTime - currentMessageStart
            if currentMessageTime > largestMessageTime {
              largestMessageTime = currentMessageTime
              largestMessageIters = currentMessageIters
              largestMessageID = Int(sentMessages - 2)
            }
            currentMessageStart = nowTime
            currentMessageIters = 0
          }
          
        } else if result.original < sentMessages {
          if result.original % 2 == 1 {
            sentMessages = result.original
          }
        }
        currentMessageIters += 1
        
        allNumIters += 1
        if allNumIters >= timeout * 10 || Int(maxSentMessages) >= Int(totalMessages) + 2 {
          returnTime = CACurrentMediaTime()
          if allNumIters >= timeout * 30 {
            fatalError()
          }
          break
        }
        
        let errorCode = returnAtomic.load(
          ordering: .sequentiallyConsistent)
        if errorCode == .max {
          fatalError()
        }
//        if errorCode != 0 {
//          returnTime = CACurrentMediaTime()
//          break
//        }
      }
      print("CPU detected messages:", sentMessages, pipeAtomic.load(ordering: .sequentiallyConsistent))
    }
    
    commandBuffer.waitUntilCompleted()
    
    // Quickly find the latency.
    let currentTime = CACurrentMediaTime()
    let driverLatency = currentTime - commandBuffer.gpuEndTime
    let measuredLatency = returnTime! - commandBuffer.gpuEndTime
    let measuredDisparity = currentTime - returnTime!
    print("Driver latency: \(format(decimals: 0, driverLatency * 1e6))")
    print("Measured latency: \(format(decimals: 0, measuredLatency * 1e6))")
    print("Measured disparity: \(format(decimals: 0, measuredDisparity * 1e6))")
    
    // Check return value.
    let errorCode = readBuffer(returnCodes_buf)[0]
    let errorData = readBuffer(returnCodes_buf)[1]
    let errorData2 = readBuffer(returnCodes_buf)[2]
    var errorMessage: String
    switch errorCode {
    case 1:
      errorMessage = "invalid arguments:"
    case 2:
      errorMessage = "success:"
    case 3:
      errorMessage = "timed out:"
    case 4:
      errorMessage = "unexpected behavior:"
    default:
      fatalError("Unknown return code")
    }
    print(errorMessage, errorData, errorData2)
    
    
    // Profile actual kernel execution time, time/iteration, time/message.
    print()
    let gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    if errorCode == 2 || errorCode == 3 {
      var i = -1
      var sentMessages = -1
      if errorCode == 2 {
        i = Int(errorData)
        sentMessages = Int(totalMessages)
      }
      if errorCode == 3 {
        i = Int(timeout)
        sentMessages = Int(errorData)
      }
      
      let iterationTime = gpuTime * 1e9 / Double(i)
      let messageTime = gpuTime * 1e9 / Double(sentMessages)
      let iterationsPerMessage = Double(i) / Double(sentMessages)
      print("Iteration time:", format(decimals: 0, iterationTime))
      print("Message time:  ", format(decimals: 0, messageTime))
      print("Iter./message: ", format(decimals: 1, iterationsPerMessage))
      print("Largest time:   \(format(decimals: 0, largestMessageTime * 1e6)) us, \(largestMessageIters) iters @ \(largestMessageID)")
      numCompletedLoops += 1
    }
    print("Total GPU time: " + format(decimals: 0, gpuTime * 1e6) + " us, " + format(decimals: 3, gpuTime) + " s")
    
    largestTimeSum += largestMessageTime
  }
  
  print()
  print("Summary: \(Int(largestTimeSum * 1e6) / Int(numCompletedLoops)) us, \(numCompletedLoops) completed")
}

func readBuffer(_ buffer: MTLBuffer) -> UnsafeMutablePointer<UInt32> {
  buffer.contents().assumingMemoryBound(to: UInt32.self)
}

func readBuffer2(_ buffer: MTLBuffer) -> UnsafeAtomic<UInt32> {
  let address = buffer.contents().assumingMemoryBound(to: UInt32.AtomicRepresentation.self)
  return UnsafeAtomic(at: address)
}

func format<T: Numeric>(decimals: Int, _ value: T) -> String {
  String(format: "%.\(decimals)f", value as! CVarArg)
}

func oldMainFunc() {
  // Initialize the context.
  let device = MTLCreateSystemDefaultDevice()!
  let library = device.makeDefaultLibrary()!
  let function = library.makeFunction(name: "vectorAddition")!
  let pipeline = try! device.makeComputePipelineState(function: function)
  
  // Initialize the buffers.
  let bufferLength = 16 * 1024 * 1024 * 400 //1024
//  precondition(device.maxBufferLength == bufferLength)
  let bufferA = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferB = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferC = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferD = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  
  // Dispatch the command.
  let queue = device.makeCommandQueue()!
  let commandBuffer = queue.makeCommandBuffer()!
  let encoder = commandBuffer.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(pipeline)
  encoder.setBuffer(bufferA, offset: 0, index: 0)
  encoder.setBuffer(bufferB, offset: 0, index: 1)
  encoder.setBuffer(bufferC, offset: 0, index: 2)
  encoder.setBuffer(bufferD, offset: 0, index: 3)
  
//  let numThreads = bufferLength / 16
  let numThreads = 10_000
  encoder.dispatchThreads(
    MTLSizeMake(numThreads, 1, 1),
    threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  
  // Report memory bandwidth and execution time (ms).
  let elapsedTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  let transferredBytes = 4 * 16 * numThreads
  let milliseconds = elapsedTime * 1000
  let bandwidth = Double(transferredBytes) / 1e9 / Double(elapsedTime)
  print("\(String(format: "%.2f", milliseconds)) ms")
  print("\(String(format: "%.2f", bandwidth)) GB/s")
}
