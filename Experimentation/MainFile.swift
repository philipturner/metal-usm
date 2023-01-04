//
//  MainFile.swift
//  USMBenchmarks
//
//  Created by Philip Turner on 12/19/22.
//

import Metal
import Atomics
import MetalKit

func mainFunc() {
  defer { print("Execution finished.") }
  
  // Determining how many heaps you can pass into `useResource(_:usage:)`
  // without causing too much driver overhead.
  //
  // Overhead is based solely on number of heap objects, not amount of memory.
  // Number of heaps - encoding overhead - scheduling overhead
  //     1 -    6 us -   32 us
  //    16 -    6 us -   32 us
  //    32 -    7 us -   37 us
  //    64 -    8 us -   43 us
  //   128 -    9 us -   54 us
  //   256 -   11 us -   72 us
  //   512 -   16 us -  117 us
  //  1024 -   26 us -  49-210 us
  //  2048 -   56 us -  72-372 us
  //  4096 -  115 us -  542-717 us
  //  8192 -  289 us -  300-833 us
  // 16384 -  677 us -  747-1033 us
  let numHeaps = 128
  let heapSizeMin = 1024 * 1024 * 1024 / numHeaps
  let heapSizeMax = heapSizeMin
  let doUseResource = true
  
  // Initialize the context.
  let device = MTLCreateSystemDefaultDevice()!
  let library = device.makeDefaultLibrary()!
  let function = library.makeFunction(name: "vectorAddition")!
  let pipeline = try! device.makeComputePipelineState(function: function)
  
  // Initialize the heaps.
  var heaps: [MTLHeap] = []
  let heapDescriptor = MTLHeapDescriptor()
  heapDescriptor.hazardTrackingMode = .tracked
  heapDescriptor.storageMode = .private
  for _ in 0..<numHeaps {
    heapDescriptor.size = Int.random(in: heapSizeMin...heapSizeMax)
    let heap = device.makeHeap(descriptor: heapDescriptor)!
    heaps.append(heap)
  }
  
  // Initialize the buffers.
  let bufferLength = 1024 //1024
//  precondition(device.maxBufferLength == bufferLength)
  let bufferA = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferB = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferC = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferD = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  
  let queue = device.makeCommandQueue()!
  
  var minEncodingOverhead: Int = 1_000_000
  var minSchedulingOverhead: Int = 1_000_000
  for _ in 0..<50 {
    // Dispatch the command.
    let encodeStart = CACurrentMediaTime()
    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    if doUseResource {
      encoder.useHeaps(heaps)
    }
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)
    encoder.setBuffer(bufferD, offset: 0, index: 3)
    
    let numThreads =  bufferLength / 16
    encoder.dispatchThreads(
      MTLSizeMake(numThreads, 1, 1),
      threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
    encoder.endEncoding()
    commandBuffer.commit()
    let encodeEnd = CACurrentMediaTime()
    commandBuffer.waitUntilCompleted()
    let waitEnd = CACurrentMediaTime()
    
    // Record various overheads.
    let encodingTime = encodeEnd - encodeStart
    let waitTime = waitEnd - encodeEnd
    let schedulingTime = commandBuffer.kernelEndTime - commandBuffer.kernelStartTime
    let executionTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
//    print()
//    print("Encoding time: \(Int(encodingTime * 1e6)) us")
//    print("Scheduling time: \(Int(schedulingTime * 1e6)) us")
//    print("Execution time: \(Int(executionTime * 1e6)) us")
//    print("Wait time: \(Int(waitTime * 1e6)) us")
    
    minEncodingOverhead = min(minEncodingOverhead, Int(encodingTime * 1e6))
    minSchedulingOverhead = min(minSchedulingOverhead, Int(schedulingTime * 1e6))
  }
  
  print()
  print("Minimum encoding overhead: \(minEncodingOverhead) us")
  print("Minimum scheduling overhead: \(minSchedulingOverhead) us")
}



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

func oldMainFunc3() {
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
  
    let temporary_file = tmpfile()!
    print(fileno(temporary_file))
    let fd = fileno(temporary_file)
    ftruncate(fd, 1024 * 1024);
  
  // 4 * 16384
  // 4 TB of virtual memory is enough to span all addresses we need.
  // Aim to set GPU VA
  //  let length = 1024 * 1024 * 1024 * 1024 * 4
  
#if os(macOS)
//  let length = 1024 * 1024 * 1024 * 1024 * 95
  let length = 1024 * 1024 * 1024 * 97850 // max length
  // 97850 GB / 2^7 * 1024 GB = 74.65% (3/4)
  // base address for largest allocation: 0x7000000000 = 448 GB
  // span: 0 GB - 448 GB --------------- 98298 GB ----- 131072 GB
#elseif os(iOS)
//  let length = 1024 * 1024 * 1024 * 52
  let length = 1024 * 1024 * 53760 // max length
  // 53760 MB / 2^9 * 1024 MB = 10.25% (1/10)
  // base address for largest allocation: 0x2a0000000 = 11008 MB
  // span: 0 GB - 11008 MB -- 64768 MB ------------------ 524288 MB
#endif
  
  // On iOS, snap the VA to 84 GB start or round down otherwise. Hope this works!
  // Try allocating 64 GB of VM on iOS, then progressively down by 8 GB steps.
  // On macOS, 95 TB of virtual memory.
  
  // 0x0000001500018000
//  let bitPattern = 0x0000001500018000 * 0b10000
//  let desiredPointer = UnsafeMutableRawPointer(bitPattern: bitPattern)!
  let actualPointer = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, Int32(-1), 0)!
  print(errno)
  print(actualPointer)
//  precondition(actualPointer == desiredPointer, "\(actualPointer) \(desiredPointer) \(errno)")
  
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
  
  do {
    let heapDesc = MTLHeapDescriptor()
    heapDesc.type = .automatic
    heapDesc.hazardTrackingMode = .untracked
    heapDesc.resourceOptions = .storageModeShared
    heapDesc.size = 1024 * 1024
    let heap = device.makeHeap(descriptor: heapDesc)!
    print(heap.makeBuffer(length: 1)!.heapOffset)
  }
  
  // Try to keep allocating until you align it right.
  struct Allocation {
    var bufferID: Int
    var gpuAddress: UInt64
    var length: Int
    var cpuOffset: Int
  }
  var successes: [Allocation] = []
  let (baseBufferID, baseVA) = allocBuffer(address: actualPointer, offset: maxCPUOffset, length: 32768)
  successes.append(Allocation(bufferID: baseBufferID, gpuAddress: baseVA, length: 32768, cpuOffset: maxCPUOffset))
  maxCPUOffset += 32768
  
  func allocate(length: Int) -> Allocation {
    // On Apple GPUs, Metal allocations are typically off by 32768 bytes.
    var currentOffset = maxCPUOffset + 32768
    var (currentBufferID, currentVA) = allocBuffer(address: actualPointer, offset: currentOffset, length: length)
    
    var numAttempts = 1
    var deltaCPU = currentOffset - baseCPUOffset
    var deltaGPU = Int(currentVA) - Int(baseVA)
    while deltaCPU != deltaGPU {
      print("Failed \(numAttempts)-th attempt, off by \(deltaCPU - deltaGPU).")
      deallocBuffer(bufferID: currentBufferID, offset: currentOffset, length: length)
      
      currentOffset = (deltaGPU - deltaCPU) + currentOffset
      (currentBufferID, currentVA) = allocBuffer(address: actualPointer, offset: currentOffset, length: length)
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
  
  let scaleFactor = 16 // 256 // 8-16 GB
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
  
  var ret: Int = 0
  var size: Int = 8
  let error = sysctlbyname("machdep.virtual_address_size", &ret, &size, nil, 0)
  guard error == 0 else {
    fatalError("Could not find sysctl value.")
  }
  print("sysctl value:", ret)
  
  
  
  // Initialize the buffers.
//  print(device.recommendedMaxWorkingSetSize / 1024 / 1024)
  let bufferLength = 3725 * 1024 * 1024 / 4
//  precondition(device.maxBufferLength == bufferLength)
  let bufferA = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferB = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferC = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  let bufferD = device.makeBuffer(
    length: bufferLength, options: .storageModeShared)!
  print(bufferA.gpuAddress)
  
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
