//
//  MainFile.swift
//  MetalAllocator
//
//  Created by Philip Turner on 1/4/23.
//

import Metal
import QuartzCore
import OrderedCollections

// Two different GPU VAs can map to the same CPU VA - sort of. Make 1/128 chunks
// of global memory, as actual Metal buffers. Then make `MTLHeap` with the
// purgeable state erasing it from memory. The buffer's complementary heap
// decides where to allocate addresses within it. Borrow the idea from PyTorch
// that will cache larger allocations, for non-1/128 chunks.
//
// The `MTLHeap` is created once by setting all zeroes, then erasing via
// purging. Next, allocate the complementary Metal buffer from bytesNoCopy.
// Perform phantom `malloc` within the `MTLHeap`, then translate to the
// complementary buffer. Only the buffer is actually `useResource`'d and aligned
// with the USM address space.

// Three tiers of heaps, each with a maximum sub-allocation size.
// Small heaps: 1/256 memory, max allocation = 1/2048 memory
// Large heaps: 1/64 memory, max allocation = 1/128 memory
// Unique heaps: >1/128 memory per device, larger buffers or when facing
// significant memory pressure
//
// Numbers above come from the 2/3 mark (actually 65/100) of device memory,
// toward the closest even-enough cutoff from a power of 2 (1.00, 1.25, 1.50,
// 1.75).
//
// 256 resident heaps will approximately double encoding latency. Assume 1/2 RAM
// is the largest realistic working set size, so derive overhead from 3/4 the
// max allocations. Overhead is (encoding, scheduling) with a baseline of
// (6 us, 32 us) per batch.
//
// Original PyTorch constants:
// kMaxSmallAlloc: 1 MB
// kMinLargeAlloc: 10 MB
// kSmallHeap: 8 MB
// kLargeHeap: 32 MB
// kXLargeHeap: 1024 MB
//
// For a 5.4 GB A15, we revise the PyTorch constants to:
// kMaxSmallAlloc: 1.75 MB
// kMinLargeAlloc: 28 MB
// kSmallHeap: 14 MB
// kLargeHeap: 56 MB
// kXLargeHeap: 56 MB
// Max resident small heaps: 449 -> 257 (1.6x, 2.0x overhead)
// Max resident large heaps: 112 -> 64 (1.3x, 1.3x overhead)
//
// For a 32 GB M1 Max, we revise the PyTorch constants to:
// kMaxSmallAlloc: 10 MB
// kMinLargeAlloc: 160 MB
// kSmallHeap: 80 MB
// kLargeHeap: 320 MB
// kXLargeHeap: 320 MB
// Max resident small heaps: 2662 -> 266 (1.6x, 2.0x overhead)
// Max resident large heaps: 666 -> 67 (1.3x, 1.3x overhead)
//
// For a 192 GB M2 Ultra, we revise the PyTorch constants to:
// kMaxSmallAlloc: 64 MB
// kMinLargeAlloc: 1024 MB
// kSmallHeap: 512 MB
// kLargeHeap: 2048 MB
// kXLargeHeap: 2048 MB
// Max resident small heaps: 15974 -> 250 (1.6x, 2.0x overhead)
// Max resident large heaps: 3994 -> 62 (1.3x, 1.3x overhead)

// If we can use `mlock` instead of `useResource`, that saves a heck ton of
// overhead and means we can allocate much more simply, with better granularity.
// Nevermind: this doesn't work.
//
// We still need a data structure to map arbitrary pointers to filled-in
// allocations, but it can be simple. For example, a 3-level tier of arrays
// going from kMaxSmallAlloc -> sqrt(kMaxSmallAlloc * 16 KB) -> 16 KB. It should
// still be searchable in batches of 4.
func mainFunc() {
  let device = MTLCreateSystemDefaultDevice()!
  let commandQueue = device.makeCommandQueue()!
  let library = device.makeDefaultLibrary()!
  let function = library.makeFunction(name: "vectorAddition")!
  let pipeline = try! device.makeComputePipelineState(function: function)
  
  // We'll search through everything in binary trees.
  // Sorted lists for allocating memory:
  // - Small heaps: sorted by increasing available size
  // - Large heaps: sorted by increasing available size
  //
  // Sorted lists for mapping memory:
  // - All heaps: sorted by address
  // Only consider the original idea (radix search) if this takes >1 us per 8
  // queries, >500 ns per 4 queries, and >250 ns per 2 queries.
  
  let pool = HeapPool(device: device)
  precondition(pool.queryTotalSize() == 0)
  
  let pointer1 = pool.allocate(size: 1024)!
  precondition(pool.queryTotalSize() == 1024)
  pointer1.assumingMemoryBound(to: UInt32.self)[0] = 7
  pool.deallocate(usmPointer: pointer1)
  precondition(pool.queryTotalSize() == 0)
  
  let bufferSize = max(4096, 768 * 1024 * 1024)
  let bufferA = pool.allocate(size: bufferSize)!
  let bufferB = pool.allocate(size: bufferSize)!
  let bufferC = pool.allocate(size: bufferSize)!
  let buffersList = [bufferA, bufferB, bufferC]
  let patternsList: [UInt64] = [4, 50, 600]
  for (buffer, var pattern) in zip(buffersList, patternsList) {
    memset_pattern8(buffer, &pattern, 4096)
  }
  
  let bufferD = pool.allocate(size: bufferSize)!
  
  let commandBuffer = commandQueue.makeCommandBuffer()!
  let encoder = commandBuffer.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(pipeline)
  encoder.useResources(pool.extractBuffers(), usage: [.read, .write])
  
  let pointers1 = [bufferA, bufferB, bufferC, bufferD].map { usmPointer in
    let (buffer, offset) = pool.getBufferAndOffset(usmPointer: usmPointer)!
    return buffer.gpuAddress + UInt64(offset)
  }
  encoder.setBytes(pointers1, length: 32, index: 0)
  encoder.dispatchThreads(
    MTLSizeMake(4096 / 16, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  
  let output = bufferD.assumingMemoryBound(to: UInt64.self)
  for i in 0..<4096 / 8 {
    guard output[i] == 654 else {
      fatalError("Element \(i) was \(output[i]), not 654.")
    }
  }
  print("succeeded! \(output[0])")
  
  pool.deallocate(usmPointer: bufferD)
  pool.deallocate(usmPointer: bufferC)
  pool.deallocate(usmPointer: bufferB)
  pool.deallocate(usmPointer: bufferA)
}

func oldMainFunc1() {
  let device = MTLCreateSystemDefaultDevice()!
  let commandQueue = device.makeCommandQueue()!
  let library = device.makeDefaultLibrary()!
  let function = library.makeFunction(name: "vectorAddition")!
  let pipeline = try! device.makeComputePipelineState(function: function)
  print("Max buffer size: \(device.maxBufferLength)")
  
  let globalMemoryRaw = fetchSysctlProperty(name: "hw.memsize")
  let globalMemory = (65 * (globalMemoryRaw >> 20) / 100) << 20
  
  // Conservative limit of 6% maximum allocatable memory.
  let virtualAddressBits = fetchSysctlProperty(
    name: "machdep.virtual_address_size")
  let virtualMemory = (1 << virtualAddressBits) / 16
  
  // Allocate the virtual memory buffer, then sub-allocate Metal buffers from
  // it. Hope this works!
  let cpuPointer = mmap(
    nil, virtualMemory, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
    0)!
  let gpuBaseVA: UInt64 = 0x11_0000_0000
  var usmAllocations: [UnsafeMutableRawPointer: MTLBuffer] = [:]
  precondition(
    UInt(bitPattern: cpuPointer) != .max, "Could not allocate virtual memory.")
  
  // TODO: We need a data structure that maps anything in a certain range to its
  // Metal buffer address. How about a tree (4x branching) where each sub-node
  // is either uninitialized, freed, occupied by one, or occupied by many.
  // That means we don't have to iterate over every virtual memory page during
  // allocation. Keep it small to reduce encoding costs, and make it explicitly
  // batched/parallelized (4x). Instead of a Swift/C++ dictionary, use a custom
  // data structure that's a multi-level, monotonically expanding set of C++
  // `std::vector`.
  //
  // Nevermind. mlock and munlock are slow, processing at 24 GB/s and 36 GB/s
  // respectively. That equals 400-600 nanoseconds/page. We can shoot off
  // parallel writes to a page table much quicker than that.
  
  func mallocUSM(size: Int) -> UnsafeMutableRawPointer {
    var targetAddress = gpuBaseVA
    var numTries = 0
    while numTries < 1000 {
      numTries += 1
      let delta = targetAddress - gpuBaseVA
      precondition(delta >= 0, "Tried to allocate below GPU base VA.")
      let usmPointer = cpuPointer + Int(Int64(bitPattern: delta))
      let buffer = device.makeBuffer(bytesNoCopy: usmPointer, length: size)!
      if buffer.gpuAddress != targetAddress {
        targetAddress = buffer.gpuAddress
        continue
      }
      
      // Lock the page and insert the buffer.
      let start = CACurrentMediaTime()
      let error = mlock(usmPointer, size)
      let end = CACurrentMediaTime()
      print("mlock: \(Double(size) / 1e9 / (end - start))")
      precondition(error == 0, "Could not lock pages: \(error)")
      precondition(usmAllocations[usmPointer] == nil, "Already exists.")
      usmAllocations[usmPointer] = buffer
      print("Tries: \(numTries)")
      return usmPointer
    }
    fatalError("Too many tries.")
  }
  func freeUSM(pointer: UnsafeMutableRawPointer) {
    // Unlock the page and remove the buffer.
    let buffer = usmAllocations.removeValue(forKey: pointer)!
    let start = CACurrentMediaTime()
    let error = munlock(pointer, buffer.length)
    let end = CACurrentMediaTime()
    print("munlock: \(Double(buffer.length) / 1e9 / (end - start))")
    precondition(error == 0, "Could not unlock pages: \(error)")
  }
  
  let bufferSize = globalMemory / 4 / 64
  freeUSM(pointer: mallocUSM(size: bufferSize))
  freeUSM(pointer: mallocUSM(size: bufferSize))
  freeUSM(pointer: mallocUSM(size: bufferSize))
  freeUSM(pointer: mallocUSM(size: bufferSize))
  let bufferA = mallocUSM(size: bufferSize)
  let bufferB = mallocUSM(size: bufferSize)
  let bufferC = mallocUSM(size: bufferSize)
  let bufferD = mallocUSM(size: bufferSize)
  print(bufferA, bufferB, bufferC, bufferD)
  
  bufferA.assumingMemoryBound(to: UInt64.self)[0] = 4
  bufferB.assumingMemoryBound(to: UInt64.self)[0] = 5
  bufferC.assumingMemoryBound(to: UInt64.self)[0] = 6
  
  let allBuffers = [usmAllocations[bufferA]!, usmAllocations[bufferB]!, usmAllocations[bufferC]!, usmAllocations[bufferD]!]
  let vaRangeMin = allBuffers.map(\.gpuAddress).min()!
  let vaRangeMax = allBuffers.map { $0.gpuAddress + UInt64($0.length) }.max()!
  
  let vaPointer = cpuPointer + Int(vaRangeMin - gpuBaseVA)
  let vaLength = Int(vaRangeMax - vaRangeMin)
  let bufferE = device.makeBuffer(bytesNoCopy: vaPointer, length: vaLength)!
  
  let preCommandBuffer = commandQueue.makeCommandBuffer()!
  let preEncoder = preCommandBuffer.makeComputeCommandEncoder()!
  preEncoder.useResources([usmAllocations[bufferA]!, usmAllocations[bufferB]!, usmAllocations[bufferC]!, usmAllocations[bufferD]!], usage: [.read, .write])
  preEncoder.endEncoding()
  preCommandBuffer.commit()
  
  let commandBuffer = commandQueue.makeCommandBuffer()!
  let encoder = commandBuffer.makeComputeCommandEncoder()!
//  encoder.useResources([usmAllocations[bufferA]!, usmAllocations[bufferB]!, usmAllocations[bufferC]!, usmAllocations[bufferD]!], usage: [.read, .write])
//  encoder.useResources([usmAllocations[bufferD]!], usage: [.read, .write])
  encoder.useResource(bufferE, usage: [.read, .write])
  encoder.setComputePipelineState(pipeline)
  
  
  
  let addressA = usmAllocations[bufferA]!.gpuAddress
  let addressB = usmAllocations[bufferB]!.gpuAddress
  let addressC = usmAllocations[bufferC]!.gpuAddress
  let addressD = usmAllocations[bufferD]!.gpuAddress
  struct Pointers1 {
    var bufferA: UInt64
    var bufferB: UInt64
    var bufferC: UInt64
    var bufferD: UInt64
  }
  print(addressA, addressB, addressC, addressD)
  var arguments = Pointers1(
    bufferA: addressA, bufferB: addressB, bufferC: addressC, bufferD: addressD)
  encoder.setBytes(&arguments, length: 4 * 8, index: 0)
//  encoder.setBuffer(usmAllocations[bufferA]!, offset: 0, index: 0)
//  encoder.setBuffer(usmAllocations[bufferB]!, offset: 0, index: 1)
//  encoder.setBuffer(usmAllocations[bufferC]!, offset: 0, index: 2)
//  encoder.setBuffer(usmAllocations[bufferD]!, offset: 0, index: 3)
  encoder.dispatchThreads(
    MTLSizeMake(1, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  
  if bufferD.assumingMemoryBound(to: UInt64.self)[0] == 15 {
    print("Succeeded: \(bufferD.assumingMemoryBound(to: UInt64.self)[0])")
  } else {
    print("Failed: \(bufferD.assumingMemoryBound(to: UInt64.self)[0])")
  }
}

func fetchSysctlProperty(name: String) -> Int {
  var ret: Int = 0
  var size: Int = 8
  let error = sysctlbyname(name, &ret, &size, nil, 0)
  precondition(error == 0, "sysctl failed.")
  return ret
}

/// Rounds an integer up to the nearest power of 2.
@inline(__always)
func roundUpToPowerOf2(_ input: Int) -> Int {
    1 << (Int.bitWidth - max(0, input - 1).leadingZeroBitCount)
}

/// Rounds an integer down to the nearest power of 2.
@inline(__always)
func roundDownToPowerOf2(_ input: Int) -> Int {
    1 << (Int.bitWidth - 1 - input.leadingZeroBitCount)
}

// In hipSYCL, this should be a separate class than the Metal allocator.
class HeapPool {
  var device: MTLDevice
  var cpuBaseVA: UnsafeMutableRawPointer
  var gpuBaseVA: UInt64
  var physicalMemoryLimit: Int
  var virtualMemoryLimit: Int
  
  // Borrowed from PyTorch MPSAllocator. We don't cache the buffers; they are
  // immediately returned to the heap. This helps reduce fragmentation, as we
  // can't afford to exceed device memory (unlike PyTorch).
  var kMaxSmallAlloc: Int
  var kMinLargeAlloc: Int
  var kSmallHeap: Int
  var kLargeHeap: Int
  
  var smallHeapsSizeSorted: OrderedSet<HeapBlock> = []
  var largeHeapsSizeSorted: OrderedSet<HeapBlock> = []
  
  // In C++, we'll query the `availableSize` property of the heap object. It's
  // just a pointer dereferencing and not an ARC retain/release, so it should
  // have no overhead.
  var heapsAddressSorted: OrderedSet<HeapBlock> = []
  
  // TODO: Don't implement this in C++. It's only for debugging in the Swift
  // prototype. We'll want something more formal for debugging anyway.
  func queryTotalSize() -> Int {
    validateAddressSorted()
    return heapsAddressSorted.reduce(0, { $0 + $1/*heap.*/.usedSize })
  }
  
  // TODO: Find a better way to cache the buffer list.
  func extractBuffers() -> [MTLBuffer] {
    heapsAddressSorted.map { $0.buffer }
  }
  
  func validateSizeSorted() {
    for set in [smallHeapsSizeSorted, largeHeapsSizeSorted] {
      if set.count > 1 {
        for i in 0..<set.count - 1 {
          let element1 = set[i]
          let element2 = set[i + 1]
          let element1_size = UInt64(element1.availableSize)
          let element2_size = UInt64(element2.availableSize)
          precondition(
            element1_size <= element2_size, "Allocations are not sorted.")
        }
      }
    }
  }
  
  func validateAddressSorted() {
    let numHeapsFromAddress = heapsAddressSorted.count
    let numHeapsFromSize =
      smallHeapsSizeSorted.count + largeHeapsSizeSorted.count
    precondition(
      numHeapsFromAddress == numHeapsFromSize,
      "An allocation wasn't removed/inserted correctly.")
    
    if heapsAddressSorted.count > 1 {
      for i in 0..<heapsAddressSorted.count - 1 {
        let element1 = heapsAddressSorted[i]
        let element2 = heapsAddressSorted[i + 1]
        let element1_size = UInt64(element1.heap.size)
        precondition(
          element1.gpuBaseVA + element1_size <= element2.gpuBaseVA,
          "Allocations are not sorted.")
      }
    }
  }
  
  init(device: MTLDevice) {
    self.device = device
    
    // MARK: - Determine Physical Memory
    
    let physicalMemoryMax = fetchSysctlProperty(name: "hw.memsize")
    self.physicalMemoryLimit = (65 * (physicalMemoryMax >> 20) / 100) << 20
    self.physicalMemoryLimit = max(physicalMemoryLimit, device.maxBufferLength)
    #if os(macOS)
    let reportedMemoryLimit = Int(device.recommendedMaxWorkingSetSize)
    self.physicalMemoryLimit = min(physicalMemoryLimit, reportedMemoryLimit)
    #endif
    
    // Small heaps are 1/256 of device memory.
    var smallHeapSize = physicalMemoryLimit / 256
    let smallHeapFloor = roundDownToPowerOf2(smallHeapSize)
    let smallHeapCeiling = roundUpToPowerOf2(smallHeapSize)
    if smallHeapFloor != smallHeapCeiling {
      // Subdivide into 4 intervals.
      let chunkSize = (smallHeapCeiling - smallHeapFloor) / 4
      var upperBound = smallHeapFloor + chunkSize
      while smallHeapSize >= upperBound {
        upperBound += chunkSize
      }
      let lowerBound = upperBound - chunkSize
      precondition(smallHeapSize < upperBound, "Invalid upper bound.")
      precondition(smallHeapSize >= lowerBound, "Invalid lower bound.")
      
      if abs(smallHeapSize - lowerBound) < abs(upperBound - smallHeapSize) {
        smallHeapSize = lowerBound
      } else {
        smallHeapSize = upperBound
      }
    }
    precondition(smallHeapSize % 8 == 0, "Small heap size is an odd number.")
    
    self.kMaxSmallAlloc = smallHeapSize / 8
    self.kMinLargeAlloc = smallHeapSize * 2
    self.kSmallHeap = smallHeapSize
    self.kLargeHeap = smallHeapSize * 4
    
    // MARK: - Allocate Virtual Memory
    
    #if os(macOS)
    // Either a terabyte or 3x the actual RAM size.
    let virtualMemoryMax = max(1 << 40, 3 * physicalMemoryMax)
    #else
    let virtualAddressBits = fetchSysctlProperty(
      name: "machdep.virtual_address_size")
    let virtualMemoryMax = (1 << virtualAddressBits) / 2
    #endif
    
    var virtualMemoryCandidate = virtualMemoryMax
    func attemptMap() -> UnsafeMutableRawPointer {
      // Each successful attempt takes ~50 us.
      mmap(
        nil, virtualMemoryCandidate, PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0)!
    }
    var cpuPointerCandidate = attemptMap()
    func cpuPointerIsValid() -> Bool {
      UInt(bitPattern: cpuPointerCandidate) != .max
    }
    func revertMap() {
      // Each call takes ~25 us.
      let error = munmap(cpuPointerCandidate, virtualMemoryCandidate)
      precondition(error == 0, "Error while ummapping memory.")
    }
    
    // If this fails on the first try, search for the actual maximum.
    if !cpuPointerIsValid() {
      while true {
        virtualMemoryCandidate /= 2
        cpuPointerCandidate = attemptMap()
        if cpuPointerIsValid() {
          break
        }
      }
      
      let largeChunkSize = virtualMemoryCandidate / 4
      let smallChunkSize = virtualMemoryCandidate / 16
      revertMap()
      virtualMemoryCandidate *= 2
      while true {
        virtualMemoryCandidate -= largeChunkSize
        cpuPointerCandidate = attemptMap()
        if cpuPointerIsValid() {
          break
        }
      }
      
      revertMap()
      virtualMemoryCandidate += largeChunkSize
      while true {
        virtualMemoryCandidate -= smallChunkSize
        cpuPointerCandidate = attemptMap()
        if cpuPointerIsValid() {
          break
        }
      }
      precondition(virtualMemoryCandidate > 0, "Virtual memory was zero.")
    }
    
    self.virtualMemoryLimit = virtualMemoryCandidate
    self.cpuBaseVA = cpuPointerCandidate
    #if os(macOS)
    self.gpuBaseVA = 0x0
    #else
    self.gpuBaseVA = 0x11_0000_0000
    let firstZoneSize = Int(0x15_0000_0000 - gpuBaseVA)
    precondition(
      virtualMemoryLimit > physicalMemoryLimit + firstZoneSize,
      "Not enough virtual memory.")
    #endif
  }
  
  private func extract(
    from set: inout OrderedSet<HeapBlock>, size: Int
  ) -> HeapBlock? {
    let firstIndex = set.firstIndex(where: { $0.availableSize >= size })
    guard let firstIndex = firstIndex else {
      return nil
    }
    return set.remove(at: firstIndex)
  }
  
  private func insert(
    _ block: HeapBlock,
    into set: inout OrderedSet<HeapBlock>,
    size: Int
  ) {
    // Insert before the first element with more capacity.
    let firstIndex = set.firstIndex(where: { $0.availableSize > size })
    if let firstIndex = firstIndex {
      set.insert(block, at: firstIndex)
    } else {
      set.append(block)
    }
  }
  
  // If this returns `nil`, stall until all GPU work has finished.
  // Perhaps it will free up some extra memory - at which point, try again.
  func allocate(size: Int) -> UnsafeMutableRawPointer? {
    precondition(size < device.maxBufferLength, "Buffer too large.")
    
    var extractedBlock: HeapBlock?
    if size < kMaxSmallAlloc {
      extractedBlock = extract(from: &smallHeapsSizeSorted, size: size)
    } else {
      extractedBlock = extract(from: &largeHeapsSizeSorted, size: size)
    }
    validateSizeSorted()
    defer {
      // Ensure everything's correct before returning.
      validateSizeSorted()
      validateAddressSorted()
    }
    
    var block: HeapBlock
    if let extractedBlock = extractedBlock {
      block = extractedBlock
    } else {
      var heapSize: Int
      if size < kMaxSmallAlloc {
        heapSize = kSmallHeap
      } else if size < kMinLargeAlloc {
        // We're not actually going to use the memory watermark like PyTorch
        // does. In low-memory situations, that would create numerous tiny
        // buffers and tank encoding performance.
        heapSize = kLargeHeap
      } else {
        // We're not going to mark heaps as "split" and treat them differently.
        // We just want to make the device's memory allocatable without major
        // fragmentation. We're not aiming for a caching-garbage collecting
        // scheme. You should expect to pay the full price of zeroing something
        // out when calling `malloc`.
        //
        // However, you should not have to pay for allocating a 500 MB heap if
        // when quickly allocating/deallocating a 32 KB buffer. We lazily
        // release zombie heaps when one fails to allocate.
        
        // Round up to 128 KB, the maximum page size witnessed in Metal Frame
        // Capture.
        let kRoundLarge = 128 * 1024
        heapSize = kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge)
      }
      
      // This should fail if you exceed device memory.
      func attemptAllocate() -> HeapBlock? {
        return HeapBlock(
          device: device, cpuBaseVA: cpuBaseVA, gpuBaseVA: gpuBaseVA,
          size: heapSize)
      }
      
      // Also remove from the set of addresses.
      func releaseHeaps(from set: inout OrderedSet<HeapBlock>) {
        while set.last?/*heap.*/.usedSize == 0 {
          let removed = set.removeLast()
          heapsAddressSorted.remove(removed)
        }
      }
      
      var possibleBlock = attemptAllocate()
      if possibleBlock == nil {
        // If this is a small heap, we know the first function call will be a
        // no-op.
        releaseHeaps(from: &smallHeapsSizeSorted)
        releaseHeaps(from: &largeHeapsSizeSorted)
        possibleBlock = attemptAllocate()
      }
      guard let possibleBlock = possibleBlock else {
        // Ran out of device memory.
        return nil
      }
      block = possibleBlock
      
      let blockUpperVA = block.gpuBaseVA + UInt64(heapSize)
      if blockUpperVA - gpuBaseVA > virtualMemoryLimit {
        // Ran out of virtual memory.
        return nil
      }
      
      // Insert new block into address list.
      let firstIndex = heapsAddressSorted.firstIndex(
        where: { $0.gpuBaseVA > block.gpuBaseVA })
      if let firstIndex = firstIndex {
        heapsAddressSorted.insert(block, at: firstIndex)
      } else {
        heapsAddressSorted.append(block)
      }
    }
    
    let usmPointer = block.allocate(size: size)
    let newSize = block.availableSize
    if size < kMaxSmallAlloc {
      insert(block, into: &smallHeapsSizeSorted, size: newSize)
    } else {
      insert(block, into: &largeHeapsSizeSorted, size: newSize)
    }
    return usmPointer
  }
  
  // We do eagerly release heap blocks, but only to preserve the O(n) cost of
  // allocating memory. Preserve one empty block perfectly equalling kSmallHeap
  // and another perfectly equalling kLargeHeap.
  func deallocate(usmPointer: UnsafeMutableRawPointer) {
    let index = heapsAddressSorted.firstIndex(where: {
      $0.cpuBaseVA <= usmPointer &&
      usmPointer < $0.cpuBaseVA + $0.heap.size
    })
    guard let index = index else {
      fatalError("Tried to deallocate an invalid USM pointer.")
    }
    
    var shouldKeepInAddressSet = false
    defer {
      if !shouldKeepInAddressSet {
        heapsAddressSorted.remove(at: index)
      }
      validateAddressSorted()
      validateSizeSorted()
    }
    
    let block = heapsAddressSorted[index]
    let heapSize = block.heap.size
    precondition(heapSize >= kSmallHeap, "Heap was too small.")
    if heapSize == kSmallHeap {
      _ = smallHeapsSizeSorted.remove(block)!
    } else {
      _ = largeHeapsSizeSorted.remove(block)!
    }
    
    // Let the block deallocate its pointer internally.
    block.deallocate(usmPointer: usmPointer)
    
    // Handle special cases where the heap's contents are completely
    // deallocated.
    if block.availableSize == heapSize {
      if heapSize == kSmallHeap {
        if smallHeapsSizeSorted.last?/*heap.*/.usedSize == 0 {
          // There's already a buffer against allocation thrashing.
          return
        }
      } else if heapSize == kLargeHeap {
        // If a heap doesn't have size `kLargeHeap`, it's either 100% occupied
        // or eagerly freed.
        if largeHeapsSizeSorted.last?/*heap.*/.usedSize == 0 {
          // There's already a buffer against allocation thrashing.
          precondition(
            largeHeapsSizeSorted.last!.availableSize == kLargeHeap,
            "Unexpected size of empty large heap.")
          return
        }
      } else {
        // This was a heap custom-allocated for a single buffer. We can throw it
        // away, as it's O(n) cost.
        //
        // NOTE: If you decide to keep such buffers, change the code block
        // directly above this. We don't want to precondition that the last
        // heap == kLargeHeap.
        return
      }
    }
    
    // We can keep the address where it is, without inserting it again.
    shouldKeepInAddressSet = true
    
    // Insert possible zombie heap back into the list.
    let newSize = block.availableSize
    if heapSize == kSmallHeap {
      insert(block, into: &smallHeapsSizeSorted, size: newSize)
    } else {
      insert(block, into: &largeHeapsSizeSorted, size: newSize)
    }
  }
  
  func getBufferAndOffset(
    usmPointer: UnsafeMutableRawPointer
  ) -> (MTLBuffer, Int)? {
    let index = heapsAddressSorted.firstIndex(where: {
      $0.cpuBaseVA <= usmPointer &&
      usmPointer < $0.cpuBaseVA + $0.heap.size
    })
    guard let index = index else {
      // Invalid USM pointer, but we don't need to crash the program.
      return nil
    }
    let heapBlock = heapsAddressSorted[index]
    
    // No need to pre-determine whether the pointer fits in this block. If it
    // doesn't, the result will be -1.
    let offset = heapBlock.getOffset(usmPointer: usmPointer)
    if offset == -1 {
      return nil
    }
    return (heapBlock.buffer, offset)
  }
}

// NOTE: In Swift, it seems the OrderedList does not binary-search when
// iterating through elements. That's okay because the C++ version will.
//
// The class itself isn't stored in any list sorted by offset. That would incur
// extra overhead when mapping pointers to buffers. Rather, we create a sorted
// set of VAs to search.
// - Nevermind.
class HeapBlock {
  var heap: MTLHeap
  var buffer: MTLBuffer
  var cpuBaseVA: UnsafeMutableRawPointer
  var gpuBaseVA: UInt64
  var heapVA: UInt64
  var availableSize: Int
  private var phantomBuffers: [Int: MTLBuffer] = [:]
  
  // Heap used size isn't a reliable way to find used size. There's a bug where
  // it returns 4 GB when you have a 4 GB heap, even though no resources are
  // allocated. The maxAvailableSize(alignment:) still says you can allocate
  // stuff on the heap.
  var usedSize: Int = 0
  
  struct Allocation: Hashable {
    var offset: Int
    var size: Int
    func hash(into hasher: inout Hasher) {
      hasher.combine(offset)
    }
  }
  var allocations: OrderedSet<Allocation> = []
  
  func validateSorted() {
    if allocations.count > 1 {
      for i in 0..<allocations.count - 1 {
        let element1 = allocations[i]
        let element2 = allocations[i + 1]
        precondition(
          element1.offset + element1.size <= element2.offset,
          "Allocations are not sorted.")
      }
    }
  }
  
  init?(
    device: MTLDevice,
    cpuBaseVA: UnsafeMutableRawPointer,
    gpuBaseVA: UInt64,
    size: Int
  ) {
    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.hazardTrackingMode = .untracked
    heapDescriptor.storageMode = .shared
    heapDescriptor.size = size
    guard let heap = device.makeHeap(descriptor: heapDescriptor) else {
      return nil
    }
    defer {
      heap.setPurgeableState(.empty)
    }
    self.heap = heap
//    precondition(heap.usedSize == 0, "Heap did not start with size 0.")
    
    var targetAddress = gpuBaseVA
    var finalBuffer: MTLBuffer?
    for _ in 0..<1024 {
      let delta = targetAddress - gpuBaseVA
      precondition(delta >= 0, "Tried to allocate below GPU base VA.")
      let usmPointer = cpuBaseVA + Int(Int64(bitPattern: delta))
      let buffer = device.makeBuffer(bytesNoCopy: usmPointer, length: size)
      guard let buffer = buffer else {
        return nil
      }
      if buffer.gpuAddress != targetAddress {
        targetAddress = buffer.gpuAddress
        continue
      }
      
      finalBuffer = buffer
      break
    }
    guard let finalBuffer = finalBuffer else {
      fatalError("Took too many attempts to allocate USM heap.")
    }
    
    let gpuAddress = finalBuffer.gpuAddress
    self.buffer = finalBuffer
    self.gpuBaseVA = gpuAddress
    self.cpuBaseVA = cpuBaseVA + Int(gpuAddress - gpuBaseVA)
    self.availableSize = heap.maxAvailableSize(alignment: 16384)
    
    // The heap doesn't give us its base address, but we can find a way to
    // acquire it anyway. First allocate a buffer, and see how it impacts
    // available size. The second buffer should come right after it, and max
    // available size should decrease by the same amount.
    //
    // If we ever see different behavior, we still have enough information to
    // reconstruct the base VA.
    precondition(
      availableSize == heap.size,
      "Heap available size mismatch: \(availableSize) != \(heap.size).")
    precondition(
      availableSize == size,
      "Heap available size mismatch: \(availableSize) != \(size).")
//    precondition(
//      heap.usedSize == 0,
//      "Heap used size not 0: \(heap.usedSize).")
    precondition(
      heap.currentAllocatedSize == size,
      "Heap allocated size not \(size): \(heap.currentAllocatedSize).")
    
    
    
    func estimateBaseVA() -> UInt64 {
      let buffer1 = heap.makeBuffer(length: 16384)!
      
      // Automatic heaps should set the offset to 0.
      var newAvailableSize = heap.maxAvailableSize(alignment: 16384)
      precondition(
        buffer1.heapOffset == 0,
        "Unexpected heap allocation behavior: \(buffer1.heapOffset) != 0")
      precondition(
        newAvailableSize == availableSize - 16384,
        "Unexpected heap allocation behavior: \(newAvailableSize) != \(availableSize - 16384).")
//      precondition(
//        heap.usedSize == 16384,
//        "Unexpected heap allocation behavior: \(heap.usedSize) != \(16384).")
      precondition(
        heap.currentAllocatedSize == size,
        "Heap allocated size not \(size): \(heap.currentAllocatedSize).")
//      precondition(heap.usedSize + newAvailableSize == heap.currentAllocatedSize)
      
      let buffer2 = heap.makeBuffer(length: 16384)!
      
      // Automatic heaps should set the offset to 0.
      newAvailableSize = heap.maxAvailableSize(alignment: 16384)
      precondition(
        buffer2.heapOffset == 0,
        "Unexpected heap allocation behavior: \(buffer2.heapOffset) != 0")
      precondition(
        buffer2.gpuAddress > buffer1.gpuAddress,
        "Unexpected heap allocation behavior: \(buffer2.gpuAddress) not > \(buffer1.gpuAddress).")
      precondition(
        buffer2.gpuAddress == buffer1.gpuAddress + 16384,
        "Unexpected heap allocation behavior: \(buffer2.gpuAddress) != \(buffer1.gpuAddress) + 16384.")
//      precondition(
//        heap.usedSize == 32768,
//        "Unexpected heap allocation behavior: \(heap.usedSize) != \(32768).")
      precondition(
        heap.currentAllocatedSize == size,
        "Heap allocated size not \(size): \(heap.currentAllocatedSize).")
      precondition(
        newAvailableSize == availableSize - 32768,
        "Unexpected heap allocation behavior: \(newAvailableSize) != \(availableSize - 32768).")
//      precondition(heap.usedSize + newAvailableSize == heap.currentAllocatedSize)
      return buffer1.gpuAddress
    }
    self.heapVA = 0
    self.heapVA = estimateBaseVA()
    precondition(self.heapVA == estimateBaseVA())
  
    // Ensure the temporary buffers deallocated.
//    precondition(
//      heap.usedSize == 0,
//      "Heap used size not 0: \(heap.usedSize).")
    precondition(
      heap.currentAllocatedSize == size,
      "Heap allocated size not \(size): \(heap.currentAllocatedSize).")
    let currentAvailableSize = heap.maxAvailableSize(alignment: 16384)
    precondition(
      availableSize == currentAvailableSize,
      "Heap available size mismatch: \(availableSize) != \(currentAvailableSize).")
  }
  
  func allocate(size: Int) -> UnsafeMutableRawPointer {
    precondition(
      size <= availableSize,
      "Didn't check size before allocating buffer from heap.")
    validateSorted()
    
//    let previousUsedSize = heap.usedSize
    let phantomBuffer = heap.makeBuffer(length: size)!
    let offset = Int(phantomBuffer.gpuAddress - heapVA)
    let phantomSize = phantomBuffer.allocatedSize
    phantomBuffers[offset] = phantomBuffer
    
    // Insert before the first element that's greater.
    let insertionIndex =
      allocations.firstIndex(where: { $0.offset > offset }) ?? allocations.count
    let allocation = Allocation(offset: offset, size: phantomSize)
    validateSorted()
    allocations.insert(allocation, at: insertionIndex)
    validateSorted()
    
    self.usedSize += size
//    // Adjust the available size.
//    let newUsedSize = heap.usedSize
//    precondition(
//      newUsedSize == previousUsedSize + phantomSize,
//      "Unexpected heap size.")
    self.availableSize = heap.maxAvailableSize(alignment: 16384)
    return cpuBaseVA + offset
  }
  
  func deallocate(usmPointer: UnsafeMutableRawPointer) {
    validateSorted()
    let offset = usmPointer - cpuBaseVA
//    let previousUsedSize = heap.usedSize
    var phantomSize: Int
    do {
      // In C++, you would explicitly release the buffer.
      let phantomBuffer = phantomBuffers.removeValue(forKey: offset)
      guard let phantomBuffer = phantomBuffer else {
        fatalError("Pointer did not originate from this heap.")
      }
      validateSorted()
      let removalIndex = allocations.firstIndex(where: { $0.offset == offset })!
      validateSorted()
      let removed = allocations.remove(at: removalIndex)
      validateSorted()
      
      phantomSize = phantomBuffer.allocatedSize
      precondition(removed.size == phantomSize)
    }
    
    self.usedSize -= phantomSize
//    // Adjust the available size.
//    let newUsedSize = heap.usedSize
//    precondition(
//      newUsedSize + phantomSize == previousUsedSize,
//      "Unexpected heap size.")
    self.availableSize = heap.maxAvailableSize(alignment: 16384)
  }
  
  // TODO: Make a more efficient traversal method after debugging, which is
  // batchable but returns failures. Unless the compiler will always inline
  // this. You need to profile and see how much overhead it invokes.
  //
  // Defer the actual optimization until later. In fact, it can be a hipSYCL
  // pull request. Just quantify the performance impact.
  func getOffset(usmPointer: UnsafeMutableRawPointer) -> Int {
    let offset = usmPointer - cpuBaseVA
    if offset >= 0 {
      let index = allocations.firstIndex(where: {
        $0.offset <= offset &&
        offset < $0.offset + $0.size
      })
      if let index = index {
//        let allocation = allocations[index]
//        if offset < allocation.offset + allocation.size {
          return offset
//        }
      }
    }
    return -1
  }
}

extension HeapBlock: Hashable {
  func hash(into hasher: inout Hasher) {
    hasher.combine(availableSize)
  }
  
  static func == (lhs: HeapBlock, rhs: HeapBlock) -> Bool {
    lhs.cpuBaseVA == rhs.cpuBaseVA
  }
}
