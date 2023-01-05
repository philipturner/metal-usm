//
//  MainFile.swift
//  MetalAllocator
//
//  Created by Philip Turner on 1/4/23.
//

import Metal
import QuartzCore

// If we can use `mlock` instead of `useResource`, that saves a heck ton of
// overhead and means we can allocate much more simply, with better granularity.
func mainFunc() {
  let device = MTLCreateSystemDefaultDevice()!
  let commandQueue = device.makeCommandQueue()!
  let library = device.makeDefaultLibrary()!
  let function = library.makeFunction(name: "vectorAddition")!
  let pipeline = try! device.makeComputePipelineState(function: function)
  
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
      
      // TODO: Test what happens when I don't lock the page (I'll have to
      // restart my Mac because the system will freeze). Then repeat this
      // nonsense on the iPhone too.
      
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
  
  
  let commandBuffer = commandQueue.makeCommandBuffer()!
  let encoder = commandBuffer.makeComputeCommandEncoder()!
//  encoder.useResources([usmAllocations[bufferA]!, usmAllocations[bufferB]!, usmAllocations[bufferC]!, usmAllocations[bufferD]!], usage: [.read, .write])
//  encoder.useResources([usmAllocations[bufferD]!], usage: [.read, .write])
  encoder.useResource(bufferE, usage: [.read, .write])
  encoder.setComputePipelineState(pipeline)
  
  // Two different GPU VAs can map to the same CPU VA - sort of. Make 1/128
  // chunks of global memory, as actual Metal buffers. Then make `MTLHeap` with
  // the purgeable state erasing it from memory. The buffer's complementary heap
  // decides where to allocate addresses within it. Borrow the idea from PyTorch
  // that will cache larger allocations, for non-1/128 chunks.
  //
  // The `MTLHeap` is created once by setting all zeroes, then erasing via
  // purging. Next, allocate the complementary Metal buffer from bytesNoCopy.
  // Perform phantom `malloc` within the `MTLHeap`, then translate to the
  // complementary buffer. Only the buffer is actually `useResource`'d and
  // aligned with the USM address space.
  
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
