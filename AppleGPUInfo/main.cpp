//
//  main.cpp
//  TestMacIOKitGPU
//
//  Created by Philip Turner on 1/3/23.
//

#include <iostream>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>

__attribute__((__noinline__))
inline void print_error_message(const char *message) {
  std::cerr << std::endl << message << std::endl;
  exit(-1);
  
}

__attribute__((__always_inline__))
inline void release_assert(bool condition, const char *message) {
  if (!condition) {
    print_error_message(message);
  }
}

// The caller must release the entry.
inline io_registry_entry_t get_gpu_entry() {
  // Class hierarchy: IOGPU -> AGXAccelerator -> AGXFamilyAccelerator
  // We could go with IOGPU, but we want to restrict this to Apple silicon.
  CFMutableDictionaryRef match_dictionary = IOServiceMatching("AGXAccelerator");
  release_assert(match_dictionary, "Could not find AGXAccelerator service.");
  
  // Get the GPU's entry object.
  io_iterator_t entry_iterator;
  kern_return_t error = IOServiceGetMatchingServices(
    kIOMainPortDefault, match_dictionary, &entry_iterator);
  release_assert(!error, "No objects match AGXAccelerator service.");
  io_registry_entry_t gpu_entry = IOIteratorNext(entry_iterator);
  release_assert(!IOIteratorNext(entry_iterator), "Found multiple GPUs.");
  
  // Release acquired objects.
  IOObjectRelease(entry_iterator);
  return gpu_entry;
}

// Number of GPU cores.
inline int64_t get_gpu_core_count(io_registry_entry_t gpu_entry) {
#if TARGET_OS_IPHONE
  // TODO: Determine the core count on iOS through something like DeviceKit.
#else
  // Get the number of cores.
  CFNumberRef gpu_core_count = (CFNumberRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault,
    0);
  release_assert(gpu_core_count, "Could not find 'gpu-core-count' property.");
  CFNumberType type = CFNumberGetType(gpu_core_count);
  release_assert(type == kCFNumberSInt64Type, "'gpu-core-count' not sInt64.");
  int64_t value;
  bool retrieved_value = CFNumberGetValue(gpu_core_count, type, &value);
  release_assert(retrieved_value, "Could not fetch 'gpu-core-count' value.");
  
  // Release acquired objects.
  CFRelease(gpu_core_count);
  return value;
#endif
}

// Clock speed in MHz.
inline int64_t get_gpu_max_clock_speed(io_registry_entry_t gpu_entry) {
  CFStringRef model = (CFStringRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("model"), kCFAllocatorDefault, 0);
  release_assert(model, "Could not find 'model' property.");
  
  // Newest data on each model's clock speed can be located at:
  // https://github.com/philipturner/metal-benchmarks
  if (CFStringHasPrefix(model, CFSTR("Apple M1"))) {
    if (CFStringHasSuffix(model, CFSTR("M1"))) {
      return 1278;
    } else if (CFStringHasSuffix(model, CFSTR("Pro"))) {
      return 1296;
    } else if (CFStringHasSuffix(model, CFSTR("Max"))) {
      return 1296;
    } else if (CFStringHasSuffix(model, CFSTR("Ultra"))) {
      return 1296;
    } else {
      // Return a default for unrecognized models.
      return 1278;
    }
  } else if (CFStringHasPrefix(model, CFSTR("Apple M2"))) {
    if (CFStringHasSuffix(model, CFSTR("M2"))) {
      return 1398;
    } else {
      // Return a default for unrecognized models.
      return 1398;
    }
  } else if (CFStringHasPrefix(model, CFSTR("Apple M"))) {
    // Return a default for unrecognized models.
    return 1398;
  } else if (CFStringHasPrefix(model, CFSTR("Apple A"))) {
    if (CFStringHasSuffix(model, CFSTR("A14"))) {
      return 1278;
    } else if (CFStringHasSuffix(model, CFSTR("A15"))) {
      return 1336;
    } else if (CFStringHasSuffix(model, CFSTR("A16"))) {
      return 1336;
    } else {
      // Return a default for unrecognized models.
      return 1336;
    }
  } else {
    // Could not extract any information about the GPU.
    return 0;
  }
}

// Each function call is expensive; the caller should cache results.
// Eventually, this should become an external library imported by hipSYCL.
int main(int argc, const char * argv[]) {
  io_registry_entry_t gpu_entry = get_gpu_entry();
  int64_t core_count = get_gpu_core_count(gpu_entry);
  int64_t max_clock_speed = get_gpu_max_clock_speed(gpu_entry);
  IOObjectRelease(gpu_entry);
  
  std::cout << "The GPU has " << core_count << " cores." << '\n';
  std::cout << "The GPU runs at " << max_clock_speed << " MHz." << '\n';
  return 0;
}
