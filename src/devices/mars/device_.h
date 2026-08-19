#ifndef INFINI_CCL_DEVICES_MARS_DEVICE_H_
#define INFINI_CCL_DEVICES_MARS_DEVICE_H_

// clang-format off
#include <hcr/hc_runtime.h>
// clang-format on

#include "checks.h"
#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kMars> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kMars>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  hcPointerAttribute_t attr;
  INFINI_CHECK_HPCC(hcPointerGetAttributes(&attr, ptr));

  if (attr.type == hcMemoryTypeDevice || attr.type == hcMemoryTypeManaged ||
      attr.type == hcMemoryTypeArray) {
    return MemorySpace::kDevice;
  }

  return MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MARS_DEVICE_H_
