#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_MARS_API_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_MARS_API_H_

#include "backends/ccl/hccl/api.h"
#include "devices/mars/runtime_.h"

namespace infini::ccl {

template <>
struct HcclDataTypeTraits<Device::Type::kMars> {
  static constexpr hcclDataType_t kBFloat16 = hcclBfloat16;
};

template <>
struct CclApi<BackendType::kHccl, Device::Type::kMars>
    : HcclApi<Device::Type::kMars> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_MARS_API_H_
