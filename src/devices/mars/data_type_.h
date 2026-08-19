#ifndef INFINI_CCL_DEVICES_MARS_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_MARS_DATA_TYPE_H_

// clang-format off
#include <common/hpcc_bfloat16.h>
#include <common/hpcc_fp16.h>
#include <hcr/hc_runtime.h>
// clang-format on

#include "data_type_impl.h"
#include "devices/mars/device_.h"

namespace infini::ccl {

template <>
struct TypeMap<Device::Type::kMars, DataType::kFloat16> {
  using type = __half;
};

template <>
struct TypeMap<Device::Type::kMars, DataType::kBFloat16> {
  using type = __hpcc_bfloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MARS_DATA_TYPE_H_
