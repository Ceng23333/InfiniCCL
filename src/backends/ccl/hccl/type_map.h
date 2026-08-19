#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_

#include <hccl.h>

#include <string>

#include "backends/ccl/common/api.h"
#include "backends/ccl/hccl/api.h"
#include "comm_impl.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device>
struct HcclDataTypeMap {
  static constexpr ConstexprMap<DataType, hcclDataType_t, 12> kMap{{{
      {DataType::kInt8, hcclInt8},
      {DataType::kInt16, hcclNumTypes},
      {DataType::kInt32, hcclInt32},
      {DataType::kInt64, hcclInt64},
      {DataType::kUInt8, hcclUint8},
      {DataType::kUInt16, hcclNumTypes},
      {DataType::kUInt32, hcclUint32},
      {DataType::kUInt64, hcclUint64},
      {DataType::kFloat32, hcclFloat32},
      {DataType::kFloat64, hcclFloat64},
      {DataType::kFloat16, hcclFloat16},
      {DataType::kBFloat16, HcclDataTypeTraits<device>::kBFloat16},
  }}};
};

static const ConstexprMap<ReductionOpType, hcclRedOp_t, 5> kHcclOpMap{{{
    {ReductionOpType::kSum, hcclSum},
    {ReductionOpType::kProd, hcclProd},
    {ReductionOpType::kMax, hcclMax},
    {ReductionOpType::kMin, hcclMin},
    {ReductionOpType::kAvg, hcclAvg},
}}};

template <Device::Type device>
inline hcclDataType_t DataTypeToHcclType(DataType dtype) {
  auto hccl_dtype = HcclDataTypeMap<device>::kMap.at(dtype);

  if (hccl_dtype == hcclNumTypes) {
    LOG(("DataType '" + std::string(kDataTypeToDesc.at(dtype)) +
         "' is not supported by the HCCL backend")
            .c_str());
  }

  return hccl_dtype;
}

inline hcclRedOp_t RedOpToHcclOp(ReductionOpType red_op) {
  return kHcclOpMap.at(red_op);
}

template <Device::Type device>
struct CclTypeMap<BackendType::kHccl, device> {
  using Api = CclApi<BackendType::kHccl, device>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType *backend_dtype) {
    auto hccl_dtype = DataTypeToHcclType<device>(dtype);
    if (hccl_dtype == hcclNumTypes) {
      return false;
    }
    *backend_dtype = hccl_dtype;
    return true;
  }

  static bool ToBackendRedOp(ReductionOpType red_op,
                             typename Api::RedOp *backend_op) {
    *backend_op = RedOpToHcclOp(red_op);
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_
