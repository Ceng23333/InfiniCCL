#ifndef INFINI_CCL_DEVICES_MARS_CASTER_CUH_
#define INFINI_CCL_DEVICES_MARS_CASTER_CUH_

#include "caster.h"
#include "data_type_.h"

namespace infini::ccl {

template <>
struct HardwareCastImpl<Device::Type::kMars, float, half> {
  __host__ __device__ static float Apply(half x) { return __half2float(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, half, float> {
  __host__ __device__ static half Apply(float x) { return __float2half(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, float, __hpcc_bfloat16> {
  __host__ __device__ static float Apply(__hpcc_bfloat16 x) {
    return __bfloat162float(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, __hpcc_bfloat16, float> {
  __host__ __device__ static __hpcc_bfloat16 Apply(float x) {
    return __float2bfloat16(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, __hpcc_bfloat16, int> {
  __host__ __device__ static __hpcc_bfloat16 Apply(int x) {
    return __int2bfloat16_rn(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, __half, int> {
  __host__ __device__ static __half Apply(int x) { return __int2half_rn(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, __hpcc_bfloat16, double> {
  __host__ __device__ static __hpcc_bfloat16 Apply(double x) {
    return __double2bfloat16(x);
  }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, __half, double> {
  __host__ __device__ static __half Apply(double x) { return __double2half(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kMars, __half, __hpcc_bfloat16> {
  __host__ __device__ static __half Apply(__hpcc_bfloat16 x) {
    return __float2half_rn(__bfloat162float(x));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MARS_CASTER_CUH_
