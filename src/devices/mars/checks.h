#ifndef INFINI_CCL_DEVICES_MARS_CHECKS_H_
#define INFINI_CCL_DEVICES_MARS_CHECKS_H_

// clang-format off
#include <hcr/hc_runtime.h>
// clang-format on

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_HPCC(result) \
  ::infini::ccl::detail::CheckHpccImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckHpccImpl(hcError_t hpcc_result, const char *file,
                                  int line) {
  if (hpcc_result != hcSuccess) {
    hcGetLastError();
    std::cerr << "HPCC error code: " << hpcc_result << " at line " << line
              << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MARS_CHECKS_H_
