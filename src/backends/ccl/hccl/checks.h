#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_CHECKS_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_CHECKS_H_

#include <hccl.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_HCCL(result) \
  ::infini::ccl::detail::CheckHcclImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckHcclImpl(hcclResult_t hccl_result, const char *file,
                                  int line) {
  if (hccl_result != hcclSuccess) {
    std::cerr << "backend(hccl) HCCL error code: " << hccl_result << " at line "
              << line << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_CHECKS_H_
