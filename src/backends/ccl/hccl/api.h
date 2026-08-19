#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_API_H_

#include <hccl.h>

#include <cstddef>

#include "backends/ccl/common/api.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct HcclDataTypeTraits;

template <Device::Type device>
struct HcclApi {
  static constexpr BackendType kBackendType = BackendType::kHccl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = hcclComm_t;
  using UniqueId = hcclUniqueId;
  using Result = hcclResult_t;
  using DataType = hcclDataType_t;
  using RedOp = hcclRedOp_t;
  using Stream = typename Runtime<device>::Stream;

  static ReturnStatus Check(Result result) {
    if (result != hcclSuccess) {
      LOG(hcclGetErrorString(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result GetUniqueId(UniqueId *id) { return hcclGetUniqueId(id); }

  static Result CommInitRank(Comm *comm, int nranks, UniqueId id, int rank) {
    return hcclCommInitRank(comm, nranks, id, rank);
  }

  static Result CommDestroy(Comm comm) { return hcclCommDestroy(comm); }

  static Result AllReduce(const void *send_buff, void *recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return hcclAllReduce(send_buff, recv_buff, count, data_type, op, comm,
                         stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_API_H_
