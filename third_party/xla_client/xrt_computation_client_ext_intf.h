#pragma once

#ifndef XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_
#define XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_

#include <cstddef>

namespace xla {

class HloModuleProto;

enum ECompileState {
  ECS_BEFORE_COMPILE,
  ECS_AFTER_COMPILE,
};

enum ECompileResult{
  ECR_ACCEPT,
  ECRT_DEFER
};

enum ERunState {
  ERS_BEFORE_RUN,
  ERS_AFTER_RUN,
};

enum ERunStatus {
  ERS_ACCEPT,
  ERS_DEFER,
};

typedef ptrdiff_t opaque_t;

/**
 * Define interface
 */
struct XrtComputationClientExternalInterface :
    public std::enable_shared_from_this<XrtComputationClientExternalInterface> {

  virtual ~XrtComputationClientExternalInterface() = default;

  virtual void OnCreate(xla::opaque_t obj) = 0;
  virtual void OnDestroy(xla::opaque_t obj) = 0;

  /**
   * @brief Called whenevr
   * @param hash
   * @param hlo_module
   * @param compile_state
   * @return
   */
  virtual ECompileResult OnCompile(
      xla::opaque_t obj,
      std::size_t hash,
      const xla::HloModuleProto &hlo_module,
      const std::vector<std::string>& devices,
      ECompileState compile_state
  ) = 0;

  /**
   * @brief
   * @param hash
   * @param run_state
   * @return
   */
  virtual ERunStatus OnExecuteComputation(
      xla::opaque_t obj,
      std::size_t hash,
      const std::string& device,
      ERunState run_state
  ) = 0;
};

}  // namespace xla

#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_
