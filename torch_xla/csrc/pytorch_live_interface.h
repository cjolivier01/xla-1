#pragma once

#include <string>

namespace xla { class HloModuleProto; }

namespace pytorch_live {

enum ECompileState {
  ECS_BEFORE_COMPILE,
  ECS_AFTER_COMPILE,
};

enum ERunState {
  ERS_BEFORE_RUN,
  ERS_AFTER_RUN,
};

enum ERunStatus {
    ERS_OK,
};

/**
 * Define interface
 */
struct IPytorchLive {
  /**
   * @brief Called whenevr
   * @param hash
   * @param hlo_module
   * @param compile_state
   * @return
   */
  virtual int on_compile(
      std::size_t hash,
      const xla::HloModuleProto &hlo_module,
      ECompileState compile_state
  ) = 0;

  /**
   * @brief
   * @param hash
   * @param run_state
   * @return
   */
  virtual ERunStatus on_run(std::size_t hash, ERunState run_state) = 0;
};

}  // namespace pytorch_live