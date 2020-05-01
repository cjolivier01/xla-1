#pragma once

#include <stack>
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

#include "tensorflow/compiler/xla/xla_client/computation_client.h"

namespace torch_xla {
enum EPythonState {
  EPS_INVALID = 0,
  EPS_IN_TRAIN_LOOP = 1,
  EPS_IN_DATA_BATCH = 2,
  EPS_IN_OPTIMIZER_STEP = 3,
  EPS_IN_DEBUG = 4,
};

//extern std::stack<int> python_state_;

EPythonState GetPythonState();
void PushPythonState(EPythonState state);
EPythonState PopPythonState();

extern std::atomic<size_t> active_parameter_count;

template<typename CB>

void XLATensor::print_tensors(const std::string& label, const std::vector<XLATensor>& tensors, CB cb) {
  std::vector<XLATensor> ats;
  for (const XLATensor& t : tensors) {
    if (cb(t)) {
      ats.reserve(tensors.size());
      ats.emplace_back(t);
    }
  }
  print_all_tensors(label, ats);
}

class THelper {
  public:

  static inline at::Tensor get_tensor(const XLATensor &t) {
    if (t.data()->tensor_data.has_value()) {
      return t.data()->tensor_data.value();
    }
    return bridge::AtenFromXlaTensor(t);
  }

  static inline bool has_tensor(const XLATensor &t) {
    const at::Tensor tensor = get_tensor(t);
    return tensor.defined();
  }

  static inline bool is_weight(const at::Tensor& tensor) {
    return tensor.requires_grad() && tensor.is_leaf();
  }

  static inline bool is_weight(const XLATensor &t) {
    at::Tensor tensor = get_tensor(t);
    if (!tensor.defined()) {
      return false;
    }
    return is_weight(tensor);
  }

  template<typename CB>
  struct _Not {
    CB cb_;
    inline bool operator()(const XLATensor &t) const {
      return !cb_(t);
    };
  };

  template<typename CB>
  static inline _Not<CB> Not(const CB cb) {
    return _Not<CB>{cb};
  }
};

class CompileWatcher {
public:
  typedef void *compiler_t;  // TODO: make this the device
  typedef size_t hash_t;
  static void NotifyCompile(compiler_t opaque, hash_t hash);
  static void NotifyExecute(compiler_t opaque, hash_t hash);
  static void NotifyStepMarker(compiler_t opaque);
  static bool IsWseRunReady(compiler_t opaque, hash_t hash);
  static bool IsWseRunReady(compiler_t opaque);
  static bool IsWseRunning(compiler_t opaque);
  static void CompileCacheHit(hash_t hash);
  static void CompileCacheMiss(hash_t hash);
  static bool IsAllowedOutput(compiler_t opaque, XLATensor tensor);
  static void SetInputsOutputs(compiler_t opaque,
                               const std::vector<at::Tensor>& input_tensors,
                               const std::vector<at::Tensor>& output_tensors);
  static std::vector<xla::ComputationClient::DataPtr> WseExecute(
      compiler_t opaque,
      hash_t hash,
      std::shared_ptr<XLATensor::Async> async);

private:
  static void Reset(compiler_t opaque);
};

}  // namespace torch_xla
