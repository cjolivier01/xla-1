#pragma once

#include <stack>
#include "tensor.h"
#include "aten_xla_bridge.h"

namespace torch_xla {
enum EPythonState {
  EPS_INVALID = 0,
  EPS_IN_TRAIN_LOOP = 1,
  EPS_IN_DATA_BATCH = 2
};

//extern std::stack<int> python_state_;

EPythonState GetPythonState();
void PushPythonState(EPythonState state);
EPythonState PopPythonState();

extern std::atomic<size_t> active_parameter_count;

template<typename CB>

void XLATensor::print_tensors(const std::vector<XLATensor>& tensors, CB cb) {
  std::vector<XLATensor> ats;
  ats.reserve(tensors.size());
  for (const XLATensor& t : tensors) {
    if (cb(t)) {
      ats.emplace_back(t);
    }
  }
  print_all_tensors(ats);
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
  typedef void *compiler_t;
  static void NotifyCompile(compiler_t opaque);
  static void NotifyRun(compiler_t opaque);
  static void NotifyMarkStep(compiler_t opaque);

  static void SetInputsOutputs(compiler_t opaque,
                               const std::vector<at::Tensor>& input_tensors,
                               const std::vector<at::Tensor>& output_tensors);


private:
  static void Reset(compiler_t opaque);
};

}  // namespace torch_xla
