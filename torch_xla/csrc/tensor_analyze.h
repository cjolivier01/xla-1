#pragma once

#include <sys/syscall.h>

#include <stack>

#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

namespace torch_xla {
enum EPythonState {
  EPS_INVALID = 0,
  EPS_IN_TRAIN_LOOP = 1,
  EPS_IN_DATA_BATCH = 2,
  EPS_IN_OPTIMIZER_STEP = 3,
  EPS_IN_DEBUG = 4,
};

//extern std::stack<int> python_state_;

EPythonState GetPythonState(pid_t tid);
void PushPythonState(EPythonState state);
void PopPythonState();

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

/**
 * This is mostly an exploratory class and will go away eventually
 */
class CompileWatcher {
public:
  typedef xla::hash_t hash_t;

  // Configuration
  static void SetDeviceProxyAddress(const std::string& device, const std::string& proxy_address);
  static void SetOutputs(const std::vector<at::Tensor>& output_tensors, bool append);

  // Notification handlers
  static void NotifyCompile(std::vector<xla::ComputationClient::CompileInstance>& instances, hash_t hash, pid_t tid);
  static void NotifyExecute(const std::string& device, hash_t hash, pid_t tid);
  static void NotifyStepMarker(const std::vector<std::string>& devices);

  // Interception and external mapping
  static bool IsReadyHash(hash_t hash, pid_t tid);
  static bool IsAllowedOutput(const XLATensor& tensor, pid_t tid);
  static bool PreProcessHlo(xla::XlaBuilder *builder, hash_t hash, pid_t tid);

private:
  static Device GetDevice();
  static void SetDeviceMapping(const std::string& from_device, const std::string& to_device);
  static const Device& GetDeviceMapping(const Device& device);
  static std::string GetDeviceMapping(const std::string& device);
  static bool IsTrainingThread(pid_t tid);
  static bool IsWseRunning(pid_t tid);
  static bool IsWseRunStep(pid_t tid);
  static void SetAllDevices(const std::vector<std::string>& all_devices);
  static bool HasWseDevices();
  static bool Reset(pid_t tid, bool reset_hash);

  //
  // Data
  //
  static std::vector<std::string> wse_devices_;
  static std::mutex device_mapping_mtx_;
  static std::unordered_map<std::string, std::pair<Device, bool>> device_mapping_;
};

inline pid_t gettid() {
  return syscall(__NR_gettid);
}

}  // namespace torch_xla
