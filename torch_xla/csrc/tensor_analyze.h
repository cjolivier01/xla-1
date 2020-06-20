#pragma once

#include <sys/syscall.h>

#include <stack>
#include <map>

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

enum EPythonState {
  EPS_INVALID = 0,
  EPS_IN_TRAIN_LOOP = 1,
  EPS_IN_MARK_STEP = 2,
  EPS_IN_DATA_BATCH = 3,
  EPS_IN_OPTIMIZER_STEP = 4,
  EPS_IN_DEBUG = 5,
};

// extern std::stack<int> python_state_;

EPythonState GetPythonState(pid_t tid);
void PushPythonState(EPythonState state);
void PopPythonState();

struct MarkStepScope {
  MarkStepScope(const std::string& device_str,
                const std::vector<std::string>& devices);
  ~MarkStepScope();
};

template <typename CB>
void XLATensor::print_tensors(const std::string& label,
                              const std::vector<XLATensor>& tensors, CB cb) {
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
  static inline at::Tensor get_tensor(const XLATensor& t) {
    if (t.data()->tensor_data.has_value()) {
      return t.data()->tensor_data.value();
    }
    return bridge::AtenFromXlaTensor(t);
  }

  static inline bool has_tensor(const XLATensor& t) {
    const at::Tensor tensor = get_tensor(t);
    return tensor.defined();
  }

  static inline bool is_weight(const at::Tensor& tensor) {
    return tensor.requires_grad() && tensor.is_leaf();
  }

  static inline bool is_weight(const XLATensor& t) {
    at::Tensor tensor = get_tensor(t);
    if (!tensor.defined()) {
      return false;
    }
    return is_weight(tensor);
  }

  template <typename CB>
  struct _Not {
    CB cb_;
    inline bool operator()(const XLATensor& t) const { return !cb_(t); };
  };

  template <typename CB>
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
  static void SetDeviceProxyAddress(const std::string& device,
                                    const std::string& proxy_address);
  static void SetOutputs(const std::vector<at::Tensor>& output_tensors,
                         bool append);

  // Notification handlers
  static void NotifyCompile(
      std::vector<xla::ComputationClient::CompileInstance>& instances,
      hash_t hash, pid_t tid);
  static void NotifyExecute(const std::string& device, hash_t hash, pid_t tid,
                            bool scheduled);
  static std::vector<xla::ComputationClient::DataPtr>
  NotifyScheduleSyncTensorsGraph(
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      XLATensor::SyncTensorCollection* coll,
      std::shared_ptr<xla::ComputationClient::Computation>& computation);

  // Interception and external mapping
  static xla::hash_t PostmarkHash(std::vector<XLATensor>* tensors,
                                  XLATensor::SyncTensorCollection& coll);
  static void OnHashChange(const xla::hash_t& prev_hash,
                           const XLATensor::SyncTensorCollection& coll);

  // static xla::hash_t OnEndHashCalc(const xla::hash_t& prev_hash, const
  // xla::hash_t& new_hash);
  static bool PreProcessHlo(xla::XlaBuilder* builder,
                            const XLATensor::SyncTensorCollection& coll);

  static bool IsSpecialLowering();

  static std::map<std::string, std::string> GetStats(bool reset_stats);

 private:
  static void NotifyStepMarkerBegin(const std::string& device_str,
                                    const std::vector<std::string>& devices);
  static void NotifyStepMarkerEnd();

  static bool IsAllowedOutput(const XLATensor& tensor,
                              XLATensor::SyncTensorCollection& coll);
  // static Device GetDevice();
  //  static void SetDeviceMapping(const std::string& from_device, const
  //  std::string& to_device); static const Device& GetDeviceMapping(const
  //  Device& device); static std::string GetDeviceMapping(const std::string&
  //  device);
  static bool IsTrainingThread(pid_t tid);
  // static bool IsWseRunning(pid_t tid);
  static bool IsQualifyingStep(pid_t tid /*, bool or_higher = false*/);
  static void SetAllDevices(const std::vector<std::string>& all_devices);
  static bool HasWseDevices();
  // static bool Reset(pid_t tid, bool reset_hash);

  //
  // Data
  //
  static std::vector<std::string> wse_devices_;
  static std::mutex device_mapping_mtx_;
  static std::unordered_map<std::string, std::pair<Device, bool>>
      device_mapping_;
  friend struct MarkStepScope;
};

inline pid_t gettid() { return syscall(__NR_gettid); }

}  // namespace torch_xla
