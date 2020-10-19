#pragma once

#include <memory>

#include "torch_xla/csrc/tensor.h"

//#define USE_PTWSE_SENTINEL

namespace torch_xla {

struct HashingState {
  explicit HashingState(const xla::hash_t& start_hash)
      : start_hash_(start_hash) {};
  const xla::hash_t start_hash_;
  xla::hash_t pre_prune_hash_ = 0;
  std::size_t pass_ = 0;
  bool fabric_run_ = false;
  bool known_executable_ =
      false;  // optimization when we know this executable already exists
};

class Sentinel {
public:
  // TO BE MOVED TO PTWSE OUT OF SENTINEL INTERFACE
  virtual void SetDeviceProxyAddress(const std::string& device,
                             const std::string& proxy_address) {}
  // TOP BE MOVED TO PTWSE OUT OF SENTINEL INTERFACE
  virtual bool WasMarkStepOnProxy() { return false; }

  // TOP BE MOVED TO PTWSE OUT OF SENTINEL INTERFACE
  virtual bool IsInitialized() { return false; }

  virtual void SetOutputs(const std::vector<at::Tensor>& output_tensors, bool append);
  //bool IsInitialized();

  /**
   * @brief Notification that a MarkStep is beginning
   * @param instances
   * @param hash
   * @param tid
   */
  virtual void NotifyStepMarkerBegin(
      const std::string& device_str, const std::vector<std::string>& devices) {}

  /**
   * @brief Notification that a MarkStep is ending
   */
  virtual void NotifyStepMarkerEnd() {}

  /**
   * @brief General notification when a compile occurs
   * @param instances
   * @param hash
   * @param tid
   */
  virtual void NotifyCompile(
      std::vector<xla::ComputationClient::CompileInstance>& instances,
      xla::hash_t hash, pid_t tid) {}

  /**
   * @brief General notifiication when an Execute occurs
   * @param computation
   * @param device
   * @param hash
   * @param tid
   */
  virtual void NotifyExecute(const xla::ComputationClient::Computation& computation,
                             const std::string& device, xla::hash_t hash, pid_t tid) {}

  /**
   * @brief Notification that a SyncTensorsGraph is occuring.  This means that
   *        a tensor sync is imminent for the given thread, which may or may not be the
   *        same tensor set/graph as the previous
   * @param tensors
   * @param tensors_data
   * @param coll
   * @param computation
   * @return
   */
  virtual std::vector<xla::ComputationClient::DataPtr> NotifyScheduleSyncTensorsGraph(
      std::vector<XLATensor>* tensors,
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      XLATensor::SyncTensorCollection* coll,
      std::shared_ptr<xla::ComputationClient::Computation>& computation) {
    return std::move(tensors_data);
  }

  // Interception and external mapping
  virtual void PostmarkHash(HashingState& state, std::vector<XLATensor>* tensors,
                            XLATensor::SyncTensorCollection& coll) {}

  virtual bool OnHashingComplete(HashingState& state, std::vector<XLATensor>* tensors,
                         XLATensor::SyncTensorCollection& coll) {
    return false;
  }

  virtual bool PreProcessHlo(xla::XlaBuilder* builder,
                     const XLATensor::SyncTensorCollection& coll) {
    return false;
  }

  virtual bool IsSpecialLoweringEnabled() { return false; }

  virtual bool IsForcingCustomLowering();
  //  void SetCompileOnly(bool compile_only);
  //  bool GetCompileOnly(XLATensor::SyncTensorCollection& coll);
  //  bool WasMarkStepOnProxy();

  /**
   * @brief Get the current Sentinel
   * @return
   */
  static std::shared_ptr<Sentinel>& GetSentinel() { return sentinel_; }

  /**
   * @brief Set the Sentinel to use
   * @param sentinel
   */
  static void SetSentinel(std::shared_ptr<Sentinel> sentinel) {
    sentinel_ = std::move(sentinel);
  }

private:
  static std::shared_ptr<Sentinel> sentinel_;
};

/**
 * @brief Convenience RAII class for wrapping a MarkStep
 */
struct MarkStepScope {
  MarkStepScope(const std::string& device_str,
                const std::vector<std::string>& devices) {
    Sentinel::GetSentinel()->NotifyStepMarkerBegin(device_str, devices);
  }
  ~MarkStepScope() {
    Sentinel::GetSentinel()->NotifyStepMarkerEnd();
  }
};


}  // namespace torch_xla