#pragma once

namespace torch_xla {

//class Sentinel {
//public:
//  void SetOutputs(const std::vector<at::Tensor>& output_tensors,
//                         bool append);
////  bool IsInitialized();
//
//  // Notification handlers
////  void NotifyCompile(
////      std::vector<xla::ComputationClient::CompileInstance>& instances,
////      hash_t hash, pid_t tid);
////  void NotifyExecute(
////      const xla::ComputationClient::Computation& computation,
////      const std::string& device, hash_t hash, pid_t tid);
//  std::vector<xla::ComputationClient::DataPtr>
//  NotifyScheduleSyncTensorsGraph(
//      std::vector<XLATensor>* tensors,
//      std::vector<xla::ComputationClient::DataPtr> tensors_data,
//      XLATensor::SyncTensorCollection* coll,
//      std::shared_ptr<xla::ComputationClient::Computation>& computation);
//
//  // Interception and external mapping
//  void PostmarkHash(HashingState& state, std::vector<XLATensor>* tensors,
//                           XLATensor::SyncTensorCollection& coll);
//  bool OnHashingComplete(HashingState& state,
//                                std::vector<XLATensor>* tensors,
//                                XLATensor::SyncTensorCollection& coll);
//
//  bool PreProcessHlo(xla::XlaBuilder* builder,
//                            const XLATensor::SyncTensorCollection& coll);
//
//  bool IsSpecialLoweringEnabled();
//
////  std::map<std::string, std::string> GetStats(bool reset_stats);
//
////  bool IsAllowedOutput(const XLATensor& tensor,
////                              XLATensor::SyncTensorCollection& coll,
////                              bool* is_restricting);
//  bool IsForcingCustomLowering();
////  void SetCompileOnly(bool compile_only);
////  bool GetCompileOnly(XLATensor::SyncTensorCollection& coll);
////  bool WasMarkStepOnProxy();
//};
//
}  // namespace torch_xla