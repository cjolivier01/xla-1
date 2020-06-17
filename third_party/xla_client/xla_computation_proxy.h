#ifndef XLA_CLIENT_XLA_COMPUTATION_PROXY_H_
#define XLA_CLIENT_XLA_COMPUTATION_PROXY_H_

#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include <memory>

namespace xla {

class XlaComputationProxy : public XrtComputationClient {
    typedef XrtComputationClient Super;
public:
  /**
   * @brief Create XlaComputationProxy object
   * @param options
   * @param topology_proto
   */
  XlaComputationProxy(
      Options options,
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto
  );
  /**
   * @brief Destroy the XlaComputationProxy object
   */
  ~XlaComputationProxy() = default;

  // Creates a Data object with no actual device handle in it. The device handle
  // will be populated in an asynchrounous fashion.
  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  // Transfers local tensor values to the TPU servers and fetches the handles.
  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  // Compiles a set of computations.
  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device, const ExecuteComputationOptions& options) override;

  void SetDeviceProxyAddress(const std::string& device, const std::string& proxy_address);

  static tensorflow::tpu::TopologyProto InitializeAndFetchTopology(
    const std::string& job, int task_no, const std::string& worker_host_port,
    const tensorflow::ConfigProto& config);

//  void SetDeviceMapping(const std::string& from_device, const std::string& to_device);
//  std::string GetDeviceMapping(const std::string& device);

  static XlaComputationProxy *Get() {
    return dynamic_cast<XlaComputationProxy *>(Super::Get());
  }

private:
  class XlaClientInfo;
  class GlobalDataHandleMapper;

  std::vector<DataPtr> TransferToServerInternal(absl::Span<const TensorSource> tensors);

//  class XrtData : public XrtData {
//  public:
//    //using XlaHandle = XrtHandle;
//    //using XlaHandlePtr = std::shared_ptr<XlaHandle>;
//
//    XrtData(std::string device, std::string proxy_device, Shape device_shape)
//    : XrtData(std::move(device), std::move(device_shape)), proxy_device_(std::move(proxy_device)) {}
//
//    XrtData(XrtComputationClient* self, std::string device, Shape shape, int64 handle)
//      : XrtData(std::move(device), std::move(device_shape)) {
//      handle_ptr = std::make_shared<XrtHandle>(
//        handle,
//        [self, this, handle]() {
//          self->ReleaseXrtData(this->device(), handle);
//        }
//      );
//    };
//    XrtData(XlaComputationProxy* self, std::string device, std::string proxy_device, Shape device_shape, int64 handle)
//      : XrtData(std::move(device), std::move(device_shape)), proxy_device_(std::move(proxy_device)) {
//      handle_ptr = std::make_shared<XrtHandle>(
//        handle,
//        [self, this, handle]() {
//          self->ReleaseXrtData(this->device(), handle);
//        }
//      );
//    }
//
//    const std::string& get_proxy_device() const { return proxy_device_; }
//
////    int64 get_handle() const { return handle_ptr_->handle; }
////    OpaqueHandle GetOpaqueHandle() override { return get_handle(); }
//
//    const std::string& device() const override { return proxy_device_.empty() ? XrtData::device() : proxy_device_; }
//
//    void Assign(const Data& data) override;
//
////    bool HasValue() const override { return handle_ptr_ != nullptr; }
//
//  private:
//    //XlaHandlePtr handle_ptr_;
//    std::string proxy_device_;
//  };

  mutable std::recursive_mutex xla_client_map_mtx_;
  std::unordered_map<std::string, std::shared_ptr<XlaClientInfo>> xla_client_map_;

  std::unique_ptr<GlobalDataHandleMapper> data_mapper_;

  std::shared_ptr<xla::ServiceInterface> GetXlaClient(const std::string& device, bool create = true);
  xla::DeviceHandle GetDeviceHandle(const std::string& device);

  mutable std::mutex proxy_executable_set_mtx_;
  std::unordered_set<uint64_t> proxy_executable_set_;

  mutable std::mutex proxy_mapping_mtx_;
  std::unordered_map<std::string, std::string> proxy_mapping_;

  std::vector<ComputationClient::DataPtr> MoveDataBetweenDevices(
    const std::vector<ComputationClient::DataPtr>& source_data,
    const std::string& to_device,
    bool release_from_source,
    bool add_mapping_entry
  );
  /**
   * @brief Is device capable of proxy?
   * @param device
   * @return
   */

  std::vector<DataPtr> NormalizeDataToDevice(absl::Span<const DataPtr> tensors, const std::string& device, bool in_place);

  ComputationClient::DataPtr TransferLiteralToServer(
    const std::string& device,
    const Literal& literal
  );
  //bool IsProxyDevice(const std::string& device) const;
  bool SetProxyForDevice(const std::string& source_device, const std::string& proxy_device);
  bool ShouldCloneDataForDevice(const std::string& device) const;
  bool IsProxyExecutable(uint64_t executable_handle) const;
  void AddProxyExecutable(uint64_t executable_handle);

  /**
   * @brief Should this device be proxied right now?
   * @param device
   * @return
   */
  void ReleaseXrtData(const std::string& device, int64 handle) override;
  void ReleaseXlaProxyData(const std::string& device, int64 handle);
};

}  // namespace xla

#endif  // XLA_CLIENT_XLA_COMPUTATION_PROXY_H_