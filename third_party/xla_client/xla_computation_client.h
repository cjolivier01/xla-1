#ifndef XLA_CLIENT_XRT_COMPUTATION_CLIENT_WSE_H_
#define XLA_CLIENT_XRT_COMPUTATION_CLIENT_WSE_H_

#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include <memory>

namespace xla {

class XlaComputationClient : public XrtComputationClient {
    typedef XrtComputationClient Super;
public:
  /**
   * @brief Create XlaComputationClient object
   * @param options
   * @param topology_proto
   */
  XlaComputationClient(
      Options options,
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto
  );
  /**
   * @brief Destroy the XlaComputationClient object
   */
  ~XlaComputationClient();

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

  static XlaComputationClient *Get() {
    return dynamic_cast<XlaComputationClient *>(Super::Get());
  }

private:
  class XlaClientInfo;
  class GlobalDataHandleMapper;
  mutable std::recursive_mutex xla_client_map_mtx_;
  std::unordered_map<std::string, std::shared_ptr<XlaClientInfo>> xla_client_map_;

//  mutable std::mutex device_mapping_mtx_;
//  std::unordered_map<std::string, std::string> device_mapping_;

  std::unique_ptr<GlobalDataHandleMapper> data_mapper_;

  template<typename CLIENT_T>
  std::shared_ptr<CLIENT_T> GetXlaClient(const std::string& device, bool create = true);
  xla::DeviceHandle GetDeviceHandle(const std::string& device);

  std::vector<ComputationClient::DataPtr> MoveDataBetweenServers(
    const std::vector<ComputationClient::DataPtr>& source_data,
    const std::string& to_device,
    bool release_from_source,
    bool add_mapping_entry
  );
  ComputationClient::DataPtr TransferLiteralToServer(const std::string& device, const Literal& literal);
  /**
   * @brief Is device capable of proxy?
   * @param device
   * @return
   */
  bool IsProxyDevice(const std::string& device) const;
  /**
   * @brief Should this device be proxied right now?
   * @param device
   * @return
   */
  bool UseProxyForDevice(const std::string& device) const;
  void ReleaseXrtData(const std::string& device, int64 handle) override;
};

}  // namespace xla

#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_WSE_H_