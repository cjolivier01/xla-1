#ifndef XLA_CLIENT_XRT_COMPUTATION_CLIENT_WSE_H_
#define XLA_CLIENT_XRT_COMPUTATION_CLIENT_WSE_H_

#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

#include <memory>

namespace xla {

class XrtComputationClientExternalInterface;

class XrtComputationClientWse : public XrtComputationClient {
    typedef XrtComputationClient Super;
public:
  /**
   * @brief Create XrtComputationClientWse object
   * @param options
   * @param topology_proto
   */
  XrtComputationClientWse(
      Options options,
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto);
  /**
   * @brief Destroy the XrtComputationClientWse object
   */
  ~XrtComputationClientWse();

  // Transfers local tensor values to the TPU servers and fetches the handles.
  virtual std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  virtual std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  // Compiles a set of computations.
  virtual std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  virtual std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device, const ExecuteComputationOptions& options) override;

  // TODO: This will be replaced with grpc
  static void SetExternalInterface(
      std::shared_ptr<XrtComputationClientExternalInterface> callback_interface
   );
};

}  // namespace xla

#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_WSE_H_