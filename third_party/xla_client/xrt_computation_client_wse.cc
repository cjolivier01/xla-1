
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_ext_intf.h"
#include "tensorflow/core/util/util.h"

#include "tensorflow/core/protobuf/tpu/topology.pb.h"

#include <stdexcept>

namespace xla {

namespace {



std::shared_ptr<XrtComputationClientExternalInterface> callback_interface_{nullptr};

xla::opaque_t GetOpaque(const XrtComputationClientWse *object_ptr) {
    return reinterpret_cast<xla::opaque_t>(object_ptr);
}

}  // namespace

XrtComputationClientWse::XrtComputationClientWse(
    Options options,
    std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto
) : XrtComputationClient(std::move(options), std::move(topology_proto)) {
    setenv("XRT_MASTER_ALLOW_SAME_TASKS", "1", true);
    std::cout << "CREATE XrtComputationClientWse" << ENDL;
    if(callback_interface_) {
        callback_interface_->OnCreate(GetOpaque(this));
    }
}

XrtComputationClientWse::~XrtComputationClientWse() {
    std::cout << "DESTROY XrtComputationClientWse" << ENDL;
    if(callback_interface_) {
        callback_interface_->OnDestroy(GetOpaque(this));
    }
}

void XrtComputationClientWse::SetExternalInterface(
    std::shared_ptr<XrtComputationClientExternalInterface> callback_interface
) {
    if(!callback_interface_) {
        callback_interface_ = callback_interface->shared_from_this();
    } else {
        if (callback_interface != callback_interface_) {
            throw std::runtime_error(
                "An attempt was made to set the Xrt callback interface more than once"
            );
        }
    }
}

ComputationClient::DataPtr XrtComputationClientWse::CreateDataPlaceholder(std::string device, Shape shape) {
    return Super::CreateDataPlaceholder(device, shape);
}

// Transfers local tensor values to the TPU servers and fetches the handles.
std::vector<ComputationClient::DataPtr> XrtComputationClientWse::TransferToServer(
  absl::Span<const TensorSource> tensors
) {
    return Super::TransferToServer(tensors);
}

// Reads the tensor literal values stored at TPU server sites, behind the
// supplied handles.
std::vector<Literal> XrtComputationClientWse::TransferFromServer(
  absl::Span<const DataPtr> handles
) {
    return Super::TransferFromServer(handles);
}

// Compiles a set of computations.
std::vector<ComputationClient::ComputationPtr> XrtComputationClientWse::Compile(
  std::vector<CompileInstance> instances
) {
    if(callback_interface_) {
        assert(instances.size() == 1);
        const ECompileResult comp_result = callback_interface_->OnCompile(
            GetOpaque(this),
            instances[0].computation.proto().id(),  // good enough or need hash from PTXLA layer?
            instances[0].computation.proto(),
            instances[0].devices,
            ECS_BEFORE_COMPILE
        );
        if (comp_result != ECRT_DEFER) {
            // We compiled it ourselves
            assert(false);
            return std::vector<ComputationClient::ComputationPtr>();
        }
    }
    return Super::Compile(std::move(instances));
}

// Executes computation with arguments and returns the result.
// The passed device must match the common device of the arguments Data.
// If options.explode_tuple is true, the output tuple will be decomposed into
// its single elements.
std::vector<ComputationClient::DataPtr> XrtComputationClientWse::ExecuteComputation(
  const Computation& computation,
  absl::Span<const DataPtr> arguments,
  const std::string& device,
  const ExecuteComputationOptions& options
) {
    if(callback_interface_) {
        ERunStatus run_status = callback_interface_->OnExecuteComputation(
            GetOpaque(this),
            computation.computation().proto().id(),
            device,
            ERS_BEFORE_RUN
        );
        if (run_status != ERS_DEFER) {
            // No data returned yet :(
            assert(false);
            return std::vector<ComputationClient::DataPtr>{nullptr};
        }
    }
    return Super::ExecuteComputation(computation, arguments, device, options);
}


}  // namespace xla