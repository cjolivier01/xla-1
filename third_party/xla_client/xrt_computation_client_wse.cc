
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_ext_intf.h"
#include "tensorflow/core/util/util.h"

#include "tensorflow/core/protobuf/tpu/topology.pb.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace xla {
namespace {

std::shared_ptr <ptxla::XrtComputationClientExternalInterface> callback_interface_{nullptr};

xla::ptxla::opaque_t GetOpaque(const XrtComputationClientWse *object_ptr) {
    return reinterpret_cast<xla::ptxla::opaque_t>(object_ptr);
}

std::vector<std::string> split(const std::string& str, const char delim) {
    std::vector<std::string> strings;
    std::size_t start;
    std::size_t end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        strings.push_back(str.substr(start, end - start));
    }
    return std::move(strings);
}

bool is_device(const std::string& found_device, const std::string& want_device) {
    const std::vector<std::string> parts = split(found_device, ':');
    if (!parts.empty()) {
        return parts[0] == want_device;
    }
    return false;
}

}  // namespace

XrtComputationClientWse::XrtComputationClientWse(
    Options options,
    std::unique_ptr <tensorflow::tpu::TopologyProto> topology_proto
) : XrtComputationClient(std::move(options), std::move(topology_proto)) {
    setenv("XRT_MASTER_ALLOW_SAME_TASKS", "1", true);
    std::cout << "CREATE XrtComputationClientWse" << ENDL;
    if (callback_interface_) {
        callback_interface_->OnCreate(GetOpaque(this));
    }
}

XrtComputationClientWse::~XrtComputationClientWse() {
    std::cout << "DESTROY XrtComputationClientWse" << ENDL;
    if (callback_interface_) {
        callback_interface_->OnDestroy(GetOpaque(this));
    }
}

void XrtComputationClientWse::SetExternalInterface(
    std::shared_ptr <ptxla::XrtComputationClientExternalInterface> callback_interface
) {
    if (!callback_interface_) {
        callback_interface_ = callback_interface->shared_from_this();
    } else {
        if (callback_interface != callback_interface_) {
            throw std::runtime_error(
                "An attempt was made to set the Xrt callback interface more than once"
            );
        }
    }
}

ComputationClient::DataPtr
XrtComputationClientWse::CreateDataPlaceholder(std::string device, Shape shape) {
    return Super::CreateDataPlaceholder(device, shape);
}

// Transfers local tensor values to the TPU servers and fetches the handles.
std::vector <ComputationClient::DataPtr> XrtComputationClientWse::TransferToServer(
    absl::Span<const TensorSource> tensors
) {
//    if (callback_interface_) {
//        std::vector<ptxla::XTensorSource> x_tensors;
//        std::pair<ptxla::EIntent, std::vector<ptxla::XDataPtr>> result =
//            callback_interface_->TransferToServer(GetOpaque(this), x_tensors);
//        if (result.first != ptxla::EI_DEFER) {
//            return std::vector<ComputationClient::DataPtr>();
//        }
//    }
    return Super::TransferToServer(tensors);
}

// Reads the tensor literal values stored at TPU server sites, behind the
// supplied handles.
std::vector <Literal> XrtComputationClientWse::TransferFromServer(
    absl::Span<const DataPtr> handles
) {
//    if (callback_interface_) {
//        std::vector<ptxla::XDataPtr> x_handles;
//        std::pair<ptxla::EIntent, std::vector<ptxla::XLiteral>> result =
//            callback_interface_->TransferFromServer(GetOpaque(this), x_handles);
//        if (result.first != ptxla::EI_DEFER) {
//            return std::vector<Literal>();
//        }
//    }
    return Super::TransferFromServer(handles);
}

// Compiles a set of computations.
std::vector <ComputationClient::ComputationPtr> XrtComputationClientWse::Compile(
    std::vector <CompileInstance> instances
) {
    std::set<std::size_t> index_of;
    std::vector<ComputationClient::ComputationPtr> results;
    results.reserve(instances.size());

    size_t this_index = 0;
    for (CompileInstance& instance : instances) {
        bool is_registered_device = is_device(instance.compilation_device, "WSE");
        if (is_registered_device) {
            ColorScope clr(Color::FG_RED);
            std::cout << "WSE DEVICE REQUESTED" << std::endl << std::flush;
        }
        // TODO: callback should be device registered
        if (!is_registered_device) {
            for (const std::string &device : instance.devices) {
                is_registered_device = is_device(device, "WSE");
                if (is_registered_device) {
                    break;
                }
            }
        }
        if (is_registered_device) {
            std::cout << "WSE DEVICE COMPILE" << std::endl << std::flush;
            if (callback_interface_) {
                const ptxla::ECompileResult comp_result = callback_interface_->OnCompile(
                    GetOpaque(this),
                    instance.computation.proto().id(),  // good enough or need hash from PTXLA layer?
                    instance.computation.proto(),
                    instance.devices,
                    ptxla::ECS_BEFORE_COMPILE
                );
                if (comp_result == ptxla::ECR_ACCEPT) {
                    assert(false);  // need to finish this
                    // We compiled it ourselves, should insert a ComputationClient::ComputationPtr
                    ComputationClient::ComputationPtr computation_ptr =
                        std::make_shared<ComputationClient::Computation>(
                            XlaComputation(instance.computation.proto()),
                            ProgramShape(instance.computation.proto().host_program_shape()),
                            instance.devices
                        );
                    index_of.insert(this_index);
                    results.push_back(computation_ptr);
                } else {
                    is_registered_device = false;
                }
            } else {
                // TEMPORARY: defer
//                std::cout << "Noc allback, deferring to CPU" << std::endl << std::flush;
//                instance.compilation_device = "CPU:0";
//                instance.devices[0] = "CPU:0";
            }
        } else {

        }
    }
    return Super::Compile(std::move(instances));
}

// Executes computation with arguments and returns the result.
// The passed device must match the common device of the arguments Data.
// If options.explode_tuple is true, the output tuple will be decomposed into
// its single elements.
std::vector <ComputationClient::DataPtr> XrtComputationClientWse::ExecuteComputation(
    const Computation &computation,
    absl::Span<const DataPtr> arguments,
    const std::string &device,
    const ExecuteComputationOptions &options
) {
//    if (callback_interface_) {
//        ptxla::ERunStatus run_status = callback_interface_->OnExecuteComputation(
//            GetOpaque(this),
//            computation.computation().proto().id(),
//            device,
//            ptxla::ERS_BEFORE_RUN
//        );
//        if (run_status != ptxla::ERS_DEFER) {
//            // No data returned yet :(
//            assert(false);
//            return std::vector < ComputationClient::DataPtr > {nullptr};
//        }
//    }
    return Super::ExecuteComputation(computation, arguments, device, options);
}

}  // namespace xla