#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_ext_intf.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/util/util.h"

#include "tensorflow/core/protobuf/tpu/topology.pb.h"

#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/service/cpu/wse_compiler.h"
#include "tensorflow/compiler/xla/primitive_util.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "absl/types/span.h"

#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <strstream>

#define START_GRPC_SERVICE

/**
 * TODO: Non-TF-linking portions of this to be moved to
 *       monolith-side after grpc boundary inserted
 */
using namespace tensorflow;

namespace xla {

int StartLocalXlaService(int port);

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
  // Check for blank in order to help out
  // when is_wse_proxy_device() returns blank
  if (!found_device.empty()) {
    const std::vector<std::string> parts = split(found_device, ':');
    if (!parts.empty()) {
      return parts[0] == want_device;
    }
  }
  return false;
}

bool is_wse_device(const std::string& found_device) {
  return is_device(found_device, "WSE");
}

bool is_wse_device(const XrtComputationClient::TensorSource& tensor_source) {
  return is_wse_device(tensor_source.device);
}

bool is_wse_device(const XrtComputationClient::DataPtr& data_ptr) {
  return is_wse_device(data_ptr->device());
}

template<typename DEST_MSG, typename SRC_MSG_ARRAY>
const DEST_MSG *get_id(const SRC_MSG_ARRAY& array, const int64 id) {
  const int64 total_count = array.size();
  for (int64 i = 0; i < total_count; ++i) {
    auto& obj = array[i];
    if (obj.id() == id) {
      return &obj;
    }
  }
  return nullptr;
}

std::string get_proxy_device(const xla::HloModuleProto& module) {
  //save_msg(module, "my_hlo_module.json");
  const int64 entry_computation_id = module.entry_computation_id();
  if (entry_computation_id) {
    const xla::HloComputationProto *computation =
      get_id<xla::HloComputationProto>(
        module.computations(),
        entry_computation_id
      );
    const int64 root_id = computation->root_id();
    if (root_id) {
      const xla::HloInstructionProto *root_instruction =
        get_id<xla::HloInstructionProto>(
          computation->instructions(),
          root_id
        );
      const xla::FrontendAttributes &frontend_attributes =
        root_instruction->frontend_attributes();
      auto iter = frontend_attributes.map().find("PROXY_DEVICE");
      if (iter != frontend_attributes.map().end()) {
        return iter->second;
      }
    }
  }
  return "";
}

bool is_wse_proxy_device(const xla::HloModuleProto& module) {
  return is_wse_device(get_proxy_device(module));
}

constexpr int XLA_SERVICE_GRPC_PORT = 50421;

using XlaClient = xla::grpc::XlaService::Stub;

std::mutex xla_server_mtx_;
std::shared_ptr<XlaClient> xla_client_;


std::shared_ptr<XlaClient> CreateXlaClientInternal(int port) {
  std::strstream ss;
  ss << "0.0.0.0" << ":" << port;
  const std::string address = ss.str();
  auto xla_service = xla::grpc::XlaService::NewStub(
    ::grpc::CreateChannel(ss.str(), ::grpc::InsecureChannelCredentials())
  );
  return xla_service;
}

std::shared_ptr<XlaClient>
  GetXlaClient(bool create_if_needed = true) {
  std::lock_guard<std::mutex> lk(xla_server_mtx_);
  if (!xla_client_.get() && create_if_needed) {
    xla_client_ = CreateXlaClientInternal(XLA_SERVICE_GRPC_PORT);
  }
  return xla_client_;
}

template<
  typename RESULT_T,
  typename CONTAINER_T,
  typename FUNC_T,
  typename CALL_T1,
  typename CALL_T2>
RESULT_T split_types(
  CONTAINER_T& all,
  FUNC_T predicate,
  CALL_T1 true_call,
  CALL_T2 false_call
) {
  std::vector<std::size_t> true_indexes;
  std::vector<std::size_t> false_indexes;
  std::vector<typename CONTAINER_T::value_type> true_items;
  std::vector<typename CONTAINER_T::value_type> false_items;

  true_indexes.reserve(all.size());
  false_indexes.reserve(all.size());
  true_items.reserve(all.size());
  false_items.reserve(all.size());
  std::size_t index = 0;
  for(auto& item : all) {
    if (predicate(item)) {
      true_indexes.emplace_back(index);
      true_items.emplace_back(std::move(item));
    } else {
      false_indexes.emplace_back(index);
      false_items.emplace_back(std::move(item));
    }
    ++index;
  }

  // TODO: 2-way multi-wait
  RESULT_T true_results = !true_items.empty() ?
                          true_call(true_items) : RESULT_T();
  RESULT_T false_results = !false_items.empty() ?
                           false_call(false_items) : RESULT_T();

  assert(true_results.size() == true_items.size());
  assert(false_results.size() == false_items.size());

  RESULT_T results(all.size());

  for (std::size_t i = 0; i < true_indexes.size(); ++i) {
    results[true_indexes[i]] = std::move(true_results[i]);
  }
  for (std::size_t i = 0; i < false_indexes.size(); ++i) {
    results[false_indexes[i]] = std::move(false_results[i]);
  }
  return std::move(results);
}
}  // namespace

XrtComputationClientWse::XrtComputationClientWse(
  Options options,
  std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto
) : XrtComputationClient(std::move(options), std::move(topology_proto)) {
  ::setenv("XRT_MASTER_ALLOW_SAME_TASKS", "1", true);
  std::cout << "CREATE XrtComputationClientWse" << ENDL;
  if (callback_interface_) {
    callback_interface_->OnCreate(GetOpaque(this));
  }

#ifdef START_GRPC_SERVICE
  xla::StartLocalXlaService(XLA_SERVICE_GRPC_PORT);
  GetXlaClient();
#endif
}

XrtComputationClientWse::~XrtComputationClientWse() {
  std::cout << "DESTROY XrtComputationClientWse" << ENDL;
  if (callback_interface_) {
    callback_interface_->OnDestroy(GetOpaque(this));
  }
  xla_client_.reset();
}

void XrtComputationClientWse::SetExternalInterface(
  std::shared_ptr <ptxla::XrtComputationClientExternalInterface> callback_interface
) {
  if (!callback_interface_) {
    callback_interface_ = callback_interface->shared_from_this();
    xla::wse::WseCompiler::SetCompilerCallback(callback_interface_);
  } else {
    if (callback_interface != callback_interface_) {
      throw std::runtime_error(
        "An attempt was made to set the Xrt callback interface more than once"
      );
    }
  }
}

void XrtComputationClientWse::ReleaseXrtData(const std::string& device, int64 handle) {
  // is it a wse device?
  assert(!is_wse_device(device));  // ever?
  auto client = GetXlaClient(false);
  if (client && handle) {
    ::grpc::ClientContext context;
    xla::UnregisterRequest request;
    xla::UnregisterResponse response;
    request.add_data()->set_handle(handle);
    assert(!is_wse_device(device));  // better if this is true, but think it's not?
    const ::grpc::Status status = client->Unregister(&context, request, &response);
    if (status.ok()) {
      return;
    }
  }
  Super::ReleaseXrtData(device, handle);
}

ComputationClient::DataPtr
XrtComputationClientWse::CreateDataPlaceholder(std::string device, Shape shape) {
  // In case we wish to create a special type of DataPtr
  return Super::CreateDataPlaceholder(device, shape);
}

template<typename PType>
void *get_data_pointer(xla::Literal& literal) {
  return literal.data<PType>().data();
}

/**
 * @brief Probably an incorrect copy function. See tensor_util.cpp
 * @param literal
 * @return
 */
void *get_data_ptr(xla::Literal& literal) {
  switch (literal.shape().element_type()) {
    case xla::PrimitiveType::PRED:
      return get_data_pointer<bool>(literal);
    case xla::PrimitiveType::F16:
      return get_data_pointer<xla::half>(literal);
    case xla::PrimitiveType::F32:
      return get_data_pointer<float>(literal);
    case xla::PrimitiveType::F64:
      return get_data_pointer<double>(literal);
    case xla::PrimitiveType::U8:
      return get_data_pointer<xla::uint8>(literal);
    case xla::PrimitiveType::S8:
      return get_data_pointer<xla::int8>(literal);
    case xla::PrimitiveType::S16:
      return get_data_pointer<xla::int16>(literal);
    case xla::PrimitiveType::U16:
      return get_data_pointer<xla::uint16>(literal);
    case xla::PrimitiveType::S32:
      return get_data_pointer<xla::int32>(literal);
    case xla::PrimitiveType::U32:
      return get_data_pointer<xla::uint32>(literal);
    case xla::PrimitiveType::S64:
      return get_data_pointer<xla::int64>(literal);
    case xla::PrimitiveType::U64:
      return get_data_pointer<xla::uint64>(literal);
    case xla::PrimitiveType::C64:
      return get_data_pointer<xla::complex64>(literal);
    case xla::PrimitiveType::C128:
      return get_data_pointer<xla::complex128>(literal);
    default:
      XLA_ERROR() << "Unsupported literal type: " << literal.shape();
  }
}


// Transfers local tensor values to the TPU servers and fetches the handles.
std::vector<ComputationClient::DataPtr> XrtComputationClientWse::TransferToServer(
  absl::Span<const TensorSource> tensors
) {
  auto results =
    split_types<std::vector<ComputationClient::DataPtr>>(
      tensors,
      [](const TensorSource& tensor_source){
        // won't work, actually... need proper device set on data ptr
        return is_wse_device(tensor_source.device);
      },
      [this](const std::vector<TensorSource>& local_tensor_sources) {
        // WSE (true)
        std::vector<ComputationClient::DataPtr> local_results;
        local_results.reserve(local_tensor_sources.size());
        for (const TensorSource& tensor_source : local_tensor_sources) {
          xla::Literal literal(tensor_source.shape);

          tensor_source.populate_fn(
            tensor_source,
            get_data_ptr(literal),
            literal.size_bytes()
          );

          ::grpc::ClientContext context;
          xla::TransferToServerRequest request;
          xla::TransferToServerResponse response;

          // set device handle?
          request.mutable_literal()->CopyFrom(literal.ToProto());

          ::grpc::Status status = GetXlaClient()->TransferToServer(&context, request, &response);
          if (!status.ok()) {
            throw std::runtime_error(status.error_message());
          }

          local_results.emplace_back(
            std::make_shared<XrtData>(
              this,
              tensor_source.device,
              tensor_source.shape,
              response.data().handle()
            )
          );
        }
        return std::move(local_results);
      },
      [this](const std::vector<TensorSource>& local_tensor_sources) {
        // OTHER (false)
        return Super::TransferToServer(local_tensor_sources);
      }
    );
  return std::move(results);
}

// Reads the tensor literal values stored at TPU server sites, behind the
// supplied handles.
std::vector<Literal> XrtComputationClientWse::TransferFromServer(
  absl::Span<const DataPtr> handles
) {
  std::vector<DataPtr> all_handles(handles.begin(), handles.end());
  std::vector<Literal> results = split_types<std::vector <Literal>>(
    all_handles,
    [](const DataPtr& data_ptr){
      // won't work, actually... need proper device set on data ptr
      return is_wse_device(data_ptr);
    },
    [this](std::vector<DataPtr>& wse_handles) {
      // WSE (true)
      std::vector<Literal> local_results;
      local_results.reserve(wse_handles.size());
      for (DataPtr& data_ptr : wse_handles) {

        ::grpc::ClientContext context;
        xla::TransferToClientRequest request;
        xla::TransferToClientResponse response;

        request.mutable_data()->set_handle(data_ptr->GetOpaqueHandle());
        request.mutable_shape_with_layout()->CopyFrom(data_ptr->shape().ToProto());

        ::grpc::Status status = GetXlaClient()->TransferToClient(
          &context, request, &response);

        if (!status.ok()) {
          throw std::runtime_error(status.error_message());
        }

        StatusOr<Literal> result =
          Literal::CreateFromProto(response.literal(), true);

        if (!result.ok()) {
          throw std::runtime_error(result.status().ToString());
        }
        local_results.emplace_back(result.ConsumeValueOrDie());
      }
      return std::move(local_results);
    },
    [this](std::vector<DataPtr>& other_handles) {
      // CPU or other (false)
      return Super::TransferFromServer(other_handles);
    }
  );
  return std::move(results);
}

// Compiles a set of computations.
std::vector<ComputationClient::ComputationPtr> XrtComputationClientWse::Compile(
  std::vector<CompileInstance> instances
) {
  //
  // TODO: ComputationPtr to return have modified HloModule and
  //       call Super with it (no proxy) on compile failure
  //
  auto results = split_types<std::vector<ComputationClient::ComputationPtr>>(
    instances,
    [](const CompileInstance& instance) -> bool {
      return xla_client_.get() &&
             is_wse_proxy_device(instance.computation.proto());
    },
    [this](std::vector<CompileInstance>& instances) {
      // WSE (true)
      std::vector<ComputationClient::ComputationPtr> local_results(instances.size());
      for (CompileInstance& instance : instances) {

        xla::CompileRequest compile_request;
        compile_request.mutable_computation()->CopyFrom(instance.computation.proto());
        xla::CompileResponse compile_response;

        bool handled = false;

        ::grpc::ClientContext context;
        const ::grpc::Status status =
          GetXlaClient()->Compile(&context, compile_request, &compile_response);
        if (status.ok()) {
            // We compiled it ourselves, should insert a ComputationClient::ComputationPtr
          ComputationClient::ComputationPtr computation_ptr =
            std::make_shared<ComputationClient::Computation>(
              XlaComputation(instance.computation.proto()),
              ProgramShape(instance.computation.proto().host_program_shape()),
              instance.devices,
              compile_response.handle().handle()
            );
          local_results.push_back(computation_ptr);
          handled = true;
        }
        if (!handled) {
          std::vector<CompileInstance> one_item;
          one_item.emplace_back(std::move(instance));
          local_results.push_back(Super::Compile(std::move(one_item))[0]);
        }
      }
      return std::move(local_results);
    },
    [this](std::vector<CompileInstance>& instances) {
      // CPU or other (false)
      return std::move(Super::Compile(std::move(instances)));
    }
  );
  return std::move(results);
}

// Executes computation with arguments and returns the result.
// The passed device must match the common device of the arguments Data.
// If options.explode_tuple is true, the output tuple will be decomposed into
// its single elements.
std::vector<ComputationClient::DataPtr> XrtComputationClientWse::ExecuteComputation(
  const Computation &computation,
  absl::Span<const DataPtr> arguments,
  const std::string &device,
  const ExecuteComputationOptions &options
) {
  if (is_wse_proxy_device(computation.computation().proto())) {
    assert(computation.execution_handle() != 0);

    ::grpc::ClientContext client_context;
    xla::ExecuteRequest request;
    xla::ExecuteResponse response;

    request.mutable_handle()->set_handle(computation.execution_handle());
    for (const DataPtr argument : arguments) {
      request.add_arguments()->set_handle(argument->GetOpaqueHandle());
    }

    ::grpc::Status status = GetXlaClient()->Execute(&client_context, request, &response);

    if (!status.ok()) {
      throw std::runtime_error(status.error_message());
    }

    xla::DeconstructTupleRequest dt_request;
    xla::DeconstructTupleResponse dt_response;
    status = GetXlaClient()->DeconstructTuple(&client_context, dt_request, &dt_response);

    if (!status.ok()) {
      throw std::runtime_error(status.error_message());
    }

    // TODO: Use BufferIn/BufferOut APIs
    std::vector<ComputationClient::DataPtr> results;  // tuple results
    results.reserve(dt_response.element_handles_size());
    for (const ::xla::GlobalDataHandle& element_handle : dt_response.element_handles()) {
      // TODO: do in parallel?
      ::xla::GetShapeRequest request;
      ::xla::GetShapeResponse response;
      request.mutable_data()->set_handle(element_handle.handle());
      status = GetXlaClient()->GetShape(&client_context, request, &response);
      if (!status.ok()) {
        throw std::runtime_error(status.error_message());
      }
      results.emplace_back(
        std::make_shared<XrtData>(
          this,
          get_proxy_device(computation.computation().proto()),
          Shape(response.shape()),
          element_handle.handle()
        )
      );
    }
    return std::move(results);
  }

  std::vector<ComputationClient::DataPtr> results =
    Super::ExecuteComputation(computation, arguments, device, options);

  return std::move(results);
}

}  // namespace xla

