#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
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

#define START_LOCAL_WSE_XLA_SERVICE

#ifdef START_LOCAL_WSE_XLA_SERVICE
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_ext_intf.h"
#endif

/**
 * TODO: Non-TF-linking portions of this to be moved to
 *       monolith-side after grpc boundary inserted
 */
using namespace tensorflow;

namespace xla {

#ifdef START_LOCAL_WSE_XLA_SERVICE
int StartLocalWseXlaService(int port);
#endif

namespace {

/**
 * @brief Force always using the proxy server for everyting
 *        (i.e. delegate everything to the grpc_service_main app)
 */
bool always_use_proxy = true;
bool wse_set_topology = false;
//const std::string ALWAYS_USE_PROXY_DEFAULT_DEVICE = "CPU:0";
const std::string ALWAYS_USE_PROXY_DEFAULT_DEVICE = "WSE:0";

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
        // A compile may have failed, in which case it
        // gets delegated back to the default device
        auto cancel_iter = frontend_attributes.map().find("CANCEL_PROXY_DEVICE");
        if (cancel_iter != frontend_attributes.map().end()) {
          if (cancel_iter->second == iter->second) {
            return "";  // this proxying was cancelled (i.e. failed compile)
          }
        }
        return iter->second;
      }
    }
  }
  return "";
}

constexpr int XLA_SERVICE_GRPC_PORT = 1685;

using XlaClient = xla::grpc::XlaService::Stub;

std::shared_ptr<XlaClient> CreateXlaClientInternal(const std::string& address) {
  auto xla_service = xla::grpc::XlaService::NewStub(
    ::grpc::CreateChannel(address, ::grpc::InsecureChannelCredentials())
  );
  return xla_service;
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

class XrtComputationClientWse::XlaClientInfo {
public:

  inline std::shared_ptr<XlaClient> operator ()() { return xla_client_; }
  inline std::shared_ptr<const XlaClient> operator ()() const { return xla_client_; }

  std::string address_;
  std::shared_ptr<XlaClient> xla_client_;
  std::vector<xla::DeviceHandle> device_handles_;
};

XrtComputationClientWse::XrtComputationClientWse(
  Options options,
  std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto
) : XrtComputationClient(std::move(options), std::move(topology_proto)) {
  ::setenv("XRT_MASTER_ALLOW_SAME_TASKS", "1", true);
  std::cout << "CREATE XrtComputationClientWse" << ENDL;
  if (callback_interface_) {
    callback_interface_->OnCreate(GetOpaque(this));
  }

#ifdef START_LOCAL_WSE_XLA_SERVICE
  xla::StartLocalWseXlaService(XLA_SERVICE_GRPC_PORT);
#endif
  if (always_use_proxy) {
    if (!IsProxyDevice(ALWAYS_USE_PROXY_DEFAULT_DEVICE)) {
      SetDeviceProxyAddress(ALWAYS_USE_PROXY_DEFAULT_DEVICE, "localhost:1685");
    }
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
    xla::wse::WseCompiler::SetCompilerCallback(callback_interface_);
  } else {
    if (callback_interface != callback_interface_) {
      throw std::runtime_error(
        "An attempt was made to set the Xrt callback interface more than once"
      );
    }
  }
}

void XrtComputationClientWse::SetDeviceProxyAddress(const std::string& device, const std::string& proxy_address) {
  std::shared_ptr<XlaClient> old_client;  // if exists, to be destroyed out of lock scope
  std::lock_guard<std::recursive_mutex> lk(xla_client_map_mtx_);
  if (device.empty()) {
    throw std::runtime_error("Invalid empty device string");
  }
  if (proxy_address.empty()) {
    // remove it
    xla_client_map_.erase(device);
    return;
  }
  auto iter = xla_client_map_.find(device);
  if (iter == xla_client_map_.end()) {
    auto new_info = std::make_shared<XlaClientInfo>();
    iter = xla_client_map_.emplace(
      std::make_pair(device, new_info)
    ).first;
    new_info->address_ = proxy_address;
  } else {
    // was already there
    if (iter->second->address_ != proxy_address) {
      // If it changed, kill the opld one (if it was created at all)
      old_client = iter->second->xla_client_;  // keep a ref until out of lock scope
      iter->second->xla_client_.reset();
      iter->second->address_ = proxy_address;
    }
  }
}

template<>
std::shared_ptr<XlaClient>
  XrtComputationClientWse::GetXlaClient<XlaClient>(const std::string& device, bool create) {
  std::lock_guard<std::recursive_mutex> lk(xla_client_map_mtx_);
  auto iter = xla_client_map_.find(device);
  if (iter == xla_client_map_.end()) {
      // No address registered for this device
      std::cout << "No proxy configured for device: " << device << ENDL;
      return nullptr;
  }
  if (!iter->second->xla_client_ && create) {
    iter->second->xla_client_ = CreateXlaClientInternal(iter->second->address_);
    if (iter->second->xla_client_) {
      ::grpc::ClientContext client_context;
      xla::GetDeviceHandlesRequest request;
      xla::GetDeviceHandlesResponse response;
      request.set_device_count(1); // HOW MANY DEVICES??
      ::grpc::Status status = iter->second->xla_client_->GetDeviceHandles(
        &client_context,
        request,
        &response
      );
      if (!status.ok()) {
        throw std::runtime_error(status.error_message());
      }
      iter->second->device_handles_.resize(0);
      iter->second->device_handles_.reserve(response.device_handles_size());
      for (const ::xla::DeviceHandle& device_handle : response.device_handles()) {
        std::cout << "device handle: " << device_handle.handle() << ENDL;
        iter->second->device_handles_.emplace_back(device_handle);
      }
    }
  }
  return iter->second->xla_client_;
}

xla::DeviceHandle XrtComputationClientWse::GetDeviceHandle(const std::string& device) {
  if (!GetXlaClient<XlaClient>(device, true)) {
    throw std::runtime_error("Failed to get XLA client for device");
  }
  std::lock_guard<std::recursive_mutex> lk(xla_client_map_mtx_);
  auto iter = xla_client_map_.find(device);
  if (iter == xla_client_map_.end()) {
    // No address registered for this device
    throw std::runtime_error("No proxy configured for device");
  }
  std::shared_ptr<XlaClientInfo> info = iter->second;
  const int64 ordinal = GetDeviceOrdinal(device);
  assert(ordinal < info->device_handles_.size());
  return info->device_handles_[ordinal];
}

bool XrtComputationClientWse::IsProxyDevice(const std::string& device) const {
  std::lock_guard<std::recursive_mutex> lk(xla_client_map_mtx_);
  return xla_client_map_.find(device) != xla_client_map_.end();
}

bool XrtComputationClientWse::UseProxyForDevice(const std::string& device) const {
  assert(!device.empty());
  return IsProxyDevice(device) && (always_use_proxy || is_wse_device(device));
}

void XrtComputationClientWse::ReleaseXrtData(const std::string& device, int64 handle) {
  // is it a wse device?
#if 1
  assert(!device.empty());
  if((device.empty() && always_use_proxy) || UseProxyForDevice(device)) {
    auto client = GetXlaClient<XlaClient>(device, always_use_proxy);
    if (client && handle) {
      ColorScope grn(Color::FG_GREEN);
      std::cout << "Releasing global data handle: " << handle << ENDL;
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
  } else {
    assert(!always_use_proxy);
    Super::ReleaseXrtData(device, handle);
  }
#endif
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
  //HEREX();
  auto results =
    split_types<std::vector<ComputationClient::DataPtr>>(
      tensors,
      [this](const TensorSource& tensor_source){
        // won't work, actually... need proper device set on data ptr
        return UseProxyForDevice(tensor_source.device);
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
          //request.set_allocated_literal(new xla::LiteralProto());
          *request.mutable_literal() = literal.ToProto();

          *request.mutable_device_handle() = GetDeviceHandle(tensor_source.device);

          ::grpc::Status status = GetXlaClient<XlaClient>(
            tensor_source.device
          )->TransferToServer(&context, request, &response);
          if (!status.ok()) {
            throw std::runtime_error(status.error_message());
          }

          std::cout << "Sent data , received handle: " << response.data().handle()
                    << ", shape=" << tensor_source.shape.ToString()
                    << ENDL;

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
        assert(!always_use_proxy);
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
    [this](const DataPtr& data_ptr){
      // won't work, actually... need proper device set on data ptr
      return UseProxyForDevice(data_ptr->device());
    },
    [this](std::vector<DataPtr>& wse_handles) {
      // WSE (true)
      std::vector<Literal> local_results;
      local_results.reserve(wse_handles.size());
      for (DataPtr& data_ptr : wse_handles) {

        ::grpc::ClientContext context;
        xla::TransferToClientRequest request;
        xla::TransferToClientResponse response;

        request.set_allocated_data(new xla::GlobalDataHandle());
        request.mutable_data()->set_handle(data_ptr->GetOpaqueHandle());

        request.set_allocated_shape_with_layout(new xla::ShapeProto());
        request.mutable_shape_with_layout()->CopyFrom(data_ptr->shape().ToProto());

        ::grpc::Status status = GetXlaClient<XlaClient>(
          data_ptr->device()
        )->TransferToClient(&context, request, &response);

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
      assert(!always_use_proxy);
      return Super::TransferFromServer(other_handles);
    }
  );
  return std::move(results);
}

// Compiles a set of computations.
std::vector<ComputationClient::ComputationPtr> XrtComputationClientWse::Compile(
  std::vector<CompileInstance> instances
) {
  //HEREX();
  //
  // TODO: ComputationPtr to return have modified HloModule and
  //       call Super with it (no proxy) on compile failure
  //
  auto results = split_types<std::vector<ComputationClient::ComputationPtr>>(
    instances,
    [this](const CompileInstance& instance) -> bool {
      const std::string device1 = instance.compilation_device;
      const std::string device2 = get_proxy_device(instance.computation.proto());
      std::string device;
      if (device1.empty() && !device2.empty()) {
        device = device2;
      } else if (!device1.empty() && device2.empty()) {
        device = device1;
      } else {
        assert(device1 == device2);  // what's this use-case?
        device = device1;
      }
      return UseProxyForDevice(device);
    },
    [this](std::vector<CompileInstance>& instances) {
      // WSE (true)
      std::vector<ComputationClient::ComputationPtr> local_results;
      local_results.reserve(instances.size());
      for (CompileInstance& instance : instances) {

        xla::CompileRequest compile_request;
        xla::CompileResponse compile_response;

        size_t param_num = 0;
        const ProgramShape program_shape = instance.computation.GetProgramShape().ValueOrDie();
        for (const Shape& parameter_shape : program_shape.parameters()) {
          ColorScope clr(Color::FG_CYAN);
          std::cout << "Compile: Param " << param_num++
                    <<  ", shape: " << parameter_shape
                    << ENDL;
          compile_request.add_input_shape_with_layout()->CopyFrom(parameter_shape.ToProto());
        }

        compile_request.set_allocated_computation(new xla::HloModuleProto());
        compile_request.mutable_computation()->CopyFrom(instance.computation.proto());

        bool handled = false;

        ::grpc::ClientContext context;
        const ::grpc::Status status =
          GetXlaClient<XlaClient>(
            instance.compilation_device
          )->Compile(&context, compile_request, &compile_response);
        if (status.ok()) {
          std::cout << "computation id: " << compile_response.handle().handle()
                    << " from proto id " << compile_request.mutable_computation()->id()
                    << ENDL;
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
        } else {
          std::cout << "Compile error: " << status.error_message() << ENDL;
        }
        if (!handled) {
          assert(!always_use_proxy);
          std::vector<CompileInstance> one_item;
          one_item.emplace_back(std::move(instance));
          auto results = Super::Compile(std::move(one_item));
          local_results.push_back(results[0]);
        }
      }
      return std::move(local_results);
    },
    [this](std::vector<CompileInstance>& instances) {
      assert(!always_use_proxy);
      // CPU or other (false)
      return std::move(Super::Compile(std::move(instances)));
    }
  );
  assert(results.size() == instances.size());
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
  //HERE();
  const std::string device1 = device;
  const std::string device2 = get_proxy_device(computation.computation().proto());
  std::string effective_device;
  if (device1.empty() && !device2.empty()) {
    effective_device = device2;
  } else if (!device1.empty() && device2.empty()) {
    effective_device = device1;
  } else {
    assert(device1 == device2);  // what's this use-case?
    effective_device = device2;  // prefer the proxy effective_device if it's specified
  }

  if (UseProxyForDevice(effective_device))  {
    assert(computation.execution_handle() != 0);

    ::grpc::ClientContext client_context_e;
    xla::ExecuteRequest request;
    xla::ExecuteResponse response;

//    std::cout << "Execution handle: " << computation.execution_handle()
//              << " " << computation.program_shape().ToString()
//              << ENDL;

    auto execution_handle = std::make_unique<xla::ExecutionHandle>();
    execution_handle->set_handle(computation.execution_handle());
    request.set_allocated_handle(execution_handle.release());

    for (const DataPtr argument : arguments) {
      //std::cout << "argument handle: " << argument->GetOpaqueHandle() << ENDL;
      request.add_arguments()->set_handle(argument->GetOpaqueHandle());
    }

    xla::ExecutionOptions eo;
    *eo.mutable_debug_options() = GetDebugOptionsFromFlags();
    *eo.add_device_handles() = GetDeviceHandle(effective_device);

    auto xla_client = GetXlaClient<XlaClient>(effective_device);

    ::grpc::Status status = xla_client->Execute(&client_context_e, request, &response);

    if (!status.ok()) {
      throw std::runtime_error(status.error_message());
    }

    xla::GetShapeRequest gs_request;

    std::vector<ComputationClient::DataPtr> results;  // tuple results
    std::vector<xla::GlobalDataHandle> result_handles;

    xla::ShapeProto response_shape;
    gs_request.mutable_data()->set_handle(response.output().handle());
    {
      ::grpc::ClientContext cc;
      xla::GetShapeResponse gs_response;
      assert(gs_request.data().handle());
      status = xla_client->GetShape(&cc, gs_request, &gs_response);
      if (!status.ok()) {
        throw std::runtime_error(status.error_message());
      }
      response_shape = gs_response.shape();
    }

    if (response_shape.element_type() == xla::PrimitiveType::TUPLE) {
      xla::DeconstructTupleRequest dt_request;
      xla::DeconstructTupleResponse dt_response;

      dt_request.mutable_tuple_handle()->set_handle(response.output().handle());

      ::grpc::ClientContext client_context_dt;
      status = xla_client->DeconstructTuple(&client_context_dt, dt_request, &dt_response);

      if (!status.ok()) {
        throw std::runtime_error(status.error_message());
      }

      results.reserve(dt_response.element_handles_size());
      result_handles.reserve(dt_response.element_handles_size());
      for (const ::xla::GlobalDataHandle& element_handle : dt_response.element_handles()) {
//        std::cout << "Tuple returned element handle: " << element_handle.handle()
//                  << std::endl << std::flush;
        result_handles.push_back(element_handle);
      }
      for (const ::xla::GlobalDataHandle& element_handle : result_handles) {
        // TODO: do in parallel?
        ::xla::GetShapeRequest request;
        ::xla::GetShapeResponse response;
        ::grpc::ClientContext client_context;
        assert(element_handle.handle());
        *request.mutable_data() = element_handle;
        status = xla_client->GetShape(&client_context, request, &response);
        if (!status.ok()) {
          throw std::runtime_error(status.error_message());
        }
        results.emplace_back(
          std::make_shared<XrtData>(
            this,
            effective_device,
            Shape(response.shape()),
            element_handle.handle()
          )
        );
      }
    } else {
      results.emplace_back(
        std::make_shared<XrtData>(this, effective_device, Shape(response_shape), response.output().handle())
      );
    }
    return std::move(results);
  }

  assert(!always_use_proxy);
  std::vector<ComputationClient::DataPtr> results =
    Super::ExecuteComputation(computation, arguments, effective_device, options);

  return std::move(results);
}

namespace {
int get_env_int(const char *s, const int dflt) {
  const char* v = getenv(s);
  if (v && *v) {
    return atoi(v);
  }
  return dflt;
}
}

tensorflow::tpu::TopologyProto XrtComputationClientWse::InitializeAndFetchTopology(
  const std::string& job,
  int task_no,
  const std::string& worker_host_port,
  const tensorflow::ConfigProto& config
) {
  std::cout << "InitializeAndFetchTopology( job=" << job
            << ", task_no= << " << task_no
            << ", worker_host_port=" << worker_host_port
            << ", config=" << msg_to_json(config)
            << std::endl << std::flush;
  const int wse_num_devices = get_env_int("WSE_NUM_DEVICES", 0);
  const int cpu_num_devices = get_env_int("CPU_NUM_DEVICES", 0);
  if (!wse_set_topology || !wse_num_devices) {
    return Super::InitializeAndFetchTopology(
      job, task_no, worker_host_port, config
    );
  }
  tensorflow::tpu::TopologyProto topology_proto;
  topology_proto.add_mesh_shape(wse_num_devices + cpu_num_devices);
  topology_proto.add_mesh_shape(1);
  topology_proto.add_mesh_shape(1);
  topology_proto.set_num_tasks(wse_num_devices);
  topology_proto.set_num_tpu_devices_per_task(wse_num_devices);
  for (int i = 0; i < wse_num_devices + cpu_num_devices; ++i) {
    topology_proto.add_device_coordinates(i);
    topology_proto.add_device_coordinates(1);
    topology_proto.add_device_coordinates(1);
  }
  return std::move(topology_proto);
}

}  // namespace xla

