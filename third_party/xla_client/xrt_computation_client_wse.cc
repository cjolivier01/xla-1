
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_ext_intf.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/util.h"

#include "tensorflow/core/protobuf/tpu/topology.pb.h"

#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"

#include "tensorflow/core/framework/tensor.pb.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/service/cpu/wse_compiler.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "absl/types/span.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
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

std::string get_proxy_device(const xla::HloModuleProto& module) {
  //save_msg(module.ToProto(), "my_hlo_module.json");
  const int64 entry_computation_id = module.entry_computation_id();
  if (entry_computation_id) {
    const xla::HloComputationProto& computation =
      module.computations()[entry_computation_id];
    const int64 root_id = computation.root_id();
    if (root_id) {
      const xla::HloInstructionProto &root_instruction =
        computation.instructions()[root_id];
      const xla::FrontendAttributes &frontend_attributes =
        root_instruction.frontend_attributes();
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

std::shared_ptr<XlaClient> GetXlaClient() {
  std::lock_guard<std::mutex> lk(xla_server_mtx_);
  if (!xla_client_.get()) {
    xla_client_ = CreateXlaClientInternal(XLA_SERVICE_GRPC_PORT);
  }
  return xla_client_;
}


}  // namespace

// Quick version of XRTMemoryManager until we can get on
// the other side of a grpc boundary
class XrtComputationClientWse::MemoryManager {
  struct TensorDataEntry {
    TensorDataEntry(
      std::shared_ptr<tensorflow::Tensor> tensor,
      bool has_data) : tensor_(tensor), has_data_(has_data) {}
    std::shared_ptr<tensorflow::Tensor> tensor_;
    bool has_data_;
  };
public:
  bool IsIn(int64 handle) {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    return tensor_map_.count(handle) != 0;
  }
  int64 Register(std::shared_ptr<tensorflow::Tensor> tensor, bool has_data) {
    assert(tensor.get());
    int64 handle;
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    while (true) {
      handle = CreateUid();
      if (tensor_map_.count(handle) == 0) {
        tensor_map_.emplace(std::make_pair(
          handle,
          std::make_shared<TensorDataEntry>(std::move(tensor), has_data))
        );
        break;
      }
    }
    return handle;
  }
  xla::Literal GetLiteral(int64 handle) {
    assert(false);  // TransferFromServer
    return xla::Literal();
  }
  std::shared_ptr<tensorflow::Tensor> Get(int64 handle) const {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    assert(tensor_map_.count(handle) != 0);
    return tensor_map_.find(handle)->second->tensor_;
  }
  bool Free(int64 handle) {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    if(tensor_map_.count(handle)) {
      tensor_map_.erase(handle);
      return true;
    }
    return false;
  }
private:

  static int64 InvalidKey() { return 0; }
  static int64 CreateUid() {
    int64 uid;
    do {
      uid = random::New64() & INT64_MAX;
    } while (uid == InvalidKey());
    return -uid + 4;  // Change a little from what XRTMemoryManager generates
  }

  mutable std::mutex mem_buffers_mtx_;
  std::unordered_map<int64, std::shared_ptr<TensorDataEntry>> tensor_map_;
};

XrtComputationClientWse::XrtComputationClientWse(
  Options options,
  std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto
) : XrtComputationClient(std::move(options), std::move(topology_proto))
  , memory_manager_(std::make_unique<XrtComputationClientWse::MemoryManager>()) {
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
  assert(!is_wse_device(device));  // better if this is true, but think it's not?
  // if it's true, then use it
  if (memory_manager_->Free(handle)) {
    return;
  }
}

ComputationClient::DataPtr
XrtComputationClientWse::CreateDataPlaceholder(std::string device, Shape shape) {
  // In case we wish to create a special type of DataPtr
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

// Reads the tensor literal values stored at TPU server sites, behind the
// supplied handles.
std::vector<Literal> XrtComputationClientWse::TransferFromServer(
  absl::Span<const DataPtr> handles
) {
  // TODO: For checkpoints, need to recognize they're not normal outputs
  //       (they actually will be the ones not filled in from the execute,
  //       so that should be sufficient)
  //       pull whole checkpoint and then put in the handle cache for any
  //       subsequent fetches until the next execute
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
        local_results.emplace_back(
          memory_manager_->GetLiteral(data_ptr->GetOpaqueHandle())
        );
      }
      return std::move(local_results);
    },
    [this](std::vector<DataPtr>& other_handles) {
      // CPU or other (false)
      return Super::TransferFromServer(other_handles);
    }
  );
  return std::move(results);
  const bool has_wse_device = std::any_of(
    handles.begin(),
    handles.end(),
    [](const DataPtr& d){return is_wse_device(d); }
  );
  if (!has_wse_device) {
    return Super::TransferFromServer(handles);
  }
  return std::move(results);
}

// Compiles a set of computations.
std::vector<ComputationClient::ComputationPtr> XrtComputationClientWse::Compile(
  std::vector<CompileInstance> instances
) {
  auto results = split_types<std::vector<ComputationClient::ComputationPtr>>(
      instances,
      [](const CompileInstance& instance) -> bool {
        return callback_interface_.get() &&
          is_wse_proxy_device(instance.computation.proto());
      },
      [this](std::vector<CompileInstance>& instances) {
        // WSE (true)
        assert(callback_interface_.get());
        std::vector<ComputationClient::ComputationPtr> local_results(instances.size());
        for (CompileInstance& instance : instances) {
          const ptxla::ECompileResult comp_result = callback_interface_->OnCompile(
            GetOpaque(this),
            instance.computation.proto().id(),  // good enough or need hash from PTXLA layer?
            instance.computation.proto(),
            instance.devices,
            ptxla::ECS_BEFORE_COMPILE
          );
          if (comp_result == ptxla::ECR_ACCEPT) {
            // We compiled it ourselves, should insert a ComputationClient::ComputationPtr
            ComputationClient::ComputationPtr computation_ptr =
              std::make_shared<ComputationClient::Computation>(
                XlaComputation(instance.computation.proto()),
                ProgramShape(instance.computation.proto().host_program_shape()),
                instance.devices
              );
            local_results.push_back(computation_ptr);
          } else {
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
  //HEREX();
  std::string wse_device_str;
  const std::vector<std::string> &devices = computation.devices();
  //std::cout << "devices: ";
  for (const std::string& this_device : devices) {
    //std::cout << device << ", ";
    if (is_wse_device(this_device)) {
      assert(devices.size() == 1);  // What to do if not one? replicas?
      wse_device_str = this_device;
      break;
    }
  }
  //std::cout << std::endl << std::flush;

  if (wse_device_str.empty()) {
    std::vector<ComputationClient::DataPtr> result;
//    for (DataPtr dp : arguments) {
//      std::cout << "argument: " << dp->shape() << std::endl;
//    }
//    assert(is_wse_device(device));
//    std::cout << std::endl << std::flush;

//    std::cout << "program shape: " << computation.program_shape().ToString() << std::endl;
//    std::cout << "program shape result: " << computation.program_shape().result().ToString()
//              << std::endl;

    const Shape &result_shape = computation.program_shape().result();
    if (result_shape.IsTuple()) {
      for (int i = 0, n = result_shape.tuple_shapes_size(); i < n; ++i) {
        const Shape &output_shape = result_shape.tuple_shapes(i);
//        std::cout << "Tuple index " << i << ": " << output_shape.ShortDebugString() << std::endl
//                  << std::flush;
      }
    } else {
      throw std::runtime_error("Expected result shape to be a tuple");
    }

    std::vector<ComputationClient::DataPtr> results(result_shape.tuple_shapes_size());

    XrtSessionCache::SessionMap session_map;
    std::map<XrtSession *, SessionWork> session_work_map;

    //std::vector<int64> tuple_elements_count(tuples.size());
    //assert(tuples.size() == 1);
    //const XrtData& xrt_data = dynamic_cast<const XrtData&>(*tuples[0]);

//    const XrtData xrt_data(device, result_shape);
//    XrtSession *session = GetSessionForDevice(
//      session_cache_.get(),
//      xrt_data.device(), &session_map
//    );
//    SessionWork *session_work = &session_work_map[session];

    //session_work->index_mapping.push_back(i);

//    tensorflow::Scope device_scope =
//      session->root()->WithDevice(TorchDeviceToXrtDevice(xrt_data.device()));
    //const std::size_t count = ShapeUtil::TupleElementCount(xrt_data.shape());

    //tuple_elements_count[i] = count;

    const size_t count = result_shape.tuple_shapes_size();

    std::vector<TensorSource> tensors;
    tensors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const Shape &output_shape = result_shape.tuple_shapes(i);

      auto populate_fn =
        [&, i](
          const xla::ComputationClient::TensorSource &source_tensor,
          void *dest_buffer, size_t dest_buffer_size
        ) {
          //std::cout << "dest buffer: " << dest_buffer << ", size=" << dest_buffer_size << ENDL;
          memset(dest_buffer, i, dest_buffer_size);
//          PopulateTensorBuffer(tensor, source_tensor.shape, dest_buffer,
//                               dest_buffer_size, device);
        };
      tensors.emplace_back(output_shape, device, std::move(populate_fn));
    }

    std::mutex lock;
    int64 total_size = 0;
    util::MultiWait mwait(tensors.size());

    for (int64 j = 0; j < count; ++j) {
//      const XrtSession::CachedNode &cached_node =
//        GetSubTupleNode(session, device_scope, device);
//      session_work->feed_inputs.insert(
//        {cached_node.holders[0], xrt_data.get_handle()});
//      tensorflow::Tensor index_tensor(tensorflow::DT_INT32,
//                                      tensorflow::TensorShape({1}));
//      index_tensor.flat<tensorflow::int32>()(0) = j;
//      session_work->feed_inputs.insert({cached_node.holders[1], index_tensor});
//      std::cout << "output tuple handle " << j << ": " << std::endl << std::flush;

//      const tensorflow::Output &output = cached_node.outputs[0];

      //session_work->outputs_handles.push_back(handle);

      //XrtComputationClient* self, std::string device, Shape device_shape, int64 handle

      //const Shape &output_shape = result_shape.tuple_shapes(j);

      const bool debug_sync = false;
      std::mutex debug_sync_mutex;

      auto converter = [&, j]() {
        std::unique_ptr<std::lock_guard<std::mutex>>(
          debug_sync ? new std::lock_guard<std::mutex>(debug_sync_mutex) : nullptr
        );
        std::cout << "locked in converter" << ENDL;
        std::string device = GetEffectiveDevice(wse_device_str);
        const std::string &xrt_device = TorchDeviceToXrtDevice(device);
        auto tensor_ptr = std::make_shared<tensorflow::Tensor>(
          GetTensorAllocator(),
          XlaTypeToDataType(tensors[j].shape.element_type()),
          MakeEquivalentTensorShape(tensors[j].shape));
        auto tdata = tensor_ptr->tensor_data();
        tensors[j].populate_fn(tensors[j], const_cast<char *>(tdata.data()), tdata.size());
        {
          std::lock_guard<std::mutex> slock(lock);
          XrtSession *session = GetSessionForXrtDevice(
            alloc_session_cache_.get(), xrt_device, &session_map
          );
          SessionWork *session_work = &session_work_map[session];
          tensorflow::Scope device_scope =
            session->root()->WithDevice(xrt_device);
          const XrtSession::CachedNode &cached_node =
            GetAllocateNode(session, device_scope, device, tensors[j].shape);
          session_work->feed_inputs.insert({cached_node.holders[0], *tensor_ptr});
          session_work->outputs_handles.push_back(cached_node.outputs[0]);
          session_work->index_mapping.push_back(j);

          total_size += tdata.size();
        }

        results[j] = std::make_shared<XrtData>(
          this,
          wse_device_str,  // should be CPU device? main device in call parameters?
          tensors[j].shape,
          memory_manager_->Register(tensor_ptr, true)  // pretend it's real for now
        );
      };
      if (!debug_sync) {
        env::ScheduleClosure(mwait.Completer(std::move(converter)));
      } else {
        converter();
      }
    }
    mwait.Wait();
    return results;
  }

  std::vector<ComputationClient::DataPtr> results =
    Super::ExecuteComputation(computation, arguments, device, options);

  return std::move(results);
}

}  // namespace xla

