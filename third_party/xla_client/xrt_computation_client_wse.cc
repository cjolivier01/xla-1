
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_ext_intf.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/core/util/util.h"

#include "tensorflow/core/protobuf/tpu/topology.pb.h"

#include "tensorflow/core/framework/tensor.pb.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/service/cpu/wse_compiler.h"

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

struct TensorBuffer {
  void allocate(size_t size) {
    data_.reset(new char[size]);
    size_ = size;
  }
  std::size_t size_;
  std::unique_ptr<char[]> data_;
};

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
    xla::wse::WseCompiler::SetCompilerCallback(callback_interface_);
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
      ColorScope clr(Color::FG_RED);
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
//                std::cout << "No callback, deferring to CPU" << std::endl << std::flush;
//                instance.compilation_device = "CPU:0";
//                instance.devices[0] = "CPU:0";
        return Super::Compile(std::move(instances));
        //ComputationClient::ComputationPtr computation_ptr = std::make_shared<ComputationClient::Computation>();
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
  //HEREX();
  std::string wse_device_str;
  const std::vector<std::string> &devices = computation.devices();
  //std::cout << "devices: ";
  for (const std::string this_device : devices) {
    //std::cout << device << ", ";
    if (is_device(this_device, "WSE")) {
      assert(devices.size() == 1);  // What to do if not one? replicas?
      wse_device_str = this_device;
      break;
    }
  }
  //std::cout << std::endl << std::flush;

  if (!wse_device_str.empty()) {
    std::vector<ComputationClient::DataPtr> result;
    for (DataPtr dp : arguments) {
      std::cout << "argument: " << dp->shape() << std::endl;
    }
    assert(is_device(device, "WSE"));
    std::cout << std::endl << std::flush;

    std::cout << "program shape: " << computation.program_shape().ToString() << std::endl;
    std::cout << "program shape result: " << computation.program_shape().result().ToString()
              << std::endl;

    const Shape &result_shape = computation.program_shape().result();
    if (result_shape.IsTuple()) {
      for (int i = 0, n = result_shape.tuple_shapes_size(); i < n; ++i) {
        const Shape &output_shape = result_shape.tuple_shapes(i);
        std::cout << "Tuple index " << i << ": " << output_shape.ShortDebugString() << std::endl
                  << std::flush;
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
          std::cout << "dest buffer: " << dest_buffer << ", size=" << dest_buffer_size << ENDL;
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

      const bool debug_sync = true;
      std::mutex debug_sync_mutex;

      auto converter = [&, j]() {
        std::unique_ptr<std::lock_guard<std::mutex>>(
          debug_sync ? new std::lock_guard<std::mutex>(debug_sync_mutex) : nullptr
        );
        std::cout << "locked in converter" << ENDL;
        std::string device = GetEffectiveDevice(wse_device_str);
        const std::string &xrt_device = TorchDeviceToXrtDevice(device);
        tensorflow::Tensor tensor(
          GetTensorAllocator(),
          XlaTypeToDataType(tensors[j].shape.element_type()),
          MakeEquivalentTensorShape(tensors[j].shape));
        auto tdata = tensor.tensor_data();
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
          session_work->feed_inputs.insert({cached_node.holders[0], tensor});
          session_work->outputs_handles.push_back(cached_node.outputs[0]);
          session_work->index_mapping.push_back(j);

          total_size += tdata.size();
        }

        results[j] = std::make_shared<XrtData>(
          this,
          device,  // should be CPU device? main device in call parameters?
          tensors[j].shape,
          tensor.scalar<int64>()()
        );
      };
      if (!debug_sync) {
        env::ScheduleClosure(mwait.Completer(std::move(converter)));
      } else {
        converter();
      }
    }
    mwait.Wait();

//    mwait.Reset(session_work_map.size());
//    std::vector<DataPtr> results(tensors.size());
//    for (auto& session_session_work : session_work_map) {
//      XrtSession* session = session_session_work.first;
//      SessionWork* session_work = &session_session_work.second;
//      auto runner = [&, session, session_work]() {
//        std::vector<tensorflow::Tensor> outputs;
//        XLA_CHECK_OK(session->session()->Run(
//          session_work->feed_inputs, session_work->outputs_handles, &outputs));
//        XLA_CHECK_EQ(outputs.size(), session_work->outputs_handles.size());
//
//        for (size_t i = 0; i < outputs.size(); ++i) {
//          size_t li = session_work->index_mapping[i];
//          results[li] = std::make_shared<XrtData>(
//            this, GetEffectiveDevice(tensors[li].device), tensors[li].shape,
//            outputs[i].scalar<int64>()());
//        }
//        CreateDataHandlesCounter()->AddValue(outputs.size());
//      };
//      env::ScheduleIoClosure(mwait.Completer(std::move(runner)));
//    }
//    mwait.Wait();


    return results;
  }

  std::vector<ComputationClient::DataPtr> results =
    Super::ExecuteComputation(computation, arguments, device, options);

  return std::move(results);
}

}  // namespace xla