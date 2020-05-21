#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/rpc/grpc_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/util.h"

#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"

// /home/chriso/src/ml_frameworks/pytorch_xla/third_party/tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h

#include "tensorflow/core/lib/random/random.h"

#include <memory>
#include <vector>
#include <strstream>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/framework/tensor.h"

namespace xla {

namespace {

// Quick version of XRTMemoryManager until we can get on
// the other side of a grpc boundary
class MemoryManager {
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
      uid = tensorflow::random::New64() & INT64_MAX;
    } while (uid == InvalidKey());
    return -uid + 4;  // Change a little from what XRTMemoryManager generates
  }

  mutable std::mutex mem_buffers_mtx_;
  std::unordered_map<int64, std::shared_ptr<TensorDataEntry>> tensor_map_;
};


}

class WseXlaService : public xla::grpc::XlaService::Service {
  typedef xla::grpc::XlaService::Service Super;
public:
  WseXlaService() : memory_manager_(std::make_unique<MemoryManager>()) {}

  virtual ~WseXlaService() {
    if (server_) {
      //server_->
    }
  }

  bool Start(int port, bool wait) {
    std::strstream ss;
    ss << "0.0.0.0" << ":" << port;
    const std::string server_address = ss.str();
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    server_ = builder.BuildAndStart();
    std::cout << "WSE Server listening on " << server_address << std::endl;
    if (wait) {
      server_->Wait();
    }
  }

protected:

  ::grpc::Status Compile(::grpc::ServerContext* context, const ::xla::CompileRequest* request, ::xla::CompileResponse* response) override {
    HEREX();
    return Super::Compile(context, request, response);
  }

  ::grpc::Status GetShape(::grpc::ServerContext* context, const ::xla::GetShapeRequest* request, ::xla::GetShapeResponse* response) override {
    assert(false);  // get from memory manager
    return ::grpc::Status::OK;
  }

  ::grpc::Status Execute(::grpc::ServerContext* context, const ::xla::ExecuteRequest* request, ::xla::ExecuteResponse* response) override {
#if 0
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
#endif
    return ::grpc::Status::OK;
  }

  ::grpc::Status DeconstructTuple(::grpc::ServerContext* context, const ::xla::DeconstructTupleRequest* request, ::xla::DeconstructTupleResponse* response) override {
    HEREX();
    return Super::DeconstructTuple(context, request, response);
  }

  ::grpc::Status TransferToClient(::grpc::ServerContext* context, const ::xla::TransferToClientRequest* request, ::xla::TransferToClientResponse* response) override {
    HEREX();
    return Super::TransferToClient(context, request, response);
  }
  // Transfers the given literal to the server to be stored in a global
  // allocation, which is returned.
  ::grpc::Status TransferToServer(::grpc::ServerContext* context, const ::xla::TransferToServerRequest* request, ::xla::TransferToServerResponse* response) {
    HEREX();
    return Super::TransferToServer(context, request, response);
  }

  // Methods used by GlobalData.
  ::grpc::Status Unregister(::grpc::ServerContext* context, const ::xla::UnregisterRequest* request, ::xla::UnregisterResponse* response) override {
    HEREX();
    // if it's true, then use it
    bool all_ok = true;
    for (const auto& data : request->data()) {
      if (!memory_manager_->Free(data.handle())) {
        all_ok = false;
      }
    }
    return all_ok ?
      ::grpc::Status::OK :
      ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Not found");
  }

private:
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<MemoryManager> memory_manager_;
};

namespace {

std::shared_ptr<WseXlaService> wse_xla_service{nullptr};

}

int StartLocalXlaService(int port) {
  wse_xla_service = std::make_shared<WseXlaService>();
  wse_xla_service->Start(port, false);
  return 0;
}

}  // namespace xla
