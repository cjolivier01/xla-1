#include "tensorflow/compiler/xla/rpc/grpc_service.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"

#include <memory>
#include <vector>
#include <strstream>

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/client_context.h>
#include <grpcpp/security/credentials.h>

namespace xla {

namespace {

typedef int64 handle_t;

// We use kDeviceBits to store the device ordinal in the handle. We store the
// device in the upper part of the int64 handle to make sure the random bits are
// in the lower part which is better when storing the handle as a key for
// unordered maps.
const int kDeviceBits = 12;

int64 MakeDeviceHandle(int64 device_ordinal, int64 rnd_value) {
  const int64 kUidMask = (static_cast<int64>(1) << (64 - kDeviceBits)) - 1;
  return (device_ordinal << (64 - kDeviceBits)) | (rnd_value & kUidMask);
}

int GetDeviceFromHandle(int64 handle) {
  return (handle >> (64 - kDeviceBits)) & ((1 << kDeviceBits) - 1);
}

std::size_t GetElementCount(const xla::ShapeProto& shape) {
  std::size_t element_count = 1;
  for (auto dim : shape.dimensions()) {
    element_count *= dim;
  }
  return element_count;
}

template<typename FILL_FUN_T>
void FillLiteral(xla::LiteralProto& literal, FILL_FUN_T fill_fn) {
  const std::size_t element_count = GetElementCount(literal.shape());
  for (std::size_t i = 0; i < element_count; ++i) {
    fill_fn(literal);
  }
}

// Quick version of XRTMemoryManager until we can get on
// the other side of a grpc boundary
class MemoryManager {
  struct LiteralDataEntry {

    LiteralDataEntry(
      std::shared_ptr<xla::LiteralProto> literal,
      const DeviceHandle& device_handle,
      bool has_data
    ) : literal_(literal),
        device_handle_(device_handle),
        has_data_(has_data) {}

    std::shared_ptr<xla::LiteralProto> literal_;
    DeviceHandle device_handle_;
    bool has_data_;
  };
public:

  MemoryManager(): distribution_(1, (1 << kDeviceBits) - 1) {}

  /**
   * @brief Check for valid handle
   * @param handle
   * @return
   */
  bool IsIn(int64 handle) const {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    return literal_map_.count(handle) != 0;
  }

  /**
   * @brief Register literal for device
   * @param literal
   * @param device_handle
   * @param has_data
   * @return
   */
  handle_t Register(
    std::shared_ptr<xla::LiteralProto> literal,
    const xla::DeviceHandle& device_handle,
    bool has_data
  ) {
    assert(literal.get());
    int64 handle;
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    while (true) {
      handle = ++next_memory_handle_ + 900000000;
      if (literal_map_.count(handle) == 0) {
        literal_map_.emplace(std::make_pair(
          handle,
          std::make_shared<LiteralDataEntry>(std::move(literal), device_handle, has_data))
        );
        literal_handle_to_device_handle_.emplace(handle, device_handle);
        break;
      }
    }
    return handle;
  }

  /**
   * @brief Get literal by handle
   * @param handle
   * @return
   */
  std::shared_ptr<xla::LiteralProto> Get(handle_t handle) const {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    auto found = literal_map_.find(handle);
    if (found == literal_map_.end()) {
      return nullptr;
    }
    return found->second->literal_;
  }

  /**
   * @brief Free memory handle
   * @param handle
   * @return
   */
  bool Free(handle_t handle) {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    if(literal_map_.count(handle)) {
      literal_map_.erase(handle);
      literal_handle_to_device_handle_.erase(handle);
      return true;
    }
    return false;
  }

  std::vector<xla::DeviceHandle> CreateDeviceHandles(size_t count) {
    std::vector<xla::DeviceHandle> results;
    results.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const int64 uid = CreateUid();
      const int ordinal = device_handles_allocated_++ + i;
      const int64 handle = MakeDeviceHandle(ordinal, uid);
      results.emplace_back(xla::DeviceHandle());
      results.rbegin()->set_handle(handle);
      results.rbegin()->set_device_count(count);
    }
    return std::move(results);
  }

  xla::DeviceHandle GetDeviceHandleFromDataHandle(const handle_t data_handle) const {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    const auto found = literal_handle_to_device_handle_.find(data_handle);
    if (found != literal_handle_to_device_handle_.end()) {
      return found->second;
    }
    return xla::DeviceHandle();
  }

  void Reset() {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    literal_map_.clear();
    literal_handle_to_device_handle_.clear();
    device_handles_allocated_ = 0;
    next_memory_handle_ = 0;
  }

  static int64 InvalidKey() { return 0; }
  int64 CreateUid() {
    int64 uid;
    do {
      uid = distribution_(generator_);
    } while (uid == InvalidKey());
    return -uid + 4;  // Change a little from what XRTMemoryManager generates
  }

  mutable std::mutex mem_buffers_mtx_;
  std::unordered_map<int64, std::shared_ptr<LiteralDataEntry>> literal_map_;
  std::unordered_map<int64, xla::DeviceHandle> literal_handle_to_device_handle_;
  std::atomic<int64> next_memory_handle_{0};
  std::atomic<int64> device_handles_allocated_{0};

  std::default_random_engine generator_;
  std::uniform_int_distribution<int> distribution_;
};

class WseXlaService : public xla::grpc::XlaService::Service {
  typedef xla::grpc::XlaService::Service Super;
public:
  WseXlaService() : memory_manager_(std::make_unique<MemoryManager>()) {}

  virtual ~WseXlaService() {}

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
    ExecutorInfoPtr exec = std::make_shared<ExecutorInfo>(*request, ++next_executor_handle_);
    std::lock_guard<std::mutex> lk(executor_map_mtx_);
    executor_info_map_.emplace(std::make_pair(exec->handle(), exec));
    response->mutable_handle()->set_handle(exec->handle());
    return ::grpc::Status::OK;
  }

  ::grpc::Status Execute(::grpc::ServerContext* context, const ::xla::ExecuteRequest* request, ::xla::ExecuteResponse* response) override {
    // Get the executable
    ExecutorInfoPtr executable;
    const handle_t execution_handle = request->handle().handle();
    {
      std::lock_guard<std::mutex> lk(executor_map_mtx_);
      auto found = executor_info_map_.find(execution_handle);
      if (found == executor_info_map_.end()) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Executable not found");
      }
      executable = found->second;
    }

    xla::DeviceHandle device_handle;

    // Parse the arguments
    for (const xla::GlobalDataHandle& argument : request->arguments()) {
      const handle_t arg_handle = argument.handle();
      if (!memory_manager_->IsIn(arg_handle)) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Argument handle not found");
      }
      if (!device_handle.handle()) {
        device_handle = memory_manager_->GetDeviceHandleFromDataHandle(arg_handle);
      }
    }
    std::vector<xla::LiteralProto> results;

    // Create the results and return the global data handles for them
    const xla::ProgramShapeProto& program_shape = executable->get_program_shape();

    //std::cout << "Result xla::ProgramShapeProto: " << msg_to_json(program_shape) << std::endl << std::flush;

    const xla::ShapeProto& result = program_shape.result();

    //std::cout << "Result xla::ShapeProto: " << msg_to_json(result) << std::endl << std::flush;

    std::list<xla::ShapeProto> result_shapes;
    const xla::PrimitiveType result_element_type = result.element_type();
    if (result_element_type == xla::PrimitiveType::TUPLE) {
      for (const xla::ShapeProto& tuple_item_shape : result.tuple_shapes()) {
        // Don't currently support nested tuples in results
        assert(tuple_item_shape.element_type() != xla::PrimitiveType::TUPLE);
        result_shapes.push_back(tuple_item_shape);
      }
    } else {
      result_shapes.push_back(result);
    }

    // Go through results, create buffers and allocate
    std::vector<handle_t> result_handles;
    result_handles.reserve(result_shapes.size());
    for (const xla::ShapeProto& result_shape : result_shapes) {
      std::shared_ptr<xla::LiteralProto> literal = std::make_shared<xla::LiteralProto>();
      *literal->mutable_shape() = result_shape;
      const xla::PrimitiveType xla_type = result_shape.element_type();
      switch (xla_type) {
        case xla::PrimitiveType::F32:
          FillLiteral(*literal, [](xla::LiteralProto& l) { l.add_f32s(0.01); });
          break;
        case xla::PrimitiveType::F64:
          FillLiteral(*literal, [](xla::LiteralProto& l) { l.add_f64s(0.01); });
          break;
        case xla::PrimitiveType::S32:
          FillLiteral(*literal, [](xla::LiteralProto& l) { l.add_s32s(1); });
          break;
        case xla::PrimitiveType::U32:
          FillLiteral(*literal, [](xla::LiteralProto& l) { l.add_u32s(1); });
          break;
        case xla::PrimitiveType::S64:
          FillLiteral(*literal, [](xla::LiteralProto& l) { l.add_s64s(1); });
          break;
        case xla::PrimitiveType::U64:
          FillLiteral(*literal, [](xla::LiteralProto& l) { l.add_s32s(1); });
          break;
        case xla::PrimitiveType::PRED:
          FillLiteral(*literal, [](xla::LiteralProto& l) { l.add_preds(false); });
          break;
        case xla::PrimitiveType::TUPLE:
        case xla::PrimitiveType::F16:
        case xla::PrimitiveType::S16:
        case xla::PrimitiveType::U16:
        case xla::PrimitiveType::BF16:
        case xla::PrimitiveType::U8:
        case xla::PrimitiveType::S8:
        case xla::PrimitiveType::C64:
        case xla::PrimitiveType::C128:
        default:
          return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Xla type not implemented");
      }
      const handle_t result_handle = memory_manager_->Register(
        literal, device_handle, true
      );
      result_handles.push_back(result_handle);
    }

    if (result_handles.size() == 1) {
      response->mutable_output()->set_handle(result_handles[0]);
    } else {
      // Return a tuple for multiple return shapes
      xla::ShapeProto tuple_shape;
      tuple_shape.set_element_type(xla::PrimitiveType::TUPLE);
      //tuple_shape.add_dimensions(result_handles.size());
      auto tuple_literal = std::make_shared<xla::LiteralProto>();
      *tuple_literal->mutable_shape() = tuple_shape;
      //tuple_item_shape.add_dimensions(result_handles.size());
      for (size_t item = 0; item < result_handles.size(); ++item) {
        const handle_t item_handle = result_handles[item];
        xla::ShapeProto tis;
        tis.add_dimensions(1);
        tis.set_element_type(xla::PrimitiveType::S64);
        *tuple_literal->mutable_shape()->add_tuple_shapes() = tis;
        assert(item_handle);
        //std::cout << "Adding handle to tuple: " << item_handle << std::endl << std::flush;
        tuple_literal->add_s64s(item_handle);
      }
      const handle_t tuple_handle = memory_manager_->Register(
        tuple_literal,
        device_handle,
        true
      );
      assert(tuple_handle);
      //std::cout << "handle for tuple: " << tuple_handle << std::endl << std::flush;
      response->mutable_output()->set_handle(tuple_handle);
    }

    ::xla::ExecutionProfile profile;
    *response->mutable_profile() = profile;

    return ::grpc::Status::OK;
  }

  ::grpc::Status DeconstructTuple(::grpc::ServerContext* context, const ::xla::DeconstructTupleRequest* request, ::xla::DeconstructTupleResponse* response) override {
    const xla::GlobalDataHandle tuple_handle = request->tuple_handle();

    std::shared_ptr<xla::LiteralProto> tuple_literal = memory_manager_->Get(tuple_handle.handle());
    if (!tuple_literal) {
      ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Memory handle not found when querying shape");
    }
    const xla::ShapeProto tuple_shape = tuple_literal->shape();
    if (tuple_shape.element_type() != xla::PrimitiveType::TUPLE) {
      return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "Handle supplie dis not a tuple");
    }

    int64 count = tuple_shape.tuple_shapes_size();
    assert(tuple_literal->s64s_size() == count);
    for (int64 j = 0; j < count; ++j) {
      const handle_t item_handle = tuple_literal->s64s(j);
      xla::GlobalDataHandle gdh;
      assert(item_handle);
      gdh.set_handle(item_handle);
      *response->add_element_handles() = gdh;
    }

    return ::grpc::Status::OK;
  }

  ::grpc::Status TransferToClient(::grpc::ServerContext* context, const ::xla::TransferToClientRequest* request, ::xla::TransferToClientResponse* response) override {
    return Super::TransferToClient(context, request, response);
  }
  // Transfers the given literal to the server to be stored in a global
  // allocation, which is returned.
  ::grpc::Status TransferToServer(::grpc::ServerContext* context, const ::xla::TransferToServerRequest* request, ::xla::TransferToServerResponse* response) {
    auto literal = std::make_shared<xla::LiteralProto>(request->literal());
    const int64 handle = memory_manager_->Register(literal, request->device_handle(), true);
    response->mutable_data()->set_handle(handle);
    return ::grpc::Status::OK;
  }

  // Methods used by GlobalData.
  ::grpc::Status Unregister(::grpc::ServerContext* context, const ::xla::UnregisterRequest* request, ::xla::UnregisterResponse* response) override {
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

  ::grpc::Status GetDeviceHandles(::grpc::ServerContext* context,
                                  const GetDeviceHandlesRequest* request,
                                  GetDeviceHandlesResponse* response) override {
    std::vector<xla::DeviceHandle> device_handles =
      memory_manager_->CreateDeviceHandles(request->device_count());
    for (const xla::DeviceHandle& device_handle : device_handles) {
      *response->mutable_device_handles()->Add() = device_handle;
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status ExecuteGraphParallel(::grpc::ServerContext* context,
                                      const ExecuteGraphParallelRequest* request,
                                      ExecuteParallelResponse* response) override {
    assert(false);
    return ::grpc::Status::OK;
  }

  ::grpc::Status WaitForExecution(::grpc::ServerContext* context,
                                  const WaitForExecutionRequest* request,
                                  WaitForExecutionResponse* response) override {
    assert(false);
    return ::grpc::Status::OK;
  }

  ::grpc::Status TransferToInfeed(::grpc::ServerContext* context,
                                  const TransferToInfeedRequest* request,
                                  TransferToInfeedResponse* response) override {
    assert(false);
    return ::grpc::Status::OK;
  }

  ::grpc::Status TransferFromOutfeed(
      ::grpc::ServerContext* context, const TransferFromOutfeedRequest* request,
      TransferFromOutfeedResponse* response) override {
    assert(false);
    return ::grpc::Status::OK;
  }

  ::grpc::Status ResetDevice(::grpc::ServerContext* context,
                             const ResetDeviceRequest* request,
                             ResetDeviceResponse* response) override {
    {
      std::lock_guard<std::mutex> lk(compile_map_mtx_);
      compile_info_map_.clear();
    }
    {
      std::lock_guard<std::mutex> lk(executor_map_mtx_);
      executor_info_map_.clear();
    }
    memory_manager_->Reset();
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetShape(::grpc::ServerContext* context,
                          const GetShapeRequest* request,
                          GetShapeResponse* response) override {
    const handle_t handle = request->data().handle();
    if (!handle) {
      return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "Invalid NULL handle in call to GetShape");
    }
    std::shared_ptr<xla::LiteralProto> literal = memory_manager_->Get(handle);
    if (!literal) {
      return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Memory handle not found when querying shape");
    }
    *response->mutable_shape() = literal->shape();
    return ::grpc::Status::OK;
  }

private:
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<MemoryManager> memory_manager_;

  struct CompileInfo {};
  using CompileInfoPtr = std::shared_ptr<CompileInfo>;

  class ExecutorInfo {
  public:
    ExecutorInfo(xla::CompileRequest compile_request, std::size_t handle)
    : compile_request_(compile_request),
      handle_(handle) {
    }
    std::size_t handle() const { return handle_; }

    const xla::ProgramShapeProto& get_program_shape() const {
      return compile_request_.computation().host_program_shape();
    }

  private:
    const xla::CompileRequest compile_request_;
    const std::size_t handle_;
  };
  using ExecutorInfoPtr = std::shared_ptr<ExecutorInfo>;

  std::mutex compile_map_mtx_;
  std::unordered_map<std::size_t, CompileInfoPtr> compile_info_map_;

  std::mutex executor_map_mtx_;
  std::unordered_map<std::size_t, ExecutorInfoPtr> executor_info_map_;
  std::atomic<std::size_t> next_executor_handle_{0};

};

std::shared_ptr<WseXlaService> wse_xla_service{nullptr};

}

int StartLocalWseXlaService(int port) {
  wse_xla_service = std::make_shared<WseXlaService>();
  wse_xla_service->Start(port, false);
  return 0;
}

// haven't gotten this to work yet in the same process for some reason won't connect
//int StartLocalCPUService(int port) {
//  int32 port = 1685;
//  bool any_address = false;
//  string platform_str = "WSE";
//  se::Platform *platform = nullptr;
//  if (!platform_str.empty()) {
//    platform = PlatformUtil::GetPlatform(platform_str).ValueOrDie();
//  }
//  std::unique_ptr<xla::GRPCService> service =
//    xla::GRPCService::NewService(platform).ConsumeValueOrDie();
//
//  ::grpc::ServerBuilder builder;
//  string server_address(
//    absl::StrFormat("%s:%d", any_address ? "[::]" : "localhost", port));
//
//  builder.SetMaxReceiveMessageSize(INT_MAX);
//  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
//  builder.RegisterService(service.get());
//  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
//
//  LOG(INFO) << "Server listening on " << server_address;
//}

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
//          PopulateTensorBuffer(literal`, source_tensor.shape, dest_buffer,
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
//      xla::LiteralProto index_tensor(tensorflow::DT_INT32,
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
        auto tensor_ptr = std::make_shared<xla::LiteralProto>(
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

}  // namespace xla
