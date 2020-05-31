#pragma once

#ifndef XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_
#define XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_

#include <vector>
#include <memory>
#include <vector>
#include <sstream>
#include <random>

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/client_context.h>
#include <grpcpp/security/credentials.h>

#ifndef COLLISION_FREE_NAMESPACE
#define COLLISION_FREE_NAMESPACE ptxla
#endif

#ifndef NOT_COMPILING_FROM_PTXLA
namespace xla {
  using XlaService = ::xla::grpc::XlaService;
}
#endif

namespace xla {
namespace COLLISION_FREE_NAMESPACE {

typedef long long int64;
typedef int64 handle_t;

// We use kDeviceBits to store the device ordinal in the handle. We store the
// device in the upper part of the int64 handle to make sure the random bits are
// in the lower part which is better when storing the handle as a key for
// unordered maps.

inline std::size_t GetElementCount(const xla::ShapeProto& shape) {
  std::size_t element_count = 1;
  for (auto dim : shape.dimensions()) {
    element_count *= dim;
  }
  return element_count;
}

template<typename FILL_FUN_T>
inline void FillLiteral(xla::LiteralProto& literal, FILL_FUN_T fill_fn) {
  const std::size_t element_count = GetElementCount(literal.shape());
  for (std::size_t i = 0; i < element_count; ++i) {
    fill_fn(literal);
  }
}

/**
 * @brief Uid utility class
 */
const int kDeviceBits = 12;
class UidUtil {

  static int64 MakeDeviceHandle(int64 device_ordinal, int64 rnd_value) {
    const int64 kUidMask = (static_cast<int64>(1) << (64 - kDeviceBits)) - 1;
    return (device_ordinal << (64 - kDeviceBits)) | (rnd_value & kUidMask);
  }

  static int64 InvalidKey() { return 0; }

public:

  static int GetDeviceOrdinalFromHandle(const xla::DeviceHandle& device_handle) {
    return (device_handle.handle() >> (64 - kDeviceBits)) & ((1 << kDeviceBits) - 1);
  }

  static int GetDeviceOrdinalFromHandle(int64 handle) {
    return (handle >> (64 - kDeviceBits)) & ((1 << kDeviceBits) - 1);
  }

  UidUtil(): distribution_(1, (1 << kDeviceBits) - 1) {}

  int64 CreateUid() {
    int64 uid;
    do {
      uid = distribution_(generator_);
    } while (uid == InvalidKey());
    return uid;  // Change a little from what XRTMemoryManager generates
  }
  int64 CreateDeviceUid(int64 device_ordinal) {
    return MakeDeviceHandle(device_ordinal, CreateUid());
  }
private:
  std::default_random_engine generator_;
  std::uniform_int_distribution<int> distribution_;
};
using UidUtilPtr = std::shared_ptr<UidUtil>;

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

  MemoryManager(UidUtilPtr uid_util_ptr): uid_util_ptr_(uid_util_ptr) {}

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
      const int ordinal = device_ordinals_allocated_++;
      const int64 handle = uid_util_ptr_->CreateDeviceUid(ordinal);
      const int check_ordinal = UidUtil::GetDeviceOrdinalFromHandle(handle);
      assert(check_ordinal == ordinal);
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
      const int64 device_ordinal = UidUtil::GetDeviceOrdinalFromHandle(found->second.handle());
      assert(device_ordinal < device_ordinals_allocated_.load());
      return found->second;
    }
    return xla::DeviceHandle();
  }

  void Reset() {
    std::lock_guard<std::mutex> lk(mem_buffers_mtx_);
    literal_map_.clear();
    literal_handle_to_device_handle_.clear();
    device_ordinals_allocated_ = 0;
    next_memory_handle_ = 0;
  }

private:
  UidUtilPtr uid_util_ptr_;
  mutable std::mutex mem_buffers_mtx_;
  std::unordered_map<int64, std::shared_ptr<LiteralDataEntry>> literal_map_;
  std::unordered_map<int64, xla::DeviceHandle> literal_handle_to_device_handle_;
  std::atomic<int64> next_memory_handle_{0};  // TODO: use uid generator
  std::atomic<int64> device_ordinals_allocated_{0};
};

/**
 * @brief ExecutableInfo
 */
class ExecutableInfo {
public:
  /**
   * @brief ExecutableInfo constructor -- holds information abotu an "executable"
   * @param compile_request
   * @param handle
   */
  ExecutableInfo(
    xla::CompileRequest compile_request,
    std::size_t handle
  ) : compile_request_(std::move(compile_request)),
      handle_(handle) {
  }

  virtual void Initialize(std::size_t device_ordinal) {
    std::cout << "Initializing executable for device ordinal: "
              << device_ordinal << std::endl << std::flush;
  }

  /**
   * @brief Get the handle
   * @return
   */
  std::size_t handle() const { return handle_; }

  /**
   * @brief Retrieve the program shape
   * @return
   */
  const xla::ProgramShapeProto &get_program_shape() const {
    return compile_request_.computation().host_program_shape();
  }
private:
  const xla::CompileRequest compile_request_;
  const std::size_t handle_;
};
using ExecutableInfoPtr = std::shared_ptr<ExecutableInfo>;


/**
 * @brief Executable Manager
 */
class ExecutableManager {
public:
  /**
   * @brief ExecutableManager constructor
   * @param uid_util_ptr UID generator
   */
  ExecutableManager(UidUtilPtr uid_util_ptr)
    : uid_util_ptr_(uid_util_ptr) {}

    /**
     * @brief Create an executable for the gin device (ala Compiler::RunBackend())
     * @param compile_request
     * @return
     */
  template<typename EXECUTABLE_T>
  std::shared_ptr<EXECUTABLE_T> Create(xla::CompileRequest compile_request) {
    std::shared_ptr<EXECUTABLE_T> result =  std::make_shared<EXECUTABLE_T>(
      std::move(compile_request),
      uid_util_ptr_->CreateUid()  // can have multiple device handles :(
    );
    std::lock_guard<std::mutex> lk(executor_map_mtx_);
    executor_info_map_.insert(std::make_pair(result->handle(), result));
    return result;
  }

  /**
   * Release an executable by handle
   */
  bool Release(handle_t executable_handle) {
    std::lock_guard<std::mutex> lk(executor_map_mtx_);
    auto iter = executor_info_map_.find(executable_handle);
    if (iter == executor_info_map_.end()) {
      return false;
    }
    executor_info_map_.erase(iter);
    return true;
  }

  /**
   * Retreive up an executable by handle
   */
  ExecutableInfoPtr GetExecutable(handle_t handle) {
    std::lock_guard<std::mutex> lk(executor_map_mtx_);
    auto found = executor_info_map_.find(handle);
    if (found == executor_info_map_.end()) {
      return nullptr;
    }
    return found->second;
  }

  /**
   * @brief Clear all executables
   */
  void Reset() {
    std::lock_guard<std::mutex> lk(executor_map_mtx_);
    executor_info_map_.clear();
  }

private:
  UidUtilPtr uid_util_ptr_;
  std::mutex executor_map_mtx_;
  std::unordered_map<std::size_t, ExecutableInfoPtr> executor_info_map_;
};

/**
 * @brief XLA Service
 */
class SimpleXlaService : public xla::XlaService::Service {
  typedef xla::XlaService::Service Super;
public:
  SimpleXlaService(UidUtilPtr uid_util_ptr)
    : uid_util_ptr_(uid_util_ptr),
      memory_manager_(std::make_unique<MemoryManager>(uid_util_ptr)),
      executable_manager_(std::make_unique<ExecutableManager>(uid_util_ptr)) {}

  virtual ~SimpleXlaService() {}

  /**
   * @brief Start
   * @param port
   * @param wait
   * @return
   */
  bool Start(int port, bool wait) {
    std::stringstream ss;
    ss << "0.0.0.0" << ":" << port;
    const std::string server_address = ss.str();
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    server_ = builder.BuildAndStart();
    std::cout << "XLA Server listening on " << server_address << std::endl;
    if (wait) {
      server_->Wait();
    }
    return true;
  }


  ::grpc::Status GetDeviceHandle(
    const ::xla::ExecuteRequest* request,
    xla::DeviceHandle *device_handle
  ) {
    // Parse the arguments
    for (const xla::GlobalDataHandle &argument : request->arguments()) {
      const handle_t arg_handle = argument.handle();
      if (!memory_manager_->IsIn(arg_handle)) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Argument handle not found");
      }
      *device_handle = memory_manager_->GetDeviceHandleFromDataHandle(arg_handle);
      if (device_handle->handle()) {
        break;
      }
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetDeviceOrdinal(const ::xla::ExecuteRequest* request, std::size_t *ordinal) {
    xla::DeviceHandle device_handle;
    ::grpc::Status status = GetDeviceHandle(request, &device_handle);
    if (!status.ok()) {
      return status;
    }
    *ordinal = UidUtil::GetDeviceOrdinalFromHandle(device_handle);
    return ::grpc::Status::OK;
  }

protected:

  /**
   * @brief Compile
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status Compile(::grpc::ServerContext* context, const ::xla::CompileRequest* request, ::xla::CompileResponse* response) override {
    std::cout << "XLA Compile" << std::endl << std::flush;
    ExecutableInfoPtr exec = executable_manager_->Create<ExecutableInfo>(*request);
    response->mutable_handle()->set_handle(exec->handle());
    return ::grpc::Status::OK;
  }

  /**
   * @brief Execute Computation
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status Execute(::grpc::ServerContext* context, const ::xla::ExecuteRequest* request, ::xla::ExecuteResponse* response) override {
    std::cout << "XLA Execute" << std::endl << std::flush;
    // Get the executable
    ExecutableInfoPtr executable = executable_manager_->GetExecutable(request->handle().handle());
    if (!executable) {
      return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Executable not found");
    }

//    xla::DeviceHandle device_handle;
//
//    // Parse the arguments
//    for (const xla::GlobalDataHandle& argument : request->arguments()) {
//      const handle_t arg_handle = argument.handle();
//      if (!memory_manager_->IsIn(arg_handle)) {
//        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Argument handle not found");
//      }
//      if (!device_handle.handle()) {
//        device_handle = memory_manager_->GetDeviceHandleFromDataHandle(arg_handle);
//      }
//    }

    xla::DeviceHandle device_handle;

    ::grpc::Status status = GetDeviceHandle(request, &device_handle);
    if (!status.ok()) {
      return status;
    }
    
    executable->Initialize(UidUtil::GetDeviceOrdinalFromHandle(device_handle));

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

  /**
   * @brief Deconstruct Tuple
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status DeconstructTuple(::grpc::ServerContext* context, const ::xla::DeconstructTupleRequest* request, ::xla::DeconstructTupleResponse* response) override {
    std::cout << "XLA DeconstructTuple" << std::endl << std::flush;
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

  /**
   * @brief Transfer to Client
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status TransferToClient(::grpc::ServerContext* context, const ::xla::TransferToClientRequest* request, ::xla::TransferToClientResponse* response) override {
    std::cout << "XLA TransferToClient" << std::endl << std::flush;
    return Super::TransferToClient(context, request, response);
  }
  // Transfers the given literal to the server to be stored in a global
  // allocation, which is returned.
  ::grpc::Status TransferToServer(::grpc::ServerContext* context, const ::xla::TransferToServerRequest* request, ::xla::TransferToServerResponse* response) {
    auto literal = std::make_shared<xla::LiteralProto>(request->literal());
//    std::cout << "TransferToServer() device handle: " << request->device_handle().handle()
//              << " (ordinal = " << UidUtil::GetDeviceOrdinalFromHandle(request->device_handle().handle()) << " )"
//              << std::endl << std::flush;
    const int64 handle = memory_manager_->Register(literal, request->device_handle(), true);
    xla::GlobalDataHandle gdh;
    gdh.set_handle(handle);
    *response->mutable_data() = gdh;
    return ::grpc::Status::OK;
  }

  /**
   * @brief Unregister
   * @param context
   * @param request
   * @param response
   * @return
   */
  // Methods used by GlobalData.
  ::grpc::Status Unregister(::grpc::ServerContext* context, const ::xla::UnregisterRequest* request, ::xla::UnregisterResponse* response) override {
    std::cout << "XLA Unregister" << std::endl << std::flush;
    // if it's true, then use it
    bool all_ok = true;
    for (const auto& data : request->data()) {
      if (!memory_manager_->Free(data.handle())) {
        if (!executable_manager_->Release(data.handle())) {
          all_ok = false;
        } else {
          std::cout << "Released executable: " << data.handle() << std::endl << std::flush;
        }
      }
    }
    return all_ok ?
           ::grpc::Status::OK :
           ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Not found");
  }

  /**
   * @brief GetDeviceHandles
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status GetDeviceHandles(::grpc::ServerContext* context,
                                  const xla::GetDeviceHandlesRequest* request,
                                  xla::GetDeviceHandlesResponse* response) override {
    std::cout << "XLA GetDeviceHandles" << std::endl << std::flush;
    std::vector<xla::DeviceHandle> device_handles =
      memory_manager_->CreateDeviceHandles(request->device_count());
    for (const xla::DeviceHandle& device_handle : device_handles) {
      *response->mutable_device_handles()->Add() = device_handle;
    }
    return ::grpc::Status::OK;
  }

  /**
   * @brief ExecuteGraphParallel
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status ExecuteGraphParallel(::grpc::ServerContext* context,
                                      const xla::ExecuteGraphParallelRequest* request,
                                      xla::ExecuteParallelResponse* response) override {
    std::cout << "XLA ExecuteGraphParallel" << std::endl << std::flush;
    assert(false);
    return ::grpc::Status::OK;
  }

  /**
   * @brief WaitForExecution
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status WaitForExecution(::grpc::ServerContext* context,
                                  const xla::WaitForExecutionRequest* request,
                                  xla::WaitForExecutionResponse* response) override {
    std::cout << "XLA WaitForExecution" << std::endl << std::flush;
    assert(false);
    return ::grpc::Status::OK;
  }

  /**
   * @brief TransferToInfeed
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status TransferToInfeed(::grpc::ServerContext* context,
                                  const xla::TransferToInfeedRequest* request,
                                  xla::TransferToInfeedResponse* response) override {
    std::cout << "XLA TransferToInfeed" << std::endl << std::flush;
    assert(false);
    return ::grpc::Status::OK;
  }

  /**
   * @brief TransferFromOutfeed
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status TransferFromOutfeed(
    ::grpc::ServerContext* context, const xla::TransferFromOutfeedRequest* request,
    xla::TransferFromOutfeedResponse* response) override {
    std::cout << "XLA TransferFromOutfeed" << std::endl << std::flush;
    assert(false);
    return ::grpc::Status::OK;
  }

  /**
   * @brief ResetDevice
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status ResetDevice(::grpc::ServerContext* context,
                             const xla::ResetDeviceRequest* request,
                             xla::ResetDeviceResponse* response) override {
    std::cout << "XLA ResetDevice" << std::endl << std::flush;
    {
      std::lock_guard<std::mutex> lk(compile_map_mtx_);
      compile_info_map_.clear();
    }
    executable_manager_->Reset();
    memory_manager_->Reset();
    return ::grpc::Status::OK;
  }

  /**
   * @brief GetShape
   * @param context
   * @param request
   * @param response
   * @return
   */
  ::grpc::Status GetShape(::grpc::ServerContext* context,
                          const xla::GetShapeRequest* request,
                          xla::GetShapeResponse* response) override {
    std::cout << "XLA GetShape" << std::endl << std::flush;
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

protected:
  UidUtilPtr uid_util_ptr_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<MemoryManager> memory_manager_;
  std::unique_ptr<ExecutableManager> executable_manager_;

  struct CompileInfo {};
  using CompileInfoPtr = std::shared_ptr<CompileInfo>;

  std::mutex compile_map_mtx_;
  std::unordered_map<std::size_t, CompileInfoPtr> compile_info_map_;

};  // class SimpleXlaService


#if 0
class HloModuleProto;

namespace COLLISION_FREE_NAMESPACE {

enum ECompileState {
  ECS_BEFORE_COMPILE,
  ECS_AFTER_COMPILE,
};

enum ECompileResult {
  ECR_ACCEPT,
  ECRT_DEFER
};

enum ERunState {
  ERS_BEFORE_RUN,
  ERS_AFTER_RUN,
};

enum ERunStatus {
  ERS_ACCEPT,
  ERS_DEFER,
};

// TODO: protobuf
using XShape = std::vector<std::size_t>;

// TODO: protobuf
class XData {
public:
  struct Info {
    virtual ~Info() {}
  };

  using OpaqueHandle = std::int64_t;

  XData(std::string device, XShape shape)
      : device_(std::move(device)), shape_(std::move(shape)) {}

  virtual ~XData() {}

  const std::string& device() const { return device_; }

  const XShape& shape() const { return shape_; }

  Info* info() const { return info_.get(); }

  std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
      std::swap(info, info_);
      return info;
  }

  virtual OpaqueHandle GetOpaqueHandle() = 0;

  virtual void Assign(const XData& data) = 0;

  virtual bool HasValue() const = 0;

private:
  std::string device_;
  XShape shape_;
  std::shared_ptr<Info> info_;
};

using XDataPtr = std::shared_ptr<XData>;

// TODO: protobuf
struct XTensorSource {
  // The PopulateFn accepts a dense buffer is standard array layout
  // (dim0-major) and deposits the source tensor data directly over the
  // provided buffer.
  using PopulateFn = std::function<void(const XTensorSource &, void *, size_t)>;

  XTensorSource() = default;

  XTensorSource(XShape shape, std::string device, PopulateFn populate_fn)
      : shape(std::move(shape)),
        device(std::move(device)),
        populate_fn(std::move(populate_fn)) {}

  XShape shape;
  std::string device;
  PopulateFn populate_fn;
};

// TODO: protobuf (just a buffer, caller will know what it is)
struct XLiteral {
  XShape _shape;
};

enum EIntent {
  EI_DEFER,
  EI_ACCEPT
};

typedef ptrdiff_t opaque_t;

/**
 * Define interface
 *
 * This class is TEMPORARY until the XRT GRPC barrier is writting to behave similarly
 * to the TPU XRT barrier (which has the same API that we need).  In the interest of POC,
 * writing as an in-process callback atm.
 * That's not to say that the GRPC callback won;t be in-process (it is for much of the XRT layer
 * in most cases, such as what calls cpu_compiler.cc), but this will also keep us from
 * needing to modify very much TensorFlow source code, since we aren't calling back from
 * within the TF codebase once the GRPC XRT layer is implemented.
 */
//struct XrtComputationClientExternalInterface :
//    public std::enable_shared_from_this<XrtComputationClientExternalInterface> {
//
//  virtual ~XrtComputationClientExternalInterface() = default;
//
//  virtual void OnCreate(xla::COLLISION_FREE_NAMESPACE::opaque_t obj) = 0;
//
//  virtual void OnDestroy(xla::COLLISION_FREE_NAMESPACE::opaque_t obj) = 0;
//
//  /**
//   * @brief Called whenevr
//   * @param hash
//   * @param hlo_module
//   * @param compile_state
//   * @return
//   */
//  virtual ECompileResult OnCompile(
//      xla::COLLISION_FREE_NAMESPACE::opaque_t obj,
//      std::size_t hash,
//      const xla::HloModuleProto &hlo_module,
//      const std::vector<std::string> &devices,
//      ECompileState compile_state
//  ) = 0;
//
//  /**
//   * @brief
//   * @param hash
//   * @param run_state
//   * @return
//   */
//  virtual ERunStatus OnExecuteComputation(
//      xla::COLLISION_FREE_NAMESPACE::opaque_t obj,
//      std::size_t hash,
//      const std::string &device,
//      ERunState run_state
//  ) = 0;
//
//  // Transfers local tensor values to the TPU servers and fetches the handles.
//  virtual std::pair<EIntent, std::vector<XDataPtr>> TransferToServer(
//      xla::COLLISION_FREE_NAMESPACE::opaque_t obj,
//      std::vector<XTensorSource>& tensors
//  ) = 0;
//
//  // Reads the tensor literal values stored at TPU server sites, behind the
//  // supplied handles.
//  virtual std::pair<EIntent, std::vector<XLiteral>> TransferFromServer(
//      xla::COLLISION_FREE_NAMESPACE::opaque_t obj,
//      std::vector<XDataPtr>& handles
//  ) = 0;
//
//};

#endif

}  // namespace ptxla
}  // namespace xla

#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_
