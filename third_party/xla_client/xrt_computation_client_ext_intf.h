#pragma once

#ifndef XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_
#define XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_

#include <cstddef>
#include <vector>
#include <functional>

namespace xla {

class HloModuleProto;

namespace ptxla {

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
 */
struct XrtComputationClientExternalInterface :
    public std::enable_shared_from_this<XrtComputationClientExternalInterface> {

  virtual ~XrtComputationClientExternalInterface() = default;

  virtual void OnCreate(xla::ptxla::opaque_t obj) = 0;

  virtual void OnDestroy(xla::ptxla::opaque_t obj) = 0;

  /**
   * @brief Called whenevr
   * @param hash
   * @param hlo_module
   * @param compile_state
   * @return
   */
  virtual ECompileResult OnCompile(
      xla::ptxla::opaque_t obj,
      std::size_t hash,
      const xla::HloModuleProto &hlo_module,
      const std::vector<std::string> &devices,
      ECompileState compile_state
  ) = 0;

  /**
   * @brief
   * @param hash
   * @param run_state
   * @return
   */
  virtual ERunStatus OnExecuteComputation(
      xla::ptxla::opaque_t obj,
      std::size_t hash,
      const std::string &device,
      ERunState run_state
  ) = 0;

  // Transfers local tensor values to the TPU servers and fetches the handles.
  virtual std::pair<EIntent, std::vector<XDataPtr>> TransferToServer(
      xla::ptxla::opaque_t obj,
      std::vector<XTensorSource>& tensors
  ) = 0;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  virtual std::pair<EIntent, std::vector<XLiteral>> TransferFromServer(
      xla::ptxla::opaque_t obj,
      std::vector<XDataPtr>& handles
  ) = 0;

};

}  // namespace ptxla
}  // namespace xla

#endif  // XLA_CLIENT_XRT_COMPUTATION_CLIENT_EXT_INTF_H_
