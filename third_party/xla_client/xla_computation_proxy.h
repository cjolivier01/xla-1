#ifndef XLA_CLIENT_XLA_COMPUTATION_PROXY_H_
#define XLA_CLIENT_XLA_COMPUTATION_PROXY_H_

#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include <memory>
#include <sstream>
#include <stdexcept>

#include <sys/syscall.h>

namespace xla {

class XlaClientInfo;

class XlaComputationProxy : public XrtComputationClient {
    typedef XrtComputationClient Super;
public:
  /**
   * @brief Create XlaComputationProxy object
   * @param options
   * @param topology_proto
   */
  XlaComputationProxy(
      Options options,
      std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto
  );
  /**
   * @brief Destroy the XlaComputationProxy object
   */
  ~XlaComputationProxy() = default;

  // Creates a Data object with no actual device handle in it. The device handle
  // will be populated in an asynchrounous fashion.
  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  // Transfers local tensor values to the TPU servers and fetches the handles.
  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  // Compiles a set of computations.
  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device, const ExecuteComputationOptions& options) override;

  static void SetDeviceProxyAddress(const std::string& device, const std::string& proxy_address);

  static tensorflow::tpu::TopologyProto InitializeAndFetchTopology(
      const std::string& job, int task_no, const std::string& worker_host_port,
      const tensorflow::ConfigProto& config);

  static XlaComputationProxy *Get() {
    return dynamic_cast<XlaComputationProxy *>(Super::Get());
  }

  static bool IsEnabled();

  static bool IsInitialized();

private:
  class GlobalDataHandleMapper;

  static bool HasProxyAddresses();

  std::vector<DataPtr> TransferToServerInternal(absl::Span<const TensorSource> tensors);

  static std::recursive_mutex xla_client_map_mtx_;
  static std::unordered_map<std::string, std::shared_ptr<XlaClientInfo>> xla_client_map_;

  std::unique_ptr<GlobalDataHandleMapper> data_mapper_;

  static std::shared_ptr<xla::ServiceInterface> GetXlaClient(const std::string& device, bool create = true);
  xla::DeviceHandle GetDeviceHandle(const std::string& device);

  mutable std::mutex proxy_executable_set_mtx_;
  std::unordered_set<uint64_t> proxy_executable_set_;

  mutable std::mutex proxy_mapping_mtx_;
  std::unordered_map<std::string, std::string> proxy_mapping_;

  std::vector<ComputationClient::DataPtr> MoveDataBetweenDevices(
    const std::vector<ComputationClient::DataPtr>& source_data,
    const std::string& to_device,
    bool release_from_source,
    bool add_mapping_entry
  );

  /**
   * @brief Is device capable of proxy?
   * @param device
   * @return
   */

  std::vector<DataPtr> NormalizeDataToDevice(absl::Span<const DataPtr> tensors, const std::string& device, bool in_place);

  ComputationClient::DataPtr TransferLiteralToServer(
    const std::string& device,
    const Literal& literal
  );
  //bool IsProxyDevice(const std::string& device) const;
  //bool SetProxyForDevice(const std::string& source_device, const std::string& proxy_device);
  bool ShouldCloneDataForDevice(const std::string& device) const;
  bool IsProxyExecutable(uint64_t executable_handle) const;
  void AddProxyExecutable(uint64_t executable_handle);

  /**
   * @brief Should this device be proxied right now?
   * @param device
   * @return
   */
  void ReleaseXrtData(const std::string& device, int64 handle) override;
  void ReleaseXlaProxyData(const std::string& device, int64 handle);
};

enum class Color {
  BG_INVALID = -1,
  FG_RESET = 0,
  FG_RED = 31,
  FG_GREEN = 32,
  FG_YELLOW = 33,
  FG_BLUE = 34,
  FG_MAGENTA = 35,
  FG_CYAN = 36,
  FG_WHITE = 37,
  FG_DEFAULT = 39,
  BG_RED = 41,
  BG_GREEN = 42,
  BG_BLUE = 44,
  BG_MAGENTA = 45,
  BG_CYAN = 46,
  BG_WHITE = 47,
  BG_DEFAULT = 49,
};

class ColorModifier {
  const Color code;
  const bool bright;
public:
  ColorModifier(Color pCode, const bool is_bright=true)
      : code(pCode), bright(is_bright) {}

  friend std::ostream &
  operator<<(std::ostream &os, const ColorModifier &mod) {
    return os << "\033[" << (mod.bright ? "1;" : "0;") << (int)mod.code << "m";
  }
};

class ColorScope {
  std::ostream& os_;
public:
  inline ColorScope(std::ostream& os, Color pCode, bool bright=true) : os_(os) {
    ColorModifier mod(pCode, bright);
    os << mod;
  }
  inline ColorScope(std::ostream& os, std::vector<Color> codes, bool bright=false) : os_(os) {
    for (auto c : codes) {
      ColorModifier mod(c, bright);
      os << mod;
    }
  }
  ColorScope(Color pCode, bool bright=true) : os_(std::cout) {
    os_ << ColorModifier(pCode, bright) << std::flush;
  }
  ~ColorScope() {
    os_ << ColorModifier(Color::FG_DEFAULT)
        << ColorModifier(Color::FG_DEFAULT)
        << ColorModifier(Color::BG_DEFAULT);
  }
};

template<typename T>
inline std::string to_string(const T& obj) {
  std::stringstream ss;
  ss << obj;
  return std::move(ss.str());
}


#define WSE_DEBUG_LOGGING

#ifdef WSE_DEBUG_LOGGING

class EnterLeave {
  static __thread int depth_;
  static const std::string library_;
  static const Color library_color_;
  const std::string label_;
  const pid_t thread_id_;
  const bool both_;
  const Color use_color_;
  static std::mutex mtx_;
public:
  static std::string concat(const char *s0, const char *s1, const char *s2) {
    std::string s;
    if (s0 && *s0) {
      s = s0;
      s += "::";
    }
    if (s1) {
      s += s1;
    }
    if (s2 && *s2) {
      s += " (";
      s += s2;
      s += ")";
    }
    return s;
  }
  inline EnterLeave(const std::string& label, bool both=true, const Color use_color = Color::BG_INVALID)
      : label_(label), thread_id_(syscall(SYS_gettid)), both_(both),
        use_color_(use_color == Color::BG_INVALID ? library_color_ : use_color) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (int x = 0; x < depth_; ++x) {
      printf("  ");
    }
    ColorScope color_scope(use_color_);
    printf("%s[tid=%d (%s)]: %s\n", both_ ? "ENTER" : "HERE", thread_id_, library_.c_str(), label.c_str());
    fflush(stdout);
    ++depth_;
  }
  inline ~EnterLeave() {
    std::lock_guard<std::mutex> lk(mtx_);
    --depth_;
    if (both_) {
      ColorScope color_scope(use_color_);
      for (int x = 0; x < depth_; ++x) {
        printf("  ");
      }
      printf("LEAVE[tid=%d (%s)]: %s\n", thread_id_, library_.c_str(), label_.c_str());
      fflush(stdout);
    }
  }
};
#else

class EnterLeave {
public:
  inline EnterLeave(const std::string& label, bool both=true) {}
};

#endif  // WSE_DEBUG_LOGGING

#ifdef WSE_DEBUG_LOGGING

std::string short_fn_name(const std::string &fn_name);

#define HEREC(__color$) EnterLeave __here(EnterLeave::concat(nullptr, ::xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), """"), true, __color$)
#define HERE() EnterLeave __here(EnterLeave::concat(nullptr, ::xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""))
#define HEREX() EnterLeave __here(EnterLeave::concat(nullptr, ::xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), false)
#define HEREXC(__color$) EnterLeave __here(EnterLeave::concat(nullptr, ::xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), false, __color$)
#define HEREXCT(__color$) EnterLeave __here(EnterLeave::concat(std::to_string(this).c_str(), ::xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), false, __color$)
#else
#define HERE() ((void)0)
#define HEREX() ((void)0)
#endif

#define ENDL std::endl << std::flush

}  // namespace xla

#endif  // XLA_CLIENT_XLA_COMPUTATION_PROXY_H_