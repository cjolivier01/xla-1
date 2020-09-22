#pragma once

#include <sys/syscall.h>

#include <map>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stack>

#include "tensorflow/compiler/xla/xla_client/types.h"
#include "tensorflow/compiler/xla/xla_client/xla_computation_proxy.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/tensor.h"

#define ENDL std::endl << std::flush

#if __cplusplus >= 201703L  // C++17
#include <shared_mutex>
using rw_mutex = std::shared_mutex;

class read_lock {
 public:
  explicit read_lock(rw_mutex& mtx) : mtx_(&mtx) { mtx_->lock_shared(); }
  ~read_lock() { mtx_->unlock_shared(); }

 private:
  rw_mutex* mtx_;
};
using write_lock = std::lock_guard<rw_mutex>;
#else
using rw_mutex = std::recursive_mutex;
using read_lock = std::lock_guard<rw_mutex>;
using write_lock = std::lock_guard<rw_mutex>;
#endif

namespace torch_xla {

enum EPythonState {
  EPS_INVALID = 0,
  EPS_IN_TRAIN_LOOP = 1,
  EPS_IN_DATA_BATCH = 2,
  EPS_IN_OPTIMIZER_STEP = 3,
  EPS_PROXY_DISABLED = 4,
  EPS_IN_DEBUG = 5,
  EPS_FORCE_PROXY = 6,
};

// extern std::stack<int> python_state_;

EPythonState GetPythonState(pid_t tid);
void PushPythonState(EPythonState state);
void PopPythonState();

class MsgException : public std::exception {
 public:
  MsgException() : std::exception() {}
  MsgException(const char* msg) : std::exception(), m_msg(msg) {}
  MsgException(const std::string& msg) : std::exception(), m_msg(msg) {}
  MsgException(const MsgException& obj)
      : std::exception(obj), m_msg(obj.m_msg) {}
  MsgException(MsgException&& rval)
      : std::exception(std::move(rval)), m_msg(std::move(rval.m_msg)) {}
  MsgException& operator=(const MsgException& rhs) {
    std::exception::operator=(rhs);
    m_msg = rhs.m_msg;
    return *this;
  }
  ~MsgException() noexcept {}

  static std::ostream& next(std::ostream& os) {
    os.put('\n');
    return os;
  }
  virtual const char* what() const noexcept override { return m_msg.c_str(); }

  template <typename T>
  MsgException& operator<<(const T& v) {
    // TODO: is there a more elegant way to do this?
    std::ostringstream oss;
    oss << v;
    m_msg.append(oss.str());
    return *this;
  }

 protected:
 private:
  std::string m_msg;
};

using sentinel_exception = MsgException;

struct MarkStepScope {
  MarkStepScope(const std::string& device_str,
                const std::vector<std::string>& devices);
  ~MarkStepScope();
  std::unique_ptr<xla::EnterLeave> el_;
};

struct HashingState {
  explicit HashingState(const xla::hash_t& start_hash)
      : start_hash_(start_hash) {};
  const xla::hash_t start_hash_;
  xla::hash_t pre_prune_hash_ = 0;
  std::size_t pass_ = 0;
  bool fabric_run_ = false;
  bool known_executable_ =
      false;  // optimization when we know this executable already exists
};

template <typename CB>
void XLATensor::print_tensors(const std::string& label,
                              const std::vector<XLATensor>& tensors, CB cb) {
  std::vector<XLATensor> ats;
  for (const XLATensor& t : tensors) {
    if (cb(t)) {
      ats.reserve(tensors.size());
      ats.emplace_back(t);
    }
  }
  print_all_tensors(label, ats);
}

class EnvFileMacro {
  static bool is_true(const std::string& s) {
    if (s.empty()) {
      return false;
    }
    const int c = ::toupper(s[0]);
    return c == 'Y' || c == 'T' || std::atoi(s.c_str()) > 0;
  }

  static bool get_env_bool(const std::string& name, bool default_value) {
    const char* s = getenv(name.c_str());
    if (!s || !*s) return default_value;
    return is_true(s);
  }

  template <class T>
  static T base_name(T const& path, T const& delims = "/\\") {
    return path.substr(path.find_last_of(delims) + 1);
  }
  template <class T>
  static T remove_extension(T const& filename) {
    typename T::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
  }
  /**
   * @brief Make macro-like name for environment variables
   *
   * @param file
   * @return std::string
   */
  static std::string file_to_macro_name(const std::string& file_name,
                                        const std::string& prefix) {
    std::stringstream ss;
    if (!prefix.empty()) {
      ss << prefix << "_";
    }
    std::string result = remove_extension(base_name(file_name));
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    ss << result;
    return ss.str();
  }

 public:
  /**
   * @brief Get a boolean from the environment variable based on a file name
   *        i.e. /usr/local/lib/my_file.hh -> VERBOSE_MY_FILE environment
   * variable
   *
   * @param file_name
   * @param default_value
   * @return true
   * @return false
   */
  static bool get_file_env_bool(const std::string& file_name,
                                bool default_value = false,
                                const std::string& prefix = "") {
    return get_env_bool(file_to_macro_name(file_name, prefix), default_value);
  }
};

/**
 * @brief Return a boolean value based upon whether the source file should
 * produce verbose output. Usage example: bool verbose = VERBOSE_FILE(false);
 *
 *        Then within the file's code, check the 'verbose' variable as needed.
 *        To set a file as verbose, set the environment variable formed from
 *        the file name:
 *
 *        i.e. /usr/local/lib/my_file.hh -> VERBOSE_MY_FILE
 *
 *        So, in this case:  'export VERBOSE_MY_FILE=1' causes verbose output
 */
#define VERBOSE_FILE(__dflt) \
  EnvFileMacro::get_file_env_bool(__FILE__, __dflt, "VERBOSE")

class THelper {
 public:
  static inline at::Tensor get_tensor(const XLATensor& t) {
    if (t.data()->tensor_data.has_value()) {
      return t.data()->tensor_data.value();
    }
    return bridge::AtenFromXlaTensor(t);
  }

  static inline bool has_tensor(const XLATensor& t) {
    const at::Tensor tensor = get_tensor(t);
    return tensor.defined();
  }

  static inline bool is_weight(const at::Tensor& tensor) {
    return tensor.requires_grad() && tensor.is_leaf();
  }

  static inline bool is_weight(const XLATensor& t) {
    at::Tensor tensor = get_tensor(t);
    if (!tensor.defined()) {
      return false;
    }
    return is_weight(tensor);
  }

  template <typename CB>
  struct _Not {
    CB cb_;
    inline bool operator()(const XLATensor& t) const { return !cb_(t); };
  };

  template <typename CB>
  static inline _Not<CB> Not(const CB cb) {
    return _Not<CB>{cb};
  }
};

/**
 * This is mostly an exploratory class and will go away eventually
 */
class XLASentinel {
 public:
  typedef xla::hash_t hash_t;

  // Configuration
  static void SetDeviceProxyAddress(const std::string& device,
                                    const std::string& proxy_address);
  static void SetOutputs(const std::vector<at::Tensor>& output_tensors,
                         bool append);
  static bool IsInitialized();

  // Notification handlers
  static void NotifyCompile(
      std::vector<xla::ComputationClient::CompileInstance>& instances,
      hash_t hash, pid_t tid);
  static void NotifyExecute(
      const xla::ComputationClient::Computation& computation,
      const std::string& device, hash_t hash, pid_t tid);
  static std::vector<xla::ComputationClient::DataPtr>
  NotifyScheduleSyncTensorsGraph(
      std::vector<XLATensor>* tensors,
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      XLATensor::SyncTensorCollection* coll,
      std::shared_ptr<xla::ComputationClient::Computation>& computation);

  // Interception and external mapping
  static void PostmarkHash(HashingState& state, std::vector<XLATensor>* tensors,
                           XLATensor::SyncTensorCollection& coll);
  static bool OnHashingComplete(HashingState& state,
                                std::vector<XLATensor>* tensors,
                                XLATensor::SyncTensorCollection& coll);

  static bool PreProcessHlo(xla::XlaBuilder* builder,
                            const XLATensor::SyncTensorCollection& coll);

  static bool IsSpecialLoweringEnabled();

  static std::map<std::string, std::string> GetStats(bool reset_stats);

  static bool IsAllowedOutput(const XLATensor& tensor,
                              XLATensor::SyncTensorCollection& coll,
                              bool* is_restricting);
  static bool IsForcingCustomLowering();
  static void SetCompileOnly(bool compile_only);
  static bool GetCompileOnly(XLATensor::SyncTensorCollection& coll);
  static bool WasMarkStepOnProxy();

 private:
  static void NotifyStepMarkerBegin(const std::string& device_str,
                                    const std::vector<std::string>& devices);
  static void NotifyStepMarkerEnd();

  static bool IsTrainingThread(pid_t tid);
  static bool IsQualifyingStep(pid_t tid /*, bool or_higher = false*/);
  static void SetAllDevices(const std::vector<std::string>& all_devices);
  static bool HasWseDevices();
  static bool PruneTensors(std::vector<XLATensor>* tensors,
                           XLATensor::SyncTensorCollection& coll);

  //
  // Data
  //
  static std::vector<std::string> wse_devices_;
  friend struct MarkStepScope;
};

inline pid_t gettid() { return syscall(__NR_gettid); }

using ColorScope = xla::ColorScope;
using EnterLeave = xla::EnterLeave;
using Color = xla::Color;

}  // namespace torch_xla
