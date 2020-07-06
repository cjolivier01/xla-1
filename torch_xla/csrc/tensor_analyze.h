#pragma once

#include "tensorflow/compiler/xla/xla_client/types.h"

#include <sys/syscall.h>

#include <stack>
#include <map>
#include <mutex>
#include <ostream>
#include <sstream>

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/tensor.h"

#define ENDL std::endl << std::flush

#if __cplusplus >= 201703L  // C++17
#include <shared_mutex>
using rw_mutex = std::shared_mutex;

class read_lock {
public:
  explicit read_lock(rw_mutex& mtx): mtx_(&mtx)  {
    mtx_->lock_shared();
  }
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
};

// extern std::stack<int> python_state_;

EPythonState GetPythonState(pid_t tid);
void PushPythonState(EPythonState state);
void PopPythonState();

class MsgException : public std::exception
{
public:
  MsgException() :
      std::exception() {}
  MsgException( const char* msg ) :
      std::exception(),
      m_msg( msg ) {}
  MsgException( const std::string& msg ) :
      std::exception(),
      m_msg( msg ) {}
  MsgException( const MsgException& obj ) :
      std::exception( obj ),
      m_msg( obj.m_msg ) {}
  MsgException( MsgException &&rval ) :
      std::exception( std::move( rval ) ),
      m_msg( std::move( rval.m_msg ) ) {}
  MsgException &operator=( const MsgException& rhs ) {
    std::exception::operator=( rhs );
    m_msg = rhs.m_msg;
    return *this;
  }
  ~MsgException() noexcept {}

  static std::ostream& next( std::ostream& os ) {
    os.put('\n');
    return os;
  }
  virtual const char* what() const noexcept override {
    return m_msg.c_str();
  }

  template<typename T>
  MsgException& operator<<( const T &v ) {
    // TODO: is there a more elegant way to do this?
    std::ostringstream oss;
    oss << v;
    m_msg.append( oss.str() );
    return *this;
  }

protected:
private:
  std::string m_msg;
};

using sentinel_exception = MsgException;

#include <ostream>
namespace Color {
enum Code {
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

class Modifier {
  const Code code;
  const bool bright;
public:
  Modifier(Color::Code pCode, const bool is_bright=true)
      : code(pCode), bright(is_bright) {}

  friend std::ostream &
  operator<<(std::ostream &os, const Modifier &mod) {
    return os << "\033[" << (mod.bright ? "1;" : "0;") << mod.code << "m";
  }
};
}  // namespace Color

class ColorScope {
  std::ostream& os_;
public:
  inline ColorScope(std::ostream& os, Color::Code pCode, bool bright=true) : os_(os) {
    Color::Modifier mod(pCode, bright);
    os << mod;
  }
  inline ColorScope(std::ostream& os, std::vector<Color::Code> codes, bool bright=false) : os_(os) {
    for (auto c : codes) {
      Color::Modifier mod(c, bright);
      os << mod;
    }
  }
  ColorScope(Color::Code pCode, bool bright=true) : os_(std::cout) {
    os_ << Color::Modifier(pCode, bright) << std::flush;
  }
  ~ColorScope() {
    os_ << Color::Modifier(Color::FG_DEFAULT) << Color::Modifier(Color::BG_DEFAULT) << std::flush;
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
  static const Color::Code library_color_;
  const std::string label_;
  const pid_t thread_id_;
  const bool both_;
  const Color::Code use_color_;
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
  inline EnterLeave(const std::string& label, bool both=true, const Color::Code use_color = Color::BG_INVALID)
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
#define HERE() EnterLeave __here(EnterLeave::concat(nullptr, short_fn_name(__PRETTY_FUNCTION__).c_str(), ""))
#define HEREX() EnterLeave __here(EnterLeave::concat(nullptr, short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), false)
#define HEREXC(__color$) EnterLeave __here(EnterLeave::concat(nullptr, ::xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), false, __color$)
#define HEREXCT(__color$) EnterLeave __here(EnterLeave::concat(std::to_string(this).c_str(), ::xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), false, __color$)
#else
#define HERE() ((void)0)
#define HEREX() ((void)0)
#endif

struct MarkStepScope {
  MarkStepScope(const std::string& device_str,
                const std::vector<std::string>& devices);
  ~MarkStepScope();
  std::unique_ptr<EnterLeave> el_;
};


struct HashingState {
  explicit HashingState(const xla::hash_t& start_hash)
  : start_hash_(start_hash),
    //non_proxy_hash_{0},
    pass_(0) {};
  //~HashingState() = default;
  const xla::hash_t start_hash_;
  xla::hash_t pre_prune_hash_{0};
  //xla::hash_t non_proxy_hash_;
  std::size_t pass_;
  bool fabric_run_ = false;
  bool known_executable_ = false;  // optimization when we know this executable already exists
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
    const char *s = getenv(name.c_str());
    if (!s || !*s) return default_value;
    return is_true(s);
  }

  template<class T>
  static T base_name(T const & path, T const & delims = "/\\") {
    return path.substr(path.find_last_of(delims) + 1);
  }
  template<class T>
  static T remove_extension(T const & filename) {
    typename T::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
  }
  /**
   * @brief Make macro-like name for environment variables
   *
   * @param file
   * @return std::string
   */
  static std::string file_to_macro_name(const std::string& file_name, const std::string& prefix) {
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
   *        i.e. /usr/local/lib/my_file.hh -> VERBOSE_MY_FILE environment variable
   *
   * @param file_name
   * @param default_value
   * @return true
   * @return false
   */
  static bool get_file_env_bool(const std::string& file_name, bool default_value=false, const std::string& prefix="") {
    return get_env_bool(file_to_macro_name(file_name, prefix), default_value);
  }
};

/**
 * @brief Return a boolean value based upon whether the source file should produce
 *        verbose output.
 *        Usage example:
 *          bool verbose = VERBOSE_FILE(false);
 *
 *        Then within the file's code, check the 'verbose' variable as needed.
 *        To set a file as verbose, set the environment variable formed from
 *        the file name:
 *
 *        i.e. /usr/local/lib/my_file.hh -> VERBOSE_MY_FILE
 *
 *        So, in this case:  'export VERBOSE_MY_FILE=1' causes verbose output
 */
#define VERBOSE_FILE(__dflt) EnvFileMacro::get_file_env_bool(__FILE__, __dflt, "VERBOSE")

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

  // Notification handlers
  static void NotifyCompile(
      std::vector<xla::ComputationClient::CompileInstance>& instances,
      hash_t hash, pid_t tid);
  static void NotifyExecute(const xla::ComputationClient::Computation& computation,
      const std::string& device, hash_t hash, pid_t tid);
  static std::vector<xla::ComputationClient::DataPtr>
  NotifyScheduleSyncTensorsGraph(
      std::vector<XLATensor>* tensors,
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      XLATensor::SyncTensorCollection* coll,
      std::shared_ptr<xla::ComputationClient::Computation>& computation);

  // Interception and external mapping
  static void PostmarkHash(HashingState& state,
      std::vector<XLATensor>* tensors,
      XLATensor::SyncTensorCollection& coll);
  static bool OnHashingComplete(HashingState& state, std::vector<XLATensor>* tensors, XLATensor::SyncTensorCollection& coll);

  static bool PreProcessHlo(xla::XlaBuilder* builder,
                            const XLATensor::SyncTensorCollection& coll);

  static bool IsSpecialLowering();

  static std::map<std::string, std::string> GetStats(bool reset_stats);

 private:
  static void NotifyStepMarkerBegin(const std::string& device_str,
                                    const std::vector<std::string>& devices);
  static void NotifyStepMarkerEnd();

  static bool IsAllowedOutput(const XLATensor& tensor,
                              XLATensor::SyncTensorCollection& coll);
  static bool IsTrainingThread(pid_t tid);
  static bool IsQualifyingStep(pid_t tid /*, bool or_higher = false*/);
  static void SetAllDevices(const std::vector<std::string>& all_devices);
  static bool HasWseDevices();
  static bool PruneTensors(std::vector<XLATensor>* tensors, XLATensor::SyncTensorCollection& coll);

  //
  // Data
  //
  static std::vector<std::string> wse_devices_;
  friend struct MarkStepScope;
};

inline pid_t gettid() { return syscall(__NR_gettid); }

}  // namespace torch_xla
