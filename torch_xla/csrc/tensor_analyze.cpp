#include "torch_xla/csrc/tensor_analyze.h"

#include <Python.h>
#include <pybind11/pybind11.h>

#include <mutex>
#include <stack>
#include <string>

#include "absl/container/node_hash_map.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/xla_computation_proxy.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/tensor.h"

#if 1
#define __ASSERT_FUNCTION __extension__ __PRETTY_FUNCTION__

void _my_assert_handler() { raise(SIGTRAP); }

#undef assert
#define assert(expr) \
  (static_cast<bool>(expr) \
       ? void(0)           \
       : _my_assert_handler() /*__assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION)*/)
#endif

#ifndef ABSL_HAVE_INTRINSIC_INT128
#error oops
#endif

/**
 * Most of this can eventually move to monolith
 */
namespace torch_xla {

namespace {
const bool verbose = VERBOSE_FILE(false);
const bool verbose_tensor_sync = verbose;
const bool verbose_output_control = verbose || false;
const bool verbose_mp = false;
const bool verbose_hash = false;
const bool verbose_notify_compile = false;
const bool verbose_notify_execute = false;
const bool verbose_remove_tensors = false;
const bool verbose_non_fabric = false;
const bool verbose_mark_step = false;
const bool disable_proxy =
    xla::sys_util::GetEnvBool("WSE_DISABLE_PROXY", false);
const bool prune_tensors_if_outputs_set = true;

constexpr std::size_t DEFAULT_CLEAN_STEPS_UNTIL_PROXY = 1;
}  // namespace

std::string mp() {
  std::stringstream ss;
  if (verbose_mp) {
    ss << "[pid=" << getpid() << "] ";
  }
  return ss.str();
}

std::string to_string(const xla::Shape& shape) {
  std::stringstream ss;
  ss << "[";
  for (std::size_t i = 0; i < shape.dimensions_size(); ++i) {
    if (i) ss << ", ";
    ss << shape.dimensions(i);
  }
  ss << "]";
  return ss.str();
}

std::string to_string(const xla::ProgramShape& ps) {
  std::stringstream ss;
  ss << "(";
  for (std::size_t i = 0; i < ps.parameters_size(); ++i) {
    if (i) ss << ", ";
    ss << to_string(ps.parameters(i));
  }
  ss << ") -> (";
  const xla::Shape& result_shape = ps.result();
  if (result_shape.element_type() != xla::PrimitiveType::TUPLE) {
    ss << to_string(result_shape);
  } else {
    for (std::size_t i = 0; i < result_shape.tuple_shapes_size(); ++i) {
      if (i) ss << ", ";
      const xla::Shape& shape = result_shape.tuple_shapes(i);
      ss << to_string(shape);
    }
  }
  ss << ")";
  return ss.str();
}

void XLATensor::print_tensor(const std::string& label,
                             const XLATensor& tensor) {
  print_tensor(label, tensor.data(), false, tensor.GetViewAliasId());
}

void XLATensor::print_tensor_ex(const std::string& label,
                                const XLATensor& tensor) {
  print_tensor_ex(label, tensor.data(), false, tensor.GetViewAliasId());
}

void XLATensor::print_tensor_ex(const std::string& label,
                                const XLATensor::Data* data, bool assert,
                                ptrdiff_t alias_id) {
  if (data->ir_value) {
    std::cout << label << " (id=" << data->unique_id << ") "
              << " IR tensor of shape: " << data->ir_value.shape().ToString()
              << std::endl
              << std::flush;
    if (assert) {
      assert(!data->xla_data);
      assert(!data->view);
    }
  } else if (data->xla_data) {
    // coming from _xla_tensors_from_aten in at least one case
    std::stringstream ss;
    ss << label << " (id=" << data->unique_id << ") "
       << " tensor, xla_data handle=";
    if (data->xla_data->HasValue()) {
      ss << data->xla_data->GetOpaqueHandle();
    } else {
      ss << "null";
    }
    ss << " on device: " << data->xla_data->device()
       << " of shape: " << data->xla_data->shape().ToString();
    std::cout << ss.str() << std::endl << std::flush;
    if (assert) {
      assert(!data->ir_value);
      assert(!data->view);
    }
  } else if (data->view) {
    std::cout << label << " (id=" << data->unique_id << ") "
              << " tensor with view of shape: "
              << data->view->shape().ToString() << std::endl
              << std::flush;
    if (assert) {
      assert(!data->ir_value);
      assert(!data->xla_data);
    }
  } else {
    std::cout << label << " (id=" << data->unique_id << ") "
              << " strange tensor of unknown size" << std::endl
              << std::flush;
  }
}

void XLATensor::print_tensor(const std::string& label,
                             const XLATensor::Data* data, bool assert,
                             ptrdiff_t alias_id) {
  ColorScope color_scope(Color::FG_CYAN, true);
  print_tensor_ex(label, data, assert, alias_id);
}

void XLATensor::print_all_tensors(const std::string& label,
                                  const std::vector<XLATensor>& tensors) {
  ColorScope cs(Color::FG_BLUE, false);
  std::cout << "------------------" << ENDL;
  std::cout << "tensors=[" << ENDL;
  for (size_t i = 0, n = tensors.size(); i < n; ++i) {
    const XLATensor& t = tensors[i];
    std::ptrdiff_t alias_id = t.GetViewAliasId();
    const bool bright = !!alias_id;
    ColorScope cs(alias_id ? Color::FG_CYAN : Color::FG_BLUE, bright);
    std::string s = "\t";
    s += label;
    XLATensor::print_tensor_ex(s, t.data(), false, t.GetViewAliasId());
    if (alias_id) {
      std::shared_ptr<Alias> alias = t.data()->view->alias();
    }
  }
  std::cout << "]" << ENDL;
  std::cout << "------------------" << ENDL;
}

struct SPythonState {
  std::stack<EPythonState> states;

  void push(EPythonState new_state, pid_t __tid = 0) {
    std::lock_guard<std::mutex> lk(python_state_map_mtx);
    const pid_t tid = __tid ? __tid : gettid();
    python_state_map[tid].states.push(new_state);
  }
  void pop() {
    std::lock_guard<std::mutex> lk(python_state_map_mtx);
    const pid_t tid = gettid();
    auto iter = python_state_map.find(tid);
    assert(iter != python_state_map.end());
    assert(!iter->second.states.empty());
    iter->second.states.pop();
    if (iter->second.states.empty()) {
      python_state_map.erase(iter);
    }
  }
  EPythonState get(pid_t tid) {
    std::lock_guard<std::mutex> lk(python_state_map_mtx);
    auto iter = python_state_map.find(tid);
    if (iter != python_state_map.end()) {
      assert(!iter->second.states.empty());
      return iter->second.states.top();
    }
    return EPS_INVALID;
  }

 private:
  std::mutex python_state_map_mtx;
  std::map<pid_t, SPythonState> python_state_map;
};
SPythonState python_state;

EPythonState GetPythonState(pid_t tid) { return python_state.get(tid); }

static void _PushPythonState(EPythonState state, pid_t __tid = 0) {
  python_state.push(state, __tid);
}

void PushPythonState(EPythonState state) { _PushPythonState(state); }

void PopPythonState() { python_state.pop(); }

namespace {

constexpr std::size_t INVALID_COUNT = std::numeric_limits<std::size_t>::max();

using Lock = std::lock_guard<std::recursive_mutex>;

struct CompileInfo {
  CompileInfo() : hash_(0U) {}
  std::atomic<std::size_t> sync_count_since_hash_change_{INVALID_COUNT};
  std::atomic<std::size_t> mark_step_count_since_last_reset_{INVALID_COUNT};
  std::unordered_set<size_t> output_ids_;

  void set_hash(XLASentinel::hash_t hash) {
    if (hash != hash_.load()) {
      hash_ = std::move(hash);
    }
  }
  XLASentinel::hash_t hash() const { return hash_; }

 private:
  std::atomic<XLASentinel::hash_t> hash_;
};

/**
 * @brief Valid proxy executable
 */
// class Executable {
//  // static constexpr uint64_t HASH_MARKER = 478925426;
//
// public:
//  explicit Executable(Int128 hash)
//      : hash_(hash)
//  // adjusted_hash_(xla::util::MHash(hash, HASH_MARKER))
//  {}
//  bool is_active() const { return active_; }
//  // xla::hash_t get_adjusted_hash() const { return adjusted_hash_; }
//  bool set_active(bool active) { active_ = active; }
//
// private:
//  const Int128 hash_;
//  // const xla::hash_t adjusted_hash_;
//  // not actually sure if we need this
//  // active anymore since transition is automatic downstream
//  bool active_{false};
//};
// using ExecutablePtr = std::shared_ptr<Executable>;

// using Int128 = std::pair<uint64_t, uint64_t>;
//
// inline Int128 H128(const xla::hash_t& h) {
//  const __int128 h1 = h.operator unsigned __int128();
//  Int128 hh = std::make_pair<uint64_t, uint64_t>(
//      (h1.operator unsigned __int128() >> 64) & 0xFFFFFFFFFFFFFFFF,
//      h1.operator unsigned __int128() & 0xFFFFFFFFFFFFFFFF
//  );
//  return std::move(hh);
//}

typedef __int128 Int128;

inline Int128 H128(const xla::hash_t& h) {
  return h.operator unsigned __int128();
}

/**
 * @brief Class to keep track of known-good executables
 */
class ExecutableCache {
  void add_executable(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    stats();
    const Int128 hh = H128(hash);
    assert(executables_.find(hh) == executables_.end());
    stats();
    auto exec = executables_.insert(hh);
    stats();
    // adjusted_hash_map_.insert({exec->get_adjusted_hash(), exec});
    // return exec;
  }

 public:
  //  bool has_executable(const XLASentinel::hash_t& hash) {
  //    Lock lk(mtx_);
  //    const Int128 hh = H128(hash);
  //    auto found = executables_.find(hh);
  //    if (found != executables_.end()) {
  //      return true;
  //    }
  //    return false;
  //  }
  bool get_executable_by_adjusted_hash(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    auto found = adjusted_hash_map_.find(hh);
    if (found != adjusted_hash_map_.end()) {
      return true;
    }
    return false;
  }
  bool has_executable(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    return executables_.count(hh) != 0;
  }
  bool has_executable_by_adjusted_hash(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    return adjusted_hash_map_.count(hh) != 0;
  }
  bool is_active_executable(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    const Int128 hh = H128(hash);
    if (has_executable(hh)) {
      return true;
    }
    //    if (exec) {
    //      return exec->is_active();
    //    }
    return false;
  }
  void activate_hash(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    stats();
    const Int128 hh = H128(hash);
    auto found = executables_.find(hh);
    if (found == executables_.end()) {
      add_executable(hh);
      //      assert(exec);
      //      exec->set_active(true);
      //      stats();
      //      XLA_COUNTER("SentinelExecutableActivate", 1);
      //      return std::move(exec);
    } else {
      // Track that we're doing this in a deterministic way and not
      // overlapping logic
      //      const bool is_active = found->second->is_active();
      //      if (!is_active) {
      //        //assert(!is_active);
      //        found->second->set_active(true);
      //        XLA_COUNTER("SentinelExecutableActivate", 1);
      //      }
      //      return found->second;
    }
  }
  void stats() {
    //    std::cout << "this->executables_.size()=" << this->executables_.size()
    //              << ENDL;
  }
  void set_adjusted_hash(const xla::hash_t& h1, const xla::hash_t& h2) {
    Lock lk(mtx_);
    assert(h1 != h2);
    const Int128 hh1 = H128(h1);
    const Int128 hh2 = H128(h2);
    auto found = executables_.find(hh1);
    if (found != executables_.end()) {
      // Should only set this once
      auto found_adjusted = adjusted_hash_map_.find(hh1);
      if (found_adjusted != adjusted_hash_map_.end()) {
        // assert(found_adjusted->second == found->second);
        assert(found_adjusted->second == hh1);
      } else {
        adjusted_hash_map_[hh2] = hh1;
        // adjusted_hash_map_.insert(hh2);
      }
    } else {
      assert(false);  // does this ever happen?
    }
  }

  //  void deactivate_hash(const XLASentinel::hash_t&
  //                           hash) {  // currently we don't need to track the
  //                                    // "active" one, so this might be
  //                                    pointless
  //    Lock lk(mtx_);
  //    const Int128 hh = H128(hash);
  //    auto found = executables_.find(hh);
  //    if (found != executables_.end()) {
  //      // should we assert that its active?  probably not
  //      // since deactivations acan come pretty randomly from any direction
  //      if (found->second->is_active()) {
  //        found->second->set_active(false);
  //        XLA_COUNTER("SentinelExecutableDeactivate", 1);
  //      }
  //    }
  //  }

 private:
  mutable std::recursive_mutex mtx_;
  // std::unordered_set<Int128> executables_;  // needs to be locked?
  std::set<Int128> executables_;  // needs to be locked?
  // absl::node_hash_map<Int128, ExecutablePtr> executables_;  // needs to be
  // locked? absl::node_hash_map<XLASentinel::hash_t, ExecutablePtr>
  // adjusted_hash_map_; absl::node_hash_map<Int128, ExecutablePtr>
  // adjusted_hash_map_;
  // std::unordered_map<Int128, Int128> adjusted_hash_map_;
  std::map<Int128, Int128> adjusted_hash_map_;
};

std::mutex compile_info_map_mtx_;
std::map<pid_t, std::shared_ptr<CompileInfo>> compile_info_map;

std::shared_ptr<CompileInfo> GetCompileInfo(pid_t tid) {
  std::lock_guard<std::mutex> lk(compile_info_map_mtx_);
  std::shared_ptr<CompileInfo> sp = compile_info_map[tid];
  if (!sp) {
    sp = compile_info_map[tid] = std::make_shared<CompileInfo>();
  }
  return std::move(sp);
}

std::shared_ptr<ExecutableCache> ex_cache = std::make_shared<ExecutableCache>();

int get_number_of_required_runs_since_reset() {
  if (disable_proxy) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::cerr << "**** WARNING **** PROXY IS DISABLED" << ENDL;
    }
    return std::numeric_limits<int>::max();
  }
  static bool trusted_model =
      xla::sys_util::GetEnvBool("XLA_TRUSTED_MODEL", false);
  if (trusted_model) {
    return 0;
  }
  static int rtc = xla::sys_util::GetEnvInt("XLA_CLEAN_STEPS_UNTIL_PROXY",
                                            DEFAULT_CLEAN_STEPS_UNTIL_PROXY);
  return rtc;
}

std::mutex init_devices_mutex;

bool thread_local is_in_mark_step = false;
bool thread_local is_clean_step = false;
bool thread_local mark_step_was_on_proxy = false;
bool thread_local prev_step_was_on_proxy = false;

std::size_t proxy_compile_count = 0;

}  // namespace

void XLASentinel::SetAllDevices(const std::vector<std::string>& all_devices) {
  wse_devices_.clear();
  wse_devices_.reserve(all_devices.size());
  for (const std::string& device_str : all_devices) {
    const Device device(device_str);
    if (device.hw_type == DeviceType::WSE) {
      wse_devices_.push_back(device_str);
    }
  }
}

bool XLASentinel::PreProcessHlo(xla::XlaBuilder* builder,
                                const XLATensor::SyncTensorCollection& coll) {
  // HEREX();
  if (HasWseDevices() && IsTrainingThread(coll.requesting_tid)) {
    if (verbose) {
      std::cout << "PreProcessHlo(): " << coll.hash << ENDL;
    }
    bool has_adjusted_exe =
        ex_cache->get_executable_by_adjusted_hash(coll.hash);
    if (has_adjusted_exe) {
      if (true /*exe->is_active()*/) {
        // Mark this for proxy
        xla::FrontendAttributes frontend_attributes;
        frontend_attributes.CopyFrom(builder->frontend_attributes());
        (*frontend_attributes.mutable_map())["PROXY_DEVICE"] =
            coll.device.ToString();
        builder->SetFrontendAttributes(frontend_attributes);

        // Sanity check that if we're pruning outputs,
        // the program shape has the same number of outputs as is expected
#ifndef NDEBUG
        std::shared_ptr<CompileInfo> compile_info =
            GetCompileInfo(coll.requesting_tid);
        const std::size_t output_ids_size = compile_info->output_ids_.empty();
        if (output_ids_size) {
          const xla::Shape& result_shape =
              builder->GetProgramShape().ValueOrDie().result();
          std::size_t output_count;
          if (result_shape.element_type() == xla::PrimitiveType::TUPLE) {
            output_count = result_shape.tuple_shapes_size();
          } else {
            output_count = 1;
          }
          if (output_count != output_ids_size) {
            throw sentinel_exception()
                << "We expected the pruned fabric program shape to have "
                << output_ids_size << " outputs, but it actually had "
                << output_count
                << " outputs.  This is probably a bug and should "
                << " be reported to the developers.";
          }
        }
#endif
        return true;
      } else {
        assert(false);  // just checking, will it ever not be?
      }
    }
  }
  return false;
}

void XLASentinel::SetDeviceProxyAddress(const std::string& device,
                                        const std::string& proxy_address) {
  // DO NOT CREATE THE COMPUTATION CLIENT!!
  xla::XlaComputationProxy::SetDeviceProxyAddress(device, proxy_address);
}

bool XLASentinel::HasWseDevices() {
  static bool got_devices = false;
  if (!got_devices) {
    std::lock_guard<std::mutex> lk(init_devices_mutex);
    if (!got_devices) {
      SetAllDevices(xla::XrtComputationClient::Get()->GetAllDevices());
      got_devices = true;
    }
  }
  return !wse_devices_.empty();
}

bool XLASentinel::PruneTensors(std::vector<XLATensor>* tensors,
                               XLATensor::SyncTensorCollection& coll) {
  if (!tensors || tensors->empty()) {
    return false;
  }
  std::vector<size_t> adjusted_indices;
  adjusted_indices.reserve(coll.indices.size());
  for (std::size_t i = 0, n = coll.indices.size(); i < n; ++i) {
    const std::size_t tensor_index = coll.indices[i];
    const XLATensor& tensor = (*tensors)[tensor_index];
    bool is_restricting;
    if (IsAllowedOutput(tensor, coll, &is_restricting)) {
      adjusted_indices.push_back(coll.indices[i]);
      if (is_restricting && verbose_output_control) {
        ColorScope clr(Color::FG_DEFAULT);
        std::stringstream ss;
        ss << "Allowing output";
        if (tensor.data()->xla_data && tensor.data()->xla_data->HasValue()) {
          ss << " HANDLE = " << tensor.data()->xla_data->GetOpaqueHandle();
        }
        XLATensor::print_tensor(ss.str(), tensor);
      }
    } else {
      if (is_restricting &&
          (verbose || verbose_output_control || verbose_remove_tensors)) {
        std::stringstream ss;
        ss << "Removing output";
        if (tensor.data()->xla_data && tensor.data()->xla_data->HasValue()) {
          ss << " HANDLE = " << tensor.data()->xla_data->GetOpaqueHandle();
        }
        XLATensor::print_tensor(ss.str(), tensor);
      }
    }
  }

  if (adjusted_indices.empty() ||
      adjusted_indices.size() != coll.indices.size()) {
    coll.indices = std::move(adjusted_indices);
    //    if (verbose) {
    //      std::cout << "PostmarkHash(): coll.hash: " << coll.hash << " -> "
    //                << exe->get_adjusted_hash() << ENDL;
    //    }
    //    coll.hash = exe->get_adjusted_hash();
    return true;
  } else {
    // Nothing left, so can't do this on proxy
    //    ex_cache->deactivate_hash(coll.hash);
    //    std::cout
    //        << "No effective allowed outputs, so reverting to standard device"
    //        << ENDL;
    return false;
  }
}

bool XLASentinel::IsTrainingThread(pid_t tid) {
  return GetPythonState(tid) == EPS_IN_TRAIN_LOOP;
}

void XLASentinel::PostmarkHash(HashingState& state,
                               std::vector<XLATensor>* tensors,
                               XLATensor::SyncTensorCollection& coll) {
  //
  // TODO: Upon same "sync tensors" hash run last time on fabric,
  //       can we simply do the prune and continue without having to
  //       to a post-order again?  I think that we can.  Can make this
  //       a setting in case there's suspicion of it causing issues later.
  //
  if (verbose) {
    std::cout << "PostMarkHash(): " << coll.hash << ENDL;
  }
#if 0
  const xla::hash_t original_hash = coll.hash;
  {
    std::cout << "ENTER XLASentinel::PostmarkHash(): " << coll.hash << ENDL;
  }
  if (!is_clean_step) {
    return original_hash;
  }
  if (coll.indices.empty() || !HasWseDevices()) {
    return original_hash;
  }
  ExecutablePtr exe = ex_cache->get_executable(coll.hash);
  if (exe) {
    if (!exe->is_active()) {
      ex_cache->activate_hash(coll.hash);
    }
  //} else if (is_qualifying_step /* ASSUMPTION: compile in mark step IsQualifyingStep(coll.requesting_tid)*/) {
  } else {
//      std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(coll->requesting_tid);
//      const std::size_t sync_count_since_hash_change =
//          compile_info->sync_count_since_hash_change_.load();
    //if (IsQualifyingStep(coll.requesting_tid)) {
    if (is_qualifying_step) {
      assert(is_in_mark_step);
      // create and activate
      exe = ex_cache->activate_hash(coll.hash);
      assert(exe);
    }
  }

  if (exe && exe->is_active()) {
    XLA_COUNTER("SentinelPostMarkHash", 1);

#if 1
    // how do we avoid running post-order twice?
    //PruneTensors();
#else

    std::vector<size_t> adjusted_indices;
    adjusted_indices.reserve(coll.indices.size());
    for (std::size_t i = 0, n = coll.indices.size(); i < n; ++i) {
      const std::size_t tensor_index = coll.indices[i];
      const XLATensor& tensor = (*tensors)[tensor_index];
      if (IsAllowedOutput(tensor, coll)) {
        adjusted_indices.push_back(coll.indices[i]);
        if (verbose_output_control) {
            ColorScope clr(Color::FG_DEFAULT);
            std::stringstream ss;
            ss << "Allowing output";
            if (tensor.data()->xla_data && tensor.data()->xla_data->HasValue()) {
                ss << " HANDLE = "
                   << tensor.data()->xla_data->GetOpaqueHandle();
            }
            XLATensor::print_tensor(ss.str(), tensor);
        }
      } else {
        if (verbose) {
          std::stringstream ss;
          ss << "Removing output";
          if (tensor.data()->xla_data && tensor.data()->xla_data->HasValue()) {
            ss << " HANDLE = "
               << tensor.data()->xla_data->GetOpaqueHandle();
          }
          XLATensor::print_tensor(ss.str(), tensor);
        }
      }
    }

    if (!adjusted_indices.empty()) {
      coll.indices = std::move(adjusted_indices);
      if (verbose) {
        std::cout << "PostmarkHash(): coll.hash: " << coll.hash << " -> "
                  << exe->get_adjusted_hash() << ENDL;
      }
      coll.hash = exe->get_adjusted_hash();
      return coll.hash;
    } else {
      // Nothing left, so can't do this on proxy
      ex_cache->deactivate_hash(coll.hash);
      std::cout
          << "No effective allowed outputs, so reverting to standard device"
          << ENDL;
    }
#endif
  }
#endif
}

/**
 * @brief Analyze the hashing situation and see if we can run this on the proxy
 * @param state
 * @param tensors
 * @param coll
 * @return
 * @note Adjusted hashing and executable's "adjusted hash" is like this:
 *
 *       For an un-pruned executable:
 *        "initial coll.hash after post-order" -> rehash( "initial coll.hash
 * after post-order" ) For a pruned executable: "initial coll.hash after
 * post-order" -> rehash( "coll.hash calculated after being sent back for prune"
 * )
 *
 *       An advantage of this is that if we are ever asked for the graph where
 * we donn't have to prune anything, it should still resolve to the same
 * executable's adjusted hash in PreProcessHlo
 *
 *       TODO: Optimize not doing two passes when we are in the same detected
 * state on first-pass entry One obvious approach is to set a flag when
 * post-order doesn't change after a prune (which so far has been the case),
 * although would like to make that optional so that the harder state can keep
 * being tested for now until there's a way to turn it on aand off and to have
 * tests for it to assure it didn't break.
 */
bool XLASentinel::OnHashingComplete(HashingState& state,
                                    std::vector<XLATensor>* tensors,
                                    XLATensor::SyncTensorCollection& coll) {
  // This only updates something if this is one we have an executable for

  static const absl::int128 PROXY_HASHING_VALUE =
      absl::MakeInt128(0xb8bc2f8ad48c431c, 0x872132d8a172a6d8);

  const std::size_t pass = state.pass_++;

  if (!is_in_mark_step /*|| !is_clean_step*/) {
    std::shared_ptr<CompileInfo> compile_info =
        GetCompileInfo(coll.requesting_tid);
    // compile_info->mark_step_count_since_last_reset_ = INVALID_COUNT;
    compile_info->mark_step_count_since_last_reset_ = 0;
    compile_info->sync_count_since_hash_change_ = INVALID_COUNT;
    return false;
  }

  if (!pass) {
    // First, see if we know about this hash already and it passed as
    // a stable-state executable.  That doesn't mean its impossible to
    // immediately do something to cause it to be switched, but we do
    // know this to have the ability to be stable
    if (ex_cache->has_executable(coll.hash)) {
      // just for optimizations ont eh second pass. Final state
      // should be deterministic with or without this flag
      state.known_executable_ = true;

      auto compile_info = GetCompileInfo(coll.requesting_tid);
      assert(compile_info->mark_step_count_since_last_reset_ !=
             INVALID_COUNT);  // maybe this is ok, we just switched to a known
                              // graph?
      ++compile_info->mark_step_count_since_last_reset_;  // is there any point
                                                          // to increment this?
      ex_cache->activate_hash(
          coll.hash);            // not sure if 'active' has a meaning anymore
      state.fabric_run_ = true;  // <-- can this state just be a thread var?
      mark_step_was_on_proxy = true;
      if (PruneTensors(tensors, coll)) {
        state.pre_prune_hash_ = coll.hash;
        coll.hash = state.start_hash_;
        return true;  // need to recalculate postorder with new inputs/outputs
      }
      return false;  // Nothing removed, so keep going (on fabric)
    } else if (prune_tensors_if_outputs_set) {
      if (is_in_mark_step) {
        if (prev_step_was_on_proxy) {
          // This is a subsequent step to a proxy step, and
          // So if we have *intentionally* set the outputs since then,
          // prune the tensors as if it were a regular mark step.
          // This is because if there is a mark step later,
          // it may try to *only* pull in those pruned tensors,
          // so if that's the case, prune them again.
          // If the resultant list of tensors to update is empty, then
          // a sync can be skipped (as it would have been had we not pruned the
          // tensors in the first place)
          if (!coll.indices.empty()) {
            std::vector<size_t> save_indices = coll.indices;
            if (PruneTensors(tensors, coll)) {
              if (coll.indices.empty()) {
                state.pre_prune_hash_ = coll.hash;
                coll.hash = state.start_hash_;
                --state.pass_;
                std::cout << "Pruned outputs on unknown executable.";
                mark_step_was_on_proxy = true;
                return true;  // need to recalculate postorder with new
                              // inputs/outputs
              } else {
                // We didn't prune everything, so allow the computation to
                // go forward
                //                            state.pre_prune_hash_ = coll.hash;
                //                            coll.hash = state.start_hash_;
                coll.indices = std::move(save_indices);
                // return false;
              }
            }
          }
        }
      }
    }

    // Note: For trusted, we don't need to analyze anything
    std::shared_ptr<CompileInfo> compile_info =
        GetCompileInfo(coll.requesting_tid);
    if (coll.hash != compile_info->hash()) {
      if (verbose || verbose_hash) {
        ColorScope clr(Color::FG_GREEN);
        std::cout << mp() << "NEW HASH: " << compile_info->hash() << " -> "
                  << coll.hash << ENDL;
      }

      compile_info->set_hash(coll.hash);
      compile_info->mark_step_count_since_last_reset_ = 0;

      // If this isn't a zero-hash (i.e. first mark step call before loop),
      // then see if it's trusted
      if (!compile_info->hash() || !IsQualifyingStep(coll.requesting_tid)) {
        return false;
      }
    }

    if (verbose || verbose_hash) {
      ColorScope clr(Color::FG_GREEN);
      std::cout << mp()
                << "SAME HASH AS LAST TIME OR TRUSTED: " << compile_info->hash()
                << ENDL;
    }
    assert(compile_info->mark_step_count_since_last_reset_ != INVALID_COUNT);
    ++compile_info->mark_step_count_since_last_reset_;
    if (IsQualifyingStep(coll.requesting_tid)) {
      if (coll.device.ordinal == 0) {
        if (verbose) {
          ColorScope clr(Color::FG_GREEN);
          std::cout << mp() << "**** QUALIFYING: " << coll.hash << ENDL;
        } else {
          std::cout << mp() << "**** Stable graph found" << ENDL;
        }
      }
      ex_cache->activate_hash(coll.hash);
      if (PruneTensors(tensors, coll)) {
        state.fabric_run_ = true;
        assert(!state.pre_prune_hash_);
        state.pre_prune_hash_ = coll.hash;
        coll.hash = state.start_hash_;
        mark_step_was_on_proxy = true;
        return true;  // need to recalculate postorder with new inputs
      }
      // Do we need to hash this differently for *our* executable
      // in case we didn't prune anything?
      const hash_t proxy_hash =
          xla::util::HashCombine(coll.hash, PROXY_HASHING_VALUE);
#ifndef NDEBUG
      // This shouldn't be the adjusted hash already or something went wrong
      assert(!ex_cache->get_executable_by_adjusted_hash(coll.hash));
#endif
      ex_cache->set_adjusted_hash(coll.hash, proxy_hash);
      coll.hash = proxy_hash;
      state.fabric_run_ = true;
      mark_step_was_on_proxy = true;
      return false;  // Nothing removed, so keep going (on fabric)
    }
  } else {
    //
    // We sent them back to recalculate
    //
    // It's possible that with different outputs, the inputs didn't change,
    // in which case, 'coll.hash' is the same as 'state.pre_prune_hash_'
    //

    assert(state.pre_prune_hash_);  // this should have been set, the hash
                                    // before the prune

    assert(state.fabric_run_);  // this should have been set the first pass,
                                // or else we got here by accident
    assert(mark_step_was_on_proxy);

    const hash_t proxy_hash =
        xla::util::HashCombine(coll.hash, PROXY_HASHING_VALUE);

#ifndef NDEBUG
    if (verbose) {
      std::cout << "Adjusted hash for proxy from " << coll.hash << " to "
                << proxy_hash << ", which had a pre-prune hash of "
                << state.pre_prune_hash_ << ENDL;
    }
    // This shouldn't be the adjusted hash already or something went wrong
    // Addendum: on second pass it should be here, right?
    assert(!ex_cache->get_executable_by_adjusted_hash(coll.hash));
#endif
    // ex_cache->modify_adjusted_hash(coll.hash, proxy_hash);
    ex_cache->set_adjusted_hash(state.pre_prune_hash_, proxy_hash);
    coll.hash = proxy_hash;
    mark_step_was_on_proxy = true;
    return false;
  }
  return false;
}

bool XLASentinel::WasMarkStepOnProxy() {
  // assert(!is_in_mark_step);
  return mark_step_was_on_proxy;
}

void XLASentinel::NotifyCompile(
    std::vector<xla::ComputationClient::CompileInstance>& instances,
    hash_t hash, pid_t tid) {
  if (verbose_notify_compile) {
    HEREX();
  }
  if (!HasWseDevices()) return;
  XLA_COUNTER("SentinelNotifyCompile", 1);
  if (is_in_mark_step) {
    XLA_COUNTER("SentinelStepMarkerCompile", 1);
  }

  if (IsTrainingThread(tid)) {
    XLA_COUNTER("SentinelMasterThreadCompile", 1);
    if (!ex_cache->has_executable_by_adjusted_hash(hash)) {
      XLA_COUNTER("SentinelNonProxyCompile", 1);
      assert(instances.size() ==
             1);  // always just one? maybe in distrib its one each.
      if (verbose || verbose_non_fabric) {
        ColorScope clr(std::cout, {Color::FG_BLUE}, false);
        std::cout
            << "** NON-FABRIC COMPILE: "
            << to_string(
                   instances[0].computation.GetProgramShape().ValueOrDie())
            << ENDL;
      }
    } else {
      XLA_COUNTER("SentinelProxyCompile", 1);
      ++proxy_compile_count;
    }
  }
}

void XLASentinel::NotifyExecute(
    const xla::ComputationClient::Computation& computation,
    const std::string& device, hash_t hash, pid_t tid) {
  if (verbose) {
    HEREX();
  }
  static std::size_t proxy_compile_only_count =
      xla::sys_util::GetEnvInt("PROXY_COMPILE_ONLY_COUNT", 0);
  if (proxy_compile_only_count &&
      proxy_compile_count >= proxy_compile_only_count) {
    std::cout << "Compile-only mode, exiting.";
    _exit(0);
  }
  if (verbose_notify_execute) {
    ColorScope clr(std::cout, {Color::FG_CYAN}, false);
    std::cout << "** EXECUTE ON TRAIN THREAD HASH: " << hash << ENDL;
  }
  XLA_COUNTER("SentinelExecute", 1);
  if (!HasWseDevices()) return;
  if (IsTrainingThread(tid)) {
    XLA_COUNTER("SentinelMasterThreadExecute", 1);
    if (!ex_cache->has_executable_by_adjusted_hash(hash)) {
      XLA_COUNTER("SentinelNonProxyExecute", 1);
      if (verbose || verbose_non_fabric) {
        ColorScope clr(std::cout, {Color::FG_BLUE}, false);
        std::cout << "** NON-FABRIC EXECUTION: "
                  << to_string(computation.program_shape()) << ENDL;
      }
    } else {
      XLA_COUNTER("SentinelProxyExecute", 1);
    }
  }
}

std::vector<xla::ComputationClient::DataPtr>
XLASentinel::NotifyScheduleSyncTensorsGraph(
    std::vector<XLATensor>* xla_tensors,
    std::vector<xla::ComputationClient::DataPtr> tensors,
    XLATensor::SyncTensorCollection* coll,
    std::shared_ptr<xla::ComputationClient::Computation>& computation) {
  if (!is_in_mark_step) {
    // Anything outside of mark step is a reset
    if (verbose_mark_step) {
      ColorScope clr(std::cout, {Color::FG_RED}, false);
      std::cout << "Sync tensor request outside of MarkStep" << ENDL;
    }
    std::shared_ptr<CompileInfo> compile_info =
        GetCompileInfo(coll->requesting_tid);
    compile_info->sync_count_since_hash_change_ = 0;
    compile_info->set_hash(0);
    return std::move(tensors);
  }

  if (verbose) {
    ColorScope cs(Color::FG_CYAN);
    std::cout << "XLASentinel::NotifyScheduleSyncTensorsGraph(): " << coll->hash
              << ENDL;
  }

  assert(tensors.size() == coll->indices.size());
  if (verbose_tensor_sync) {
    std::size_t index = 0;
    std::for_each(
        tensors.begin(), tensors.end(), [coll, &index, xla_tensors](auto& t) {
          ColorScope cs(Color::FG_CYAN);
          std::cout << (index + 1) << " " << coll->hash
                    << ": SyncTensorsGraph tensor shape: " << t->shape();
          if (t->HasValue()) {
            std::cout << ", handle = " << t->GetOpaqueHandle();
          }
          std::cout << " ";
          XLATensor::print_tensor("", (*xla_tensors)[coll->indices[index++]]);
          // std::cout << ENDL;
        });
  }
  return std::move(tensors);
}

void XLASentinel::NotifyStepMarkerBegin(
    const std::string& device_str, const std::vector<std::string>& devices) {
  is_clean_step = false;
  is_in_mark_step = true;
  prev_step_was_on_proxy = mark_step_was_on_proxy;
  mark_step_was_on_proxy = false;
  XLA_COUNTER("SentinelStepMarker", 1);

  static bool registered_step_requirement = false;
  if (!registered_step_requirement) {
    XLA_VALUE_METRIC("SentinelRequiredStepsSinceReset",
                     get_number_of_required_runs_since_reset());
  }
  if (verbose) {
    ColorScope clr(Color::FG_YELLOW);
    std::cout << "*************** XLASentinel::NotifyStepMarker: device="
              << device_str << std::endl
              << std::flush;
  }
  const pid_t tid = gettid();
  if (!IsTrainingThread(tid)) {
    assert(GetPythonState(tid) == EPS_INVALID);
    // The assumption is that only the training thread can call
    // _XLAC._mark_step()
    _PushPythonState(EPS_IN_TRAIN_LOOP, tid);
  }
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t current_sync_count =
      compile_info->sync_count_since_hash_change_.load();
  if (current_sync_count == INVALID_COUNT) {
    // It's the first MarkStep, so just return (at top of training loop)
    compile_info->mark_step_count_since_last_reset_ = 0;
    compile_info->sync_count_since_hash_change_ = 0;
    if (verbose_mark_step) {
      std::cout << "Unclean or precursor step detected" << ENDL;
    }
    return;
  }
  is_clean_step = true;
  //  if (verbose_mark_step) {
  //    std::cout << "Clean step detected" << ENDL;
  //  }
  ++compile_info->mark_step_count_since_last_reset_;
  // const std::size_t step = ++compile_info->mark_step_count_since_last_reset_;
  // is_clean_step = compile_info->mark_step_count_since_last_reset_.load() > 0;
  if (is_clean_step) {
    XLA_COUNTER("SentinelCleanSteps", 1);
  }
  //  is_qualifying_step = IsQualifyingStep(tid);
  //  if (is_qualifying_step) {
  //    XLA_COUNTER("SentinelQualifyingSteps", 1);
  //  }
}

void XLASentinel::NotifyStepMarkerEnd() {
  assert(is_in_mark_step);

#if 1  // TURNED ON FOR HEADLESS TEST
  const pid_t tid = gettid();
  auto compile_info = GetCompileInfo(tid);
  compile_info->output_ids_.clear();
#endif

  is_in_mark_step = false;
  is_clean_step = false;
  // is_qualifying_step = false;
}

bool XLASentinel::IsSpecialLoweringEnabled() {
  static bool allow_special_compile =
      xla::sys_util::GetEnvBool("XLA_ALLOW_SPECIAL_LOWERING", false);
  return allow_special_compile /*&& is_qualifying_step*/;
}

bool XLASentinel::IsForcingCustomLowering() {
  static int val = xla::sys_util::GetEnvInt("XLA_ALLOW_SPECIAL_LOWERING", 0);
  return val == 2;
}

bool XLASentinel::IsQualifyingStep(pid_t tid /*, bool or_higher*/) {
  assert(is_in_mark_step);  // shouldn't we always be? then we can just call
                            // once in MarkStep
  if (!is_in_mark_step) {
    return false;
  }
  if (!HasWseDevices()) {
    return false;
  }
  if (tid == gettid() && GetPythonState(tid) == EPS_FORCE_PROXY) {
    return true;
  }
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t mark_step_count_since_reset =
      compile_info->mark_step_count_since_last_reset_.load();
  const int steps_required = get_number_of_required_runs_since_reset();
  if (steps_required == std::numeric_limits<int>::max()) {
    // This is the case of simply being turned off/disabled
    return false;
  }
  if (is_clean_step) {
    if (!steps_required) {
      return true;
    }
    if (steps_required < 0) {
      // force on step
      const auto force_on_step = static_cast<std::size_t>(-steps_required);
      if (force_on_step == mark_step_count_since_reset) {
        return true;
      }
    }
  }
  if (!mark_step_count_since_reset) {
    // The first step is superfluous since it's the top of the dataset iterator
    // loop, before any graph is built. This also takes care of disqualifying
    // due to spurious compiles within the train loop
    return false;
  }
  bool ready;
  if (!steps_required) {
    ready = true;  // always ready
  } else {
    // const bool ready = mark_step_count_since_reset - 1 == steps_required;
    // ready = mark_step_count_since_reset - 1 == steps_required;
    ready = mark_step_count_since_reset - 1 > steps_required;
//    if (mark_step_count_since_reset - 1 != steps_required) {
//      std::cout << "Over-qualifying step by "
//                << ((mark_step_count_since_reset - 1) - steps_required)
//                << " steps!" << std::endl
//                << std::flush;
//    }
  }
  if (ready) {
    assert(is_clean_step);  // validate it coincides with clean step logic

    if (!xla::XlaComputationProxy::IsEnabled()) {
      return false;
    }

    if (GetPythonState(tid) == EPS_PROXY_DISABLED) {
      ColorScope clr(std::cout, {Color::BG_MAGENTA, Color::FG_YELLOW});
      std::cout << "Qualifying step, but proxy is disabled" << ENDL;
      return false;
    }

    if (verbose) {
      ColorScope clr(std::cout, {Color::BG_BLUE, Color::FG_YELLOW});
      std::cout << "Run ready" << std::endl << std::flush;
    }
  }
  return ready;
}

bool XLASentinel::IsInitialized() {
  return xla::XlaComputationProxy::IsInitialized();
}

void XLASentinel::SetOutputs(const std::vector<at::Tensor>& output_tensors,
                             bool append) {
  if (!HasWseDevices()) {
    return;
  }
  const pid_t tid = gettid();
  assert(IsTrainingThread(tid));
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  if (!append) {
    compile_info->output_ids_.clear();
  }
  for (const at::Tensor& tensor : output_tensors) {
    XLATensor xla_tensor = bridge::GetXlaTensor(tensor);
    const bool added =
        compile_info->output_ids_.insert(xla_tensor.data()->unique_id).second;
    assert(added);
  }
}

bool XLASentinel::IsAllowedOutput(const XLATensor& tensor,
                                  XLATensor::SyncTensorCollection& coll,
                                  bool* is_restricting) {
  if (!is_clean_step || !is_in_mark_step) {
    return true;
  }

  assert(HasWseDevices());  // whyxla?
  assert(is_in_mark_step);  // gets cleared at end of step
  assert(is_clean_step);    // otherwise, why are you asking?
  std::shared_ptr<CompileInfo> compile_info =
      GetCompileInfo(coll.requesting_tid);
  if (compile_info->output_ids_.empty()) {
    if (is_restricting) {
      *is_restricting = false;
    }
    return true;
  }
  if (is_restricting) {
    *is_restricting = true;
  }
  return compile_info->output_ids_.find(tensor.data()->unique_id) !=
         compile_info->output_ids_.end();
}

}  // namespace torch_xla
