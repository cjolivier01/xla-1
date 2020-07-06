#include "torch_xla/csrc/tensor_analyze.h"

#include <Python.h>
#include <pybind11/pybind11.h>

#include <mutex>
#include <stack>
#include <string>

#include "absl/container/node_hash_map.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/xla_computation_proxy.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/tensor.h"

#define __ASSERT_FUNCTION __extension__ __PRETTY_FUNCTION__

#undef assert
#define assert(expr)       \
  (static_cast<bool>(expr) \
       ? void(0)           \
       : __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))

/**
 * Most of this can eventually move to monolith
 */
namespace torch_xla {

bool verbose = VERBOSE_FILE(false);
bool verbose_tensor_sync = verbose;
bool verbose_output_control = verbose || false;
bool verbose_mp = true;

constexpr size_t DEFAULT_CLEAN_STEPS_UNTIL_PROXY = 1;

#ifdef WSE_DEBUG_LOGGING
__thread int EnterLeave::depth_ = 0;
const std::string EnterLeave::library_ = "ptxla";
const Color::Code EnterLeave::library_color_ = Color::FG_BLUE;
std::mutex EnterLeave::mtx_;
#endif

const char *prev_char(const char *original, const char *start, char c) {
  while(start > original && *start != c) {
    --start;
  }
  return start;
}

std::string mp() {
  std::stringstream ss;
  if (verbose_mp) {
    ss << "[pid=" << getpid() << "] ";
  }
  return ss.str();
}

std::string short_fn_name(const std::string &fn_name) {
  std::string result = fn_name;
  //std::cout << "fn_name=" << fn_name << ENDL;
  const char *start = fn_name.c_str();
  const char *s = strchr(start, '(');
  if (s && *s && s > start) {
    //std::cout << "s: " << s << ENDL;
    if (const char *s0 = prev_char(start, s - 1, ' ')) {
      //std::cout << "s0: " << s0 << ENDL;
      if (*s0 == ' ') {
        ++s0;
      }
      const size_t sz = s - s0 + 1;
      //std::cout << "sz: " << sz << std::endl << std::flush;
      result = std::string(s0, sz);
      result.append(")");
    }
  }
  return result;
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

MarkStepScope::MarkStepScope(
    const std::string& device_str,
    const std::vector<std::string>& devices) {
  if (verbose) {
    el_ = std::make_unique<EnterLeave>("*** MARK STEP", verbose, Color::FG_RED);
  }
  XLASentinel::NotifyStepMarkerBegin(device_str, devices);
}

MarkStepScope::~MarkStepScope() { XLASentinel::NotifyStepMarkerEnd(); }

namespace {

constexpr std::size_t INVALID_COUNT = std::numeric_limits<std::size_t>::max();

using Lock = std::lock_guard<std::recursive_mutex>;

struct CompileInfo {
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
  std::atomic<XLASentinel::hash_t> hash_{0};
};

/**
 * @brief Valid proxy executable
 */
class Executable {
  //static constexpr uint64_t HASH_MARKER = 478925426;

 public:
  explicit Executable(xla::hash_t hash)
      : hash_(hash)
        //adjusted_hash_(xla::util::MHash(hash, HASH_MARKER))
        {}
  bool is_active() const { return active_; }
  //xla::hash_t get_adjusted_hash() const { return adjusted_hash_; }
  bool set_active(bool active) { active_ = active; }

 private:

  const xla::hash_t hash_;
  //const xla::hash_t adjusted_hash_;
  // not actually sure if we need this
  // active anymore since transition is automatic downstream
  bool active_{false};
};
using ExecutablePtr = std::shared_ptr<Executable>;

/**
 * @brief Class to keep track of known-good executables
 */
class ExecutableCache {
  ExecutablePtr add_executable(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    assert(executables_.find(hash) == executables_.end());
    auto exec = executables_.insert({hash, std::make_shared<Executable>(hash)})
                    .first->second;
    //adjusted_hash_map_.insert({exec->get_adjusted_hash(), exec});
    return exec;
  }

 public:

//  ~ExecutableCache() {
//    std::cout << "end of executable cache" << ENDL;
//  }

  ExecutablePtr get_executable(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    auto found = executables_.find(hash);
    if (found != executables_.end()) {
      return found->second;
    }
    return nullptr;
  }
  ExecutablePtr get_executable_by_adjusted_hash(
      const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    auto found = adjusted_hash_map_.find(hash);
    if (found != adjusted_hash_map_.end()) {
      return found->second;
    }
    return nullptr;
  }
  bool has_executable(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    return executables_.count(hash) != 0;
  }
  bool has_executable_by_adjusted_hash(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    return adjusted_hash_map_.count(hash) != 0;
  }
  bool is_active_executable(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    auto exec = get_executable(hash);
    if (exec) {
      return exec->is_active();
    }
    return false;
  }
  ExecutablePtr activate_hash(const XLASentinel::hash_t& hash) {
    Lock lk(mtx_);
    auto found = executables_.find(hash);
    if (found == executables_.end()) {
      auto exec = add_executable(hash);
      exec->set_active(true);
      XLA_COUNTER("SentinelExecutableActivate", 1);
      return std::move(exec);
    } else {
      // Track that we're doing this in a deterministic way and not
      // overlapping logic
      const bool is_active = found->second->is_active();
      if (!is_active) {
        assert(!is_active);
        found->second->set_active(true);
        XLA_COUNTER("SentinelExecutableActivate", 1);
      }
      return found->second;
    }
  }

//  void modify_adjusted_hash(const xla::hash_t& h1, const xla::hash_t& h2) {
//    Lock lk(mtx_);
//    assert(h1 != h2);
//    assert(executables_.count(h1) != 0);
//    auto found = adjusted_hash_map_.find(h1);
//    if (found != adjusted_hash_map_.end()) {
//      auto exe = std::move(found->second);
//      adjusted_hash_map_.erase(found);
//      adjusted_hash_map_.insert({h2, std::move(exe)});
//    }
//  }

  void set_adjusted_hash(const xla::hash_t& h1, const xla::hash_t& h2) {
    Lock lk(mtx_);
    assert(h1 != h2);
    auto found = executables_.find(h1);
    if (found != executables_.end()) {
      // Should only set this once
      auto found_adjusted = adjusted_hash_map_.find(h1);
      if (found_adjusted != adjusted_hash_map_.end()) {
        assert(found_adjusted->second == found->second);
      } else {
        adjusted_hash_map_[h2] = found->second;
      }
    } else {
      assert(false);  // does this ever happen?
    }
  }

  void deactivate_hash(const XLASentinel::hash_t& hash) {  // currently we don't need to track the "active" one, so this might be pointless
    Lock lk(mtx_);
    auto found = executables_.find(hash);
    if (found != executables_.end()) {
      // should we assert that its active?  probably not
      // since deactivations acan come pretty randomly from any direction
      if (found->second->is_active()) {
        found->second->set_active(false);
        XLA_COUNTER("SentinelExecutableDeactivate", 1);
      }
    }
  }

 private:
  mutable std::recursive_mutex mtx_;
  absl::node_hash_map<XLASentinel::hash_t, ExecutablePtr>
      executables_;  // needs to be locked?
  absl::node_hash_map<XLASentinel::hash_t, ExecutablePtr> adjusted_hash_map_;
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

size_t get_number_of_required_runs_since_reset() {
  static bool trusted_model =
      xla::sys_util::GetEnvBool("XLA_TRUSTED_MODEL", false);
  if (trusted_model) {
    return 0;
  }
  static size_t rtc = xla::sys_util::GetEnvInt("XLA_CLEAN_STEPS_UNTIL_PROXY",
                                               DEFAULT_CLEAN_STEPS_UNTIL_PROXY);
  return rtc;
}

std::mutex init_devices_mutex;

bool __thread is_in_mark_step = false;
bool __thread is_clean_step = false;
//bool __thread is_qualifying_step = false;

}  // namespace

std::vector<std::string> XLASentinel::wse_devices_;

void XLASentinel::SetAllDevices(
    const std::vector<std::string>& all_devices) {
  wse_devices_.clear();
  wse_devices_.reserve(all_devices.size());
  for (const std::string& device_str : all_devices) {
    const Device device(device_str);
    if (device.hw_type == DeviceType::WSE) {
      wse_devices_.push_back(device_str);
    }
  }
}

bool XLASentinel::PreProcessHlo(
    xla::XlaBuilder* builder, const XLATensor::SyncTensorCollection& coll) {
  //HEREX();
  if (HasWseDevices() && IsTrainingThread(coll.requesting_tid)) {
    if (verbose) {
      std::cout << "PreProcessHlo(): " << coll.hash << ENDL;
    }
    ExecutablePtr exe = ex_cache->get_executable_by_adjusted_hash(coll.hash);
    if (exe) {
      if (exe->is_active()) {
        // Mark this for proxy
        xla::FrontendAttributes frontend_attributes;
        frontend_attributes.CopyFrom(builder->frontend_attributes());
        (*frontend_attributes.mutable_map())["PROXY_DEVICE"] =
            coll.device.ToString();
        builder->SetFrontendAttributes(frontend_attributes);

        // Sanity check that if we're pruning outputs,
        // the program shape has the same number of outputs as is expected
#ifndef NDEBUG
        std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(coll.requesting_tid);
        const std::size_t output_ids_size = compile_info->output_ids_.empty();
        if (output_ids_size) {
          const xla::Shape& result_shape = builder->GetProgramShape().ValueOrDie().result();
          std::size_t output_count;
          if (result_shape.element_type() == xla::PrimitiveType::TUPLE) {
            output_count = result_shape.tuple_shapes_size();
          } else {
            output_count = 1;
          }
          if(output_count != output_ids_size) {
            throw sentinel_exception()
              << "We expected the pruned fabric program shape to have "
              << output_ids_size << " outputs, but it actually had "
              << output_count << " outputs.  This is probably a bug and should "
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
  xla::ComputationClient* cc = xla::XrtComputationClient::Get();
  auto computation_client = dynamic_cast<xla::XlaComputationProxy*>(cc);
  if (computation_client) {
    computation_client->SetDeviceProxyAddress(device, proxy_address);
  } else {
    throw std::runtime_error("Device proxying is not enabled");
  }
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

bool XLASentinel::PruneTensors(std::vector<XLATensor>* tensors, XLATensor::SyncTensorCollection& coll) {
  if (!tensors || tensors->empty()) {
    return false;
  }
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

  if (!adjusted_indices.empty() && adjusted_indices.size() != coll.indices.size()) {
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

void XLASentinel::PostmarkHash(
    HashingState& state,
    std::vector<XLATensor>* tensors, XLATensor::SyncTensorCollection& coll) {
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
 *        "initial coll.hash after post-order" -> rehash( "initial coll.hash after post-order" )
 *       For a pruned executable:
 *        "initial coll.hash after post-order" -> rehash( "coll.hash calculated after being sent back for prune" )
 *
 *       An advantage of this is that if we are ever asked for the graph where we donn't have to prune
 *       anything, it should still resolve to the same executable's adjusted hash in PreProcessHlo
 *
 *       TODO: Optimize not doing two passes when we are in the same detected state on first-pass entry
 *             One obvious approach is to set a flag when post-order doesn't change after a prune (which
 *             so far has been the case), although would like to make that optional so that the harder state
 *             can keep being tested for now until there's a way to turn it on aand off and to have tests for it
 *             to assure it didn't break.
 */
bool XLASentinel::OnHashingComplete(
    HashingState& state,
    std::vector<XLATensor>* tensors,
    XLATensor::SyncTensorCollection& coll
) {
  // This only updates something if this is one we have an executable for

  static const absl::int128 PROXY_HASHING_VALUE = absl::MakeInt128(
      0xb8bc2f8ad48c431c,
      0x872132d8a172a6d8
  );

  const std::size_t pass = state.pass_++;

  if (!is_in_mark_step /*|| !is_clean_step*/) {
    std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(coll.requesting_tid);
    //compile_info->mark_step_count_since_last_reset_ = INVALID_COUNT;
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
      assert(compile_info->mark_step_count_since_last_reset_ != INVALID_COUNT);  // maybe this is ok, we just switched to a known graph?
      ++compile_info->mark_step_count_since_last_reset_;  // is there any point to increment this?
      ex_cache->activate_hash(coll.hash);  // not sure if 'active' has a meaning anymore
      state.fabric_run_ = true;
      if (PruneTensors(tensors, coll)) {
        state.pre_prune_hash_ = coll.hash;
        coll.hash = state.start_hash_;
        return true;  // need to recalculate postorder with new inputs
      }
      return false;  // Nothing removed, so keep going (on fabric)
    }

    // Note: For trusted, we don't need to analyze anything
    std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(coll.requesting_tid);
    if (coll.hash != compile_info->hash()) {
      if (true || verbose) {
        ColorScope clr(Color::FG_GREEN);
        std::cout << mp() << "NEW HASH: " << compile_info->hash()
                  << " -> " << coll.hash
                  << ENDL;
      }

      compile_info->set_hash(coll.hash);
      compile_info->mark_step_count_since_last_reset_ = 0;

      // If this isn't a zero-hash (i.e. first mark step call before loop),
      // then see if it's trusted
      if (!compile_info->hash() || !IsQualifyingStep(coll.requesting_tid)) {
        return false;
      }
    }

    ColorScope clr(Color::FG_GREEN);
    std::cout << mp()
              << "SAME HASH AS LAST TIME OR TRUSTED HASH: "
              << compile_info->hash()
              << ENDL;
    assert(compile_info->mark_step_count_since_last_reset_ != INVALID_COUNT);
    ++compile_info->mark_step_count_since_last_reset_;
    if (IsQualifyingStep(coll.requesting_tid)) {
      std::cout << mp() << "**** QUALIFYING: " << coll.hash << ENDL;
      ex_cache->activate_hash(coll.hash);
      if (PruneTensors(tensors, coll)) {
        state.fabric_run_ = true;
        assert(!state.pre_prune_hash_);
        state.pre_prune_hash_ = coll.hash;
        coll.hash = state.start_hash_;
        return true;  // need to recalculate postorder with new inputs
      }
      // Do we need to hash this differently for *our* executable
      // in case we didn't prune anything?
      const hash_t proxy_hash = xla::util::HashCombine(coll.hash, PROXY_HASHING_VALUE);
#ifndef NDEBUG
      // This shouldn't be the adjusted hash already or something went wrong
      assert(!ex_cache->get_executable_by_adjusted_hash(coll.hash));
#endif
      ex_cache->set_adjusted_hash(coll.hash, proxy_hash);
      coll.hash = proxy_hash;
      state.fabric_run_ = true;
      return false;  // Nothing removed, so keep going (on fabric)
    }
  } else {
    //
    // We sent them back to recalculate
    //
    // It's possible that with different outputs, the inputs didn't change,
    // in which case, 'coll.hash' is the same as 'state.pre_prune_hash_'
    //

    assert(state.pre_prune_hash_);  // this should have been set, the hash before the prune

    assert(state.fabric_run_);  // this should have been set the first pass,
                                // or else we got here by accident

    const hash_t proxy_hash = xla::util::HashCombine(coll.hash, PROXY_HASHING_VALUE);

#ifndef NDEBUG
    if (verbose) {
      std::cout << "Adjusted hash for proxy from " << coll.hash
                << " to " << proxy_hash
                << ", which had a pre-prune hash of " << state.pre_prune_hash_
                << ENDL;
    }
    // This shouldn't be the adjusted hash already or something went wrong
    // Addendum: on second pass it should be here, right?
    assert(!ex_cache->get_executable_by_adjusted_hash(coll.hash));
#endif
    //ex_cache->modify_adjusted_hash(coll.hash, proxy_hash);
    ex_cache->set_adjusted_hash(state.pre_prune_hash_, proxy_hash);
    coll.hash = proxy_hash;
    return false;
  }
}

void XLASentinel::NotifyCompile(
    std::vector<xla::ComputationClient::CompileInstance>& instances,
    hash_t hash, pid_t tid) {
  //HEREX();
  if (!HasWseDevices()) return;
  XLA_COUNTER("SentinelNotifyCompile", 1);
  if (is_in_mark_step) {
    XLA_COUNTER("SentinelStepMarkerCompile", 1);
  }

  if (IsTrainingThread(tid)) {
    XLA_COUNTER("SentinelMasterThreadCompile", 1);
    if (!ex_cache->has_executable_by_adjusted_hash(hash)) {
      XLA_COUNTER("SentinelNonProxyCompile", 1);
      ColorScope clr(std::cout, {Color::FG_BLUE}, false);
      assert(instances.size() == 1);  // always just one? maybe in distrib its one each.
      std::cout << "** NON-FABRIC COMPILE: "
                << to_string(instances[0].computation.GetProgramShape().ValueOrDie())
                << ENDL;
    } else {
      XLA_COUNTER("SentinelProxyCompile", 1);
    }
  }
}

void XLASentinel::NotifyExecute(
    const xla::ComputationClient::Computation& computation,
    const std::string& device,
    hash_t hash,
    pid_t tid
) {
  if (!HasWseDevices()) return;
  if (verbose) {
    HEREX();
  }
  XLA_COUNTER("SentinelExecute", 1);
  if (IsTrainingThread(tid)) {
    XLA_COUNTER("SentinelMasterThreadExecute", 1);
    if(!ex_cache->has_executable_by_adjusted_hash(hash)) {
      XLA_COUNTER("SentinelNonProxyExecute", 1);
      ColorScope clr(std::cout, {Color::FG_BLUE}, false);
      std::cout << "** NON-FABRIC EXECUTION: "
                << to_string(computation.program_shape())
                << ENDL;
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

  //if (!HasWseDevices()) return std::move(tensors);
  if (!is_in_mark_step) {
    // Anything outside of mark step is a reset
    std::shared_ptr<CompileInfo> compile_info =
        GetCompileInfo(coll->requesting_tid);
    compile_info->sync_count_since_hash_change_ = 0;
    compile_info->set_hash(0);
    return std::move(tensors);
  }

  if (verbose) {
    ColorScope cs(Color::FG_CYAN);
    std::cout << "XLASentinel::NotifyScheduleSyncTensorsGraph(): "
              << coll->hash << ENDL;
  }

  assert(tensors.size() == coll->indices.size());
  if (verbose_tensor_sync) {
    std::size_t index = 0;
    std::for_each(
        tensors.begin(), tensors.end(), [coll, &index, xla_tensors](auto &t) {
          ColorScope cs(Color::FG_CYAN);
          std::cout << (index + 1) << " " << coll->hash
          << ": SyncTensorsGraph tensor shape: " << t->shape();
          if (t->HasValue()) {
            std::cout << ", handle = " << t->GetOpaqueHandle();
          }
          std::cout << " ";
          XLATensor::print_tensor("", (*xla_tensors)[coll->indices[index++]]);
          //std::cout << ENDL;
        }
    );
  }

#if 0
  std::shared_ptr<CompileInfo> compile_info =
      GetCompileInfo(coll->requesting_tid);
  if (!compile_info->hash()) {
    compile_info->set_hash(coll->hash);
    compile_info->sync_count_since_hash_change_ = 0;
  } else if (coll->hash == compile_info->hash()) {
    ++compile_info->sync_count_since_hash_change_;
#if 0
    ++compile_info->mark_step_count_since_last_reset_;  // new

    auto exe = ex_cache->get_executable_by_adjusted_hash(coll->hash);
    if (/*exe && exe->is_active() &&*/
      IsQualifyingStep(coll->requesting_tid)) {
      //is_qualifying_step) {

      // vvv SAME AS IN POST MARK HASH
      std::vector<size_t> adjusted_indices;
      adjusted_indices.reserve(coll->indices.size());

      std::vector<xla::ComputationClient::DataPtr> new_tensors;
      new_tensors.reserve(coll->indices.size());
      //assert(xla_tensors->size() == coll->indices.size());   // assuming these are the same? or I have to use indexes?
      for (std::size_t i = 0, n = coll->indices.size(); i < n; ++i) {
        const std::size_t tensor_index = coll->indices[i];
        const XLATensor& tensor = (*xla_tensors)[tensor_index];
        //const XLATensor& tensor = (*xla_tensors)[i];
        if (IsAllowedOutput(tensor, *coll)) {
          adjusted_indices.push_back(coll->indices[i]);
          new_tensors.push_back(tensors[i]);
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
        coll->indices = std::move(adjusted_indices);
        if (verbose) {
          std::cout << "PostmarkHash(): coll.hash: " << coll->hash << " -> "
                    << exe->get_adjusted_hash() << ENDL;
        }
        coll->hash = exe->get_adjusted_hash();
        return std::move(new_tensors);
      }
      // ^^^ SAME AS IN POST MARK HASH
    }
#endif

  } else {
    ColorScope clr(Color::FG_CYAN);
    std::cout << "ScheduleSyncTensorsGraph() MarkStep hash change: "
              << compile_info->hash() << " -> " << coll->hash
              << ", is_clean_step = " << is_clean_step
              << ENDL;

    // Disable any old executables?
    //ex_cache->deactivate_current(compile_info->hash());
    auto current_exec = ex_cache->get_executable(coll->hash);
    if (current_exec && current_exec->is_active()) {
      current_exec->set_active(false);
    }
    compile_info->set_hash(coll->hash);
    //compile_info->sync_count_since_hash_change_ = 1;
    compile_info->sync_count_since_hash_change_ = 0;
    compile_info->mark_step_count_since_last_reset_ = 0;
  }
#endif
  return std::move(tensors);
}

void XLASentinel::NotifyStepMarkerBegin(
    const std::string& device_str, const std::vector<std::string>& devices) {
  is_clean_step = false;
  is_in_mark_step = true;
  XLA_COUNTER("SentinelStepMarker", 1);

  static bool registered_step_requirement = false;
  if (!registered_step_requirement) {
    XLA_VALUE_METRIC(
        "SentinelRequiredStepsSinceReset",
        get_number_of_required_runs_since_reset()
    );
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
    return;
  }
  is_clean_step = true;
  ++compile_info->mark_step_count_since_last_reset_;
  //const std::size_t step = ++compile_info->mark_step_count_since_last_reset_;
  //is_clean_step = compile_info->mark_step_count_since_last_reset_.load() > 0;
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

//  const pid_t tid = gettid();
//  auto compile_info = GetCompileInfo(tid);
//  compile_info->output_ids_.clear();

  is_in_mark_step = false;
  is_clean_step = false;
  //is_qualifying_step = false;
}

bool XLASentinel::IsSpecialLowering() {
  static bool allow_special_compile =
      xla::sys_util::GetEnvBool("XLA_ALLOW_SPECIAL_LOWERING", false);
  return allow_special_compile /*&& is_qualifying_step*/;
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
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t mark_step_count_since_reset =
      compile_info->mark_step_count_since_last_reset_.load();
  const std::size_t steps_required = get_number_of_required_runs_since_reset();
  if (!steps_required && is_clean_step) {
    return true;
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
    //const bool ready = mark_step_count_since_reset - 1 == steps_required;
    ready = mark_step_count_since_reset - 1 == steps_required;
  }
  if (ready) {
    assert(is_clean_step);  // validate it coincides with clean step logic
    if (verbose) {
      ColorScope clr({Color::BG_BLUE, Color::FG_YELLOW});
      std::cout << "Run ready" << std::endl << std::flush;
    }
  }
  return ready;
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
                                  XLATensor::SyncTensorCollection& coll) {
  assert(HasWseDevices());  // why?
  assert(is_in_mark_step);  // gets cleared at end of step
  assert(is_clean_step);    // otherwise, why are you asking?
  std::shared_ptr<CompileInfo> compile_info =
      GetCompileInfo(coll.requesting_tid);
  if (compile_info->output_ids_.empty()) {
    return true;
  }
  return compile_info->output_ids_.find(tensor.data()->unique_id) !=
         compile_info->output_ids_.end();
}

}  // namespace torch_xla
