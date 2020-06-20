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
#include "tensorflow/core/util/util.h"
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

bool verbose = false;

// const bool IGNORE_FIRST_MARK_STEP = true;
// const bool ENABLE_DEVICE_REMAPPING = true;
// const bool REQUIRE_INPUTS_OUTPUTS_SET = false;
// constexpr size_t DEFAULT_STEPS_TILL_COMPILE = 15;
constexpr size_t DEFAULT_STEPS_TILL_COMPILE = 1;

bool is_true(const char* s) {
  if (s && *s) {
    const char c = ::tolower(*s);
    if (c == 'y' || c == 't') {
      return true;
    }
    return atoi(s) > 0;
  }
  return false;
}

bool get_env_bool(const char* s, const bool dflt) {
  const char* v = getenv(s);
  if (v && *v) {
    return is_true(v);
  }
  return dflt;
}

int XLATensor::get_rank(const XLATensor::Data* data) {
  if (data->ir_value) {
    return data->ir_value.shape().rank();
  } else if (data->xla_data) {
    return data->xla_data->shape().rank();
  } else if (data->view) {
    return data->view->shape().rank();
  } else {
    return -1;
  }
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
    std::cout << label << " (id=" << data->unique_id << ") "
              << " tensor with no xla_data handle="
              << data->xla_data->GetOpaqueHandle()
              << " on device: " << data->xla_data->device()
              << " of shape: " << data->xla_data->shape().ToString()
              << std::endl
              << std::flush;
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

MarkStepScope::MarkStepScope(const std::string& device_str,
                             const std::vector<std::string>& devices) {
  CompileWatcher::NotifyStepMarkerBegin(device_str, devices);
}

MarkStepScope::~MarkStepScope() { CompileWatcher::NotifyStepMarkerEnd(); }

namespace {

template <typename DEST_MSG, typename SRC_MSG_ARRAY>
const DEST_MSG* get_id(const SRC_MSG_ARRAY& array, const int64_t id) {
  const int64_t total_count = array.size();
  for (int64_t i = 0; i < total_count; ++i) {
    auto& obj = array[i];
    if (obj.id() == id) {
      return &obj;
    }
  }
  return nullptr;
}

std::string get_proxy_device(const xla::HloModuleProto& module) {
  // save_msg(module, "my_hlo_module.json");
  const int64_t entry_computation_id = module.entry_computation_id();
  if (entry_computation_id) {
    auto computation = get_id<xla::HloComputationProto>(module.computations(),
                                                        entry_computation_id);
    const int64_t root_id = computation->root_id();
    if (root_id) {
      auto root_instruction = get_id<xla::HloInstructionProto>(
          computation->instructions(), root_id);
      const xla::FrontendAttributes& frontend_attributes =
          root_instruction->frontend_attributes();
      auto iter = frontend_attributes.map().find("PROXY_DEVICE");
      if (iter != frontend_attributes.map().end()) {
        // A compile may have failed, in which case it
        // gets delegated back to the default device
        auto cancel_iter =
            frontend_attributes.map().find("CANCEL_PROXY_DEVICE");
        if (cancel_iter != frontend_attributes.map().end()) {
          if (cancel_iter->second == iter->second) {
            return "";  // this proxying was cancelled (i.e. failed compile)
          }
        }
        return iter->second;
      }
    }
  }
  return "";
}

using Lock = std::lock_guard<std::recursive_mutex>;

struct CompileInfo {
  std::atomic<size_t> sync_count_since_hash_change_{0};
  // std::atomic<size_t> run_count_at_last_mark_step_{0};
  std::atomic<size_t> mark_step_count_since_last_reset_{0};
  std::unordered_set<size_t> output_ids_;

  void set_hash(CompileWatcher::hash_t hash) {
    if (hash != hash_.load()) {
      // std::cout << "hash " << hash_.load() << " -> " << hash << std::endl <<
      // std::flush; std::cout << "Setting hash to: " << hash << std::endl <<
      // std::flush;
      hash_ = std::move(hash);
    }
  }
  //  bool hash_equal(const CompileWatcher::hash_t& hash) const {
  //    return hash == hash_.load();
  //  }
  CompileWatcher::hash_t hash() const { return hash_; }

 private:
  std::atomic<CompileWatcher::hash_t> hash_{0};
};

/**
 * @brief Valid proxy executable
 */
class Executable {
  static constexpr uint64_t HASH_MARKER = 478925426;

 public:
  explicit Executable(xla::hash_t hash)
      : hash_(hash), adjusted_hash_(xla::util::MHash(hash, HASH_MARKER)) {}
  bool set_active(bool active) { active_ = active; }
  bool is_active() const { return active_; }
  //  bool set_compiled(bool compiled) { compiled_ = compiled; }
  //  bool is_compiled() const { return compiled_; }
  xla::hash_t get_adjusted_hash() const { return adjusted_hash_; }

 private:
  const xla::hash_t hash_;
  const xla::hash_t adjusted_hash_;
  bool active_{false};
  //  bool compiled_{false};
};
using ExecutablePtr = std::shared_ptr<Executable>;

/**
 * @brief Class to keep track of known-good executables
 */
class ExecutableCache {
  ExecutablePtr add_executable(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    assert(executables_.find(hash) == executables_.end());
    auto exec = executables_.insert({hash, std::make_shared<Executable>(hash)})
                    .first->second;
    adjusted_hash_map_.insert({exec->get_adjusted_hash(), exec});
    return exec;
  }

 public:
  ExecutablePtr get_executable(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    auto found = executables_.find(hash);
    if (found != executables_.end()) {
      return found->second;
    }
    return nullptr;
  }
  ExecutablePtr get_executable_by_adjusted_hash(
      const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    auto found = adjusted_hash_map_.find(hash);
    if (found != adjusted_hash_map_.end()) {
      return found->second;
    }
    return nullptr;
  }
  bool has_executable(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    return executables_.count(hash) != 0;
  }
  bool has_executable_by_adjusted_hash(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    return adjusted_hash_map_.count(hash) != 0;
  }
  bool is_active_executable(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    auto exec = get_executable(hash);
    if (exec) {
      return exec->is_active();
    }
    return false;
  }
  ExecutablePtr activate_hash(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    auto found = executables_.find(hash);
    if (found == executables_.end()) {
      auto exec = add_executable(hash);
      exec->set_active(true);
      return std::move(exec);
    } else {
      // Track that we're doing this in a deterministic way and not
      // overlapping logic
      const bool is_active = found->second->is_active();
      assert(!is_active);
      found->second->set_active(true);
      return found->second;
    }
  }
  void modify_adjusted_hash(const xla::hash_t& h1, const xla::hash_t& h2) {
    Lock lk(mtx_);
    assert(h1 != h2);
    auto found = adjusted_hash_map_.find(h1);
    if (found != adjusted_hash_map_.end()) {
      auto exe = std::move(found->second);
      adjusted_hash_map_.erase(found);
      adjusted_hash_map_.insert({h2, std::move(exe)});
    }
  }
  void deactivate_hash(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    auto found = executables_.find(hash);
    if (found != executables_.end()) {
      // should we asser that its active?  probably not
      // since deactivations acan come pretty randomly from any direction
      found->second->set_active(false);
    }
  }

 private:
  mutable std::recursive_mutex mtx_;
  absl::node_hash_map<CompileWatcher::hash_t, ExecutablePtr>
      executables_;  // needs to be locked?
  absl::node_hash_map<CompileWatcher::hash_t, ExecutablePtr> adjusted_hash_map_;
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

namespace {
std::shared_ptr<ExecutableCache> ex_cache = std::make_shared<ExecutableCache>();
}

size_t get_number_of_required_runs_since_reset() {
  static bool trusted_model =
      xla::sys_util::GetEnvBool("XLA_TRUSTED_MODEL", false);
  if (trusted_model) {
    return 0;
  }
  static size_t rtc = xla::sys_util::GetEnvInt("XLA_RUNS_TILL_COMPILE",
                                               DEFAULT_STEPS_TILL_COMPILE);
  return rtc;
}

std::mutex init_devices_mutex;

bool __thread is_in_mark_step = false;
bool __thread is_clean_step = false;
bool __thread is_qualifying_step = false;

}  // namespace

std::vector<std::string> CompileWatcher::wse_devices_;

void CompileWatcher::SetAllDevices(
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

bool CompileWatcher::PreProcessHlo(
    xla::XlaBuilder* builder, const XLATensor::SyncTensorCollection& coll) {
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
        // assert(coll.device.ToString() == GetDevice().ToString());  // get rid
        // of GetDevice() as soon as this passes
        //(*frontend_attributes.mutable_map())["PROXY_DEVICE"] =
        // GetDevice().ToString();
        (*frontend_attributes.mutable_map())["PROXY_DEVICE"] =
            coll.device.ToString();
        builder->SetFrontendAttributes(frontend_attributes);
        return true;
      }
    }
  }
  return false;
}

void CompileWatcher::SetDeviceProxyAddress(const std::string& device,
                                           const std::string& proxy_address) {
  xla::ComputationClient* cc = xla::XrtComputationClient::Get();
  auto computation_client = dynamic_cast<xla::XlaComputationProxy*>(cc);
  if (computation_client) {
    computation_client->SetDeviceProxyAddress(device, proxy_address);
  } else {
    throw std::runtime_error("Device proxying is not enabled");
  }
}

bool CompileWatcher::HasWseDevices() {
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

bool CompileWatcher::IsTrainingThread(pid_t tid) {
  return GetPythonState(tid) == EPS_IN_TRAIN_LOOP;
}

void CompileWatcher::OnHashChange(const xla::hash_t& prev_hash,
                                  const XLATensor::SyncTensorCollection& coll) {
  // This only updates something if this is one we have an executable for
  ex_cache->modify_adjusted_hash(prev_hash, coll.hash);
}

xla::hash_t CompileWatcher::PostmarkHash(
    std::vector<XLATensor>* tensors, XLATensor::SyncTensorCollection& coll) {
  const xla::hash_t original_hash = coll.hash;
  if (!is_clean_step) {
    return original_hash;
  }
  ExecutablePtr exe = ex_cache->get_executable(coll.hash);
  if (exe) {
    if (!exe->is_active()) {
      exe->set_active(true);
    }
  } else if (IsQualifyingStep(coll.requesting_tid)) {
    // create and activate
    exe = ex_cache->activate_hash(coll.hash);
  }

  if (exe && exe->is_active()) {
    std::vector<size_t> adjusted_indices;
    adjusted_indices.reserve(coll.indices.size());
    for (std::size_t i = 0, n = coll.indices.size(); i < n; ++i) {
      const std::size_t tensor_index = coll.indices[i];
      const XLATensor& tensor = (*tensors)[tensor_index];
      if (IsAllowedOutput(tensor, coll)) {
        adjusted_indices.push_back(coll.indices[i]);
      } else {
        if (verbose) {
          XLATensor::print_tensor("Removing output", tensor);
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
      exe->set_active(false);
      std::cout
          << "No effective allowed outputs, so reverting to standard device"
          << ENDL;
    }
  }
}

void CompileWatcher::NotifyCompile(
    std::vector<xla::ComputationClient::CompileInstance>& instances,
    hash_t hash, pid_t tid) {}

void CompileWatcher::NotifyExecute(const std::string& device, hash_t hash,
                                   pid_t tid, bool scheduled) {
  // Just notify when the training thread tries to run something off of the
  // proxy
  if (IsTrainingThread(tid) && !ex_cache->has_executable_by_adjusted_hash(hash)) {
    ColorScope clr(std::cout, {Color::FG_RED, Color::BG_BLUE}, false);
    std::cout << "** NON-FABRIC EXECUTION" << ENDL;
  }
}

std::vector<xla::ComputationClient::DataPtr>
CompileWatcher::NotifyScheduleSyncTensorsGraph(
    std::vector<xla::ComputationClient::DataPtr> tensors,
    XLATensor::SyncTensorCollection* coll,
    std::shared_ptr<xla::ComputationClient::Computation>& computation) {
  if (!is_in_mark_step) {
    // Anything outside of mark step is a reset
    std::shared_ptr<CompileInfo> compile_info =
        GetCompileInfo(coll->requesting_tid);
    compile_info->sync_count_since_hash_change_ = 0;
    compile_info->set_hash(0);
    return std::move(tensors);
  }

  //  std::for_each(tensors.begin(), tensors.end(), [](auto& t){
  //    std::cout << "SyncTensorsGraph tensor shape: " << t->shape() << ENDL;
  //  });
  // XLATensor::print_all_tensors("SyncTensorsGraph tensors", tensors);

  std::shared_ptr<CompileInfo> compile_info =
      GetCompileInfo(coll->requesting_tid);
  if (!compile_info->hash()) {
    compile_info->set_hash(coll->hash);
    compile_info->sync_count_since_hash_change_ = 1;
  } else if (coll->hash == compile_info->hash()) {
    ++compile_info->sync_count_since_hash_change_;
  } else {
    ColorScope clr(Color::FG_CYAN);
    std::cout << "ScheduleSyncTensorsGraph() MarkStep hash change: "
              << compile_info->hash() << " -> " << coll->hash << ENDL;
    compile_info->set_hash(coll->hash);
    compile_info->sync_count_since_hash_change_ = 1;
  }
  return std::move(tensors);
}

void CompileWatcher::NotifyStepMarkerBegin(
    const std::string& device_str, const std::vector<std::string>& devices) {
  is_clean_step = false;
  is_in_mark_step = true;
  if (verbose) {
    ColorScope clr(Color::FG_YELLOW);
    std::cout << "*************** CompileWatcher::NotifyStepMarker: device="
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
  if (!current_sync_count) {
    compile_info->mark_step_count_since_last_reset_ = 0;
    return;
  }
  const std::size_t step = ++compile_info->mark_step_count_since_last_reset_;
  is_clean_step = compile_info->mark_step_count_since_last_reset_.load() > 0;
  is_qualifying_step = IsQualifyingStep(tid);
  if (is_qualifying_step) {
    ColorScope red(Color::FG_RED);
    std::cout << "BEGIN WseRunStep() at step since sync: " << step << std::endl
              << std::flush;
  }
}

void CompileWatcher::NotifyStepMarkerEnd() {
  assert(is_in_mark_step);
  const pid_t tid = gettid();
  if (IsQualifyingStep(tid)) {
    ColorScope red(Color::FG_RED);
    std::cout << "END WseRunStep()" << std::endl << std::flush;
  }
  auto compile_info = GetCompileInfo(tid);
  compile_info->output_ids_.clear();

  is_in_mark_step = false;
  is_clean_step = false;
  is_qualifying_step = false;
}

bool CompileWatcher::IsSpecialLowering() {
  static bool allow_special_compile =
      xla::sys_util::GetEnvBool("XLA_ALLOW_SPECIAL_LOWERING", false);
  return allow_special_compile && is_qualifying_step;
}

// Device CompileWatcher::GetDevice() {
//    if (HasWseDevices()) {
//      return Device(*wse_devices_.begin());
//    }
//    return Device(DeviceType::CPU, 0);
//}

bool CompileWatcher::IsQualifyingStep(pid_t tid /*, bool or_higher*/) {
  assert(is_in_mark_step);  // shouldn't we always be? then we can just call
                            // once in MarkStep
  if (!is_in_mark_step) {
    return false;
  }
  if (!HasWseDevices() /*|| !IsTrainingThread(tid)*/) {
    return false;
  }
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t mark_step_count_since_reset =
      compile_info->mark_step_count_since_last_reset_.load();
  if (!mark_step_count_since_reset) {
    // The first step is superfluous since it's the top of the dataset interator
    // loop, before any graph is built. This also takes care of disqualifying
    // due to spurious compiles within the train loop
    return false;
  }
  const std::size_t steps_required = get_number_of_required_runs_since_reset();
  const bool ready = mark_step_count_since_reset - 1 == steps_required;
  if (ready) {
    assert(is_clean_step);  // validate it coincides with clean step logic
    ColorScope clr({Color::BG_BLUE, Color::FG_YELLOW});
    std::cout << "Run ready" << std::endl << std::flush;
  }
  return ready;
}

void CompileWatcher::SetOutputs(const std::vector<at::Tensor>& output_tensors,
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

bool CompileWatcher::IsAllowedOutput(const XLATensor& tensor,
                                     XLATensor::SyncTensorCollection& coll) {
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

// void CompileWatcher::SetDeviceMapping(const std::string& from_device, const
// std::string& to_device) {
//  std::lock_guard<std::mutex> lk(device_mapping_mtx_);
//  assert(!from_device.empty());
//  assert(from_device != to_device);
//  if (to_device.empty()) {
//    device_mapping_.erase(from_device);
//  } else {
//    device_mapping_[from_device] = std::make_pair(Device(to_device), true);
//  }
//}
//
// std::mutex CompileWatcher::device_mapping_mtx_;
// std::unordered_map<std::string, std::pair<Device, bool>>
// CompileWatcher::device_mapping_;
//
// std::string CompileWatcher::GetDeviceMapping(const std::string& device) {
//  assert(!device.empty());
//  std::lock_guard<std::mutex> lk(device_mapping_mtx_);
//  auto iter = device_mapping_.find(device);
//  if (iter != device_mapping_.end() && iter->second.second /* enabled? */) {
//    return iter->second.first.ToString();
//  }
//  return device;
//}
//
// const torch_xla::Device& CompileWatcher::GetDeviceMapping(const Device&
// device) {
//  std::lock_guard<std::mutex> lk(device_mapping_mtx_);
//  auto iter = device_mapping_.find(device.ToString());
//  if (iter != device_mapping_.end() && iter->second.second /* enabled? */) {
//    return iter->second.first;
//  }
//  return device;
//}

}  // namespace torch_xla
