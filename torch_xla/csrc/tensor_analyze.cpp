#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_analyze.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xla_computation_proxy.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/core/util/util.h"

#include "absl/container/node_hash_map.h"

#include <string>
#include <stack>
#include <mutex>
#include <Python.h>

#include <pybind11/pybind11.h>

#undef assert
#  define assert(expr)							\
     (static_cast <bool> (expr)						\
      ? void (0)							\
      : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))


/**
 * Most of this can eventually move to monolith
 */
namespace torch_xla {
//std::atomic<size_t> active_parameter_count{0};

bool verbose = true;

const bool IGNORE_FIRST_MARK_STEP = true;
const bool ENABLE_DEVICE_REMAPPING = true;
const bool REQUIRE_INPUTS_OUTPUTS_SET = false;
constexpr size_t DEFAULT_STEPS_TILL_COMPILE = 15;

bool is_true(const char *s) {
  if (s && *s) {
    const char c = ::tolower(*s);
    if (c == 'y' || c == 't') {
      return true;
    }
    return atoi(s) > 0;
  }
  return false;
}

bool get_env_bool(const char *s, const bool dflt) {
  const char *v = getenv(s);
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
  } else if(data->view) {
    return data->view->shape().rank();
  } else {
    return -1;
  }
}

void XLATensor::print_tensor(const std::string& label, const XLATensor& tensor) {
  print_tensor(label, tensor.data(), false, tensor.GetViewAliasId());
}

void XLATensor::print_tensor_ex(const std::string& label, const XLATensor& tensor) {
  print_tensor_ex(label, tensor.data(), false, tensor.GetViewAliasId());
}

void XLATensor::print_tensor_ex(const std::string& label,
    const XLATensor::Data* data, bool assert, ptrdiff_t alias_id) {
  if (data->ir_value) {
    std::cout << label << " (id=" << data->unique_id << ") "
              << " IR tensor of shape: " << data->ir_value.shape().ToString()
              << std::endl << std::flush;
    if (assert) {
      assert(!data->xla_data);
      assert(!data->view);
    }
  } else if (data->xla_data) {
    // coming from _xla_tensors_from_aten in at least one case
    std::cout << label << " (id=" << data->unique_id << ") "
              << " tensor with no ir_value of shape: "
              << data->xla_data->shape().ToString()
              << std::endl << std::flush;
    if (assert) {
      assert(!data->ir_value);
      assert(!data->view);
    }
  } else if(data->view) {
    std::cout << label << " (id=" << data->unique_id << ") "
              << " tensor with view of shape: "
              << data->view->shape().ToString()
              << std::endl << std::flush;
    if (assert) {
      assert(!data->ir_value);
      assert(!data->xla_data);
    }
  } else {
    std::cout << label << " (id=" << data->unique_id << ") "
              << " strange tensor of unknown size"
              << std::endl << std::flush;
  }
}

void XLATensor::print_tensor(const std::string& label, const XLATensor::Data* data, bool assert, ptrdiff_t alias_id) {
  ColorScope color_scope(Color::FG_CYAN, true);
  print_tensor_ex(label, data, assert, alias_id);
}

void XLATensor::print_all_tensors(const std::string& label, const std::vector<XLATensor>& tensors) {
  ColorScope cs(Color::FG_BLUE, false);
  std::cout << "------------------" << ENDL;
  std::cout << "tensors=[" << ENDL;
  for (size_t i = 0, n = tensors.size(); i < n; ++i) {
    const XLATensor &t = tensors[i];
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

  void push(EPythonState new_state, pid_t __tid=0) {
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

EPythonState GetPythonState(pid_t tid) {
  return python_state.get(tid);
}

static void _PushPythonState(EPythonState state, pid_t __tid=0) {
  python_state.push(state, __tid);
}


void PushPythonState(EPythonState state) {
  _PushPythonState(state);
}

void PopPythonState() {
  python_state.pop();
}

namespace {

template<typename DEST_MSG, typename SRC_MSG_ARRAY>
const DEST_MSG *get_id(const SRC_MSG_ARRAY& array, const int64_t id) {
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
  //save_msg(module, "my_hlo_module.json");
  const int64_t entry_computation_id = module.entry_computation_id();
  if (entry_computation_id) {
    auto computation = get_id<xla::HloComputationProto>(
      module.computations(),
      entry_computation_id
    );
    const int64_t root_id = computation->root_id();
    if (root_id) {
      auto root_instruction = get_id<xla::HloInstructionProto>(
        computation->instructions(),
        root_id
      );
      const xla::FrontendAttributes &frontend_attributes =
        root_instruction->frontend_attributes();
      auto iter = frontend_attributes.map().find("PROXY_DEVICE");
      if (iter != frontend_attributes.map().end()) {
        // A compile may have failed, in which case it
        // gets delegated back to the default device
        auto cancel_iter = frontend_attributes.map().find("CANCEL_PROXY_DEVICE");
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
  //std::atomic<size_t> run_count_at_last_mark_step_{0};
  std::atomic<size_t> mark_step_count_since_last_reset_{0};
  std::unordered_set<size_t> output_ids_;

  void set_hash(CompileWatcher::hash_t hash) {
    if (hash != hash_.load()) {
      //std::cout << "hash " << hash_.load() << " -> " << hash << std::endl << std::flush;
      //std::cout << "Setting hash to: " << hash << std::endl << std::flush;
      hash_ = std::move(hash);
    }
  }
  bool hash_equal(const CompileWatcher::hash_t& hash) const {
    return hash == hash_.load();
  }
  CompileWatcher::hash_t hash() const { return hash_; }
private:
  std::atomic<CompileWatcher::hash_t> hash_{0};
};

class Executable {
  static constexpr uint64_t HASH_MARKER = 478925426;
public:
  explicit Executable(xla::hash_t hash) : hash_(hash), adjusted_hash_(xla::util::MHash(hash, HASH_MARKER)) {}
  bool set_active(bool active) { active_ = active; }
  bool is_active() const { return active_; }
  bool set_compiled(bool compiled) { compiled_ = compiled; }
  bool is_compiled() const { return compiled_; }
  xla::hash_t get_adjusted_hash() const { return adjusted_hash_; }
private:
  const xla::hash_t hash_;
  const xla::hash_t adjusted_hash_;
  bool active_{false};
  bool compiled_{false};
};
using ExecutablePtr = std::shared_ptr<Executable>;

class ExecutableCache {
  ExecutablePtr add_executable(const CompileWatcher::hash_t& hash) {
    Lock lk(mtx_);
    assert(executables_.find(hash) == executables_.end());
    auto exec = executables_.insert(
      {hash, std::make_shared<Executable>(hash)}
    ).first->second;
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
  ExecutablePtr get_executable_by_adjusted_hash(const CompileWatcher::hash_t& hash) {
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
  absl::node_hash_map<CompileWatcher::hash_t, ExecutablePtr> executables_;  // needs to be locked?
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
std::shared_ptr<ExecutableCache> ex_cache  = std::make_shared<ExecutableCache>();
}

size_t get_number_of_required_runs_since_reset() {
  static bool trusted_model = xla::sys_util::GetEnvBool("XLA_TRUSTED_MODEL", false);
  if (trusted_model) {
    return 0;
  }
  static size_t rtc =
    xla::sys_util::GetEnvInt(
      "XLA_RUNS_TILL_COMPILE",
      DEFAULT_STEPS_TILL_COMPILE
    );
  return rtc;
}

std::mutex init_devices_mutex;

bool __thread is_in_mark_step = false;
bool __thread is_clean_step = false;

}  // anonymoud namespace

std::vector<std::string> CompileWatcher::wse_devices_;

void CompileWatcher::SetAllDevices(const std::vector<std::string>& all_devices) {
  wse_devices_.clear();
  wse_devices_.reserve(all_devices.size());
  for (const std::string& device_str : all_devices) {
    const Device device(device_str);
    if (device.hw_type == DeviceType::WSE) {
      wse_devices_.push_back(device_str);
    }
  }
}

bool CompileWatcher::PreProcessHlo(xla::XlaBuilder *builder, const XLATensor::SyncTensorCollection& coll) {
  if (HasWseDevices() && IsTrainingThread(coll.requesting_tid)) {
    std::cout << "PreProcessHlo(): " << coll.hash << ENDL;
    ExecutablePtr exe = ex_cache->get_executable_by_adjusted_hash(coll.hash);
    if (exe) {
      if (exe->is_active()) {
        // Mark this for proxy
        xla::FrontendAttributes frontend_attributes;
        frontend_attributes.CopyFrom(builder->frontend_attributes());
        (*frontend_attributes.mutable_map())["PROXY_DEVICE"] = GetDevice().ToString();
        builder->SetFrontendAttributes(frontend_attributes);
        return true;
      }
    }
//    std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
//    bool active = false;
    // Probably should just set to compiled here
//    if (IsQualifyingStep(tid)) {
//      // TODO: preapproved executables should be able to start on first valid step marker
//      //       where no other executions happenned within the mark step boundary
//      auto exec = ex_cache->get_executable(hash);
//      if (exec) {
//        exec->set_active(true);
//      } else {
//        ex_cache->add_executable(hash)->set_active(true);
//      }
//      active = true;
//    }
//    if (active /*|| compile_info->is_approved_hash(hash)*/) {
//      // This is a WSE-viable compile
//      xla::FrontendAttributes frontend_attributes;
//      frontend_attributes.CopyFrom(builder->frontend_attributes());
//      (*frontend_attributes.mutable_map())["PROXY_DEVICE"] = GetDevice().ToString();
//      builder->SetFrontendAttributes(frontend_attributes);
//      return true;
//    }
  }
  return false;
}

void CompileWatcher::SetDeviceProxyAddress(
  const std::string& device, const std::string& proxy_address) {
  xla::ComputationClient *cc = xla::XrtComputationClient::Get();
  auto computation_client = dynamic_cast<xla::XlaComputationProxy *>(cc);
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

void CompileWatcher::OnHashChange(const xla::hash_t& prev_hash, const XLATensor::SyncTensorCollection& coll) {
  // This only updates something if this is one we have an executable for
  ex_cache->modify_adjusted_hash(prev_hash, coll.hash);
}

xla::hash_t CompileWatcher::PostmarkHash(std::vector<XLATensor>* tensors, XLATensor::SyncTensorCollection& coll) {
  const xla::hash_t original_hash = coll.hash;
  if (!is_clean_step) {
    return original_hash;
  }
  ExecutablePtr exe = ex_cache->get_executable(coll.hash);
  if (exe) {
    if (!exe->is_active()) {
      exe->set_active(true);
    }
  } else if(IsQualifyingStep(coll.requesting_tid)) {
    // create and activate
    exe = ex_cache->activate_hash(coll.hash);
  }

  if (exe && exe->is_active()) {
    //std::vector<XLATensor> adjusted_tensors;
    //adjusted_tensors.reserve(tensors->size());
    std::vector<size_t> adjusted_indices;
    adjusted_indices.reserve(coll.indices.size());
    for (std::size_t i = 0, n = coll.indices.size(); i < n; ++i) {
      const std::size_t tensor_index = coll.indices[i];
      const XLATensor& tensor = (*tensors)[tensor_index];
      if (IsAllowedOutput(tensor, coll)) {
        //adjusted_tensors.emplace_back(tensor);
        adjusted_indices.push_back(coll.indices[i]);
      } else {
        XLATensor::print_tensor("Removing output", tensor);
      }
    }
    if (!adjusted_indices.empty()) {
      //*tensors = std::move(adjusted_tensors);
      coll.indices = std::move(adjusted_indices);
      std::cout << "PostmarkHash(): coll.hash: " << coll.hash << " -> " << exe->get_adjusted_hash() << ENDL;
      coll.hash = exe->get_adjusted_hash();
      return coll.hash;
    } else {
      // Nothing left, so can't do this on proxy
      exe->set_active(false);
      std::cout << "No effective allowed outputs, so reverting to standard device" << ENDL;
    }
  }
}

void CompileWatcher::NotifyCompile(
    std::vector<xla::ComputationClient::CompileInstance>& instances,
    hash_t hash,
    pid_t tid
) {
  // TODO: may need to know if it fails, and, for instance, mark it as not compilable
  ExecutablePtr exec = ex_cache->get_executable_by_adjusted_hash(hash);
  if (exec) {
    exec->set_compiled(true);
  }
#if 0
  if (!HasWseDevices() || !IsTrainingThread(tid)) {
    return;
  }
  HEREX();

  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);

  std::cout << "NotifyCompile(): ";
  if (hash != compile_info->hash()) {
    std::cout << "hash = " << hash << " ( current = " << compile_info->hash() << " )" << std::endl << std::flush;
  } else {
    std::cout << "Hash is unchanged" << std::endl << std::flush;
  }

  assert(instances.size() == 1);  // what to do if more than one?
  if (compile_info->hash_equal(hash)) {
    std::cout << "Compiling same hash again?" << std::endl << std::flush;
    return;
  }
  const std::string proxy_device = get_proxy_device(instances[0].computation.proto());
  if (!proxy_device.empty() && proxy_device == GetDevice().ToString()) {
    ColorScope color(Color::FG_RED);
    std::cout << "Presumably compiling for WSE device" << ENDL;
    // Set hash to WSE's hash value
    assert(compile_info->hash_equal(hash));
    compile_info->add_approved_hash(hash);
    //compile_info->set_hash(hash);
  //} else if (!IsWseRunning(tid)) {
  }
//  else if (!compile_info->is_approved_hash(hash)) {
//    std::cout << "Compiling hash=" << hash << ENDL;
//    Reset(tid, true);
//    compile_info->set_hash(hash);
//  }
  else {
    if (!compile_info->is_approved_hash(hash)) {
      // Should drop out of WSE device...
      ColorScope color(Color::FG_RED);
      std::cout << "COMPILING FOR TRAINING WHILE RUNNING" << std::endl << std::flush;
      //Reset(tid, true);
    }
  }
#endif
}

void CompileWatcher::NotifyExecute(const std::string& device, hash_t hash, pid_t tid, bool scheduled) {
#if 0
  if (!HasWseDevices() || !IsTrainingThread(tid)) {
    return;
  }
  HEREX();
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);

  if (hash != compile_info->hash()) {
    std::cout << "NotifyExecute(): hash = " << hash
              << " ( current = " << compile_info->hash() << " )"
              << std::endl << std::flush;
  }

  bool new_hash = false;
  if (!compile_info->hash()) {
      compile_info->set_hash(hash);
      new_hash = true;
  }

  if (compile_info->is_approved_hash(hash)) {
    return;
  }

  if (compile_info->hash_equal(hash)) {
    ++compile_info->sync_count_since_hash_change_;
#if 0
    // Here we can determine if more runs than steps are occuring
    if (compile_info->sync_count_since_hash_change_ == get_number_of_required_runs_since_reset()) {
      if (compile_info->sync_count_since_hash_change_ <= compile_info->mark_step_count_since_last_reset_) {
        ColorScope clr(Color::FG_GREEN);
        // TODO: Should also have a check that everything required is available,
        //  like grads and whatnot in the live tensors.
        //  Maybe even inspect the proposed HLO graph for compatibility.
        std::cout << "**** ELIGIBLE FOR WSE COMPILE ****"
                  << ", hash=" << hash << ", device=" << device << ENDL;
        if (ENABLE_DEVICE_REMAPPING) {
          SetDeviceMapping(device, GetDevice().ToString());
        }
      } else {
        // THIS COULD BE ASYNC
//              std::cout << "TOO MANY RUNS PER STEP: " << compile_info->sync_count_since_hash_change_
//                        << ", hash=" << hash << ", device=" << device << std::endl << std::flush;
      }
    } else {
//          if (!IsWseRunning()) {
//              std::cout << "REPEAT RUN " << compile_info->sync_count_since_hash_change_
//                        << ", hash=" << hash << std::endl << std::flush;
//          }
    }
#endif
  } else {
    Reset(tid, false);
    // Switched to another graph
//    std::cout << "RESETTING EXECUTION COUNTER FROM " << compile_info->sync_count_since_hash_change_.load()
//              << ", hash " << compile_info->hash() << " -> " << hash
//              << ", device=" << device << std::endl << std::flush;
    compile_info->set_hash(hash);
    assert(compile_info->sync_count_since_hash_change_.load() == 0);
    ++compile_info->sync_count_since_hash_change_;
  }
#endif
}

std::vector<xla::ComputationClient::DataPtr> CompileWatcher::NotifyScheduleSyncTensorsGraph(
  std::vector<xla::ComputationClient::DataPtr> tensors,
  XLATensor::SyncTensorCollection* coll,
  std::shared_ptr<xla::ComputationClient::Computation>& computation
) {
  if (!is_in_mark_step) {
    // Anything outside of mark step is a reset
    std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(coll->requesting_tid);
    compile_info->sync_count_since_hash_change_ = 0;
    compile_info->set_hash(0);
    return std::move(tensors);
  }

//  std::for_each(tensors.begin(), tensors.end(), [](auto& t){
//    std::cout << "SyncTensorsGraph tensor shape: " << t->shape() << ENDL;
//  });
  //XLATensor::print_all_tensors("SyncTensorsGraph tensors", tensors);

  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(coll->requesting_tid);
  if (!compile_info->hash()) {
    compile_info->set_hash(coll->hash);
    compile_info->sync_count_since_hash_change_ = 1;
  } else if (coll->hash == compile_info->hash()) {
    ++compile_info->sync_count_since_hash_change_;
  } else {
    ColorScope clr(Color::FG_CYAN);
    std::cout << "ScheduleSyncTensorsGraph() hash change: " << compile_info->hash() << " -> " << coll->hash << ENDL;
    compile_info->set_hash(coll->hash);
    compile_info->sync_count_since_hash_change_ = 1;
  }
  return std::move(tensors);
}

void CompileWatcher::NotifyStepMarkerBegin(const std::string& device_str, const std::vector<std::string>& devices) {
  is_clean_step = false;
  is_in_mark_step = true;
  {
    ColorScope clr(Color::FG_YELLOW);
    std::cout << "*************** CompileWatcher::NotifyStepMarker: device=" << device_str << std::endl << std::flush;
  }
  const pid_t tid = gettid();
  if (!IsTrainingThread(tid)) {
    assert(GetPythonState(tid) == EPS_INVALID);
    // The assumption is that only the training thread can call _XLAC._mark_step()
    _PushPythonState(EPS_IN_TRAIN_LOOP, tid);
  }
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t current_sync_count = compile_info->sync_count_since_hash_change_.load();
  if (!current_sync_count) {
    compile_info->mark_step_count_since_last_reset_ = 0;
    return;
  }
  //if (compile_info->run_count_at_last_mark_step_.load() == compile_info->sync_count_since_hash_change_.load()) {
    // First or superfluous step marker
    //return;
  //}
  ++compile_info->mark_step_count_since_last_reset_;
  //assert(current_run_count > compile_info->run_count_at_last_mark_step_.load());
  //compile_info->run_count_at_last_mark_step_ = current_run_count;
  is_clean_step = compile_info->mark_step_count_since_last_reset_.load() > 0;
  if (IsQualifyingStep(tid)) {
    ColorScope red(Color::FG_RED);
    std::cout << "BEGIN WseRunStep()" << std::endl << std::flush;
    //compile_info->
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
}

Device CompileWatcher::GetDevice() {
    if (HasWseDevices()) {
      return Device(*wse_devices_.begin());
    }
    return Device(DeviceType::CPU, 0);
}

//bool CompileWatcher::Reset(pid_t tid, bool reset_hash) {
//  if(!IsTrainingThread(tid)) {
//    EPythonState state = GetPythonState(tid);
//    if (state == EPS_INVALID) {
//      return false;
//    }
//    assert(state == EPS_IN_TRAIN_LOOP);
//  }
//  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
//
//  compile_info->sync_count_since_hash_change_ = 0;
//  compile_info->run_count_at_last_mark_step_ = 0;
//  compile_info->mark_step_count_since_last_reset_ = 0;
//  compile_info->output_ids_.clear();
//  if (reset_hash) {
//    compile_info->set_hash(0);
//  }
//  // reset mark_step_, or that's the same thing as sync_count_since_hash_change_?
//  return true;
//}

bool CompileWatcher::IsQualifyingStep(pid_t tid/*, bool or_higher*/) {
  if (!is_in_mark_step) {
    return false;
  }
  if (!HasWseDevices() /*|| !IsTrainingThread(tid)*/) {
    return false;
  }
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  const std::size_t mark_step_count_since_reset = compile_info->mark_step_count_since_last_reset_.load(); 
  if (!mark_step_count_since_reset) {
    // The first step is superfluous since it's the top of the dataset interator loop,
    // before any graph is built.
    // This also takes care of disqualifying due to spurious compiles
    // within the train loop
    return false;
  }
#if 1
  const std::size_t steps_required = get_number_of_required_runs_since_reset();
//  const bool ready = !or_higher ?
//                     (mark_step_count_since_reset - 1 == steps_required)
//                     : (mark_step_count_since_reset - 1 >= steps_required);
  const bool ready = mark_step_count_since_reset - 1 == steps_required;
#else
  const std::size_t runs_since_compile = get_number_of_required_runs_since_reset();
  const std::size_t current_runs_since_compile_count = compile_info->sync_count_since_hash_change_.load();
  const bool ready = runs_since_compile == current_runs_since_compile_count;
  if (ready) {
    std::cout << "WseRunReady" << std::endl << std::flush;
  } else if (current_runs_since_compile_count < runs_since_compile) {
    std::cout << "Before compile window" << std::endl << std::flush;
  } else
  if (runs_since_compile > current_runs_since_compile_count) {
    std::cout << "Past compile window" << std::endl << std::flush;
  }
#endif
  if (ready) {
    assert(is_clean_step); // validate it coincides with clean step logic
    ColorScope clr({Color::BG_BLUE, Color::FG_YELLOW});
    std::cout << "Run ready" << std::endl << std::flush;
  }
  return ready;
}

//bool CompileWatcher::IsWseRunning(pid_t tid) {
//  if (!HasWseDevices() || !IsTrainingThread(tid)) {
//    return false;
//  }
//  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
//  return compile_info->sync_count_since_hash_change_ >= get_number_of_required_runs_since_reset();
//}

void CompileWatcher::SetOutputs(const std::vector<at::Tensor>& output_tensors, bool append) {
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
    const bool added = compile_info->output_ids_.insert(
        xla_tensor.data()->unique_id).second;
    assert(added);
  }
}

bool CompileWatcher::IsAllowedOutput(const XLATensor& tensor, XLATensor::SyncTensorCollection& coll) {
  assert(is_in_mark_step);  // gets cleared at end of step
  assert(is_clean_step); // otherwise, why are you asking?
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(coll.requesting_tid);
  if (compile_info->output_ids_.empty()) {
    return true;
  }
  return compile_info->output_ids_.find(tensor.data()->unique_id) !=
    compile_info->output_ids_.end();
}

void CompileWatcher::SetDeviceMapping(const std::string& from_device, const std::string& to_device) {
  std::lock_guard<std::mutex> lk(device_mapping_mtx_);
  assert(!from_device.empty());
  assert(from_device != to_device);
  if (to_device.empty()) {
    device_mapping_.erase(from_device);
  } else {
    device_mapping_[from_device] = std::make_pair(Device(to_device), true);
  }
}

std::mutex CompileWatcher::device_mapping_mtx_;
std::unordered_map<std::string, std::pair<Device, bool>> CompileWatcher::device_mapping_;

std::string CompileWatcher::GetDeviceMapping(const std::string& device) {
  assert(!device.empty());
  std::lock_guard<std::mutex> lk(device_mapping_mtx_);
  auto iter = device_mapping_.find(device);
  if (iter != device_mapping_.end() && iter->second.second /* enabled? */) {
    return iter->second.first.ToString();
  }
  return device;
}

const torch_xla::Device& CompileWatcher::GetDeviceMapping(const Device& device) {
  std::lock_guard<std::mutex> lk(device_mapping_mtx_);
  auto iter = device_mapping_.find(device.ToString());
  if (iter != device_mapping_.end() && iter->second.second /* enabled? */) {
    return iter->second.first;
  }
  return device;
}

}  // namespace torch_xla
