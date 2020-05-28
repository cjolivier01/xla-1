#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util_cer.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/core/util/util.h"

#include <string>
#include <stack>
#include <mutex>
#include <Python.h>

#include <pybind11/pybind11.h>

/**
 * Most of this can eventually move to monolith
 */
namespace torch_xla {
//std::atomic<size_t> active_parameter_count{0};

bool verbose = false;
const bool IGNORE_FIRST_MARK_STEP = true;

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
    std::cout << label << " (id=" << data->unique_id << ", type = " << data->tensor_type << ") "
              << " IR tensor of shape: " << data->ir_value.shape().ToString()
              << std::endl << std::flush;
    if (assert) {
      assert(!data->xla_data);
      assert(!data->view);
    }
  } else if (data->xla_data) {
    // coming from _xla_tensors_from_aten in at least one case
    std::cout << label << " (id=" << data->unique_id << ", type = " << data->tensor_type << ") "
              << " tensor with no ir_value of shape: "
              << data->xla_data->shape().ToString()
              << std::endl << std::flush;
    if (assert) {
      assert(!data->ir_value);
      assert(!data->view);
    }
  } else if(data->view) {
    std::cout << label << " (id=" << data->unique_id << ", type = " << data->tensor_type << ") "
              << " tensor with view of shape: "
              << data->view->shape().ToString()
              << std::endl << std::flush;
    if (assert) {
      assert(!data->ir_value);
      assert(!data->xla_data);
    }
  } else {
    std::cout << label << " (id=" << data->unique_id << ", type = " << data->tensor_type << ") "
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

struct CompileInfo {
  CompileWatcher::compiler_t last_opaque_ = nullptr;
  std::atomic<size_t> run_count_{0};
  std::atomic<size_t> mark_step_{0};
  std::atomic<size_t> mark_steps_since_reset_{0};
  std::atomic<EPythonState> python_state_{EPS_INVALID};
  //std::shared_ptr<std::vector<at::Tensor>> input_tensors_;
  //std::shared_ptr<std::vector<at::Tensor>> output_tensors_;
  std::unique_ptr<CompileWatcher::hash_t> hash_;
  std::unordered_set<size_t> output_ids_;

  std::shared_ptr<std::vector<XLATensor>> removed_outputs_;
  std::unordered_set<size_t> removed_output_ids_;

  void set_hash(CompileWatcher::hash_t hash) {
      std::cout << "Setting hash to: " << hash << std::endl << std::flush;
      hash_ = std::make_unique<CompileWatcher::hash_t>(hash);
  }
};

std::mutex compile_info_map_mtx_;
//std::map<CompileWatcher::compiler_t, std::shared_ptr<CompileInfo>> compile_info_map;
std::map<pid_t, std::shared_ptr<CompileInfo>> compile_info_map;

//std::shared_ptr<CompileInfo> GetCompileInfo(CompileWatcher::compiler_t opaque) {
std::shared_ptr<CompileInfo> GetCompileInfo(pid_t opaque) {
  std::lock_guard<std::mutex> lk(compile_info_map_mtx_);
  std::shared_ptr<CompileInfo> sp = compile_info_map[opaque];
  if (!sp) {
    sp = compile_info_map[opaque] = std::make_shared<CompileInfo>();
  }
  return std::move(sp);
}

//const size_t MARK_STEPS_TILL_COMPILE = 15;

size_t get_runs_till_compile() {
  const size_t DEFAULT_RUNS_TILL_COMPILE = 15;
  static size_t rtc =
    xla::sys_util::GetEnvInt(
      "XLA_RUNS_TILL_COMPILE",
      DEFAULT_RUNS_TILL_COMPILE
    );
  return rtc;
}

static std::mutex init_devices_mutex;
}  // namespace

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

bool CompileWatcher::PreProcessHlo(compiler_t opaque, xla::XlaBuilder *builder, pid_t tid) {
  if (!HasWseDevices() || !IsTrainingThread(tid) || !IsWseRunReady(opaque, tid)) {
    return false;
  }
  xla::FrontendAttributes frontend_attributes;
  frontend_attributes.CopyFrom(builder->frontend_attributes());
  (*frontend_attributes.mutable_map())["PROXY_DEVICE"] = GetDevice().ToString();
  builder->SetFrontendAttributes(frontend_attributes);
  return true;
}

void CompileWatcher::SetDeviceProxyAddress(
  const std::string& device, const std::string& proxy_address) {
  xla::ComputationClient *cc = xla::XrtComputationClient::Get();
  xla::XrtComputationClientWse *computation_client =
    dynamic_cast<xla::XrtComputationClientWse *>(cc);
  if (computation_client) {
    computation_client->SetDeviceProxyAddress(device, proxy_address);
  } else {
    throw std::runtime_error("Device proxying is not enabled");
  }
  HEREX();
}

bool CompileWatcher::HasWseDevices() {
  static bool got_devices = false;
  if (!got_devices) {
    std::lock_guard<std::mutex> lk(init_devices_mutex);
    if (!got_devices) {
      SetAllDevices(xla::XrtComputationClient::Get()->GetAllDevices());
    }
  }
  return !wse_devices_.empty();
}

bool CompileWatcher::IsTrainingThread(pid_t tid) {
  return GetPythonState(tid) == EPS_IN_TRAIN_LOOP;
}

void CompileWatcher::NotifyCompile(
    compiler_t opaque,
    std::vector<xla::ComputationClient::CompileInstance>& instances,
    hash_t hash,
    pid_t tid
) {
  if (!HasWseDevices() || !IsTrainingThread(tid)) {
    return;
  }
  HEREX();
  if (IsWseRunReady(opaque, tid)) {
//      {
//          ColorScope clr(Color::FG_GREEN);
//          std::cout << "SET FOR WSE COMPILE" << std::endl << std::flush;
//      }
//      const std::string wse_device = GetDevice();
//      assert(instances.size() == 1);
//      std::vector<std::string>& devices = instances[0].devices;
//      assert(std::find(devices.begin(), devices.end(), wse_device) == devices.end());
//      assert(devices.size() == 1);
//      devices.push_back(GetDevice());
  } else if (!IsWseRunning(opaque, tid)) {
    std::cout << "Compiling hash=" << hash << ENDL;
    assert(opaque != nullptr);
    Reset(opaque, tid, true);
    std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
    compile_info->python_state_ = GetPythonState(tid);
    compile_info->set_hash(hash);
  } else {
    std::cout << "COMPILING WHILE RUNNING" << std::endl << std::flush;
  }
}

void CompileWatcher::NotifyExecute(compiler_t opaque, std::string& device, hash_t hash, pid_t tid) {
  if (!HasWseDevices() || !IsTrainingThread(tid)) {
    return;
  }
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  bool new_hash = false;
  if (!compile_info->hash_.get()) {
      compile_info->set_hash(hash);
      new_hash = true;
  }
  if (opaque == compile_info->last_opaque_ &&
      compile_info->hash_.get() && hash == *compile_info->hash_) {
    // Here we can determine if more runs than steps are occuring
    if (compile_info->run_count_++ == get_runs_till_compile()) {
      if (compile_info->run_count_ <= compile_info->mark_steps_since_reset_) {
        ColorScope clr(Color::FG_GREEN);
        // TODO: Should also have a check that everything required is available,
        //  like grads and whatnot in the live tensors.
        //  Maybe even inspect the proposed HLO graph for compatibility.
        std::cout << "**** ELIGIBLE FOR WSE COMPILE ****"
                  << ", hash=" << hash << ", device=" << device << ENDL;
      } else {
        // THIS COULD BE ASYNC
//              std::cout << "TOO MANY RUNS PER STEP: " << compile_info->run_count_
//                        << ", hash=" << hash << ", device=" << device << std::endl << std::flush;
      }
    } else {
//          if (!IsWseRunning(opaque)) {
//              std::cout << "REPEAT RUN " << compile_info->run_count_
//                        << ", hash=" << hash << std::endl << std::flush;
//          }
    }
  } else if(IsWseRunReady(opaque, tid)) {
    // Set new hash since this is WSE version of the same graph
    ColorScope clr(Color::FG_BLUE);
    std::cout << "Resetting hash to WSE's compile" << std::endl << std::flush;
    compile_info->set_hash(hash);
  } else {
    ColorScope clr(Color::FG_RED);
    if (!compile_info->hash_.get()) {
      // TODO: need to recognize ineligible (i.e. dataset fetch) vis python scope check
      std::cout << "No hash" << std::endl;
    }
    if (Reset(opaque, !new_hash, tid)) {
      std::cout << "RESETTING EXECUTION COUNTER FROM " << compile_info->run_count_.load()
                << ", hash " << *compile_info->hash_ << " -> " << hash
                << ", device=" << device << std::endl << std::flush;
    }
  }
}

static __thread std::atomic<unsigned long long> total_mark_steps{0};

void CompileWatcher::NotifyStepMarker(compiler_t opaque, const std::vector<std::string>& devices) {
  const pid_t tid = gettid();
//  assert(IsTrainingThread(tid));
//  if (!IsTrainingThread(tid)) {
//    return;
//  }
  if (++total_mark_steps == 1 && IGNORE_FIRST_MARK_STEP) {
    return;
  }
  if (!IsTrainingThread(tid)) {
    assert(GetPythonState(tid) == EPS_INVALID);
    // The assumption is that only the training thread can call _XLAC._mark_step()
    _PushPythonState(EPS_IN_TRAIN_LOOP, tid);
  }
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  //if (!compile_info->output_ids_.empty()) {
    const size_t total_steps = ++compile_info->mark_step_;
    const size_t steps_since_reset = ++compile_info->mark_steps_since_reset_;
    ColorScope clr(IsWseRunning(opaque, gettid()) ? Color::FG_YELLOW : Color::FG_WHITE);
    //std::cout << "Mark step: " << steps_since_reset << "/" << total_steps << std::endl << std::flush;
  //}
}

Device CompileWatcher::GetDevice() {
    if (HasWseDevices()) {
      return Device(*wse_devices_.begin());
    }
    return Device(DeviceType::CPU, 0);
}

bool CompileWatcher::Reset(compiler_t opaque, pid_t tid, bool reset_hash) {
  if(!IsTrainingThread(tid)) {
    EPythonState state = GetPythonState(tid);
    if (state == EPS_INVALID) {
      return false;
    }
    assert(state == EPS_IN_TRAIN_LOOP);
  }
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  assert(compile_info->last_opaque_ == nullptr || compile_info->last_opaque_ == opaque);
  compile_info->last_opaque_ = opaque;
  compile_info->run_count_ = 0;
  compile_info->mark_steps_since_reset_ = 0;
  compile_info->python_state_ = EPS_INVALID;
  //compile_info->input_tensors_.reset();
  //compile_info->output_tensors_.reset();
  if (reset_hash) {
    compile_info->hash_.reset();
  }
  // reset mark_step_, or that's the same thing as run_count_?
  return true;
}

// TODO: This should be based on the computation cache hash value
bool CompileWatcher::IsWseRunReady(compiler_t opaque, hash_t hash, pid_t tid) {
  if (!HasWseDevices() || !IsTrainingThread(tid)) {
    return false;
  }
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  return compile_info->run_count_ == get_runs_till_compile() &&
      compile_info->hash_.get() &&
      *compile_info->hash_ == hash;
}

bool CompileWatcher::IsWseRunReady(compiler_t opaque, pid_t tid) {
  if (!HasWseDevices() || !IsTrainingThread(tid)) {
    return false;
  }
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  const bool ready = compile_info->run_count_ == get_runs_till_compile();
  if (ready) {
    std::cout << "WseRunReady" << std::endl << std::flush;
  }
  return ready;
}

bool CompileWatcher::IsWseRunning(compiler_t opaque, pid_t tid) {
  if (!HasWseDevices() || !IsTrainingThread(tid)) {
    return false;
  }
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  return compile_info->run_count_ >= get_runs_till_compile();
}

void CompileWatcher::SetInputsOutputs(compiler_t opaque,
                                      const std::vector<at::Tensor>& input_tensors,
                                      const std::vector<at::Tensor>& output_tensors,
                                      bool append) {
  if (!HasWseDevices()) {
    return;
  }
  const pid_t tid = gettid();
  assert(IsTrainingThread(tid));
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
//  if (!append || !compile_info->input_tensors_) {
//      compile_info->input_tensors_ = std::make_shared<std::vector<at::Tensor>>(input_tensors);
//  }
//  if (!append || !compile_info->output_tensors_) {
//      compile_info->output_tensors_ = std::make_shared<std::vector<at::Tensor>>(output_tensors);
//  }

  if (verbose) {
      for (const at::Tensor &tensor : input_tensors) {
          XLATensor xla_tensor = bridge::GetXlaTensor(tensor);
          ColorScope color_scope(Color::FG_YELLOW, true);
          XLATensor::print_tensor_ex("Input", xla_tensor);
      }
  }
  if (!append) {
    compile_info->output_ids_.clear();
  }
  for (const at::Tensor& tensor : output_tensors) {
    XLATensor xla_tensor = bridge::GetXlaTensor(tensor);
    if (verbose) {
        ColorScope color_scope(Color::FG_YELLOW, true);
        XLATensor::print_tensor("Output", xla_tensor);
    }
    const bool added = compile_info->output_ids_.insert(
        xla_tensor.data()->unique_id).second;
    assert(added);
  }
}

//std::vector<xla::ComputationClient::DataPtr> CompileWatcher::WseExecute(
//    compiler_t opaque,
//    hash_t hash,
//    std::shared_ptr<XLATensor::Async> async) {
//}

void CompileWatcher::ResetConsideredSyncOutputs(compiler_t opaque, pid_t tid) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  compile_info->removed_outputs_.reset();
}

bool CompileWatcher::IsAllowedOutput(compiler_t opaque, XLATensor tensor, pid_t tid) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(tid);
  if (!IsWseRunning(opaque, tid)) {
    return true;
  }
  if (compile_info->output_ids_.empty()) {
    return true;  // they haven't specified, so try everything
  }
  const bool found =  compile_info->output_ids_.find(tensor.data()->unique_id)
      != compile_info->output_ids_.end();
  // TODO: Ensure that the directly-ensuing compile is this set of input/outputs
  return found;
}

}  // namespace torch_xla
