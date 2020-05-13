#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util_cer.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

#include "tensorflow/compiler/xla/xla_client/xrt_computation_client_wse.h"
#include "tensorflow/core/util/util.h"

#include <string>
#include <stack>
#include <mutex>

/**
 * Most of this can eventually move to monolith
 */
namespace torch_xla {
std::atomic<size_t> active_parameter_count{0};

bool verbose = false;

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
};

static thread_local SPythonState python_state;

EPythonState GetPythonState() {
  return python_state.states.empty() ? EPS_INVALID : python_state.states.top();
}

void PushPythonState(EPythonState state) {
  python_state.states.push(state);
}

EPythonState PopPythonState() {
  const EPythonState current_state = GetPythonState();
  python_state.states.pop();
  return current_state;
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
      std::cout << "Setting has to: " << hash << std::endl << std::flush;
      hash_ = std::make_unique<CompileWatcher::hash_t>(hash);
  }
};

std::mutex compile_info_map_mtx_;
std::map<CompileWatcher::compiler_t, std::shared_ptr<CompileInfo>> compile_info_map;

std::shared_ptr<CompileInfo> GetCompileInfo(CompileWatcher::compiler_t opaque) {
  std::lock_guard<std::mutex> lk(compile_info_map_mtx_);
  std::shared_ptr<CompileInfo> sp = compile_info_map[opaque];
  if (!sp) {
    sp = compile_info_map[opaque] = std::make_shared<CompileInfo>();
  }
  return std::move(sp);
}

const size_t MARK_STEPS_TILL_COMPILE = 3;
const size_t RUNS_TILL_COMPILE = 3;

}  // namespace

void CompileWatcher::SetLiveInterface(std::shared_ptr<xla::ptxla::XrtComputationClientExternalInterface> interface) {
    xla::XrtComputationClientWse::SetExternalInterface(interface);
}

void CompileWatcher::NotifyCompile(
    compiler_t opaque,
    std::vector<xla::ComputationClient::CompileInstance>& instances,
    hash_t hash
) {
  if (IsWseRunReady(opaque)) {
      {
          ColorScope clr(Color::FG_GREEN);
          std::cout << "SET FOR WSE COMPILE" << std::endl << std::flush;
      }
      const std::string wse_device = GetDevice();
      assert(instances.size() == 1);
      std::vector<std::string>& devices = instances[0].devices;
      assert(std::find(devices.begin(), devices.end(), wse_device) == devices.end());
      assert(devices.size() == 1);
      devices[0] = GetDevice();
  } else if (!IsWseRunning(opaque)) {
    std::cout << "Compiling hash=" << hash << ENDL;
    assert(opaque != nullptr);
    Reset(opaque);
    std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
    compile_info->python_state_ = GetPythonState();
    compile_info->set_hash(hash);
  } else {
    std::cout << "COMPILING WHILE RUNNING" << std::endl << std::flush;
  }
}

void CompileWatcher::NotifyExecute(compiler_t opaque, std::string& device, hash_t hash) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  bool new_hash = false;
  if (!compile_info->hash_.get()) {
      compile_info->set_hash(hash);
      new_hash = true;
  }
  if (opaque == compile_info->last_opaque_ &&
      compile_info->hash_.get() && hash == *compile_info->hash_) {
      // Here we can determine if more runs than steps are occuring
      if (compile_info->run_count_++ == RUNS_TILL_COMPILE) {
          if (compile_info->run_count_ <= compile_info->mark_steps_since_reset_) {
              ColorScope clr(Color::FG_RED);
              // TODO: Should also have a check that everything required is available,
              //  like grads and whatnot in the live tensors.
              //  Maybe even inspect the proposed HLO graph for compatibility.
              std::cout << "**** ELIGIBLE FOR WSE COMPILE ****"
                        << ", hash=" << hash << ", device=" << device << ENDL;
          } else {
              std::cout << "TOO MANY RUNS PER STEP: " << compile_info->run_count_
                        << ", hash=" << hash << ", device=" << device << std::endl << std::flush;
          }
      } else {
//          if (!IsWseRunning(opaque)) {
//              std::cout << "REPEAT RUN " << compile_info->run_count_
//                        << ", hash=" << hash << std::endl << std::flush;
//          }
      }
  } else {
    if (!compile_info->hash_.get()) {
      // TODO: need to recognize ineligible (i.e. dataset fetch) vis python scope check
      std::cout << "No hash" << std::endl;
    }
    std::cout << "RESETTING EXECUTION COUNTER FROM " << compile_info->run_count_.load()
              << ", hash " << *compile_info->hash_ << " -> " << hash
              << ", device=" << device << std::endl << std::flush;
    Reset(opaque, !new_hash);
  }
}

void CompileWatcher::NotifyStepMarker(compiler_t opaque, const std::vector<std::string>& devices) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  //if (!compile_info->output_ids_.empty()) {
    const size_t total_steps = ++compile_info->mark_step_;
    const size_t steps_since_reset = ++compile_info->mark_steps_since_reset_;
    ColorScope clr(IsWseRunning(opaque) ? Color::FG_YELLOW : Color::FG_WHITE);
    std::cout << "Mark step: " << steps_since_reset << "/" << total_steps << std::endl << std::flush;
  //}
}

std::string CompileWatcher::GetDevice() {
    return "CPU:1";  // TODO: work out code path for unrecognized device type
}

void CompileWatcher::Reset(compiler_t opaque, bool reset_hash) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
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
}

// TODO: This should be based on the computation cache hash value
bool CompileWatcher::IsWseRunReady(compiler_t opaque, hash_t hash) {
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  return compile_info->run_count_ == RUNS_TILL_COMPILE &&
      compile_info->hash_.get() &&
      *compile_info->hash_ == hash;
}

bool CompileWatcher::IsWseRunReady(compiler_t opaque) {
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  return compile_info->run_count_ == RUNS_TILL_COMPILE;
}

bool CompileWatcher::IsWseRunning(compiler_t opaque) {
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  return compile_info->run_count_ >= RUNS_TILL_COMPILE;
}

void CompileWatcher::SetInputsOutputs(compiler_t opaque,
                                      const std::vector<at::Tensor>& input_tensors,
                                      const std::vector<at::Tensor>& output_tensors,
                                      bool append) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
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

void CompileWatcher::CompileCacheHit(hash_t hash) {

}

void CompileWatcher::CompileCacheMiss(hash_t hash) {

}

std::vector<xla::ComputationClient::DataPtr> CompileWatcher::WseExecute(
    compiler_t opaque,
    hash_t hash,
    std::shared_ptr<XLATensor::Async> async) {
}

void CompileWatcher::ResetConsideredSyncOutputs(compiler_t opaque) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  compile_info->removed_outputs_.reset();
}

bool CompileWatcher::IsAllowedOutput(compiler_t opaque, XLATensor tensor) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  if (!IsWseRunning(opaque)) {
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
