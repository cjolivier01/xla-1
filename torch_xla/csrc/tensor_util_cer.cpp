#include "torch_xla/csrc/tensor.h"
#include "tensorflow/core/util/util.h"
#include "torch_xla/csrc/tensor_util_cer.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

#include <string>
#include <stack>
#include <mutex>

namespace torch_xla {
std::atomic<size_t> active_parameter_count{0};

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

void XLATensor::print_tensor(const std::string& label, const XLATensor& tensor, bool assert) {
  print_tensor(label, tensor.data(), assert);
}

void XLATensor::print_tensor_ex(const std::string& label, const XLATensor& tensor, bool assert) {
  print_tensor_ex(label, tensor.data(), assert);
}

void XLATensor::print_tensor_ex(const std::string& label, const XLATensor::Data* data, bool assert) {
  std::cout << "[" << syscall(SYS_gettid) << "] ";
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

void XLATensor::print_tensor(const std::string& label, const XLATensor::Data* data, bool assert) {
  ColorScope color_scope(Color::FG_CYAN, true);
  print_tensor_ex(label, data, assert);
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
    XLATensor::print_tensor_ex(s, t.data());
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
  if (state == EPS_IN_DEBUG) {
    std::cout << ">>> ENTER DEBUG MODE" << std::endl << std::flush;
  }
  python_state.states.push(state);
}

EPythonState PopPythonState() {
  const EPythonState current_state = GetPythonState();
  if (current_state == EPS_IN_DEBUG) {
    std::cout << ">>> LEAVE DEBUG MODE" << std::endl << std::flush;
  }
  python_state.states.pop();
  return current_state;
}

namespace {

struct CompileInfo {
  CompileWatcher::compiler_t last_opaque_ = nullptr;
  std::atomic<size_t> run_count_{0};
  std::atomic<size_t> mark_step_{0};
  std::atomic<EPythonState> python_state_{EPS_INVALID};
  std::shared_ptr<std::vector<at::Tensor>> input_tensors_;
  std::shared_ptr<std::vector<at::Tensor>> output_tensors_;
  std::unique_ptr<CompileWatcher::hash_t> hash_;
  std::unordered_set<size_t> output_ids_;
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

const size_t RUNS_TILL_COMPILE = 3;

}  // namespace

void CompileWatcher::NotifyCompile(compiler_t opaque, hash_t hash) {
  if (!IsWseRunning(opaque)) {
    assert(opaque != nullptr);
    Reset(opaque);
    std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
    compile_info->python_state_ = GetPythonState();
    compile_info->hash_ = std::make_unique<CompileWatcher::hash_t>(hash);
  } else {
    std::cout << "COMPILING WHILE RUNNING" << std::endl << std::flush;
  }
}

void CompileWatcher::NotifyExecute(compiler_t opaque, hash_t hash) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  if (compile_info->output_ids_.empty()) {
    return;
  }
  if (opaque == compile_info->last_opaque_ &&
      compile_info->hash_.get() && hash == *compile_info->hash_) {
    //if (compile_info->python_state_ == EPS_IN_TRAIN_LOOP) {
      if (compile_info->run_count_++ == RUNS_TILL_COMPILE) {
        ColorScope clr(Color::FG_RED);
        // TODO: Should also have a check that everything required is available,
        //  like grads and whatnot in the live tensors.
        //  Maybe even inspect the proposed HLO graph for compatibility.
        std::cout << "**** ELIGIBLE FOR WSE COMPILE ****" << ENDL;
      }
    //}
  } else {
    Reset(opaque);
  }
}

void CompileWatcher::NotifyStepMarker(compiler_t opaque) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  if (!compile_info->output_ids_.empty()) {
    ++compile_info->mark_step_;
  }
}

void CompileWatcher::Reset(compiler_t opaque) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  assert(compile_info->last_opaque_ == nullptr || compile_info->last_opaque_ == opaque);
  compile_info->last_opaque_ = opaque;
  compile_info->run_count_ = 0;
  compile_info->python_state_ = EPS_INVALID;
  compile_info->input_tensors_.reset();
  compile_info->output_tensors_.reset();
  compile_info->hash_.reset();
  // reset mark_step_, or that's the same thing as run_count_?
}

// TODO: This should be based on the computation cache hash value
bool CompileWatcher::IsWseRunReady(compiler_t opaque, hash_t hash) {
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  return compile_info->run_count_ == RUNS_TILL_COMPILE &&
    //compile_info->python_state_ == EPS_IN_TRAIN_LOOP &&
      compile_info->hash_.get() &&
      *compile_info->hash_ == hash;
}

bool CompileWatcher::IsWseRunReady(compiler_t opaque) {
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  return compile_info->run_count_ == RUNS_TILL_COMPILE //&&
         //compile_info->python_state_ == EPS_IN_TRAIN_LOOP
         ;
}

bool CompileWatcher::IsWseRunning(compiler_t opaque) {
  const std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  return compile_info->run_count_ >= RUNS_TILL_COMPILE //&&
         //compile_info->python_state_ == EPS_IN_TRAIN_LOOP
         ;
}

void CompileWatcher::SetInputsOutputs(compiler_t opaque,
                                      const std::vector<at::Tensor>& input_tensors,
                                      const std::vector<at::Tensor>& output_tensors) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  compile_info->input_tensors_ = std::make_shared<std::vector<at::Tensor>>(input_tensors);
  compile_info->output_tensors_ = std::make_shared<std::vector<at::Tensor>>(output_tensors);

  for (const at::Tensor& tensor : *compile_info->input_tensors_) {
    XLATensor xla_tensor = bridge::GetXlaTensor(tensor);
    ColorScope color_scope(Color::FG_YELLOW, true);
    //XLATensor::print_tensor_ex("Input", xla_tensor);
  }
  compile_info->output_ids_.clear();
  for (const at::Tensor& tensor : *compile_info->output_tensors_) {
    XLATensor xla_tensor = bridge::GetXlaTensor(tensor);
    ColorScope color_scope(Color::FG_YELLOW, true);
    //XLATensor::print_tensor("Output", xla_tensor);
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
  //std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  HERE();
}

bool CompileWatcher::IsAllowedOutput(compiler_t opaque, XLATensor tensor) {
  std::shared_ptr<CompileInfo> compile_info = GetCompileInfo(opaque);
  if (!IsWseRunning(opaque)) {
    return true;
  }
  const bool found =  compile_info->output_ids_.find(tensor.data()->unique_id)
      != compile_info->output_ids_.end();
  return found;
}

}  // namespace torch_xla
