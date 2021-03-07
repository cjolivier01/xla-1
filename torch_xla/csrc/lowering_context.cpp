#include "torch_xla/csrc/lowering_context.h"

#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/python_util.h"
#include "ATen/pytorch_scope.h"

namespace torch_xla {
namespace ir {
namespace {

class HloMetadataSetter {
 public:
  HloMetadataSetter(LoweringContext* loctx, const Node* node) {
    if (ShouldPopulateXlaOpMetadata()) {
      PopulateXlaOpMetadata(loctx, node);
      loctx_ = loctx;
    }
  }

  ~HloMetadataSetter() {
    if (loctx_ != nullptr) {
      loctx_->builder()->ClearOpMetadata();
    }
  }

 private:
  static bool ShouldPopulateXlaOpMetadata() {
    static bool op_metadata = xla::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
    return op_metadata;
  }

  static void PopulateXlaOpMetadata(LoweringContext* loctx, const Node* node) {
    xla::OpMetadata metadata;
    // NOTE: we apply some string manipulation as xprof backend utility
    // for nesting/grouping traces depends on certain op name/type
    // patterns for classification.
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/utils/tf_op_utils.cc#L55
    std::string op_type =
        absl::StrReplaceAll(node->op().ToString(), {{":", "_"}});
    metadata.set_op_type(op_type);
    const ir::MetaData& nmeta = node->metadata();
    std::string op_name_prefix;
    if (!nmeta.scope.empty()) {
      op_name_prefix =
          absl::StrCat(absl::StrReplaceAll(nmeta.scope, {{":", "_"}}), "/");
    }
    metadata.set_op_name(absl::StrCat(op_name_prefix, op_type));

    if (!nmeta.frame_info.empty()) {
      const SourceLocation& frame = nmeta.frame_info.front();
      std::string::size_type pos = frame.file.find_last_of('/');
      if (pos == std::string::npos) {
        pos = 0;
      } else {
        ++pos;
      }
      metadata.set_source_file(frame.function + "@" + frame.file.substr(pos));
      metadata.set_source_line(frame.line);
    }
    loctx->builder()->SetOpMetadata(std::move(metadata));
  }

  LoweringContext* loctx_ = nullptr;
};

}  // namespace

LoweringContext::LoweringContext(const std::string& name, Device device)
    : builder_(name), device_(std::move(device)) {}

LoweringContext::LoweringContext(const std::string& name, Device device,
                                 absl::Span<const Node* const> post_order,
                                 Util::EmissionMap emit_status)
    : builder_(name),
      device_(std::move(device)),
      emit_status_(std::move(emit_status)) {
  LinkAutogradNodes(post_order);
  for (auto node : post_order) {
    LowerNode(node);
  }
}

void LoweringContext::LinkAutogradNodes(absl::Span<const Node* const> post_order) {
  autograd_nodes_.clear();
  fwd_nodes_.clear();
  int64_t pytorch_grad_fn_seq_nr = -1;
  for (auto node : post_order) {
    if (!node->IsAutograd(&pytorch_grad_fn_seq_nr)) {
      autograd_nodes_[pytorch_grad_fn_seq_nr].emplace(node);
    } else {
      fwd_nodes_[pytorch_grad_fn_seq_nr].emplace(node);
    }
  }
}

xla::XlaOp LoweringContext::GetParameter(
    const std::shared_ptr<xla::ComputationClient::Data>& data) {
  xla::ComputationClient::Data::OpaqueHandle handle = data->GetOpaqueHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    xla::XlaOp param =
        xla::Parameter(builder(), parameters_.size(), data->shape(),
                       absl::StrCat("p", parameters_.size()));
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(data);
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

const std::vector<xla::ComputationClient::DataPtr>&
LoweringContext::GetParametersData() const {
  return parameters_;
}

const std::vector<size_t>& LoweringContext::GetParameterSequence() const {
  return parameter_sequence_;
}

size_t LoweringContext::AddResult(xla::XlaOp op) {
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

xla::XlaOp LoweringContext::GetResult(size_t index) const {
  return root_tuple_.at(index);
}

void LoweringContext::SetResult(size_t index, xla::XlaOp op) {
  root_tuple_.at(index) = std::move(op);
}

xla::StatusOr<xla::XlaComputation> LoweringContext::Build() {
  if (!root_tuple_.empty()) {
    xla::XlaOp root = xla::Tuple(builder(), root_tuple_);
    return builder()->Build(root);
  }
  return builder()->Build();
}

xla::StatusOr<xla::XlaComputation> LoweringContext::Build(xla::XlaOp root) {
  XLA_CHECK(root_tuple_.empty());
  return builder()->Build(root);
}

void LoweringContext::AssignOutputOp(const Output& output, xla::XlaOp op) {
  emitted_outputs_[output] = std::move(op);
}

xla::XlaOp LoweringContext::GetOutputOp(const Output& output) {
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
    LinkAutogradNodes(post_order);
    for (auto node : post_order) {
      LowerNode(node);
    }
    // At this point the output better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    XLA_CHECK(it != emitted_outputs_.end())
        << "No XLA operation emitted for output: " << output;
  }
  return it->second;
}


xla::XlaOp LoweringContext::GetOutputOp(
    const Output& output, const std::vector<ir::Value>& boundaries) {
  std::unordered_set<const ir::Node*> boundary_set;
  std::for_each(boundaries.begin(), boundaries.end(), [&boundary_set](auto& p){
    boundary_set.emplace(p.node.get());
  });
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
    LinkAutogradNodes(post_order);
    for (auto node : post_order) {
      LowerNode(node);
    }
    // At this point the outpout better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    XLA_CHECK(it != emitted_outputs_.end())
        << "No XLA operation emitted for output: " << output;
  }
  return it->second;
}

static bool replace(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = str.find(from);
  if(start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

/*static*/ void SetAsBackwardNodeOf(
    std::unordered_map<std::string, std::string>& bwd_fattr,
    const std::string& this_node_key,
    const Node& fwd_node
) {
  const auto& fwd_fattr = fwd_node.metadata().frontend_attributes;
  std::size_t count = 0;
  for (const auto& item : fwd_fattr) {
    const std::string& fwd_key = item.first;
    auto found_pos = fwd_key.find(std::string(pytorch_ptwse::PartitionScope::MATCHED_OP));
    if (found_pos != std::string::npos) {
      ++count;

      std::string rev_ref_key = fwd_key;
      if (replace(rev_ref_key, pytorch_ptwse::PartitionScope::MATCHED_OP, pytorch_ptwse::PartitionScope::REVERSE_OF_OP)) {
        //assert(bwd_fattr.count(rev_ref_key) == 0);
        if (bwd_fattr.count(rev_ref_key)) {
          // Ugh what to do here
          bwd_fattr[rev_ref_key] += "," + fwd_key;
        } else {
          bwd_fattr.emplace(std::move(rev_ref_key), fwd_key);
        }
      }
    }
  }
}

XlaOpVector LoweringContext::LowerNode(const Node* node) {
  XlaOpVector result_ops;
  try {
    HloMetadataSetter meta_setter(this, node);
    // Attempt to linke backward nodes to their forward nodes
    std::unordered_map<std::string, std::string> extra_attributes;
#if 0
    int64_t seq_nr;
    if (node->IsAutograd(&seq_nr) && seq_nr >= 0) {
      // Get list of nodes this is bwd for
      auto found_fwd_nodes_iter = fwd_nodes_.find(seq_nr);
      if (found_fwd_nodes_iter != fwd_nodes_.end()) {
        for (const Node *fwd : found_fwd_nodes_iter->second) {
          // We want the highest scope, right?

          for (const auto& bwd_fattr : node->metadata().frontend_attributes) {
            const std::string& bwd_match_op_key = bwd_fattr.first;
            auto found_pos = bwd_match_op_key.find(std::string(pytorch_ptwse::PartitionScope::MATCHED_OP));
            if (found_pos != std::string::npos) {
              SetAsBackwardNodeOf(extra_attributes, bwd_match_op_key, *fwd);
            }
          }

          // SetAsBackwardNodeOf(extra_attributes, this_node_key, *fwd, *node);

//          std::map<int, std::string> scope_keys;
//          for (const auto& item : fwd->metadata().frontend_attributes) {
//            const std::string& fwd_key = item.first;
//            if (strstr(fwd_key.c_str(), ".MATCHED_OP.")) {
//              auto parts = absl::StrSplit(fwd_key, '.');
//              const uint64_t scope = std::atol(std::string(*parts.begin()).c_str());
//              // Should be no way they got merged
//              assert(!scope_keys.count(scope));
//              scope_keys.emplace(scope, item.second);
//            }
//          }
//          if (!scope_keys.empty()) {
//            const std::string& fwd_key = scope_keys.rbegin()->second;
//            assert(!fwd_key.empty());
//            std::string new_key = fwd_key;
//            const std::string to_replace = "MATCHED_OP";
//            new_key.replace(new_key.find(to_replace.c_str()), to_replace.length(), "REVERSE_OF_OP");
//            // This is wrong but lets debug
//            extra_attributes.emplace(new_key, scope_keys.rbegin()->second);
//          }
        }
      }
    }
#endif
    pytorch_ptwse::FrontendAttributeSetter<ir::Node> frontend_attribute_scope_(
        builder(), node->metadata().frontend_attributes, std::move(extra_attributes));
    result_ops = node->Lower(this);
  } catch (const std::exception& ex) {
    ReportBuilderError(node, ex.what());
  }
  if (!builder()->first_error().ok()) {
    ReportBuilderError(node, /*error_msg=*/nullptr);
  }
  return result_ops;
}

void LoweringContext::ReportBuilderError(const Node* node,
                                         const char* error_msg) {
  std::stringstream ss;
  ss << "Error while lowering: " << node->ToString() << "\n";
  if (!builder()->first_error().ok()) {
    ss << "XLA builder error: " << builder()->GetCurrentStatus() << "\n";
  }
  if (error_msg != nullptr) {
    ss << "Error: " << error_msg << "\n";
  }
  const ir::MetaData& nmeta = node->metadata();
  if (!nmeta.scope.empty()) {
    ss << "Scope: " << nmeta.scope << "\n";
  }
  ss << nmeta.frame_info;
  throw std::runtime_error(ss.str());
}

}  // namespace ir
}  // namespace torch_xla
