#include "torch_xla/csrc/lower_custom.h"

#include <sstream>
#include <unordered_map>

namespace torch_xla {
namespace ir {

namespace {

class FrontendAttributeScope {
 public:
  FrontendAttributeScope(
      xla::XlaBuilder* builder, const std::string& op_name,
      const std::string op_type,
      const std::unordered_map<std::string, std::string>& attributes)
      : builder_(builder) {
    if (!attributes.empty()) {
      set_ = true;
      xla::FrontendAttributes frontend_attributes;
      frontend_attributes.CopyFrom(builder_->frontend_attributes());
      for (const auto& item : attributes) {
        (*frontend_attributes.mutable_map())[item.first] = item.second;
      }
      save_ = builder->SwapFrontendAttributes(frontend_attributes);
      // metadata_ = std::move(builder->metadata_);
      xla::OpMetadata metadata = metadata_;
      metadata.set_op_name(op_name);
      metadata.set_op_type(op_type);
      metadata.set_source_file(as_tf1_source_string(attributes));
      builder->SetOpMetadata(std::move(metadata));
    }
  }
  ~FrontendAttributeScope() {
    if (set_) {
      builder_->ClearOpMetadata();
      builder_->SetFrontendAttributes(save_);
    }
  }
  std::string Dump() {
    std::stringstream ss;
    for (const auto& item : builder_->frontend_attributes().map()) {
      ss << item.first << " -> " << item.second << ", ";
    }
    return ss.str();
  }

 private:
  static std::string as_tf1_source_string(
      const std::unordered_map<std::string, std::string>& attributes) {
    std::stringstream ss;
    std::size_t x = 0;
    for (const auto& item : attributes) {
      if (x++) ss << "|";
      ss << item.first << ";" << item.second;
    }
    return std::move(ss.str());
  }
  xla::XlaBuilder* builder_;
  xla::FrontendAttributes save_;
  xla::OpMetadata metadata_;
  bool set_ = false;
};

}  // anonymous namespace

xla::XlaOp CustomLowerOp(const std::string& name, const ir::Node* node,
                         const std::string& quark_type,
                         const std::string& quark_flavor,
                         std::unordered_map<std::string, std::string> config,
                         LoweringContext* loctx) {
  const std::vector<Output>& operands = node->operands();
  std::vector<xla::XlaOp> input_ops;
  input_ops.reserve(operands.size());
  for (const ir::Output& input : operands) {
    xla::XlaOp input_op = loctx->GetOutputOp(input);
    input_ops.emplace_back(input_op);
  }

  std::stringstream ss;
  if (!node->metadata().scope.empty()) {
    ss << node->metadata().scope;
    ss << ".";
  }
  ss << name;

  xla::HloInstructionProto instr;
  *instr.mutable_shape() = node->shape().ToProto();
  *instr.mutable_name() = ss.str();

  xla::HloOpcode opcode;
  switch (input_ops.size()) {
    case 1:
      opcode = xla::HloOpcode::kQuarkUnary;
      break;
    case 2:
      opcode = xla::HloOpcode::kQuarkBinary;
      break;
    case 3:
      opcode = xla::HloOpcode::kQuarkTernary;
      break;
    default:
      throw std::runtime_error("Invalid number of inputs for quark operation");
  }
  assert(config.count("quark_type") == 0);
  assert(config.count("quark_flavor") == 0);
  config["quark_type"] = quark_type;
  config["quark_flavor"] = quark_flavor;
  const ir::FrontendAttributeScope fa(loctx->builder(), name, quark_type,
                                      std::move(config));
  return loctx->builder()
      ->AddInstructionEx(std::move(instr), opcode, input_ops)
      .ValueOrDie();
}

}  // namespace ir
}  // namespace torch_xla
