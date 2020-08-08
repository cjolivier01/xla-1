#pragma once

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"

#include <string>

namespace torch_xla {
namespace ir {

xla::XlaOp CustomLowerOp(const std::string& name,
                         const ir::Node *node,
                         const std::string& quark_type,
                         const std::string& quark_flavor,
                         std::unordered_map<std::string, std::string> config,
                         LoweringContext* loctx);

}  // namespace ir
}  // namespace torch_xla
