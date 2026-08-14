// lib/passes/cse.cpp - common-subexpression elimination for the atom IR (plan
// todo 13).
//
// Value-numbered CSE within block scopes: an op that is structurally
// identical to an earlier one reuses the earlier result.
//
// KEY: (op kind, per-op attributes, const payloads, operands, result types),
// built as a canonical string per op (see makeCseKey).
//
// CSE-ABLE (pure, side-effect-free, value is a pure function of operands):
//   const/add/sub/mul/div/fma/max/min/cmp/select/math/thread_id/block_id/
//   block_dim/grid_dim/reduce
//
// EXCLUDED:
//   load/load_vec - memory reads: an intervening store or barrier can change
//     the value, and proving no such interference is a memory analysis outside
//     this pass's scope.
//   alloc_shared/barrier/store/store_vec/copy/yield/loop_for - side effects
//     or structural ops.
//
// DOMINANCE: a scope reuses only values defined earlier in the same scope or
// an enclosing scope. Each nested scope gets a COPY of the parent's table at
// the point of the loop_for (values defined after the loop do not leak into
// the body), and the copy does not leak back. All uses of the duplicated
// result are rewritten to the canonical value, then the op is erased.
//
// Correct under the printer's shadow-avoiding value numbering: the pass
// rewrites Value handles structurally, never names, so a nested scope may
// reuse an outer %vN name with no effect on correctness.
//
// Self-registration contract (todos 14-18): see lib/passes/CMakeLists.txt.

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"

#include <cstdio>
#include <expected>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kernelh::ir {
namespace {

bool isCseable(OpKind kind) {
  switch (kind) {
  case OpKind::Const:
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Fma:
  case OpKind::Max:
  case OpKind::Min:
  case OpKind::Cmp:
  case OpKind::Select:
  case OpKind::Math:
  case OpKind::ThreadId:
  case OpKind::BlockId:
  case OpKind::BlockDim:
  case OpKind::GridDim:
  case OpKind::Reduce:
    return true;
  case OpKind::Load:
  case OpKind::Store:
  case OpKind::LoadVec:
  case OpKind::StoreVec:
  case OpKind::AllocShared:
  case OpKind::Barrier:
  case OpKind::LoopFor:
  case OpKind::Yield:
  case OpKind::Copy:
    return false;
  }
  return false;
}

// Stable identity of a Value for the CSE key (defining pointer address +
// result/arg index). Only compared within a single pass run.
std::string valueKey(const Value &value) {
  char buffer[64];
  switch (value.kind()) {
  case Value::Kind::None:
    return "none";
  case Value::Kind::OpResult:
    std::snprintf(buffer, sizeof(buffer), "o:%p:%lld",
                  static_cast<const void *>(value.op()),
                  static_cast<long long>(value.index()));
    return buffer;
  case Value::Kind::BlockArg:
    std::snprintf(buffer, sizeof(buffer), "b:%p:%lld",
                  static_cast<const void *>(value.block()),
                  static_cast<long long>(value.index()));
    return buffer;
  }
  return "none";
}

// Canonical key: kind | attrs | payloads | operand identities | result types.
// Two ops produce the same key iff they are interchangeable.
std::string makeCseKey(const Op &op) {
  std::string key;
  auto append = [&key](std::string_view piece) {
    key.append(piece);
    key.push_back('|');
  };
  append(std::to_string(static_cast<int>(op.kind)));
  append(std::to_string(static_cast<int>(op.pred)));
  append(std::to_string(static_cast<int>(op.mathKind)));
  append(std::to_string(static_cast<int>(op.reduceKind)));
  append(std::to_string(op.reduceDim));
  append(std::to_string(static_cast<int>(op.idDim)));
  append(op.intValue ? "i" + std::to_string(*op.intValue) : "-");
  append(op.floatValue ? "f" + std::to_string(*op.floatValue) : "-");
  for (const Value &operand : op.operands) {
    append(valueKey(operand));
  }
  for (const Type &type : op.resultTypes) {
    append(type.toString());
  }
  return key;
}

template <typename Fn>
void walkBlocks(Block &block, const Fn &fn) {
  fn(block);
  for (const auto &op : block.ops) {
    if (op->region != nullptr) {
      walkBlocks(*op->region, fn);
    }
  }
}

void rewriteAllUses(Kernel &kernel, const Value &from, const Value &to) {
  walkBlocks(kernel.body, [&](Block &block) {
    for (const auto &op : block.ops) {
      for (Value &operand : op->operands) {
        if (operand == from) {
          operand = to;
        }
      }
    }
  });
}

void cseBlock(Block &block, Kernel &kernel,
              std::map<std::string, Value> &table, bool &changed) {
  size_t i = 0;
  while (i < block.ops.size()) {
    Op &op = *block.ops[i];
    if (isCseable(op.kind)) {
      const std::string key = makeCseKey(op);
      const auto found = table.find(key);
      if (found != table.end()) {
        rewriteAllUses(kernel, op.result(), found->second);
        block.ops.erase(block.ops.begin() + i);
        changed = true;
        continue;
      }
      table.emplace(key, op.result());
    }
    if (op.region != nullptr) {
      // Nested scope: copy the parent's table at this point (outer values
      // dominate the body; values defined after the loop are absent). The
      // copy is discarded on return, so nothing leaks back.
      std::map<std::string, Value> nested = table;
      cseBlock(*op.region, kernel, nested, changed);
    }
    ++i;
  }
}

void cseKernel(Kernel &kernel) {
  bool changed = false;
  std::map<std::string, Value> table;
  cseBlock(kernel.body, kernel, table, changed);
}

} // namespace

std::expected<void, std::string> csePass(Module &module) {
  for (const auto &kernel : module.kernels) {
    cseKernel(*kernel);
  }
  return {};
}

// Self-registration (contract: lib/passes/CMakeLists.txt).
static const bool kCseRegistered =
    (kernelh::ir::registerPass("cse", csePass), true);

} // namespace kernelh::ir
