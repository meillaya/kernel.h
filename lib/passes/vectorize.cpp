// lib/passes/vectorize.cpp - the vectorize pass for the atom IR (plan todo
// 17).
//
// When the kernel's schedule attr `vector_width` = N (> 1), the pass rewrites
// contiguous atom.load / atom.store on global or shared tensors into
// atom.load_vec / atom.store_vec of width N (VectorType vector<NxE>, the
// todo-9 core inventory ops) and widens the pointwise compute between them.
// `vector_width` = 1 (the default) makes the pass a no-op.
//
// DIVISIBILITY CHECK (pinned by the plan): every tile extent must be
// divisible by N, else the pass returns a std::expected diagnostic naming the
// offending extent. The check runs BEFORE any rewriting, so a rejected kernel
// is left untouched.
//
// WIDENING RULE (fixpoint). A value's type is widened to vector<NxE> iff:
//   - its defining op is a load on a global/shared tensor of rank >= 1 (the
//     vector covers N contiguous elements of the last dim), or an elementwise
//     compute op whose operands are ALL widened values of ONE type
//     (add/sub/mul/div/max/min/fma/select; select's i32 condition stays
//     scalar), AND
//   - EVERY use of the value can legally consume a vector: a store (becomes
//     store_vec), or a compute op that itself widens (all of ITS operands
//     widen). Anything else - atom.cmp, atom.math, atom.reduce, atom.copy,
//     loop_for iter_args, yield - blocks widening, and the whole chain stays
//     scalar.
// The set is computed as a greatest-fixpoint by iteratively removing
// candidates that violate the rules (monotone, order-independent), so no
// mixed scalar/vector op can survive in the output.
//
// DOCUMENTED LIMITATION (the IR has no vector splat const - atom.const
// results are scalar or tensor only, per the verifier): a widened value can
// never meet a scalar operand. A chain like add(load, const) therefore stays
// entirely scalar; only pure load->compute->store chains over the same
// vectorizable domain are widened. atom.math has no vector form in this IR
// either and is never widened (the verifier's element-type extraction is
// scalar-only). Both cases degrade gracefully: the chain remains scalar and
// verifier-clean instead of producing mixed-typed ops.
//
// Self-registration contract (todos 14-18): see lib/passes/CMakeLists.txt.

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"

#include <cstdint>
#include <expected>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kernelh::ir {
namespace {

template <typename Fn>
void walkBlocks(Block &block, const Fn &fn) {
  fn(block);
  for (const auto &op : block.ops) {
    if (op->region != nullptr) {
      walkBlocks(*op->region, fn);
    }
  }
}

// True when the load can widen: a load on a global/shared tensor of rank >= 1
// (the vector covers N contiguous elements of the last dim).
bool isWidenableLoad(const Op &op) {
  if (op.kind != OpKind::Load || op.operands.empty() ||
      !op.operands[0].type().isTensor()) {
    return false;
  }
  const TensorType tensor = op.operands[0].type().asTensor();
  return tensor.rank() >= 1 && (tensor.space == MemSpace::Global ||
                                tensor.space == MemSpace::Shared);
}

// Elementwise compute ops that have a vector form in this IR (the verifier's
// "operands and result must share one type" rule admits any type, vector
// included). atom.math / atom.cmp are NOT here on purpose (see the limitation
// note above).
bool isWidenableCompute(OpKind kind) {
  switch (kind) {
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Max:
  case OpKind::Min:
  case OpKind::Fma:
  case OpKind::Select:
    return true;
  default:
    return false;
  }
}

// Every use of every op result: defining op -> list of using ops. Ops live in
// unique_ptr trees and are never moved during analysis, so raw pointers stay
// stable.
using UseMap = std::unordered_map<const Op *, std::vector<const Op *>>;

void collectUses(Block &block, UseMap &uses) {
  for (const auto &op : block.ops) {
    for (const Value &operand : op->operands) {
      if (operand.kind() == Value::Kind::OpResult && operand.op() != nullptr) {
        uses[operand.op()].push_back(op.get());
      }
    }
    if (op->region != nullptr) {
      collectUses(*op->region, uses);
    }
  }
}

bool allOperandsWiden(const Op &op, const std::set<const Op *> &vectorized) {
  for (const Value &operand : op.operands) {
    if (operand.kind() != Value::Kind::OpResult ||
        vectorized.count(operand.op()) == 0) {
      return false;
    }
  }
  return true;
}

// True when a use of a widened value can legally consume a vector-typed
// value. A store is always fine (it becomes store_vec). A compute op is fine
// only when IT is also widened (its result then becomes vector) and every one
// of its operands widens - a compute op left scalar can never see a vector
// operand (mixed types are not expressible in this IR).
bool useAcceptsVectors(const Op &use, const std::set<const Op *> &vectorized) {
  switch (use.kind) {
  case OpKind::Store:
    return true;
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Max:
  case OpKind::Min:
  case OpKind::Fma:
    return vectorized.count(&use) != 0 && allOperandsWiden(use, vectorized);
  case OpKind::Select:
    // The condition (operand 0) stays an i32 scalar; only both arms widen.
    return vectorized.count(&use) != 0 && use.operands.size() >= 3 &&
           use.operands[1].kind() == Value::Kind::OpResult &&
           use.operands[2].kind() == Value::Kind::OpResult &&
           vectorized.count(use.operands[1].op()) != 0 &&
           vectorized.count(use.operands[2].op()) != 0;
  default:
    return false;
  }
}

// True when `op`'s membership in the vectorized set is still consistent.
bool opSatisfies(const Op &op, const UseMap &uses,
                 const std::set<const Op *> &vectorized) {
  switch (op.kind) {
  case OpKind::Load:
    break; // structural eligibility was checked when the candidate was added
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Max:
  case OpKind::Min:
  case OpKind::Fma:
    // Every operand must widen AND all operands must share one type (the
    // verifier's rule; e.g. vector<4xf32> with vector<4xi32> is not a legal
    // op, so both chains stay scalar).
    if (!allOperandsWiden(op, vectorized)) {
      return false;
    }
    for (size_t i = 1; i < op.operands.size(); ++i) {
      if (op.operands[i].type() != op.operands[0].type()) {
        return false;
      }
    }
    break;
  case OpKind::Select:
    if (op.operands.size() < 3 ||
        op.operands[1].kind() != Value::Kind::OpResult ||
        op.operands[2].kind() != Value::Kind::OpResult ||
        vectorized.count(op.operands[1].op()) == 0 ||
        vectorized.count(op.operands[2].op()) == 0 ||
        op.operands[1].type() != op.operands[2].type()) {
      return false;
    }
    break;
  default:
    return false;
  }
  const auto it = uses.find(&op);
  if (it != uses.end()) {
    for (const Op *use : it->second) {
      if (!useAcceptsVectors(*use, vectorized)) {
        return false;
      }
    }
  }
  return true;
}

// Greatest fixpoint: start from every structurally eligible op, then remove
// candidates whose widening is inconsistent with the current set until
// stable. Monotone removal - the result is order-independent and no mixed
// scalar/vector op survives.
std::set<const Op *> computeVectorizedOps(Kernel &kernel) {
  UseMap uses;
  collectUses(kernel.body, uses);

  std::set<const Op *> vectorized;
  walkBlocks(kernel.body, [&](Block &block) {
    for (const auto &op : block.ops) {
      if (isWidenableLoad(*op) || isWidenableCompute(op->kind)) {
        vectorized.insert(op.get());
      }
    }
  });

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto it = vectorized.begin(); it != vectorized.end();) {
      if (opSatisfies(**it, uses, vectorized)) {
        ++it;
      } else {
        it = vectorized.erase(it);
        changed = true;
      }
    }
  }
  return vectorized;
}

void rewriteKernel(Kernel &kernel, const std::set<const Op *> &vectorized,
                   int64_t width) {
  walkBlocks(kernel.body, [&](Block &block) {
    for (const auto &op : block.ops) {
      switch (op->kind) {
      case OpKind::Load:
        if (vectorized.count(op.get()) != 0) {
          const TensorType tensor = op->operands[0].type().asTensor();
          op->kind = OpKind::LoadVec;
          op->resultTypes[0] = Type::vector(VectorType{width, tensor.elem});
        }
        break;
      case OpKind::Add:
      case OpKind::Sub:
      case OpKind::Mul:
      case OpKind::Div:
      case OpKind::Max:
      case OpKind::Min:
      case OpKind::Fma:
        if (vectorized.count(op.get()) != 0) {
          // Every operand is vector<NxE> of one type (fixpoint invariant);
          // the result must share it.
          op->resultTypes[0] = op->operands[0].type();
        }
        break;
      case OpKind::Select:
        if (vectorized.count(op.get()) != 0) {
          op->resultTypes[0] = op->operands[1].type();
        }
        break;
      case OpKind::Store:
        if (!op->operands.empty() &&
            op->operands[0].kind() == Value::Kind::OpResult &&
            vectorized.count(op->operands[0].op()) != 0) {
          op->kind = OpKind::StoreVec;
        }
        break;
      default:
        break;
      }
    }
  });
}

std::expected<void, std::string> vectorizeKernel(Kernel &kernel) {
  const int64_t width = kernel.schedule.vector_width;
  if (width < 1) {
    return std::unexpected("vectorize: kernel '" + kernel.name +
                           "': invalid vector width " + std::to_string(width));
  }
  if (width == 1) {
    return {}; // no-op
  }
  // Pinned divisibility check: every tile extent must be divisible by N.
  // Runs before any rewriting, so a rejected kernel is left untouched.
  for (const int64_t extent : kernel.schedule.tile) {
    if (extent <= 0) {
      return std::unexpected("vectorize: kernel '" + kernel.name +
                             "': tile extent " + std::to_string(extent) +
                             " must be positive");
    }
    if (extent % width != 0) {
      return std::unexpected(
          "vectorize: kernel '" + kernel.name + "': tile extent " +
          std::to_string(extent) + " is not divisible by vector width " +
          std::to_string(width));
    }
  }
  const std::set<const Op *> vectorized = computeVectorizedOps(kernel);
  if (!vectorized.empty()) {
    rewriteKernel(kernel, vectorized, width);
  }
  return {};
}

} // namespace

std::expected<void, std::string> vectorizePass(Module &module) {
  for (const auto &kernel : module.kernels) {
    if (const auto result = vectorizeKernel(*kernel); !result) {
      return result;
    }
  }
  return {};
}

// Self-registration (contract: lib/passes/CMakeLists.txt).
static const bool kVectorizeRegistered =
    (kernelh::ir::registerPass("vectorize", vectorizePass), true);

} // namespace kernelh::ir
