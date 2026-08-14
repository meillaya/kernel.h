// lib/passes/canonicalize.cpp - the canonicalize pass for the atom IR (plan
// todo 13).
//
// Three rewrites, run to a joint fixpoint over every kernel:
//
//  1. CONSTANT FOLDING - a pure elementwise op whose operands are ALL
//     atom.const folds in place into a single atom.const (kind=Const, payload
//     set, operands cleared so the old operands become dead). Fold set:
//       add/sub/mul/div/fma/max/min/cmp/math
//     Integer add/sub/mul use __builtin_*_overflow so an int64 overflow does
//     NOT fold (the op is left intact). Integer div does NOT fold when the
//     divisor is zero (UB in this IR) or on INT64_MIN / -1. An i32 result must
//     fit int32_t (the verifier's rule) or the fold is refused. Float div and
//     math fold with IEEE semantics (x/0 -> inf, 0/0 -> NaN). Integer fma is
//     excluded from the fold set (its hardware meaning is ambiguous).
//
//  2. IDENTITY ELISION - a bin op with ONE const operand collapses to the
//     identity operand:
//       x + 0, 0 + x, x - 0, x * 1, 1 * x, x / 1 -> x   (int and float)
//       x * 0 -> const 0                                 (INTEGER only)
//     Float x * 0 is deliberately NOT elided: IEEE 0*x is NaN when x is inf
//     or NaN, so the fold would change semantics. All uses of the elided
//     result are rewritten to the identity operand, then the op is erased.
//
//  3. DEAD-OP ELIMINATION - iterative fixpoint: remove pure ops whose results
//     are unused anywhere in the kernel. Side-effecting ops (store/store_vec/
//     barrier/copy/yield) and atom.loop_for are NEVER removed (loop_for is
//     conservatively kept even when its results are dead).
//
// Self-registration contract (todos 14-18): see lib/passes/CMakeLists.txt.

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kernelh::ir {
namespace {

ScalarType elemTypeOf(const Type &type) {
  return type.isTensor() ? type.asTensor().elem : type.asScalar();
}

bool isConstValue(const Value &value) {
  return value.kind() == Value::Kind::OpResult && value.op() != nullptr &&
         value.op()->kind == OpKind::Const;
}

const Op &asConstOp(const Value &value) { return *value.op(); }

template <typename Fn>
void walkBlocks(Block &block, const Fn &fn) {
  fn(block);
  for (const auto &op : block.ops) {
    if (op->region != nullptr) {
      walkBlocks(*op->region, fn);
    }
  }
}

// Rewrites every use of `from` to `to` in `kernel`. Only called when `to`
// dominates every use site (elision rewrites to an operand of the same op, so
// it always dominates the erased result's uses).
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

// A folded literal; which member carries the value is decided by the element
// type of the op being folded.
struct Lit {
  int64_t i = 0;
  double f = 0.0;
};

// Attempts to fold `op` (all operands must be const - checked by the caller)
// into a single literal. Returns nullopt when no fold is defined (int div by
// zero, INT64_MIN / -1, integer fma, i32 overflow, int64 overflow).
std::optional<Lit> tryFold(const Op &op) {
  // atom.cmp: scalar operands, folds to an i32 0/1 boolean.
  if (op.kind == OpKind::Cmp) {
    if (op.operands.size() < 2) {
      return std::nullopt;
    }
    const ScalarType elem = elemTypeOf(op.operands[0].type());
    const bool isFloat = isFloatType(elem);
    const Op &a = asConstOp(op.operands[0]);
    const Op &b = asConstOp(op.operands[1]);
    bool result = false;
    switch (op.pred) {
    case CmpPred::Lt:
      result = isFloat ? *a.floatValue < *b.floatValue
                       : *a.intValue < *b.intValue;
      break;
    case CmpPred::Le:
      result = isFloat ? *a.floatValue <= *b.floatValue
                       : *a.intValue <= *b.intValue;
      break;
    case CmpPred::Gt:
      result = isFloat ? *a.floatValue > *b.floatValue
                       : *a.intValue > *b.intValue;
      break;
    case CmpPred::Ge:
      result = isFloat ? *a.floatValue >= *b.floatValue
                       : *a.intValue >= *b.intValue;
      break;
    case CmpPred::Eq:
      result = isFloat ? *a.floatValue == *b.floatValue
                       : *a.intValue == *b.intValue;
      break;
    case CmpPred::Ne:
      result = isFloat ? *a.floatValue != *b.floatValue
                       : *a.intValue != *b.intValue;
      break;
    }
    return Lit{result ? 1 : 0, 0.0};
  }

  // atom.math: unary; verifier restricts element types to f16/f32.
  if (op.kind == OpKind::Math) {
    if (op.operands.empty()) {
      return std::nullopt;
    }
    if (!isFloatType(elemTypeOf(op.operands[0].type()))) {
      return std::nullopt;
    }
    const double a = *asConstOp(op.operands[0]).floatValue;
    double r = 0.0;
    switch (op.mathKind) {
    case MathKind::Exp:
      r = std::exp(a);
      break;
    case MathKind::Log:
      r = std::log(a);
      break;
    case MathKind::Sqrt:
      r = std::sqrt(a);
      break;
    case MathKind::Rsqrt:
      r = 1.0 / std::sqrt(a); // 0 -> +inf (IEEE)
      break;
    case MathKind::Tanh:
      r = std::tanh(a);
      break;
    }
    return Lit{0, r};
  }

  // The rest of the fold set is elementwise bin ops.
  const ScalarType elem = elemTypeOf(op.operands[0].type());
  const bool isFloat = isFloatType(elem);

  if (isFloat) {
    const double a = *asConstOp(op.operands[0]).floatValue;
    const double b = *asConstOp(op.operands[1]).floatValue;
    switch (op.kind) {
    case OpKind::Add:
      return Lit{0, a + b};
    case OpKind::Sub:
      return Lit{0, a - b};
    case OpKind::Mul:
      return Lit{0, a * b};
    case OpKind::Div:
      return Lit{0, a / b}; // IEEE: x/0 -> inf, 0/0 -> NaN
    case OpKind::Fma:
      return Lit{0, std::fma(a, b, *asConstOp(op.operands[2]).floatValue)};
    case OpKind::Max:
      return Lit{0, std::fmax(a, b)};
    case OpKind::Min:
      return Lit{0, std::fmin(a, b)};
    default:
      return std::nullopt;
    }
  }

  const int64_t a = *asConstOp(op.operands[0]).intValue;
  const int64_t b = *asConstOp(op.operands[1]).intValue;
  int64_t r = 0;
  switch (op.kind) {
  case OpKind::Add:
    if (__builtin_add_overflow(a, b, &r)) {
      return std::nullopt;
    }
    break;
  case OpKind::Sub:
    if (__builtin_sub_overflow(a, b, &r)) {
      return std::nullopt;
    }
    break;
  case OpKind::Mul:
    if (__builtin_mul_overflow(a, b, &r)) {
      return std::nullopt;
    }
    break;
  case OpKind::Div:
    // Integer div by zero does NOT fold (UB in this IR); neither does the
    // INT64_MIN / -1 overflow. C++ `/` truncates toward zero, matching the
    // IR's documented signed-integer division.
    if (b == 0) {
      return std::nullopt;
    }
    if (a == std::numeric_limits<int64_t>::min() && b == -1) {
      return std::nullopt;
    }
    r = a / b;
    break;
  case OpKind::Max:
    r = std::max(a, b);
    break;
  case OpKind::Min:
    r = std::min(a, b);
    break;
  default:
    return std::nullopt; // incl. integer fma, excluded from the fold set
  }
  // The folded constant must fit its result type (the verifier requires i32
  // consts to fit int32_t; i64/index always fit after the checked math).
  if (elem == ScalarType::I32 &&
      (r < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
       r > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))) {
    return std::nullopt;
  }
  return Lit{r, 0.0};
}

// Folds `op` in place into a const matching its result element type. Operands
// and payload fields are cleared so the const prints exactly like a parsed
// atom.const and its former operands become dead for DCE.
void foldInPlace(Op &op, const Lit &lit) {
  const ScalarType elem = elemTypeOf(op.resultTypes[0]);
  op.kind = OpKind::Const;
  op.operands.clear();
  op.region.reset();
  op.pred = CmpPred::Lt;
  op.mathKind = MathKind::Exp;
  op.reduceKind = ReduceKind::Sum;
  op.reduceDim = 0;
  op.idDim = IdDim::X;
  if (isFloatType(elem)) {
    op.intValue.reset();
    op.floatValue = lit.f;
  } else {
    op.floatValue.reset();
    op.intValue = lit.i;
  }
}

// Identity elision: the op collapses to one of its operands.
//   x + 0, 0 + x, x - 0, x * 1, 1 * x, x / 1 -> x
// The const operand must carry the identity value for the op's element type.
std::optional<Value> tryIdentityElision(const Op &op) {
  switch (op.kind) {
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
    break;
  default:
    // Any other op (store_vec has a vector operand[0], etc.) is never an
    // identity elision; leave before inspecting element types.
    return std::nullopt;
  }
  if (op.operands.size() < 2) {
    return std::nullopt;
  }
  const ScalarType elem = elemTypeOf(op.operands[0].type());
  const bool isFloat = isFloatType(elem);
  const auto isZero = [isFloat](const Op &c) {
    return isFloat ? (c.floatValue.has_value() && *c.floatValue == 0.0)
                   : (c.intValue.has_value() && *c.intValue == 0);
  };
  const auto isOne = [isFloat](const Op &c) {
    return isFloat ? (c.floatValue.has_value() && *c.floatValue == 1.0)
                   : (c.intValue.has_value() && *c.intValue == 1);
  };
  const auto isZeroConst = [&](const Value &v) {
    return isConstValue(v) && isZero(asConstOp(v));
  };
  const auto isOneConst = [&](const Value &v) {
    return isConstValue(v) && isOne(asConstOp(v));
  };
  switch (op.kind) {
  case OpKind::Add:
    if (isZeroConst(op.operands[0])) {
      return op.operands[1];
    }
    if (isZeroConst(op.operands[1])) {
      return op.operands[0];
    }
    break;
  case OpKind::Sub:
    if (isZeroConst(op.operands[1])) {
      return op.operands[0];
    }
    break;
  case OpKind::Mul:
    if (isOneConst(op.operands[0])) {
      return op.operands[1];
    }
    if (isOneConst(op.operands[1])) {
      return op.operands[0];
    }
    break;
  case OpKind::Div:
    if (isOneConst(op.operands[1])) {
      return op.operands[0];
    }
    break;
  default:
    break;
  }
  return std::nullopt;
}

// Integer x * 0 -> const 0 (and 0 * x). NOT done for floats: IEEE 0*x is NaN
// when x is inf or NaN, so the fold would change semantics.
std::optional<Lit> tryIntZeroFold(const Op &op) {
  if (op.kind != OpKind::Mul || op.operands.size() < 2) {
    return std::nullopt;
  }
  if (isFloatType(elemTypeOf(op.operands[0].type()))) {
    return std::nullopt;
  }
  for (const Value &operand : op.operands) {
    if (isConstValue(operand) && *asConstOp(operand).intValue == 0) {
      return Lit{0, 0.0};
    }
  }
  return std::nullopt;
}

bool foldElideBlock(Block &block, Kernel &kernel) {
  bool changed = false;
  size_t i = 0;
  while (i < block.ops.size()) {
    Op &op = *block.ops[i];
    if (op.region != nullptr) {
      changed |= foldElideBlock(*op.region, kernel);
    }
    if (op.kind != OpKind::Const) {
      // 1. Constant folding: every operand const + foldable kind.
      bool allConst = !op.operands.empty();
      for (const Value &operand : op.operands) {
        if (!isConstValue(operand)) {
          allConst = false;
          break;
        }
      }
      if (allConst) {
        if (const auto lit = tryFold(op)) {
          foldInPlace(op, *lit);
          changed = true;
          continue;
        }
      }
      // 2a. Identity elision: rewrite uses of op.result() to the identity
      //     operand, then erase the op.
      if (const auto identity = tryIdentityElision(op)) {
        rewriteAllUses(kernel, op.result(), *identity);
        block.ops.erase(block.ops.begin() + i);
        changed = true;
        continue;
      }
      // 2b. Integer x*0 -> const 0 (in place).
      if (const auto zero = tryIntZeroFold(op)) {
        foldInPlace(op, *zero);
        changed = true;
        continue;
      }
    }
    ++i;
  }
  return changed;
}

// Pure ops (no side effects) are removable when dead. Side-effecting ops
// (store/store_vec/barrier/copy/yield) and atom.loop_for (conservative) are
// never removed.
bool isDceRemovable(OpKind kind) {
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
  case OpKind::Load:
  case OpKind::LoadVec:
  case OpKind::AllocShared:
  case OpKind::ThreadId:
  case OpKind::BlockId:
  case OpKind::BlockDim:
  case OpKind::GridDim:
  case OpKind::Reduce:
    return true;
  case OpKind::Store:
  case OpKind::StoreVec:
  case OpKind::Barrier:
  case OpKind::Copy:
  case OpKind::Yield:
  case OpKind::LoopFor:
    return false;
  }
  return false;
}

// True when `value` is used anywhere in the block tree rooted at `block`.
// A value's legal use sites are exactly the subtree rooted at its defining
// block (structured SSA), so the root can always be the defining block.
bool valueHasUse(const Block &block, const Value &value) {
  for (const auto &op : block.ops) {
    for (const Value &operand : op->operands) {
      if (operand == value) {
        return true;
      }
    }
    if (op->region != nullptr && valueHasUse(*op->region, value)) {
      return true;
    }
  }
  return false;
}

bool dceBlock(Block &block) {
  bool changed = false;
  size_t i = 0;
  while (i < block.ops.size()) {
    Op *op = block.ops[i].get();
    if (op->region != nullptr) {
      changed |= dceBlock(*op->region);
    }
    if (isDceRemovable(op->kind) && !valueHasUse(block, op->result())) {
      block.ops.erase(block.ops.begin() + i);
      changed = true;
      continue;
    }
    ++i;
  }
  return changed;
}

bool canonicalizeKernel(Kernel &kernel) {
  bool changed = true;
  while (changed) {
    changed = foldElideBlock(kernel.body, kernel);
    changed |= dceBlock(kernel.body);
  }
  return true;
}

} // namespace

std::expected<void, std::string> canonicalizePass(Module &module) {
  for (const auto &kernel : module.kernels) {
    canonicalizeKernel(*kernel);
  }
  return {};
}

// Self-registration (contract: lib/passes/CMakeLists.txt).
static const bool kCanonicalizeRegistered =
    (kernelh::ir::registerPass("canonicalize", canonicalizePass), true);

} // namespace kernelh::ir
