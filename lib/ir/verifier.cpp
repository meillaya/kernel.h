// Structural verifier for the atom IR (plan todo 9).
//
// Returns ALL violations (never early-exits). See verifier.hpp for the rule
// list. The LDS limit check is the SINGLE function verifyKernelSharedLimit
// using kDefaultLdsLimitBytes - todo 10a swaps that constant for the
// device-limits table lookup.

#include "kernelh/ir/verifier.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace kernelh::ir {

namespace {

// ---------------------------------------------------------------------------
// Scope bookkeeping collected in one pre-order walk: every block's lexical
// parent + owner op, and every op's containing block + position. Dominance is
// checked against THIS structure (not the stored parent pointers), so a
// manually-built module cannot trick the verifier with stale pointers.
// ---------------------------------------------------------------------------
struct ScopeInfo {
  std::unordered_map<const Block *, const Block *> blockParent;
  std::unordered_map<const Block *, const Op *> blockOwner;
  std::unordered_map<const Op *, const Block *> opBlock;
  std::unordered_map<const Op *, int64_t> opPos;
};

void collectScopeInfo(const Block &block, const Block *parentBlock,
                      const Op *ownerOp, ScopeInfo &info) {
  info.blockParent[&block] = parentBlock;
  info.blockOwner[&block] = ownerOp;
  for (size_t i = 0; i < block.ops.size(); ++i) {
    const Op *op = block.ops[i].get();
    info.opBlock[op] = &block;
    info.opPos[op] = static_cast<int64_t>(i);
    if (op->region != nullptr) {
      collectScopeInfo(*op->region, &block, op, info);
    }
  }
}

// True when the value `v` (defined in defBlock) is visible at the use site
// (op at position `usePos` in block `useBlock`). Structured-SSA rule: the
// definition must dominate the use.
bool valueDominates(const Value &v, const Block *useBlock, int64_t usePos,
                    const ScopeInfo &info) {
  const Block *defBlock = nullptr;
  bool isBlockArg = false;
  int64_t defPos = -1;
  if (v.kind() == Value::Kind::BlockArg) {
    defBlock = v.block();
    isBlockArg = true;
  } else {
    auto blockIt = info.opBlock.find(v.op());
    if (blockIt == info.opBlock.end()) {
      return false; // definition outside the module -> not dominated
    }
    defBlock = blockIt->second;
    defPos = info.opPos.at(v.op());
  }

  if (defBlock == useBlock) {
    return isBlockArg || defPos < usePos;
  }
  // useBlock must be strictly nested inside defBlock; walk up to the child of
  // defBlock that owns the region containing the use.
  const Block *child = useBlock;
  while (child != nullptr && info.blockParent.at(child) != defBlock) {
    child = info.blockParent.at(child);
  }
  if (child == nullptr) {
    return false; // defBlock is not an ancestor of useBlock
  }
  if (isBlockArg) {
    return true; // block args dominate everything in their block's regions
  }
  const Op *owner = info.blockOwner.at(child);
  return defPos < info.opPos.at(owner);
}

// True when `t` is structurally well-formed (tensor extents and vector widths
// positive; scalars trivially valid).
bool typeIsValid(const Type &t) {
  if (t.isTensor()) {
    for (int64_t extent : t.asTensor().shape) {
      if (extent <= 0) {
        return false;
      }
    }
    return true;
  }
  if (t.isVector()) {
    return t.asVector().width > 0;
  }
  return t.isScalar();
}

void addError(Diagnostics &diags, std::string message) {
  diags.push_back(Diagnostic{Diagnostic::Severity::Error, std::move(message)});
}

// Convenience: all operands after `first` are the same type as operand[0].
bool allSameType(const Op &op, size_t first, size_t count) {
  for (size_t i = first + 1; i < first + count; ++i) {
    if (op.operands[i].type() != op.operands[first].type()) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// LDS limit check - THE single place the shared-memory budget is enforced.
// todo 10a swaps kDefaultLdsLimitBytes for the device-limits table lookup
// inside this one function.
// ---------------------------------------------------------------------------
void verifyKernelSharedLimit(const Kernel &kernel, Diagnostics &diags) {
  const int64_t total = kernelTotalSharedBytes(kernel);
  if (total > kDefaultLdsLimitBytes) {
    addError(diags,
             "shared memory usage " + std::to_string(total) +
                 " bytes exceeds the " +
                 std::to_string(kDefaultLdsLimitBytes) +
                 "-byte LDS limit (kDefaultLdsLimitBytes)");
  }
}

// ---------------------------------------------------------------------------
// Op rules
// ---------------------------------------------------------------------------

void verifyOp(const Op &op, const Block &block, int64_t pos,
              const ScopeInfo &info, Diagnostics &diags);

void verifyYieldOperands(const Op &yield, const Op &loop, Diagnostics &diags) {
  if (yield.operands.size() != loop.resultTypes.size()) {
    addError(diags,
             "atom.yield operand count (" +
                 std::to_string(yield.operands.size()) +
                 ") does not match atom.loop_for iter_args count (" +
                 std::to_string(loop.resultTypes.size()) + ")");
    return;
  }
  for (size_t i = 0; i < yield.operands.size(); ++i) {
    if (yield.operands[i].type() != loop.resultTypes[i]) {
      addError(diags,
               "atom.yield operand " + std::to_string(i) +
                   " type mismatch: expected " +
                   loop.resultTypes[i].toString() + ", got " +
                   yield.operands[i].type().toString());
    }
  }
}

void verifyLoopFor(const Op &op, const Block &block, Diagnostics &diags) {
  const size_t numIterArgs = op.resultTypes.size();
  if (op.operands.size() != 3 + numIterArgs) {
    addError(diags,
             "atom.loop_for operand count (" +
                 std::to_string(op.operands.size()) +
                 ") does not match lb/ub/step + " +
                 std::to_string(numIterArgs) + " iter_args");
    return;
  }
  for (size_t i = 0; i < 3; ++i) {
    if (!op.operands[i].type().isScalar() ||
        op.operands[i].type().asScalar() != ScalarType::Index) {
      addError(diags,
               "atom.loop_for bound/step operand " + std::to_string(i) +
                   " must have type index");
    }
  }
  if (op.region == nullptr) {
    addError(diags, "atom.loop_for requires a body block");
    return;
  }
  const Block &body = *op.region;
  // Body block args: [induction(index)] + one per iter-arg type.
  if (body.args.size() != 1 + numIterArgs) {
    addError(diags,
             "atom.loop_for body block argument count (" +
                 std::to_string(body.args.size()) +
                 ") does not match 1 induction + " +
                 std::to_string(numIterArgs) + " iter args");
  } else {
    if (!body.args[0].type.isScalar() ||
        body.args[0].type.asScalar() != ScalarType::Index) {
      addError(diags,
               "atom.loop_for induction block argument must have type index");
    }
    for (size_t i = 0; i < numIterArgs; ++i) {
      if (body.args[1 + i].type != op.resultTypes[i]) {
        addError(diags,
                 "atom.loop_for iter-arg block argument " + std::to_string(i) +
                     " type mismatch: expected " +
                     op.resultTypes[i].toString() + ", got " +
                     body.args[1 + i].type.toString());
      }
    }
  }

  // Body must end with exactly one atom.yield whose operands match the loop's
  // iter-arg types. A non-terminal yield is also a violation.
  if (body.ops.empty()) {
    addError(diags, "atom.loop_for body must terminate with an atom.yield");
    return;
  }
  const Op *last = body.ops.back().get();
  if (last->kind != OpKind::Yield) {
    addError(diags,
             "atom.loop_for body must terminate with an atom.yield (last op "
             "is '" +
                 std::string(toString(last->kind)) + "')");
    return;
  }
  for (size_t i = 0; i + 1 < body.ops.size(); ++i) {
    if (body.ops[i]->kind == OpKind::Yield) {
      addError(diags,
               "atom.yield must be the last operation of its block");
    }
  }
  verifyYieldOperands(*last, op, diags);
}

// Main dispatch. `block`/`pos` are the use site for dominance checks.
void verifyOp(const Op &op, const Block &block, int64_t pos,
              const ScopeInfo &info, Diagnostics &diags) {
  // -- Structural: region parent pointers must be consistent with the tree. --
  if (op.region != nullptr) {
    if (op.region->parentBlock != &block) {
      addError(diags,
               "inconsistent block nesting: region parent block does not "
               "match the containing block");
    }
    if (op.region->parentOp != &op) {
      addError(diags,
               "inconsistent block nesting: region parent op does not match "
               "the owning op");
    }
  }

  // -- Operands: validity, type well-formedness, dominance. --
  for (const Value &operand : op.operands) {
    if (!operand.valid()) {
      addError(diags, "invalid (null) operand on atom." +
                          std::string(toString(op.kind)));
      continue;
    }
    if (!typeIsValid(operand.type())) {
      addError(diags,
               "operand has an invalid type on atom." +
                   std::string(toString(op.kind)));
    }
    if (!valueDominates(operand, &block, pos, info)) {
      addError(diags,
               "SSA dominance violation: operand of atom." +
                   std::string(toString(op.kind)) +
                   " is not dominated by its definition");
    }
  }
  for (const Type &result : op.resultTypes) {
    if (!typeIsValid(result)) {
      addError(diags,
               "result has an invalid type on atom." +
                   std::string(toString(op.kind)));
    }
  }

  auto scalarType = [](const Value &v) { return v.type().asScalar(); };

  switch (op.kind) {
  case OpKind::Const: {
    if (op.resultTypes.size() != 1) {
      addError(diags, "atom.const must produce exactly one result");
      break;
    }
    const Type &result = op.resultTypes[0];
    if (result.isVector()) {
      addError(diags,
               "atom.const result cannot be a vector type (vector consts are "
               "not part of the op inventory)");
      break;
    }
    const ScalarType elem =
        result.isTensor() ? result.asTensor().elem : result.asScalar();
    const bool isFloat = isFloatType(elem);
    const bool hasInt = op.intValue.has_value();
    const bool hasFloat = op.floatValue.has_value();
    if (isFloat) {
      if (!hasFloat || hasInt) {
        addError(diags,
                 "atom.const with float element type requires a float "
                 "literal");
      }
    } else {
      if (!hasInt || hasFloat) {
        addError(diags,
                 "atom.const with integer element type requires an integer "
                 "literal");
      }
      if (elem == ScalarType::I32 && hasInt &&
          (op.intValue.value() <
               static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
           op.intValue.value() >
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()))) {
        addError(diags, "atom.const value " + std::to_string(*op.intValue) +
                            " does not fit i32");
      }
    }
    break;
  }
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Max:
  case OpKind::Min: {
    if (op.operands.size() != 2 || op.resultTypes.size() != 1) {
      addError(diags, "atom." + std::string(toString(op.kind)) +
                          " requires 2 operands and 1 result");
      break;
    }
    if (op.operands[0].type() != op.operands[1].type() ||
        op.resultTypes[0] != op.operands[0].type()) {
      addError(diags, "type mismatch in atom." +
                          std::string(toString(op.kind)) + ": operands and "
                          "result must share one type");
    }
    break;
  }
  case OpKind::Fma: {
    if (op.operands.size() != 3 || op.resultTypes.size() != 1) {
      addError(diags, "atom.fma requires 3 operands and 1 result");
      break;
    }
    if (!allSameType(op, 0, 3) || op.resultTypes[0] != op.operands[0].type()) {
      addError(diags,
               "type mismatch in atom.fma: all operands and the result must "
               "share one type");
    }
    break;
  }
  case OpKind::Cmp: {
    if (op.operands.size() != 2 || op.resultTypes.size() != 1) {
      addError(diags, "atom.cmp requires 2 operands and 1 result");
      break;
    }
    if (!op.operands[0].type().isScalar() ||
        !op.operands[1].type().isScalar() ||
        op.operands[0].type() != op.operands[1].type()) {
      addError(diags,
               "type mismatch in atom.cmp: operands must be scalars of one "
               "type");
    }
    if (!op.resultTypes[0].isScalar() ||
        op.resultTypes[0].asScalar() != ScalarType::I32) {
      addError(diags,
               "atom.cmp result must be i32 (the boolean carrier; there is no "
               "i1 type in this IR)");
    }
    break;
  }
  case OpKind::Select: {
    if (op.operands.size() != 3 || op.resultTypes.size() != 1) {
      addError(diags, "atom.select requires 3 operands and 1 result");
      break;
    }
    if (!op.operands[0].type().isScalar() ||
        scalarType(op.operands[0]) != ScalarType::I32) {
      addError(diags,
               "type mismatch in atom.select: condition must be an i32 "
               "scalar");
    }
    if (op.operands[1].type() != op.operands[2].type() ||
        op.resultTypes[0] != op.operands[1].type()) {
      addError(diags,
               "type mismatch in atom.select: both arms and the result must "
               "share one type");
    }
    break;
  }
  case OpKind::Math: {
    if (op.operands.size() != 1 || op.resultTypes.size() != 1) {
      addError(diags, "atom.math requires 1 operand and 1 result");
      break;
    }
    if (op.resultTypes[0] != op.operands[0].type()) {
      addError(diags, "type mismatch in atom.math: result must match operand");
      break;
    }
    const ScalarType elem =
        op.operands[0].type().isTensor()
            ? op.operands[0].type().asTensor().elem
            : op.operands[0].type().asScalar();
    if (!isFloatType(elem)) {
      addError(diags,
               "atom.math is only defined for f16/f32 element types");
    }
    break;
  }
  case OpKind::Load: {
    if (op.resultTypes.size() != 1 || op.operands.empty() ||
        !op.operands[0].type().isTensor()) {
      addError(diags,
               "atom.load requires a tensor operand and one index operand "
               "per dim");
      break;
    }
    const TensorType t = op.operands[0].type().asTensor();
    if (op.operands.size() != 1 + static_cast<size_t>(t.rank())) {
      addError(diags,
               "atom.load index count (" +
                   std::to_string(op.operands.size() - 1) +
                   ") does not match tensor rank (" +
                   std::to_string(t.rank()) + ")");
      break;
    }
    for (size_t i = 1; i < op.operands.size(); ++i) {
      if (!op.operands[i].type().isScalar() ||
          scalarType(op.operands[i]) != ScalarType::Index) {
        addError(diags, "atom.load index operands must have type index");
      }
    }
    if (!op.resultTypes[0].isScalar() ||
        op.resultTypes[0].asScalar() != t.elem) {
      addError(diags,
               "atom.load result must be the tensor element type");
    }
    break;
  }
  case OpKind::Store: {
    if (!op.resultTypes.empty() || op.operands.size() < 2 ||
        !op.operands[1].type().isTensor()) {
      addError(diags,
               "atom.store requires a value operand, a tensor operand, and "
               "one index operand per dim");
      break;
    }
    const TensorType t = op.operands[1].type().asTensor();
    if (!op.operands[0].type().isScalar() ||
        op.operands[0].type().asScalar() != t.elem) {
      addError(diags,
               "atom.store value type must match the tensor element type");
    }
    if (op.operands.size() != 2 + static_cast<size_t>(t.rank())) {
      addError(diags,
               "atom.store index count (" +
                   std::to_string(op.operands.size() - 2) +
                   ") does not match tensor rank (" +
                   std::to_string(t.rank()) + ")");
      break;
    }
    for (size_t i = 2; i < op.operands.size(); ++i) {
      if (!op.operands[i].type().isScalar() ||
          scalarType(op.operands[i]) != ScalarType::Index) {
        addError(diags, "atom.store index operands must have type index");
      }
    }
    break;
  }
  case OpKind::LoadVec: {
    if (op.resultTypes.size() != 1 || op.operands.empty() ||
        !op.operands[0].type().isTensor() ||
        !op.resultTypes[0].isVector()) {
      addError(diags,
               "atom.load_vec requires a tensor operand, index operands, and "
               "a vector<WxE> result");
      break;
    }
    const TensorType t = op.operands[0].type().asTensor();
    if (op.operands.size() != 1 + static_cast<size_t>(t.rank())) {
      addError(diags,
               "atom.load_vec index count (" +
                   std::to_string(op.operands.size() - 1) +
                   ") does not match tensor rank (" +
                   std::to_string(t.rank()) + ")");
      break;
    }
    for (size_t i = 1; i < op.operands.size(); ++i) {
      if (!op.operands[i].type().isScalar() ||
          scalarType(op.operands[i]) != ScalarType::Index) {
        addError(diags, "atom.load_vec index operands must have type index");
      }
    }
    if (op.resultTypes[0].asVector().elem != t.elem) {
      addError(diags,
               "atom.load_vec vector element type must match the tensor "
               "element type");
    }
    break;
  }
  case OpKind::StoreVec: {
    if (!op.resultTypes.empty() || op.operands.size() < 2 ||
        !op.operands[0].type().isVector() ||
        !op.operands[1].type().isTensor()) {
      addError(diags,
               "atom.store_vec requires a vector operand, a tensor operand, "
               "and one index operand per dim");
      break;
    }
    const TensorType t = op.operands[1].type().asTensor();
    if (op.operands[0].type().asVector().elem != t.elem) {
      addError(diags,
               "atom.store_vec vector element type must match the tensor "
               "element type");
    }
    if (op.operands.size() != 2 + static_cast<size_t>(t.rank())) {
      addError(diags,
               "atom.store_vec index count (" +
                   std::to_string(op.operands.size() - 2) +
                   ") does not match tensor rank (" +
                   std::to_string(t.rank()) + ")");
      break;
    }
    for (size_t i = 2; i < op.operands.size(); ++i) {
      if (!op.operands[i].type().isScalar() ||
          scalarType(op.operands[i]) != ScalarType::Index) {
        addError(diags, "atom.store_vec index operands must have type index");
      }
    }
    break;
  }
  case OpKind::AllocShared: {
    if (!op.operands.empty() || op.resultTypes.size() != 1 ||
        !op.resultTypes[0].isTensor() ||
        op.resultTypes[0].asTensor().space != MemSpace::Shared) {
      addError(diags,
               "atom.alloc_shared requires exactly one tensor result in the "
               "shared memory space");
    }
    break;
  }
  case OpKind::Barrier: {
    if (!op.operands.empty() || !op.resultTypes.empty()) {
      addError(diags, "atom.barrier takes no operands and no results");
    }
    // Barrier placement: only kernel-body blocks and loop_for body blocks.
    // In v1 those are the only block kinds (atom.reduce is a leaf op with no
    // region), so this is structurally vacuous - but implemented generically
    // so it extends when regions grow.
    const Op *owner = info.blockOwner.at(&block);
    if (owner != nullptr && owner->kind != OpKind::LoopFor) {
      addError(diags,
               "atom.barrier is only allowed in kernel-body blocks and "
               "atom.loop_for body blocks");
    }
    break;
  }
  case OpKind::ThreadId:
  case OpKind::BlockId:
  case OpKind::BlockDim:
  case OpKind::GridDim: {
    if (!op.operands.empty() || op.resultTypes.size() != 1 ||
        !op.resultTypes[0].isScalar() ||
        op.resultTypes[0].asScalar() != ScalarType::Index) {
      addError(diags, "atom." + std::string(toString(op.kind)) +
                          " must produce exactly one index result");
    }
    break;
  }
  case OpKind::LoopFor:
    verifyLoopFor(op, block, diags);
    break;
  case OpKind::Yield: {
    if (!op.resultTypes.empty()) {
      addError(diags, "atom.yield produces no results");
    }
    const Op *owner = info.blockOwner.at(&block);
    if (owner == nullptr || owner->kind != OpKind::LoopFor) {
      addError(diags,
               "atom.yield is only valid inside an atom.loop_for body");
      break;
    }
    if (pos != static_cast<int64_t>(block.ops.size()) - 1) {
      addError(diags, "atom.yield must be the last operation of its block");
    }
    verifyYieldOperands(op, *owner, diags);
    break;
  }
  case OpKind::Reduce: {
    if (op.operands.size() != 1 || op.resultTypes.size() != 1 ||
        !op.operands[0].type().isTensor()) {
      addError(diags,
               "atom.reduce requires one tensor operand and one scalar "
               "result");
      break;
    }
    const TensorType t = op.operands[0].type().asTensor();
    if (op.reduceDim < 0 || op.reduceDim >= t.rank()) {
      addError(diags, "atom.reduce dim " + std::to_string(op.reduceDim) +
                          " is out of range for tensor rank " +
                          std::to_string(t.rank()));
    }
    if (!op.resultTypes[0].isScalar() ||
        op.resultTypes[0].asScalar() != t.elem) {
      addError(diags,
               "atom.reduce result must be the tensor element type");
    }
    break;
  }
  case OpKind::Copy: {
    if (!op.resultTypes.empty() || op.operands.size() != 2 ||
        !op.operands[0].type().isTensor() ||
        !op.operands[1].type().isTensor()) {
      addError(diags,
               "atom.copy requires two tensor operands of identical shape "
               "and element type");
      break;
    }
    const TensorType src = op.operands[0].type().asTensor();
    const TensorType dst = op.operands[1].type().asTensor();
    if (src.shape != dst.shape || src.elem != dst.elem) {
      addError(diags,
               "atom.copy source and destination must have identical shape "
               "and element type");
    }
    break;
  }
  }
}

void verifyKernelAttrs(const Kernel &kernel, Diagnostics &diags) {
  for (int64_t t : kernel.schedule.tile) {
    if (t <= 0) {
      addError(diags, "invalid schedule attribute: tile size must be positive "
                          "(got " + std::to_string(t) + ")");
    }
  }
  if (kernel.schedule.vector_width < 1) {
    addError(diags, "invalid schedule attribute: vector_width must be >= 1 "
                        "(got " +
                    std::to_string(kernel.schedule.vector_width) + ")");
  }
  if (kernel.schedule.unroll < 1) {
    addError(diags, "invalid schedule attribute: unroll must be >= 1 (got " +
                        std::to_string(kernel.schedule.unroll) + ")");
  }
  for (const auto &[key, value] : kernel.shape) {
    if (value <= 0) {
      addError(diags, "invalid shape attribute: '" + key +
                          "' must be positive (got " + std::to_string(value) +
                          ")");
    }
  }
}

void verifyBlock(const Block &block, const ScopeInfo &info,
                 Diagnostics &diags) {
  // Kernel body block args = kernel parameters; shared tensors cannot be
  // passed in from the host (loop body args have no such restriction).
  const bool isKernelBody = block.parentBlock == nullptr;
  if (isKernelBody) {
    for (size_t i = 0; i < block.args.size(); ++i) {
      if (block.args[i].type.isTensor() &&
          block.args[i].type.asTensor().space == MemSpace::Shared) {
        addError(diags,
                 "kernel argument cannot be a shared-space tensor");
      }
    }
  }
  for (size_t i = 0; i < block.args.size(); ++i) {
    if (!typeIsValid(block.args[i].type)) {
      addError(diags, "block argument " + std::to_string(i) +
                          " has an invalid type");
    }
  }
  for (size_t i = 0; i < block.ops.size(); ++i) {
    const Op &op = *block.ops[i];
    verifyOp(op, block, static_cast<int64_t>(i), info, diags);
    if (op.region != nullptr) {
      verifyBlock(*op.region, info, diags);
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool isModuleValid(const Module &module) {
  for (const Diagnostic &d : verifyModule(module)) {
    if (d.severity == Diagnostic::Severity::Error) {
      return false;
    }
  }
  return true;
}

Diagnostics verifyModule(const Module &module) {
  Diagnostics diags;
  ScopeInfo info;
  for (const auto &kernel : module.kernels) {
    collectScopeInfo(kernel->body, nullptr, nullptr, info);
  }
  for (const auto &kernel : module.kernels) {
    verifyKernelAttrs(*kernel, diags);
    verifyKernelSharedLimit(*kernel, diags);
    verifyBlock(kernel->body, info, diags);
  }
  return diags;
}

} // namespace kernelh::ir
