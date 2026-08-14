// lib/passes/unroll.cpp - the unroll pass for the atom IR (plan todo 18).
//
// The pass consumes the kernel's schedule attr `unroll` and rewrites every
// atom.loop_for with a COMPILE-TIME (constant) trip count:
//
//   * FULL UNROLL: when the trip count is <= 8 AND the `unroll` factor is >=
//     the trip count, the loop is removed entirely and the body is emitted
//     `trip` times back-to-back in the enclosing block, with the induction
//     replaced by its constant value for each copy.
//
//   * PARTIAL UNROLL: otherwise (when `unroll` > 1) the loop is unrolled by
//     the `unroll` factor. The main loop keeps `ceil(trip/factor)` trips
//     over a body replicated `factor` times (copy i reads induction
//     mainInduction + i*step), and the REMAINING iterations run in a CLEANUP
//     loop after it - a second loop from lb + (trip - R)*step to ub that
//     executes the ORIGINAL body once per leftover iteration (R = trip %
//     factor). When R == 0 no cleanup loop is emitted and the main loop keeps
//     its original bounds.
//
//   * ITER_ARGS CARRY CHAIN (pinned): the loop's iter_args are replicated
//     across the copies. Copy 0 receives the loop's incoming iter-arg values;
//     copy i receives copy i-1's yield values (same count, same order); the
//     final copy's yield values become the loop's results (full unroll) or
//     the rebuilt loop's results (partial; the cleanup loop then receives the
//     main loop's results as its own iter args).
//
// DOCUMENTED NON-ACTIONS (all pinned by the plan):
//   - `unroll` = 1 (the default) makes the pass a no-op.
//   - A loop whose lb/ub/step are not all atom.const (non-constant trip
//     count) is LEFT ALONE - there is no dynamic unrolling in v1.
//   - Negative/non-positive steps and empty bounds (ub <= lb) are left alone.
//   - A partial-unroll factor that exceeds the trip count (main loop trip
//     would be 0) is left alone: the whole loop would be a degenerate cleanup.
//   - Loops are unrolled BOTTOM-UP (innermost first), so a loop nested inside
//     an unrolled loop is unrolled first and its unrolled form is cloned
//     along with the enclosing body.
//
// An INVALID `unroll` factor (0 or negative) is a DIAGNOSTIC ERROR: the pass
// returns std::expected with a message naming the kernel and the factor (the
// verifier has no unroll check; passes may run on unverified modules).
//
// Self-registration contract (todos 14-18): see lib/passes/CMakeLists.txt.

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <iterator>
#include <memory>
#include <optional>
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

// The integer value of a constant op (any constant with an int payload;
// bounds are index-typed in verifier-clean modules).
std::optional<int64_t> constIntValue(const Value &value) {
  if (value.kind() == Value::Kind::OpResult && value.op() != nullptr &&
      value.op()->kind == OpKind::Const && value.op()->intValue.has_value()) {
    return *value.op()->intValue;
  }
  return std::nullopt;
}

void rewriteUses(Block &block, const Value &from, const Value &to) {
  for (const auto &op : block.ops) {
    for (Value &operand : op->operands) {
      if (operand == from) {
        operand = to;
      }
    }
    if (op->region != nullptr) {
      rewriteUses(*op->region, from, to);
    }
  }
}

// ---------------------------------------------------------------------------
// Deep cloning. The unrolled copies are structural duplicates of the loop
// body. Two remap tables make a clone a self-contained SSA slice:
//   - argMaps: original block -> replacement values for its block args
//     (set for the loop body being cloned, and identity for each nested
//     loop body),
//   - opMap:   original op -> its clone (registered in clone order, so an
//     operand referencing an earlier op in the body resolves to that op's
//     clone).
// References to anything OUTSIDE the cloned slice (kernel-body block args,
// parent-scope values) are kept as-is - the copies land in a scope where
// those values dominate. parentBlock/parentOp follow the parser/builder
// convention: a region's parentBlock is the block CONTAINING the loop and
// its parentOp is the loop op itself.
// ---------------------------------------------------------------------------

struct CloneContext {
  std::unordered_map<const Block *, std::vector<Value>> argMaps;
  std::unordered_map<const Op *, Op *> opMap;
};

std::unique_ptr<Op> cloneOp(const Op &op, CloneContext &ctx,
                            Block *containingBlock) {
  auto clone = std::make_unique<Op>();
  clone->kind = op.kind;
  clone->resultTypes = op.resultTypes;
  clone->intValue = op.intValue;
  clone->floatValue = op.floatValue;
  clone->pred = op.pred;
  clone->mathKind = op.mathKind;
  clone->reduceKind = op.reduceKind;
  clone->reduceDim = op.reduceDim;
  clone->idDim = op.idDim;
  clone->operands.reserve(op.operands.size());
  for (const Value &operand : op.operands) {
    if (operand.kind() == Value::Kind::BlockArg) {
      const auto argIt = ctx.argMaps.find(operand.block());
      if (argIt != ctx.argMaps.end()) {
        clone->operands.push_back(
            argIt->second[static_cast<size_t>(operand.index())]);
        continue;
      }
      clone->operands.push_back(operand); // outside the cloned slice
    } else if (operand.kind() == Value::Kind::OpResult) {
      const auto opIt = ctx.opMap.find(operand.op());
      if (opIt != ctx.opMap.end()) {
        clone->operands.push_back(
            opIt->second->result(operand.index()));
        continue;
      }
      clone->operands.push_back(operand); // outside the cloned slice
    } else {
      clone->operands.push_back(operand);
    }
  }
  if (op.region != nullptr) {
    clone->region = std::make_unique<Block>();
    clone->region->args = op.region->args;
    clone->region->parentBlock = containingBlock;
    clone->region->parentOp = clone.get();
    std::vector<Value> nestedArgs;
    nestedArgs.reserve(op.region->args.size());
    for (size_t i = 0; i < op.region->args.size(); ++i) {
      nestedArgs.push_back(
          Value::blockArg(clone->region.get(), static_cast<int64_t>(i)));
    }
    ctx.argMaps[op.region.get()] = std::move(nestedArgs);
    for (const auto &nested : op.region->ops) {
      clone->region->ops.push_back(
          cloneOp(*nested, ctx, clone->region.get()));
    }
    ctx.argMaps.erase(op.region.get());
  }
  ctx.opMap[&op] = clone.get();
  return clone;
}

// Clones every op of `loop`'s body (including the trailing atom.yield) into
// `dstBlock`-scoped ops; the body's block args are rewritten through
// `argMap` (argMap[0] = the copy's induction, argMap[1+j] = the copy's
// incoming iter-arg value).
std::vector<std::unique_ptr<Op>> cloneBody(const Op &loop,
                                           const std::vector<Value> &argMap,
                                           Block *dstBlock) {
  std::vector<std::unique_ptr<Op>> out;
  const Block &body = *loop.region;
  out.reserve(body.ops.size());
  CloneContext ctx;
  ctx.argMaps[&body] = argMap;
  for (const auto &op : body.ops) {
    out.push_back(cloneOp(*op, ctx, dstBlock));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Standalone op construction (the builder helpers only append; unroll must
// build ops detached and splice them in at a chosen position).
// ---------------------------------------------------------------------------

std::unique_ptr<Op> makeIndexConstOp(int64_t value) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Const;
  op->resultTypes.push_back(Type::scalar(ScalarType::Index));
  op->intValue = value;
  return op;
}

std::unique_ptr<Op> makeIndexAddOp(Value a, Value b) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Add;
  op->operands = {a, b};
  op->resultTypes.push_back(a.type());
  return op;
}

std::unique_ptr<Op> makeYieldOp(std::vector<Value> values) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Yield;
  op->operands = std::move(values);
  return op;
}

// Replaces `target` in `block.ops` with `replacement` (in order). The target
// op is destroyed after all clones were taken and all uses rewritten.
void replaceOp(Block &block, Op *target,
               std::vector<std::unique_ptr<Op>> &&replacement) {
  auto &ops = block.ops;
  const auto it = std::find_if(ops.begin(), ops.end(),
                               [target](const std::unique_ptr<Op> &p) {
                                 return p.get() == target;
                               });
  if (it == ops.end()) {
    return; // unreachable: `target` came from this exact list
  }
  const size_t pos = static_cast<size_t>(it - ops.begin());
  ops.erase(ops.begin() + static_cast<std::ptrdiff_t>(pos));
  ops.insert(ops.begin() + static_cast<std::ptrdiff_t>(pos),
             std::make_move_iterator(replacement.begin()),
             std::make_move_iterator(replacement.end()));
}

// ---------------------------------------------------------------------------
// The per-loop transformation
// ---------------------------------------------------------------------------

enum class UnrollMode { None, Full, Partial };

// Decides what to do with one loop; `factor` is the kernel's unroll attr
// (>= 2 here; 1 and below were already handled).
UnrollMode planUnroll(const Op &loop, int64_t factor, int64_t &tripOut,
                      int64_t &remainderOut, int64_t &mainTripOut,
                      int64_t &lbOut, int64_t &ubOut, int64_t &stepOut) {
  if (loop.operands.size() < 3 || loop.region == nullptr ||
      loop.region->args.size() != 1 + loop.resultTypes.size() ||
      loop.operands.size() != 3 + loop.resultTypes.size()) {
    return UnrollMode::None; // malformed; the verifier would reject it
  }
  const std::optional<int64_t> lb = constIntValue(loop.operands[0]);
  const std::optional<int64_t> ub = constIntValue(loop.operands[1]);
  const std::optional<int64_t> step = constIntValue(loop.operands[2]);
  if (!lb || !ub || !step) {
    return UnrollMode::None; // non-constant trip count: no dynamic unrolling
  }
  if (*step <= 0 || *ub <= *lb) {
    return UnrollMode::None; // non-positive step / empty loop: leave alone
  }
  const int64_t trip = (*ub - *lb + *step - 1) / *step;
  if (trip <= 0) {
    return UnrollMode::None;
  }
  tripOut = trip;
  lbOut = *lb;
  ubOut = *ub;
  stepOut = *step;
  if (trip <= 8 && factor >= trip) {
    return UnrollMode::Full;
  }
  const int64_t remainder = trip % factor;
  const int64_t mainTrip = (trip - remainder) / factor;
  if (mainTrip == 0) {
    return UnrollMode::None; // factor exceeds the trip: degenerate cleanup
  }
  remainderOut = remainder;
  mainTripOut = mainTrip;
  return UnrollMode::Partial;
}

void fullyUnroll(Kernel &kernel, Op &loop, int64_t lbVal, int64_t stepVal,
                 int64_t trip) {
  Block *parent = loop.region->parentBlock;
  if (parent == nullptr) {
    return;
  }
  std::vector<std::unique_ptr<Op>> replacement;
  std::vector<Value> incoming(loop.operands.begin() + 3, loop.operands.end());
  for (int64_t i = 0; i < trip; ++i) {
    std::vector<Value> argMap;
    if (i == 0) {
      argMap.push_back(loop.operands[0]); // reuse the original lb const
    } else {
      std::unique_ptr<Op> c = makeIndexConstOp(lbVal + i * stepVal);
      argMap.push_back(c->result());
      replacement.push_back(std::move(c));
    }
    argMap.insert(argMap.end(), incoming.begin(), incoming.end());
    std::vector<std::unique_ptr<Op>> cloned = cloneBody(loop, argMap, parent);
    if (cloned.empty() || cloned.back()->kind != OpKind::Yield) {
      return; // malformed body; leave the loop alone
    }
    incoming = cloned.back()->operands;
    cloned.pop_back();
    for (auto &op : cloned) {
      replacement.push_back(std::move(op));
    }
  }
  for (size_t j = 0; j < loop.resultTypes.size(); ++j) {
    rewriteUses(kernel.body, loop.result(static_cast<int64_t>(j)),
                incoming[j]);
  }
  replaceOp(*parent, &loop, std::move(replacement));
}

// Builds the factor-copy body of the partial-unroll MAIN loop: `offsets[i]`
// is a constant i*step in the enclosing scope; copy 0's induction is the
// main loop's induction itself, copy i's is induction + offsets[i-1]. The
// trailing yield of the last copy carries the loop's results.
std::vector<std::unique_ptr<Op>> buildFactorCopies(
    const Op &loop, const std::vector<Value> &offsets, Block *dstBlock,
    const Value &induction) {
  std::vector<std::unique_ptr<Op>> out;
  std::vector<Value> incoming(loop.operands.begin() + 3, loop.operands.end());
  const size_t copies = offsets.size() + 1;
  for (size_t i = 0; i < copies; ++i) {
    std::vector<Value> argMap;
    if (i == 0) {
      argMap.push_back(induction);
    } else {
      std::unique_ptr<Op> addOp = makeIndexAddOp(induction, offsets[i - 1]);
      argMap.push_back(addOp->result());
      out.push_back(std::move(addOp));
    }
    argMap.insert(argMap.end(), incoming.begin(), incoming.end());
    std::vector<std::unique_ptr<Op>> cloned = cloneBody(loop, argMap, dstBlock);
    if (cloned.empty() || cloned.back()->kind != OpKind::Yield) {
      return {}; // malformed body; signal the caller to leave the loop alone
    }
    incoming = cloned.back()->operands;
    cloned.pop_back();
    for (auto &op : cloned) {
      out.push_back(std::move(op));
    }
  }
  out.push_back(makeYieldOp(std::move(incoming)));
  return out;
}

void partiallyUnroll(Kernel &kernel, Op &loop, int64_t lbVal, int64_t stepVal,
                     int64_t factor, int64_t mainTrip, int64_t remainder) {
  Block *parent = loop.region->parentBlock;
  if (parent == nullptr) {
    return;
  }
  std::vector<std::unique_ptr<Op>> replacement;

  // Loop-invariant consts, emitted before the loops: the per-copy induction
  // offsets i*step (copies 1..factor-1) and, when a cleanup loop is needed,
  // the main loop's upper bound mid = lb + mainTrip*factor*step.
  std::vector<Value> offsets;
  offsets.reserve(static_cast<size_t>(factor - 1));
  for (int64_t i = 1; i < factor; ++i) {
    std::unique_ptr<Op> c = makeIndexConstOp(i * stepVal);
    offsets.push_back(c->result());
    replacement.push_back(std::move(c));
  }
  std::optional<Value> mid;
  if (remainder != 0) {
    std::unique_ptr<Op> c =
        makeIndexConstOp(lbVal + mainTrip * factor * stepVal);
    mid = c->result();
    replacement.push_back(std::move(c));
  }

  // The main loop: bounds [lb, mid (or ub)], the original iter args, and the
  // factor-copy body. Its results feed the cleanup loop's iter args.
  auto mainLoop = std::make_unique<Op>();
  mainLoop->kind = OpKind::LoopFor;
  mainLoop->operands.push_back(loop.operands[0]); // lb
  mainLoop->operands.push_back(mid.has_value() ? *mid : loop.operands[1]);
  mainLoop->operands.push_back(loop.operands[2]); // step
  for (size_t i = 3; i < loop.operands.size(); ++i) {
    mainLoop->operands.push_back(loop.operands[i]); // iter args ride along
  }
  mainLoop->resultTypes = loop.resultTypes;
  auto mainBody = std::make_unique<Block>();
  mainBody->args = loop.region->args;
  mainBody->parentBlock = parent;
  mainBody->parentOp = mainLoop.get();
  const Value mainInduction = Value::blockArg(mainBody.get(), 0);
  mainBody->ops = buildFactorCopies(loop, offsets, mainBody.get(),
                                    mainInduction);
  if (mainBody->ops.empty()) {
    return; // malformed body; leave the loop alone
  }
  mainLoop->region = std::move(mainBody);

  Op *resultLoop = mainLoop.get();
  std::unique_ptr<Op> cleanup;
  if (remainder != 0) {
    // The cleanup loop: bounds [mid, ub], the MAIN loop's results as iter
    // args, and the ORIGINAL body (one copy) for the leftover iterations.
    cleanup = std::make_unique<Op>();
    cleanup->kind = OpKind::LoopFor;
    cleanup->operands.push_back(*mid);
    cleanup->operands.push_back(loop.operands[1]); // ub
    cleanup->operands.push_back(loop.operands[2]); // step
    for (size_t j = 0; j < loop.resultTypes.size(); ++j) {
      cleanup->operands.push_back(mainLoop->result(static_cast<int64_t>(j)));
    }
    cleanup->resultTypes = loop.resultTypes;
    auto cleanupBody = std::make_unique<Block>();
    cleanupBody->args = loop.region->args;
    cleanupBody->parentBlock = parent;
    cleanupBody->parentOp = cleanup.get();
    std::vector<Value> argMap;
    argMap.reserve(cleanupBody->args.size());
    for (size_t i = 0; i < cleanupBody->args.size(); ++i) {
      argMap.push_back(
          Value::blockArg(cleanupBody.get(), static_cast<int64_t>(i)));
    }
    cleanupBody->ops = cloneBody(loop, argMap, parent);
    if (cleanupBody->ops.empty()) {
      return; // malformed body; leave the loop alone
    }
    cleanup->region = std::move(cleanupBody);
    resultLoop = cleanup.get();
  }

  replacement.push_back(std::move(mainLoop));
  if (cleanup != nullptr) {
    replacement.push_back(std::move(cleanup));
  }

  for (size_t j = 0; j < loop.resultTypes.size(); ++j) {
    rewriteUses(kernel.body, loop.result(static_cast<int64_t>(j)),
                resultLoop->result(static_cast<int64_t>(j)));
  }
  replaceOp(*parent, &loop, std::move(replacement));
}

void unrollLoop(Kernel &kernel, Op &loop, int64_t factor) {
  int64_t trip = 0;
  int64_t remainder = 0;
  int64_t mainTrip = 0;
  int64_t lbVal = 0;
  int64_t ubVal = 0;
  int64_t stepVal = 0;
  switch (planUnroll(loop, factor, trip, remainder, mainTrip, lbVal, ubVal,
                     stepVal)) {
  case UnrollMode::Full:
    fullyUnroll(kernel, loop, lbVal, stepVal, trip);
    break;
  case UnrollMode::Partial:
    partiallyUnroll(kernel, loop, lbVal, stepVal, factor, mainTrip,
                    remainder);
    break;
  case UnrollMode::None:
    break; // documented non-actions
  }
}

std::expected<void, std::string> unrollKernel(Kernel &kernel) {
  const int64_t factor = kernel.schedule.unroll;
  if (factor < 1) {
    return std::unexpected("unroll: kernel '" + kernel.name +
                           "': invalid unroll factor " +
                           std::to_string(factor));
  }
  if (factor == 1) {
    return {}; // documented no-op
  }
  // Pre-order walk: parents before children. Processing the list in REVERSE
  // unrolls innermost loops first, so a loop nested inside an unrolled loop
  // is already unrolled (and its unrolled form cloned) when the enclosing
  // loop is rewritten; the collected pointers stay valid.
  std::vector<Op *> loops;
  walkBlocks(kernel.body, [&](Block &block) {
    for (const auto &op : block.ops) {
      if (op->kind == OpKind::LoopFor) {
        loops.push_back(op.get());
      }
    }
  });
  for (auto it = loops.rbegin(); it != loops.rend(); ++it) {
    unrollLoop(kernel, **it, factor);
  }
  return {};
}

} // namespace

std::expected<void, std::string> unrollPass(Module &module) {
  for (const auto &kernel : module.kernels) {
    if (const auto result = unrollKernel(*kernel); !result) {
      return result;
    }
  }
  return {};
}

// Self-registration (contract: lib/passes/CMakeLists.txt).
static const bool kUnrollRegistered =
    (kernelh::ir::registerPass("unroll", unrollPass), true);

} // namespace kernelh::ir
