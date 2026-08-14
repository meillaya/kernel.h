// lib/passes/tiling.cpp - the tile pass for the atom IR (plan todo 15).
//
// ---------------------------------------------------------------------------
// TILING CONTRACT (consumed by the kernel templates, todos 25-31)
// ---------------------------------------------------------------------------
// The pass consumes a kernel's schedule attrs and rewrites its iteration
// domain into a grid-level block loop (block_id-derived offsets) plus an
// intra-block tile loop. It only acts when
//
//   tile = [T, ...]  (non-empty)   AND   map_blocks = true
//
// and the kernel body contains the SERIAL TILE SKELETON: the outermost loop
// (the first op with a region) is the DOMAIN LOOP over dimension 0,
//
//   %lo = atom.const 0 : index
//   %hi = atom.const E : index      // E = full problem extent, >= 1
//   %st = atom.const 1 : index
//   atom.loop_for %lo to %hi step %st { ^bb0(%i: index): ... }
//
// with the induction %i feeding the tensor indices of the body's loads and
// stores. The pass rewrites the domain loop into the intra-block tile loop
// and makes the outer iteration implicit over gpu blocks:
//
//   * map_blocks:  base = atom.block_id x * T is emitted in the kernel body
//     (block_id-derived offset; no serial outer loop anymore - the grid
//     iterates over the blocks). Every use of the induction inside the body
//     is replaced by the GLOBAL index gi = base + i, and the loop bound
//     becomes the tile extent T. tile[0] is the only extent consumed in v1;
//     the remaining tile entries (BN/BK) ride along for later passes.
//
//   * map_warps:   the tile is additionally PARTITIONED ACROSS WAVEFRONTS.
//     The wavefront index is wf = atom.thread_id / wavefront_size, where
//     wavefront_size comes from the device-limits table
//     (kernelh::device_limits_for("gfx1101").wavefront_size - todo 10 table;
//     never hardcoded). With W = ceil(atom.block_dim / wavefront_size)
//     wavefronts per block, the in-tile row is i * W + wf and the loop trip
//     count becomes ceil(T / W): consecutive wavefronts own strided slices
//     of the tile. W is runtime, so the trip count and the row offset are
//     runtime expressions (atom.div/atom.add on index values).
//
//   * BOUNDARY PREDICATES (pinned): loads/stores must never go out of
//     bounds. Two guards are emitted (atom.cmp lt, combined with atom.min
//     when both apply, the i32 boolean convention):
//       %ok_tile = atom.cmp lt <in-tile row>, <T>   (map_warps only: W is
//                                                    runtime, so divisibility
//                                                    of T by W is unknowable)
//       %ok_ext  = atom.cmp lt <gi>, <E>            (only when E % T != 0)
//     Every load/store index that depends on gi is then replaced by
//     atom.select %ok, <index>, %c0 - PREDICATED INDEXING: out-of-range
//     accesses land on element 0, which is always in bounds, so no access
//     can ever touch an out-of-bounds element.
//
//   * The schedule attrs ride along unchanged and the tiled kernel remains
//     verifier-clean.
//
// NO-OP KERNELS (documented): empty tile, map_blocks == false, no outermost
// loop, or a domain loop that does not match the skeleton (bounds not the
// pinned consts / step not 1 / extent < 1). A NON-POSITIVE tile extent is a
// DIAGNOSTIC ERROR: the pass returns std::expected with a message naming the
// kernel and the offending extent (the verifier also rejects it, but passes
// may run on unverified modules).
//
// Self-registration contract (todos 14-18): see lib/passes/CMakeLists.txt.

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kernelh::ir {
namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

bool isConstOp(const Value &value) {
  return value.kind() == Value::Kind::OpResult && value.op() != nullptr &&
         value.op()->kind == OpKind::Const;
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

// True when `value`'s definition chain (the defining op's operands,
// recursively) reaches `target`. SSA has no cycles, but a visited set keeps
// the walk linear.
bool dependsOn(const Value &value, const Value &target,
               std::unordered_set<const Op *> &visited) {
  if (value == target) {
    return true;
  }
  if (value.kind() != Value::Kind::OpResult || value.op() == nullptr) {
    return false;
  }
  Op *op = value.op();
  if (!visited.insert(op).second) {
    return false;
  }
  for (const Value &operand : op->operands) {
    if (dependsOn(operand, target, visited)) {
      return true;
    }
  }
  return false;
}

bool dependsOn(const Value &value, const Value &target) {
  std::unordered_set<const Op *> visited;
  return dependsOn(value, target, visited);
}

// True when any op anywhere in the block tree uses a value that depends on
// `target` (directly or transitively).
bool anyOpDependsOn(const Block &block, const Value &target) {
  for (const auto &op : block.ops) {
    for (const Value &operand : op->operands) {
      if (dependsOn(operand, target)) {
        return true;
      }
    }
    if (op->region != nullptr && anyOpDependsOn(*op->region, target)) {
      return true;
    }
  }
  return false;
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

// True for the four memory ops whose indices must be boundary-guarded.
bool isMemoryOp(OpKind kind) {
  switch (kind) {
  case OpKind::Load:
  case OpKind::LoadVec:
  case OpKind::Store:
  case OpKind::StoreVec:
    return true;
  default:
    return false;
  }
}

// First index operand of a memory op (skips the value/tensor operands).
size_t firstIndexOperand(const Op &op) {
  return (op.kind == OpKind::Store || op.kind == OpKind::StoreVec) ? 2 : 1;
}

// ---------------------------------------------------------------------------
// Standalone op construction. Ops are built detached and inserted into a
// block at a chosen position (the builder helpers only append).
// ---------------------------------------------------------------------------

std::unique_ptr<Op> makeIdOp(OpKind kind, IdDim dim) {
  auto op = std::make_unique<Op>();
  op->kind = kind;
  op->idDim = dim;
  op->resultTypes.push_back(Type::scalar(ScalarType::Index));
  return op;
}

std::unique_ptr<Op> makeIndexConstOp(int64_t value) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Const;
  op->resultTypes.push_back(Type::scalar(ScalarType::Index));
  op->intValue = value;
  return op;
}

std::unique_ptr<Op> makeBinOp(OpKind kind, Value a, Value b) {
  auto op = std::make_unique<Op>();
  op->kind = kind;
  op->operands = {a, b};
  op->resultTypes.push_back(a.type());
  return op;
}

std::unique_ptr<Op> makeCmpOp(Value a, Value b) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Cmp;
  op->pred = CmpPred::Lt;
  op->operands = {a, b};
  op->resultTypes.push_back(Type::scalar(ScalarType::I32));
  return op;
}

std::unique_ptr<Op> makeSelectOp(Value cond, Value a, Value b) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Select;
  op->operands = {cond, a, b};
  op->resultTypes.push_back(a.type());
  return op;
}

// ---------------------------------------------------------------------------
// The per-kernel transformation
// ---------------------------------------------------------------------------

struct TileOutcome {
  bool changed = false;
  std::string error; // non-empty = hard diagnostic (pass aborts)
};

TileOutcome tileKernel(Kernel &kernel) {
  TileOutcome out;
  const ScheduleAttrs &sched = kernel.schedule;

  // Documented no-ops: no tile attr, or blocks not mapped to the grid.
  if (sched.tile.empty() || !sched.map_blocks) {
    return out;
  }
  // Non-positive tile extents are a diagnostic error (the verifier rejects
  // them too, but passes may run on unverified modules).
  for (int64_t t : sched.tile) {
    if (t <= 0) {
      out.error = "kernel '" + kernel.name + "' has a non-positive tile "
                  "extent " + std::to_string(t) +
                  "; tile sizes must be positive";
      return out;
    }
  }
  const int64_t tileExtent = sched.tile[0];
  const bool mapWarps = sched.map_warps;

  // The domain loop: the outermost region op of the kernel body.
  Op *domain = nullptr;
  for (const auto &op : kernel.body.ops) {
    if (op->region != nullptr) {
      domain = op.get();
      break;
    }
  }
  if (domain == nullptr || domain->kind != OpKind::LoopFor ||
      domain->operands.size() < 3) {
    return out; // no domain loop -> documented no-op
  }
  const Value lb = domain->operands[0];
  const Value ub = domain->operands[1];
  const Value step = domain->operands[2];
  // Skeleton check: lb const 0, ub const E >= 1, step const 1.
  if (!isConstOp(lb) || !lb.op()->intValue.has_value() ||
      *lb.op()->intValue != 0 || !isConstOp(ub) || !ub.op()->intValue.has_value() ||
      !isConstOp(step) || !step.op()->intValue.has_value() ||
      *step.op()->intValue != 1) {
    return out; // skeleton mismatch -> documented no-op
  }
  const int64_t extent = *ub.op()->intValue;
  if (extent < 1) {
    return out;
  }

  Block &body = kernel.body;
  Block &loopBody = *domain->region;
  const Value induction = Value::blockArg(&loopBody, 0);
  const bool inductionUsed = anyOpDependsOn(loopBody, induction);

  // -------------------------------------------------------------------------
  // 1. Pre-loop ops, inserted into the kernel body before the domain loop:
  //    the block_id-derived base offset and, with map_warps, the wavefront
  //    partition math. Their results dominate the tiled loop.
  // -------------------------------------------------------------------------
  const int64_t wavefrontSize =
      kernelh::device_limits_for("gfx1101").wavefront_size; // todo 10 table

  std::vector<std::unique_ptr<Op>> preOps;
  auto emitPre = [&](std::unique_ptr<Op> op) {
    const Value result = op->result();
    preOps.push_back(std::move(op));
    return result;
  };

  const Value c0 = lb;       // const 0 (reused: guard fallback index)
  const Value cExtent = ub;  // const E (reused: extent guard)
  const Value c1 = step;     // const 1 (reused: step + increments)

  const Value blockId = emitPre(makeIdOp(OpKind::BlockId, IdDim::X));
  const Value cTile = emitPre(makeIndexConstOp(tileExtent));
  const Value base = emitPre(makeBinOp(OpKind::Mul, blockId, cTile));

  Value loopUpperBound = cTile; // without map_warps the tile loop runs [0, T)
  std::optional<Value> wavefront;  // wf = thread_id / wavefront_size
  std::optional<Value> wavefronts; // W  = ceil(block_dim / wavefront_size)
  if (mapWarps) {
    const Value cWave = emitPre(makeIndexConstOp(wavefrontSize));
    const Value threadId = emitPre(makeIdOp(OpKind::ThreadId, IdDim::X));
    wavefront = emitPre(makeBinOp(OpKind::Div, threadId, cWave));
    const Value blockDim = emitPre(makeIdOp(OpKind::BlockDim, IdDim::X));
    const Value blockDimM1 = emitPre(makeBinOp(OpKind::Sub, blockDim, c1));
    const Value blockDimRounded =
        emitPre(makeBinOp(OpKind::Add, blockDimM1, cWave));
    wavefronts = emitPre(makeBinOp(OpKind::Div, blockDimRounded, cWave));
    // Trip count = ceil(T / W): each wavefront strides through the tile.
    const Value wfM1 = emitPre(makeBinOp(OpKind::Sub, *wavefronts, c1));
    const Value tripT = emitPre(makeBinOp(OpKind::Add, wfM1, cTile));
    loopUpperBound = emitPre(makeBinOp(OpKind::Div, tripT, *wavefronts));
  }

  // -------------------------------------------------------------------------
  // 2. Loop-body prologue: the in-tile row + the global index gi = base +
  //    row, then the boundary guards. Inserted at the top of the loop body
  //    so every rewritten use is dominated.
  // -------------------------------------------------------------------------
  std::vector<std::unique_ptr<Op>> prologue;
  auto emitPrologue = [&](std::unique_ptr<Op> op) {
    const Value result = op->result();
    prologue.push_back(std::move(op));
    return result;
  };

  Value row = induction;
  std::optional<Value> gi;
  if (inductionUsed) {
    if (mapWarps) {
      row = emitPrologue(makeBinOp(OpKind::Mul, induction, *wavefronts));
      row = emitPrologue(makeBinOp(OpKind::Add, row, *wavefront));
    }
    gi = emitPrologue(makeBinOp(OpKind::Add, base, row));
    rewriteUses(loopBody, induction, *gi);
  }

  // Boundary predicates are only useful when an out-of-range index is
  // possible AND some memory access actually depends on gi.
  const bool tileGuardNeeded = mapWarps;           // W is runtime
  const bool extentGuardNeeded = (extent % tileExtent) != 0;
  bool anyGuardedAccess = false;
  if (gi.has_value()) {
    walkBlocks(loopBody, [&](Block &block) {
      for (const auto &op : block.ops) {
        if (isMemoryOp(op->kind)) {
          for (size_t i = firstIndexOperand(*op); i < op->operands.size();
               ++i) {
            if (dependsOn(op->operands[i], *gi)) {
              anyGuardedAccess = true;
            }
          }
        }
      }
    });
  }

  std::optional<Value> ok;
  if (anyGuardedAccess && (tileGuardNeeded || extentGuardNeeded)) {
    if (tileGuardNeeded) {
      ok = emitPrologue(makeCmpOp(row, cTile)); // in-tile row < T
    }
    if (extentGuardNeeded) {
      const Value okExtent = emitPrologue(makeCmpOp(*gi, cExtent)); // gi < E
      if (ok.has_value()) {
        ok = emitPrologue(makeBinOp(OpKind::Min, *ok, okExtent)); // both
      } else {
        ok = okExtent;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 3. Predicated indexing: every load/store index that depends on gi is
  //    replaced by atom.select %ok, <index>, %c0 - out-of-range accesses
  //    land on element 0, which is always in bounds (never out-of-bounds).
  // -------------------------------------------------------------------------
  if (ok.has_value()) {
    walkBlocks(loopBody, [&](Block &block) {
      std::vector<std::unique_ptr<Op>> rebuilt;
      rebuilt.reserve(block.ops.size());
      for (auto &opPtr : block.ops) {
        Op &op = *opPtr;
        if (isMemoryOp(op.kind)) {
          for (size_t i = firstIndexOperand(op); i < op.operands.size(); ++i) {
            if (dependsOn(op.operands[i], *gi)) {
              auto sel = makeSelectOp(*ok, op.operands[i], c0);
              const Value selResult = sel->result();
              rebuilt.push_back(std::move(sel));
              op.operands[i] = selResult;
            }
          }
        }
        rebuilt.push_back(std::move(opPtr));
      }
      block.ops = std::move(rebuilt);
    });
  }

  if (!prologue.empty()) {
    loopBody.ops.insert(loopBody.ops.begin(),
                        std::make_move_iterator(prologue.begin()),
                        std::make_move_iterator(prologue.end()));
  }

  // -------------------------------------------------------------------------
  // 4. Rebuild the loop with the tiled bounds; the body block is moved in
  //    place (its parentBlock stays the kernel body; parentOp now points at
  //    the new loop).
  // -------------------------------------------------------------------------
  auto newLoop = std::make_unique<Op>();
  newLoop->kind = OpKind::LoopFor;
  newLoop->operands.push_back(lb);
  newLoop->operands.push_back(loopUpperBound);
  newLoop->operands.push_back(step);
  for (size_t i = 3; i < domain->operands.size(); ++i) {
    newLoop->operands.push_back(domain->operands[i]); // iter args ride along
  }
  newLoop->resultTypes = domain->resultTypes;

  // Uses of the OLD loop's results (iter-arg loops) resolve to the new
  // loop's results. Done before the region moves so the whole tree is
  // walked.
  for (size_t i = 0; i < domain->resultTypes.size(); ++i) {
    rewriteUses(kernel.body, domain->result(static_cast<int64_t>(i)),
                newLoop->result(static_cast<int64_t>(i)));
  }

  newLoop->region = std::move(domain->region);
  newLoop->region->parentOp = newLoop.get();

  auto &ops = body.ops;
  const auto it = std::find_if(ops.begin(), ops.end(),
                               [domain](const std::unique_ptr<Op> &p) {
                                 return p.get() == domain;
                               });
  if (it == ops.end()) {
    return out; // unreachable: `domain` came from this exact list
  }
  const size_t pos = static_cast<size_t>(it - ops.begin());
  ops[pos] = std::move(newLoop); // destroys the old domain op
  ops.insert(ops.begin() + static_cast<std::ptrdiff_t>(pos),
             std::make_move_iterator(preOps.begin()),
             std::make_move_iterator(preOps.end()));

  out.changed = true;
  return out;
}

} // namespace

std::expected<void, std::string> tilePass(Module &module) {
  for (const auto &kernel : module.kernels) {
    const TileOutcome outcome = tileKernel(*kernel);
    if (!outcome.error.empty()) {
      return std::unexpected(outcome.error);
    }
  }
  return {};
}

// Self-registration (contract: lib/passes/CMakeLists.txt).
static const bool kTileRegistered =
    (kernelh::ir::registerPass("tile", tilePass), true);

} // namespace kernelh::ir
