// lib/passes/promote_shared.cpp - the promote-shared pass for the atom IR
// (plan todo 16).
//
// When a kernel's schedule attr `use_shared_memory` is true, the pass stages
// the kernel's TILE LOADS (the per-thread loads from global tensors inside
// the tiled iteration domain) through STATIC shared memory:
//
//   - one atom.alloc_shared buffer per staged global tensor, sized from the
//     schedule tile dims x element size (a compile-time constant - plan G9:
//     NO dynamic shared memory; the size comes from TensorType::sizeBytes,
//     the same element-size table the verifier uses),
//   - a store stage per tile loop: either a whole-tile atom.copy global->shared
//     (fires when the staged tensor IS the tile - atom.copy requires identical
//     operand shapes, so this is the only legal whole-tile copy), or per-element
//     store staging when the tensor is larger than the tile,
//   - barriers per the PINNED RULE (plan G20): ONE atom.barrier AFTER the
//     last shared store of a tile stage and BEFORE the first shared load of
//     that stage, and ONE atom.barrier at the END of each tile iteration
//     (before the next stage's stores overwrite shared),
//   - the compute's global loads are rewritten to loads from the shared
//     buffers (the compute section no longer touches global memory for the
//     staged tensors).
//
// CAPACITY CHECK (pinned): the total static shared after promotion (the
// kernel's existing alloc_shared bytes + the planned buffers) must fit the
// device LDS budget read from the device-limits table
// (kernelh::device_limits_for("gfx1101").lds_bytes_per_workgroup - NEVER
// hardcoded). On overflow the pass returns a std::expected error naming the
// tile config; the module is left UNTOUCHED (the error feeds the launch
// verifier / autotuner later).
//
// FIRING SHAPE (consumes the OUTPUT of the tile pass - the contract in
// lib/passes/tiling.cpp): the kernel body holds one or more TILE loops (the
// rewritten block-level loop); the tile loop's body (or a nested loop body
// below it - e.g. the N/K loops of a tiled GEMM) contains the tile loads:
// atom.load of a global-space tensor at indices that are either tile-local
// or block_id-derived offsets (possibly wrapped in the tile pass's boundary
// atom.select guards). Loads in the tile loop's subtree are promoted; the
// staging structure goes at the tile-loop body top.
//
// SHARED-SHAPE RULE: for a rank-r tile load, an index whose definition chain
// reaches atom.block_id maps to the next OUTER tile dim (BM, BN, ... in order
// of appearance); a PURE index maps to the remaining tile dims in REVERSE
// order (BK, then BN, ... - this is what gives A (BM x BK) and B (BK x BN)
// their distinct buffers from the one tile vector). A rank-1 load maps to
// tile[0]. STORE-MODE INDEX RULE: the shared index of a load index add(a, b)
// with exactly one block_id-bearing operand is the OTHER operand; a pure
// index is used as-is. A load whose index cannot decompose stays global
// (best-effort). Store mode only promotes loads in the tile loop's DIRECT
// body (the per-element store must precede the barrier - a documented v1
// limitation); copy-mode loads are promoted at any depth (the whole-tile
// copy has no ordering dependency on the loads).
//
// If use_shared_memory is false the pass is a no-op for that kernel. The pass
// is idempotent: a second run sees no promotable global loads (copy mode
// removes them; store mode leaves them consumed only by shared stores, which
// the planner detects and skips).
//
// Self-registration contract (todos 14-18): see lib/passes/CMakeLists.txt.

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace kernelh::ir {
namespace {

// True when `value`'s definition chain (const/arith/select over id ops)
// reaches an atom.block_id. The IR is a tree (structured SSA), so the
// recursion terminates.
bool containsBlockId(const Value &value) {
  if (value.kind() != Value::Kind::OpResult || value.op() == nullptr) {
    return false;
  }
  const Op &op = *value.op();
  switch (op.kind) {
  case OpKind::BlockId:
    return true;
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Max:
  case OpKind::Min:
    return op.operands.size() == 2 &&
           (containsBlockId(op.operands[0]) ||
            containsBlockId(op.operands[1]));
  case OpKind::Select:
    return op.operands.size() == 3 &&
           (containsBlockId(op.operands[1]) ||
            containsBlockId(op.operands[2]));
  default:
    return false;
  }
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

template <typename Fn>
void walkBlocks(const Block &block, const Fn &fn) {
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

// Like rewriteAllUses, but leaves `except` untouched (store mode: the store
// keeps consuming the global load it was created for).
void rewriteUsesExcept(Kernel &kernel, const Value &from, const Value &to,
                       const Op *except) {
  walkBlocks(kernel.body, [&](const Block &block) {
    for (const auto &op : block.ops) {
      if (op.get() == except) {
        continue;
      }
      for (Value &operand : op->operands) {
        if (operand == from) {
          operand = to;
        }
      }
    }
  });
}

// True when every use of the load's result is a store into a shared-space
// tensor - the shape a store-mode promotion run leaves behind (the load then
// only feeds its own store). Dead loads (no uses at all) are skipped too.
bool alreadyStagedOrDead(const Kernel &kernel, const Op &load) {
  bool hasUse = false;
  bool allUsesAreSharedStores = true;
  walkBlocks(kernel.body, [&](const Block &block) {
    for (const auto &op : block.ops) {
      for (const Value &operand : op->operands) {
        if (operand == load.result()) {
          hasUse = true;
          if (op->kind != OpKind::Store || op->operands.size() < 2 ||
              !op->operands[1].type().isTensor() ||
              op->operands[1].type().asTensor().space != MemSpace::Shared) {
            allUsesAreSharedStores = false;
          }
        }
      }
    }
  });
  return !hasUse || allUsesAreSharedStores;
}

// Derives the static shared buffer type for a rank-r tile load from the
// schedule tile vector (rule documented at the top of this file). Returns
// nullopt when no shape can be derived (e.g. tile too short for the rank).
std::optional<TensorType> deriveSharedType(const Kernel &kernel,
                                           const Op &load) {
  if (load.operands.empty() || !load.operands[0].type().isTensor()) {
    return std::nullopt;
  }
  const TensorType global = load.operands[0].type().asTensor();
  const std::vector<int64_t> &tile = kernel.schedule.tile;
  const size_t rank = global.shape.size();
  if (rank == 0 || tile.empty() || tile.size() < rank) {
    return std::nullopt;
  }
  std::vector<int64_t> shape;
  if (rank == 1) {
    shape.push_back(tile[0]);
  } else {
    size_t outer = 0;               // next outer tile dim (BM, BN, ...)
    size_t inner = tile.size() - 1; // next remaining tile dim (BK, ...)
    for (size_t d = 0; d < rank; ++d) {
      if (containsBlockId(load.operands[1 + d])) {
        shape.push_back(tile[outer]);
        ++outer;
      } else {
        shape.push_back(tile[inner]);
        --inner;
      }
    }
  }
  return TensorType{std::move(shape), global.elem, MemSpace::Shared};
}

// The tile-local part of a load index: for add(a, b) with exactly one
// block_id-bearing operand, the OTHER operand; a pure index maps to itself.
// Value::none() when no tile-local part exists (block-only index).
Value tileLocalPart(const Value &index) {
  if (!containsBlockId(index)) {
    return index;
  }
  if (index.kind() != Value::Kind::OpResult || index.op() == nullptr ||
      index.op()->kind != OpKind::Add || index.op()->operands.size() != 2) {
    return Value::none();
  }
  const Op &op = *index.op();
  if (containsBlockId(op.operands[0]) && !containsBlockId(op.operands[1])) {
    return op.operands[1];
  }
  if (containsBlockId(op.operands[1]) && !containsBlockId(op.operands[0])) {
    return op.operands[0];
  }
  return Value::none();
}

// ---------------------------------------------------------------------------
// Planning (read-only; the capacity check runs on the plan before anything is
// rewritten, so an overflow leaves the module untouched).
// ---------------------------------------------------------------------------

struct GroupPlan {
  Value tensor;          // the global tensor being staged
  TensorType sharedType; // static tile-shaped shared buffer type
  bool wholeTileCopy;    // true: the tensor IS the tile -> atom.copy fires
  std::vector<Op *> loads; // the tile loads of `tensor` in the loop subtree
};

struct LoopPlan {
  Block *body = nullptr; // the tile loop's direct body
  std::vector<GroupPlan> groups;
  int64_t addedBytes = 0;
};

// Finds the block that directly owns `target` inside the subtree of `root`.
Block *blockContaining(Block &root, const Op *target) {
  for (const auto &op : root.ops) {
    if (op.get() == target) {
      return &root;
    }
    if (op->region != nullptr) {
      if (Block *found = blockContaining(*op->region, target)) {
        return found;
      }
    }
  }
  return nullptr;
}

// Plans the promotion of one tile-loop body. Copy-mode loads are planned
// anywhere in the subtree; store-mode loads only in the loop's DIRECT body
// (the per-element store must precede the barrier). Returns false when the
// body has no promotable loads.
bool planLoopBody(const Kernel &kernel, Block &body, LoopPlan &plan) {
  plan.body = &body;
  const auto collectLoads = [&](Block &block, bool direct) {
    for (const auto &op : block.ops) {
      if (op->kind != OpKind::Load || op->operands.empty()) {
        continue;
      }
      const Type &tensorType = op->operands[0].type();
      if (!tensorType.isTensor() ||
          tensorType.asTensor().space != MemSpace::Global) {
        continue;
      }
      if (alreadyStagedOrDead(kernel, *op)) {
        continue; // store-mode idempotency / dead-load skip
      }
      GroupPlan *group = nullptr;
      for (GroupPlan &g : plan.groups) {
        if (g.tensor == op->operands[0]) {
          group = &g;
          break;
        }
      }
      if (group == nullptr) {
        const auto shared = deriveSharedType(kernel, *op);
        if (!shared) {
          continue; // cannot derive a tile shape - leave this load global
        }
        const TensorType global = tensorType.asTensor();
        GroupPlan g;
        g.tensor = op->operands[0];
        g.sharedType = *shared;
        // atom.copy requires identical operand shapes: the whole-tile copy
        // is only legal when the staged tensor IS the tile.
        g.wholeTileCopy = (global.shape == shared->shape);
        plan.groups.push_back(std::move(g));
        group = &plan.groups.back();
      }
      if (!group->wholeTileCopy && !direct) {
        continue; // store mode: nested loads stay global (v1 limitation)
      }
      if (!group->wholeTileCopy) {
        // Store mode: every index must decompose into a tile-local part.
        bool decomposable = true;
        for (size_t d = 1; d < op->operands.size(); ++d) {
          if (tileLocalPart(op->operands[d]).kind() == Value::Kind::None) {
            decomposable = false;
            break;
          }
        }
        if (!decomposable) {
          continue;
        }
      }
      group->loads.push_back(op.get());
    }
  };
  collectLoads(body, true);
  for (const auto &op : body.ops) {
    if (op->region != nullptr) {
      walkBlocks(*op->region, [&](Block &block) {
        collectLoads(block, false);
      });
    }
  }
  for (GroupPlan &g : plan.groups) {
    plan.addedBytes += g.sharedType.sizeBytes();
  }
  return !plan.groups.empty();
}

// Capacity check (pinned): existing shared bytes + the planned buffers vs the
// device LDS budget. On overflow returns an error naming the tile config.
std::expected<void, std::string> checkCapacity(
    const Kernel &kernel, const std::vector<LoopPlan> &plans,
    const kernelh::DeviceLimits &limits) {
  int64_t total = kernelTotalSharedBytes(kernel);
  for (const LoopPlan &plan : plans) {
    total += plan.addedBytes;
  }
  const int64_t limit = limits.lds_bytes_per_workgroup;
  if (total > limit) {
    std::string tileText;
    for (size_t i = 0; i < kernel.schedule.tile.size(); ++i) {
      if (i != 0) {
        tileText += ", ";
      }
      tileText += std::to_string(kernel.schedule.tile[i]);
    }
    return std::unexpected(
        "shared memory usage " + std::to_string(total) +
        " bytes for tile config [" + tileText + "] exceeds the " +
        std::to_string(limit) + "-byte LDS limit for target '" +
        limits.name + "'");
  }
  return {};
}

// ---------------------------------------------------------------------------
// Rewriting (runs only after the capacity check passed).
// ---------------------------------------------------------------------------

// Splices every op of `scratch` into `block` at `position`.
void spliceAt(Block &block, size_t position, Block &scratch) {
  block.ops.insert(
      block.ops.begin() + static_cast<std::ptrdiff_t>(position),
      std::make_move_iterator(scratch.ops.begin()),
      std::make_move_iterator(scratch.ops.end()));
}

// Replaces the copy-mode loads in place with shared loads (the shared loads
// land exactly where the global loads were, so dominance is preserved at any
// nesting depth), then erases the global loads.
void replaceCopyModeLoads(Kernel &kernel, Block &tileBody,
                          const GroupPlan &group, const Value &buffer) {
  for (const Op *load : group.loads) {
    Block *loadBlock = blockContaining(tileBody, load);
    if (loadBlock == nullptr) {
      continue; // defensive: load no longer in the tree
    }
    const auto it = std::find_if(
        loadBlock->ops.begin(), loadBlock->ops.end(),
        [load](const std::unique_ptr<Op> &p) { return p.get() == load; });
    if (it == loadBlock->ops.end()) {
      continue;
    }
    const size_t pos = static_cast<size_t>(it - loadBlock->ops.begin());
    Block scratch;
    std::vector<Value> indices(load->operands.begin() + 1,
                               load->operands.end());
    Op *sharedLoad = builder::makeLoad(scratch, buffer, indices);
    spliceAt(*loadBlock, pos, scratch);
    rewriteAllUses(kernel, load->result(), sharedLoad->result());
    // The shared load now sits at `pos`; the global load shifted to pos + 1.
    loadBlock->ops.erase(loadBlock->ops.begin() +
                         static_cast<std::ptrdiff_t>(pos + 1));
  }
}

// Rebuilds one tile-loop body with the pinned stage structure:
//
//   [alloc_shared...]                   (block-tile scope)
//   [atom.copy global->shared ...]      (copy mode: the whole-tile stores)
//   [global loads + shared stores ...]  (store mode: the store stage)
//   [atom.barrier]                      (after the last shared store,
//                                        before the first shared load)
//   [shared loads ...]                  (the compute's loads, rewritten)
//   ... compute ops ...
//   [atom.barrier]                      (end of the tile iteration)
//   [atom.yield]
void applyLoopPlan(Kernel &kernel, const LoopPlan &plan) {
  Block &body = *plan.body;

  std::vector<Op *> originalOps;
  for (const auto &op : body.ops) {
    originalOps.push_back(op.get());
  }
  std::vector<bool> isStoreLoad(originalOps.size(), false);
  for (size_t i = 0; i < originalOps.size(); ++i) {
    for (const GroupPlan &g : plan.groups) {
      if (g.wholeTileCopy) {
        continue;
      }
      for (const Op *load : g.loads) {
        if (load == originalOps[i]) {
          isStoreLoad[i] = true;
        }
      }
    }
  }

  Block rebuilt;
  std::vector<Value> buffers;
  for (const GroupPlan &g : plan.groups) {
    buffers.push_back(builder::makeAllocShared(rebuilt, g.sharedType)->result());
  }
  std::vector<Value> loadBuffer(originalOps.size());
  std::vector<std::vector<Value>> loadLocalIndices(originalOps.size());
  std::vector<Op *> loadStore(originalOps.size(), nullptr);
  // Rewrites deferred until the rebuilt body is attached (the tree must be
  // complete before any use-rewriting walk).
  std::vector<std::tuple<Value, Value, const Op *>> deferredRewrites;
  size_t groupIdx = 0;
  for (const GroupPlan &g : plan.groups) {
    if (g.wholeTileCopy) {
      builder::makeCopy(rebuilt, g.tensor, buffers[groupIdx]);
    } else {
      // Move the promoted global loads to the store stage and store each
      // loaded value into its buffer at the decomposed tile-local indices.
      for (const Op *load : g.loads) {
        for (size_t i = 0; i < originalOps.size(); ++i) {
          if (originalOps[i] != load) {
            continue;
          }
          rebuilt.ops.push_back(std::move(body.ops[i]));
          std::vector<Value> local;
          for (size_t d = 1; d < load->operands.size(); ++d) {
            local.push_back(tileLocalPart(load->operands[d]));
          }
          loadLocalIndices[i] = local;
          loadBuffer[i] = buffers[groupIdx];
          loadStore[i] =
              builder::makeStore(rebuilt, load->result(), buffers[groupIdx],
                                 local);
          break;
        }
      }
    }
    ++groupIdx;
  }
  // One barrier: after the last shared store, before the first shared load
  // (pinned rule).
  builder::makeBarrier(rebuilt);
  // The compute's loads: one shared load per store-mode load; the global
  // loads' uses are rewritten to them (except the store's own use) once the
  // tree is complete.
  groupIdx = 0;
  for (const GroupPlan &g : plan.groups) {
    if (g.wholeTileCopy) {
      ++groupIdx;
      continue;
    }
    for (const Op *load : g.loads) {
      for (size_t i = 0; i < originalOps.size(); ++i) {
        if (originalOps[i] != load) {
          continue;
        }
        Op *sharedLoad =
            builder::makeLoad(rebuilt, loadBuffer[i], loadLocalIndices[i]);
        deferredRewrites.emplace_back(load->result(), sharedLoad->result(),
                                      loadStore[i]);
        break;
      }
    }
    ++groupIdx;
  }
  // Remaining ops in order; the end-of-iteration barrier goes right before
  // the yield (pinned rule).
  for (size_t i = 0; i < originalOps.size(); ++i) {
    if (isStoreLoad[i]) {
      continue;
    }
    if (originalOps[i]->kind == OpKind::Yield) {
      builder::makeBarrier(rebuilt);
    }
    rebuilt.ops.push_back(std::move(body.ops[i]));
  }
  body.ops = std::move(rebuilt.ops);

  // The tree is complete: apply the deferred use rewrites.
  for (const auto &[from, to, except] : deferredRewrites) {
    rewriteUsesExcept(kernel, from, to, except);
  }
  // Copy-mode loads are replaced in place at their original positions
  // (any nesting depth).
  groupIdx = 0;
  for (const GroupPlan &g : plan.groups) {
    if (g.wholeTileCopy) {
      replaceCopyModeLoads(kernel, body, g, buffers[groupIdx]);
    }
    ++groupIdx;
  }
}

std::expected<void, std::string> promoteSharedKernel(
    Kernel &kernel, const kernelh::DeviceLimits &limits) {
  // Plan every tile loop (direct child of the kernel body).
  std::vector<LoopPlan> plans;
  for (const auto &op : kernel.body.ops) {
    if (op->kind == OpKind::LoopFor && op->region != nullptr) {
      LoopPlan plan;
      if (planLoopBody(kernel, *op->region, plan)) {
        plans.push_back(std::move(plan));
      }
    }
  }
  if (plans.empty()) {
    return {}; // nothing to promote
  }
  // Capacity check BEFORE any rewrite: overflow leaves the module untouched
  // and fails with a diagnostic naming the tile config.
  if (auto error = checkCapacity(kernel, plans, limits); !error) {
    return error;
  }
  for (const LoopPlan &plan : plans) {
    applyLoopPlan(kernel, plan);
  }
  return {};
}

} // namespace

std::expected<void, std::string> promoteSharedPass(Module &module) {
  // The LDS budget comes from the device-limits table (single source of
  // truth - never hardcoded). "gfx1101" is a table entry, so this never
  // throws.
  const kernelh::DeviceLimits &limits =
      kernelh::device_limits_for("gfx1101");
  for (const auto &kernel : module.kernels) {
    if (!kernel->schedule.use_shared_memory) {
      continue; // documented no-op for this kernel
    }
    if (auto error = promoteSharedKernel(*kernel, limits); !error) {
      return error;
    }
  }
  return {};
}

// Self-registration (contract: lib/passes/CMakeLists.txt).
static const bool kPromoteSharedRegistered =
    (kernelh::ir::registerPass("promote-shared", promoteSharedPass), true);

} // namespace kernelh::ir
