// lib/passes/fusion.cpp - the fuse pass for the atom IR (plan todo 14).
//
// Fuses producer-consumer ELEMENTWISE chains that share the same iteration
// domain into ONE atom.loop_for nest, eliminating the intermediate tensor when
// its only use is inside the fused chain.
//
// Target shape (before):
//
//   %t  = atom.alloc_shared : tensor<Nxf32, shared>        // intermediate
//   %p  = atom.loop_for %lb to %ub step %st iter_args() -> () {
//     ^bb0(%i: index):
//       ... elementwise chain (loads, arith, math) ...
//       atom.store %w, %t[%i] : tensor<Nxf32, shared>       // producer store
//       atom.yield
//   }
//   %c  = atom.loop_for %lb to %ub step %st iter_args() -> () {
//     ^bb0(%j: index):
//       %u  = atom.load %t[%j] : tensor<Nxf32, shared>      // consumer load
//       %z  = atom.math tanh %u : f32
//       atom.store %z, %out[%j] : tensor<Nxf32, global>
//       atom.yield
//   }
//
// After:
//
//   %p  = atom.loop_for %lb to %ub step %st iter_args() -> () {
//     ^bb0(%i: index):
//       ... elementwise chain ...
//       %z  = atom.math tanh %w : f32                        // load of %t gone
//       atom.store %z, %out[%i] : tensor<Nxf32, global>
//       atom.yield
//   }
//
// and the intermediate %t is eliminated: the alloc_shared is dead and erased.
//
// FIRE CONDITIONS (pinned; all must hold, else the pair is left untouched):
//
//   (a) SAME DOMAIN: the producer and consumer loops must be in the SAME
//       block, producer before consumer, with handle-identical lb/ub/step
//       values (the same SSA definitions). Both loops must be elementwise:
//       zero iter_args (no loop-carried values), a body of pure elementwise
//       ops (const/add/sub/mul/div/fma/max/min/cmp/select/math/load/store/
//       thread_id/block_id/block_dim/grid_dim) ending in an empty atom.yield.
//       Vector memory ops, atom.copy, nested loop_for, atom.alloc_shared and
//       atom.barrier in the bodies disqualify the loop.
//
//   (b) SINGLE USE: the intermediate tensor %t is used ONLY by the fused
//       chain - exactly one atom.store to %t (the producer store, whose index
//       is the producer's induction variable) and the consumer's atom.loads of
//       %t (each indexed by the consumer's induction variable). ANY other use
//       of %t anywhere in the kernel (an extra load, a store, an atom.copy, an
//       atom.reduce operand, a use in a nested scope) blocks fusion. The
//       intermediate must be rank-1; the index correspondence store[i] ->
//       load[i] is exactly the loop-domain correspondence.
//
//   (c) NO BARRIER: there is no atom.barrier between the two loops in their
//       parent block (a barrier is a memory-ordering point: the consumer's
//       loads must read what the producer's stores wrote, and fusing across a
//       barrier would hoist those loads before it). Barriers inside either
//       loop body also disqualify the loop (see (a)).
//
//   (d) NO REDUCTION BOUNDARY: fusion never crosses an atom.reduce - the
//       consumer's loads of %t must feed pure elementwise computation, so an
//       atom.reduce anywhere in either body disqualifies the loop (a), a
//       reduction-style consumer loop (non-empty iter_args) is disqualified
//       by (a), and an atom.reduce whose operand is %t is a use outside the
//       chain, blocked by (b). Elementwise producers/consumers AROUND a
//       reduction still fuse with each other; the reduction itself is never
//       part of a fused chain.
//
//   (e) VERIFIER CLEAN: the rewrite is type-preserving (the producer's stored
//       value and the consumer's loaded value both have the intermediate's
//       element type) and dominance-preserving (the producer body is placed
//       before the consumer body in the fused loop), so a verified-clean
//       input stays verified-clean. Passes run on verified-clean IR (the
//       tool verifies at --load-atom time); callers may re-verify.
//
// CHAINS: fusion runs to a fixpoint per block, so a chain of N loops
// (t1 = f(...); t2 = g(t1); out = h(t2)) collapses to one loop in N-1 fusion
// steps. A consumer that loads several intermediates fuses with each
// producer in turn.
//
// Self-registration contract (todos 14-18): see lib/passes/CMakeLists.txt.

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/value.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

namespace kernelh::ir {
namespace {

// Pure elementwise op kinds allowed in a fuse-eligible loop body. Anything
// else (barrier, reduce, copy, vector memory ops, nested loops, alloc_shared)
// disqualifies the loop: the pinned semantics fuse only elementwise chains
// and never across barriers or reductions.
bool isElementwiseOpKind(OpKind kind) {
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
  case OpKind::Store:
  case OpKind::ThreadId:
  case OpKind::BlockId:
  case OpKind::BlockDim:
  case OpKind::GridDim:
    return true;
  case OpKind::LoadVec:
  case OpKind::StoreVec:
  case OpKind::AllocShared:
  case OpKind::Barrier:
  case OpKind::LoopFor:
  case OpKind::Yield:
  case OpKind::Reduce:
  case OpKind::Copy:
    return false;
  }
  return false;
}

// A fuse-eligible loop: atom.loop_for with EXACTLY lb/ub/step operands (zero
// iter_args -> no loop-carried values, so no reduction-style loop), a body
// with one block argument (the induction var), only elementwise ops, and an
// EMPTY atom.yield as the last op.
bool isElementwiseLoop(const Op &op) {
  if (op.kind != OpKind::LoopFor) {
    return false;
  }
  if (op.operands.size() != 3) {
    return false; // lb, ub, step - any iter_args disqualifies
  }
  if (op.region == nullptr || op.region->args.size() != 1 ||
      op.region->ops.empty()) {
    return false;
  }
  for (size_t i = 0; i + 1 < op.region->ops.size(); ++i) {
    if (!isElementwiseOpKind(op.region->ops[i]->kind)) {
      return false;
    }
  }
  const Op &last = *op.region->ops.back();
  return last.kind == OpKind::Yield && last.operands.empty();
}

// Fire condition (a): handle-identical lb/ub/step (same SSA definitions).
bool sameDomain(const Op &a, const Op &b) {
  return a.operands[0] == b.operands[0] && a.operands[1] == b.operands[1] &&
         a.operands[2] == b.operands[2];
}

// Fire condition (c): no atom.barrier strictly between the two loops.
bool barrierBetween(const Block &block, size_t first, size_t last) {
  for (size_t i = first + 1; i < last; ++i) {
    if (block.ops[i]->kind == OpKind::Barrier) {
      return true;
    }
  }
  return false;
}

// Fire condition (b): every use of `intermediate` in the whole kernel is
// either the single producer store or one of the consumer's loads.
bool usesOnlyChain(const Block &block, const Value &intermediate,
                   const Op *producerStore, const std::vector<Op *> &cLoads) {
  for (const auto &op : block.ops) {
    for (const Value &operand : op->operands) {
      if (operand != intermediate) {
        continue;
      }
      bool allowed = (op.get() == producerStore);
      if (!allowed) {
        for (const Op *load : cLoads) {
          if (op.get() == load) {
            allowed = true;
            break;
          }
        }
      }
      if (!allowed) {
        return false;
      }
    }
    if (op->region != nullptr &&
        !usesOnlyChain(*op->region, intermediate, producerStore, cLoads)) {
      return false;
    }
  }
  return true;
}

// True when `value` is used anywhere in the block tree rooted at `block`.
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

// Erases `victim` from whichever block in the tree owns it (it must exist).
void eraseOpFromBlocks(Block &block, const Op *victim) {
  for (size_t i = 0; i < block.ops.size(); ++i) {
    if (block.ops[i].get() == victim) {
      block.ops.erase(block.ops.begin() + i);
      return;
    }
    if (block.ops[i]->region != nullptr) {
      eraseOpFromBlocks(*block.ops[i]->region, victim);
    }
  }
}

// A fusable (producer, consumer) pair plus the intermediate chain metadata.
struct Fusion {
  size_t pIndex = 0;
  size_t cIndex = 0;
  Op *producer = nullptr;
  Op *consumer = nullptr;
  Op *store = nullptr;   // the producer's single store to `intermediate`
  Value intermediate;    // the tensor being eliminated
  Value storeValue;      // the stored scalar replacing the consumer's loads
  Value pInduction;      // producer's induction var
  Value cInduction;      // consumer's induction var
};

// Finds the first fusable pair in `block` (fixpoint loop). All fire
// conditions are checked here; returns false when nothing fuses.
bool findFusion(Block &block, const Kernel &kernel, Fusion &out) {
  for (size_t i = 0; i < block.ops.size(); ++i) {
    Op &p = *block.ops[i];
    if (!isElementwiseLoop(p)) {
      continue;
    }
    const Value pInd = Value::blockArg(p.region.get(), 0);
    for (const auto &bodyOp : p.region->ops) {
      if (bodyOp->kind != OpKind::Store) {
        continue;
      }
      Op &s = *bodyOp;
      const Value t = s.operands[1];
      const Type tType = t.type(); // copy: avoid the dangling-ref bug class
      if (!tType.isTensor() || tType.asTensor().rank() != 1) {
        continue; // only rank-1 intermediates fuse (single induction var)
      }
      // The producer store must write exactly the current iteration's
      // element: store[%i].
      if (s.operands.size() != 3 || s.operands[2] != pInd) {
        continue;
      }
      for (size_t j = i + 1; j < block.ops.size(); ++j) {
        Op &c = *block.ops[j];
        if (!isElementwiseLoop(c) || !sameDomain(p, c)) {
          continue;
        }
        if (barrierBetween(block, i, j)) {
          continue; // fire condition (c)
        }
        // The consumer's loads of `t` must each read the current iteration's
        // element: load[%j].
        const Value cInd = Value::blockArg(c.region.get(), 0);
        std::vector<Op *> cLoads;
        bool loadsOk = true;
        for (const auto &cop : c.region->ops) {
          if (cop->kind != OpKind::Load || cop->operands[0] != t) {
            continue;
          }
          if (cop->operands.size() != 2 || cop->operands[1] != cInd) {
            loadsOk = false;
            break;
          }
          cLoads.push_back(cop.get());
        }
        if (!loadsOk) {
          continue;
        }
        if (!usesOnlyChain(kernel.body, t, &s, cLoads)) {
          continue; // fire condition (b): extra use of the intermediate
        }
        out = Fusion{i, j, &p, &c, &s, t, s.operands[0], pInd, cInd};
        return true;
      }
    }
  }
  return false;
}

// Merges the consumer into the producer: the producer's body (minus its
// store to the intermediate and its terminator) is followed by the
// consumer's body with every load of the intermediate replaced by the
// producer's stored value and the consumer's induction var rewritten to the
// producer's. The consumer loop is erased; the producer loop is the fused
// nest. Then the intermediate's def is erased when it is a now-dead
// alloc_shared.
void applyFusion(Block &block, Kernel &kernel, const Fusion &f) {
  Block &pBody = *f.producer->region;
  Block &cBody = *f.consumer->region;

  // Consumer ops: induction var -> producer's induction var.
  for (const auto &op : cBody.ops) {
    for (Value &operand : op->operands) {
      if (operand == f.cInduction) {
        operand = f.pInduction;
      }
    }
  }
  // Consumer ops: loads of the intermediate -> the producer's stored value.
  std::vector<std::unique_ptr<Op>> fused;
  fused.reserve(pBody.ops.size() + cBody.ops.size());
  for (auto &op : pBody.ops) {
    if (op.get() == f.store || op->kind == OpKind::Yield) {
      continue; // the producer's store and terminator are dropped
    }
    fused.push_back(std::move(op));
  }
  for (auto &op : cBody.ops) {
    if (op->kind == OpKind::Load && op->operands[0] == f.intermediate) {
      continue; // the consumer's loads of the intermediate are dropped
    }
    for (Value &operand : op->operands) {
      if (operand.kind() == Value::Kind::OpResult &&
          operand.op() != nullptr && operand.op()->kind == OpKind::Load &&
          operand.op()->operands.size() >= 1 &&
          operand.op()->operands[0] == f.intermediate) {
        operand = f.storeValue; // use the producer's stored value
      }
    }
    fused.push_back(std::move(op));
  }
  pBody.ops = std::move(fused);

  // The consumer loop is gone; the producer loop (at pIndex) is the nest.
  block.ops.erase(block.ops.begin() + f.cIndex);

  // Eliminate the intermediate: a now-dead alloc_shared is erased.
  if (f.intermediate.kind() == Value::Kind::OpResult &&
      f.intermediate.op() != nullptr &&
      f.intermediate.op()->kind == OpKind::AllocShared &&
      !valueHasUse(kernel.body, f.intermediate)) {
    eraseOpFromBlocks(kernel.body, f.intermediate.op());
  }
}

// Recursive per-block fusion driver (defined below; kernel-level fixpoint
// needs it first).
void fuseBlocks(Block &block, Kernel &kernel, bool &changed);

// Fixpoint over the whole kernel: fusion can also qualify pairs in nested
// blocks, and a nested fusion can free uses of an intermediate, so the walk
// repeats until a full pass makes no change. Each fusion removes exactly one
// loop and creates no new ones, so this terminates.
void fuseKernel(Kernel &kernel) {
  bool changed = true;
  while (changed) {
    changed = false;
    fuseBlocks(kernel.body, kernel, changed);
  }
}

void fuseBlocks(Block &block, Kernel &kernel, bool &changed) {
  Fusion f;
  while (findFusion(block, kernel, f)) {
    applyFusion(block, kernel, f);
    changed = true;
  }
  for (const auto &op : block.ops) {
    if (op->region != nullptr) {
      fuseBlocks(*op->region, kernel, changed);
    }
  }
}

} // namespace

std::expected<void, std::string> fusePass(Module &module) {
  for (const auto &kernel : module.kernels) {
    fuseKernel(*kernel);
  }
  return {};
}

// Self-registration (contract: lib/passes/CMakeLists.txt).
static const bool kFuseRegistered =
    (kernelh::ir::registerPass("fuse", fusePass), true);

} // namespace kernelh::ir
