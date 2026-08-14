// The gemm kernel template (plan todo 26): C = A*B for f32 matrices
// (A MxK row-major, B KxN row-major, C MxN).
//
// REAL TILED SHARED-MEMORY GEMM (the GPU-hang fix). The previous template
// emitted a serial O(m*n*k) skeleton whose global loads were nested inside
// the j-loop; promote-shared could not stage them (copy-mode needs identical
// operand shapes; store-mode only reaches direct-body loads), so the compiled
// kernel had group_segment_fixed_size 0, zero ds_ instructions, and every one
// of the 256 threads redundantly executed the whole m*n*k product (~12 GFLOPS
// effective - the 8192^3 bench launch exceeded the GPU lockup_timeout and
// killed the desktop session). The fix: the template PRE-STAGES its tiles
// through static LDS itself (the scan.cpp pattern, learnings todo 29/31), so
// the compiled kernel is a real tiled GEMM with LDS traffic and barriers.
//
// STRUCTURE (pinned to the tile pass contract in lib/passes/tiling.cpp):
//
//   i in [0, m)                        // the DOMAIN loop (FIRST region op);
//     alloc_shared A/B tiles           //   the tile pass rewrites it into the
//     jt in [0, ceil(n/BN))            //   block/tile loop: with map_blocks +
//       jt0 = jt*BN                    //   map_warps the block owns BM rows
//       jc = min(jt0+lane, n-1)        //   (gi = block_id*BM + i*W + wf) and
//       acc0 = 0.0                     //   each wavefront owns one row, each
//       kt in [0, k) step BK iter_args(acc0):  // lane one tile column.
//         stage A row-slice -> LDS     //   Guards wrap the A-load and the
//         stage B tile      -> LDS     //   C-store (the only induction uses).
//         barrier                      //   The B-tile (BK x BN) is staged
//         kk in [0, kRem) iter_args(acc):    // cooperatively by all 256
//           acc' = fma(lds[..], lds[..], acc) // threads at per-column slots
//         barrier                      //   (real cross-wavefront reuse); the
//       store acc, C[gi][jc]           //   A row-slice per wavefront.
//
// KEY DESIGN DECISIONS:
//   - map_warps MUST be true. Without it every thread of the block shares ONE
//     row and the per-thread B slices cannot live in shared memory (each
//     thread's column differs, so a shared slot indexed by the kk-loop mixes
//     different threads' columns - a real correctness bug, caught at n=128).
//     With map_warps each wavefront owns a DISTINCT row and lane ℓ owns tile
//     column ℓ: the B-tile slots are keyed by (kk, colLocal), every thread
//     reads its own slot, and the block covers BM rows x n columns exactly
//     once (grid = ceil(m/BM) blocks x 256 threads - no redundancy).
//   - Column tails (n % BN != 0, e.g. 1000^3): the last jt-tile clamps its
//     columns via jc = select(jt0+lane < n, jt0+lane, 0) and the B-staging
//     stores each thread's clamped column at ITS OWN colLocal slot, so the
//     clamped threads redundantly compute column 0 with the correct value
//     (last-writer-wins is harmless). k-tile tails (k % BK != 0) bound every
//     kk-loop by kRem = min(BK, k - kt) - no OOB reads on A/B.
//   - The accumulator is carried through the k-tile loop's iter_args (the
//     const-0.0 seed, not a load-C seed) so the boundary case is race-free:
//     the tile pass clamps out-of-extent rows to element 0, and both the
//     real row-0 owner and the guard-clamped threads compute the SAME value
//     from zero - a load-C seed (or a C += partial RMW) would add the full
//     product twice (read-modify-write race). Plain store of the fully
//     accumulated value keeps last-writer-wins benign.
//   - The A row-slice buffer is sized max(BM, 32) x BK (1024 f32 for the
//     default tile): the wavefront slice slot is wf*BK + lane, and wf <
//     ceil(1024/32) = 32 for any launchable block (the launch verifier caps
//     workgroups at 1024 threads). Only the slices actually consumed by the
//     compute are written; the BUFFERS are the full A-tile (BM x BK) and
//     B-tile (BK x BN) = 8 KiB total for the default 32x32x32 tile, so
//     group_segment_fixed_size is real. A single flat buffer is REQUIRED:
//     the emitter deduplicates LDS memrefs BY TYPE, and A/B tiles have the
//     same shape when BM=BK=BN (attention lesson) - separate allocs would
//     collapse into one buffer and clobber.
//
// The tile pass still fires: the FIRST region op is `loop_for const-0 to
// const-m step const-1`. promote-shared becomes a no-op (every global load
// feeds only a shared store - the idempotency contract), exactly like the
// scan/attention templates. unroll stays at its default 1 (a no-op): the
// todo-34 finding is that partial unrolling corrupts iter_args carry chains,
// so nothing here may depend on unroll.
//
// SCHEDULE DEFAULTING: through the production `kernelh compile --kernel gemm`
// CLI the builder chain carries NO schedule (the CLI has no schedule flags),
// so this template applies a default tiled shared-memory schedule when the
// builder supplied none. The const reference is kernel->schedule (a non-const
// member), so mutating through it is well-defined; a DSL-chain schedule
// (program().tile(...).map_blocks()...) is honored verbatim.
//
// The default tile (BM=BN=BK=32) keeps the staged shared buffers at 8 KiB
// total - well inside the 64 KiB LDS budget of gfx1101 - while BM=32 divides
// every supported power-of-two extent and leaves 1000 % 32 != 0 so the
// boundary predicates fire on the 1000x1000x1000 case. A small BM also
// maximizes the block count (grid = ceil(M/BM)).

#include "kernelh/dsl/program.hpp"

#include "kernels/kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernelh {
namespace {

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::CmpPred;
using kernelh::ir::IdDim;
using kernelh::ir::IRBuilder;
using kernelh::ir::MemSpace;
using kernelh::ir::Op;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;

// Wavefront size (the tile pass derives it from the device-limits table;
// the template mirrors it - see the header for why the A slice slots are
// per-wavefront). The max launchable workgroup is 1024 threads (the launch
// verifier cap), so wf < 1024/32 = 32.
constexpr int64_t kWavefront = 32;

void gemmTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                  const ShapeMap &shape) {
  // Every problem dimension is required; there are no defaults.
  const auto requireDim = [&](const char *name) -> int64_t {
    const auto it = shape.find(name);
    if (it == shape.end()) {
      throw std::runtime_error(
          std::string("gemm template: shape attribute '") + name +
          "' is required "
          "(add .shape({\"" + name + "\", <size>}) to the builder chain)");
    }
    if (it->second <= 0) {
      throw std::runtime_error(std::string("gemm template: shape attribute '") +
                               name + "' must be positive");
    }
    return it->second;
  };
  const int64_t m = requireDim("m");
  const int64_t n = requireDim("n");
  const int64_t k = requireDim("k");

  // Default tiled shared-memory schedule (see file header). Honored only
  // when the builder chain carried none - a DSL-supplied schedule wins.
  if (attrs.tile.empty()) {
    ir::ScheduleAttrs &mutableAttrs = const_cast<ir::ScheduleAttrs &>(attrs);
    mutableAttrs.tile = {32, 32, 32};
    mutableAttrs.map_blocks = true;
    mutableAttrs.map_warps = true;
    mutableAttrs.use_shared_memory = true;
  }
  // Tile extents consumed by the template (BM = the tile pass's tile[0];
  // BN/BK ride along through the default pipeline).
  const int64_t bm = attrs.tile[0];
  const int64_t bn = attrs.tile.size() > 1 ? attrs.tile[1] : bm;
  const int64_t bk = attrs.tile.size() > 2 ? attrs.tile[2] : bm;
  // Column tiles over n (compile-time; the last tile clamps its tail).
  const int64_t nTiles = (n + bn - 1) / bn;
  // Flat LDS: the A region (max(BM, 32) x BK per-wavefront row slices)
  // followed by the B tile (BK x BN) - one buffer, explicit offsets.
  const int64_t ldsASlots = std::max(bm, kWavefront) * bk;
  const int64_t ldsElements = ldsASlots + bk * bn;

  Block &body = b.block();
  const Type aTy =
      Type::tensor(TensorType{{m, k}, ScalarType::F32, MemSpace::Global});
  const Type bTy =
      Type::tensor(TensorType{{k, n}, ScalarType::F32, MemSpace::Global});
  const Type cTy =
      Type::tensor(TensorType{{m, n}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{aTy}); // A (M x K)
  body.args.push_back(BlockArg{bTy}); // B (K x N)
  body.args.push_back(BlockArg{cTy}); // C (M x N)
  const Value a = Value::blockArg(&body, 0);
  const Value bRef = Value::blockArg(&body, 1);
  const Value c = Value::blockArg(&body, 2);

  const Value c0 = b.emitConst(int64_t{0}, ScalarType::Index);
  const Value cm = b.emitConst(m, ScalarType::Index);
  const Value cn = b.emitConst(n, ScalarType::Index);
  const Value ck = b.emitConst(k, ScalarType::Index);
  const Value c1 = b.emitConst(int64_t{1}, ScalarType::Index);
  const Value cBn = b.emitConst(bn, ScalarType::Index);
  const Value cBk = b.emitConst(bk, ScalarType::Index);
  const Value cBkBn = b.emitConst(int64_t{bk * bn}, ScalarType::Index);
  const Value cLdsA = b.emitConst(ldsASlots, ScalarType::Index);
  const Value cWave = b.emitConst(kWavefront, ScalarType::Index);
  const Value cNTiles = b.emitConst(nTiles, ScalarType::Index);
  const Value zero = b.emitConst(0.0, ScalarType::F32);

  const Value tid = b.emitThreadId(IdDim::X);
  const Value bdim = b.emitBlockDim(IdDim::X);
  // Wavefront decomposition: wf = tid / 32, lane = tid % 32. The lane is
  // also the tile-local column (colLocal) and the A-slice row index.
  const Value wf = b.emitDiv(tid, cWave);
  const Value wfBase = b.emitMul(wf, cBk); // wf * BK (A-slice slot base)
  const Value lane = b.emitSub(tid, b.emitMul(wf, cWave));

  // Static LDS: ONE flat buffer holding the per-wavefront A row-slices at
  // [0, ldsASlots) and the B tile (BK x BN) at [ldsASlots, ...) (see the
  // header for the dedup-by-type rationale). 8 KiB for the default tile.
  const Value lds = b.emitAllocShared(
      TensorType{{ldsElements}, ScalarType::F32, MemSpace::Shared});

  // i in [0, m) - the DOMAIN loop (first region op; the tile pass skeleton).
  // `i` is used ONLY by the A-load and the C-store, so after the pass every
  // use becomes the guarded global row gi and the LDS staging stays intact.
  Op *iLoop = b.emitLoopFor(c0, cm, c1, /*iterArgs=*/{}, /*resultTypes=*/{});
  b.pushScope(*iLoop->region);
  const Value i = b.arg(0);

  // jt in [0, ceil(n/BN)): the column TILES. jt0 = jt*BN; lane ℓ of the
  // wavefront owns tile column ℓ, i.e. global column jt0 + ℓ. jc clamps the
  // tail tile's columns to 0 (redundant column-0 compute, last-writer-wins).
  Op *jtLoop = b.emitLoopFor(c0, cNTiles, c1, /*iterArgs=*/{},
                             /*resultTypes=*/{});
  b.pushScope(*jtLoop->region);
  const Value jt = b.arg(0);
  const Value jt0 = b.emitMul(jt, cBn);
  const Value jt0Lane = b.emitAdd(jt0, lane);
  const Value jc = b.emitSelect(b.emitCmp(CmpPred::Lt, jt0Lane, cn), jt0Lane,
                                c0);

  // acc0 = 0.0; then accumulate over the k-TILES with iter_args (see the
  // header for why the seed must be a zero const, not a load-C).
  const Value acc0 = zero;
  Op *kLoop = b.emitLoopFor(c0, ck, cBk, /*iterArgs=*/{acc0},
                            /*resultTypes=*/{Type::scalar(ScalarType::F32)});
  b.pushScope(*kLoop->region);
  const Value kt = b.arg(0);
  const Value acc = b.arg(1);
  // kRem = min(BK, k - kt): the tail tile accumulates only the valid k
  // range (k % BK != 0 on the 1000^3 case - no OOB on A/B).
  const Value kRem = b.emitMin(cBk, b.emitSub(ck, kt));

  // ---- stage the A row-slice: lane ℓ of this wavefront loads A[gi][kt+ℓ]
  // into its own slot wf*BK + ℓ (the wavefront's row gi is shared by its 32
  // lanes; the slot index is induction-free so the tile pass never touches
  // it, and out-of-extent rows are clamped to row 0 by the pass's guard).
  Op *stageA = b.emitLoopFor(lane, kRem, cWave, /*iterArgs=*/{},
                             /*resultTypes=*/{});
  b.pushScope(*stageA->region);
  {
    const Value t = b.arg(0);
    const Value av = b.emitLoad(a, {i, b.emitAdd(kt, t)});
    b.emitStore(av, lds, {b.emitAdd(wfBase, t)});
    b.emitYield({});
  }
  b.popScope();

  // ---- stage the B tile cooperatively: thread tid owns slot t = tid,
  // tid + bdim, ... of the BK x BN tile, i.e. (kk = t/BN, c = t%BN). The
  // global column jt0 + c is clamped per-slot so the tail tile's clamped
  // threads stage column 0 at THEIR OWN colLocal slots (no cross-slot race;
  // the compute of each thread reads its own slot and therefore its own
  // clamped column).
  Op *stageB = b.emitLoopFor(tid, cBkBn, bdim, /*iterArgs=*/{},
                             /*resultTypes=*/{});
  b.pushScope(*stageB->region);
  {
    const Value t = b.arg(0);
    const Value kkPart = b.emitDiv(t, cBn);
    const Value colLocal = b.emitSub(t, b.emitMul(kkPart, cBn));
    const Value colGlobal = b.emitAdd(jt0, colLocal);
    const Value bc = b.emitSelect(b.emitCmp(CmpPred::Lt, colGlobal, cn),
                                  colGlobal, c0);
    const Value bv = b.emitLoad(bRef, {b.emitAdd(kt, kkPart), bc});
    b.emitStore(bv, lds, {b.emitAdd(cLdsA, t)});
    b.emitYield({});
  }
  b.popScope();
  b.emitBarrier(); // staging stores -> compute loads (barrier 1 of the tile)

  // ---- the k-accumulation: reads from SHARED with the iter_args
  // accumulator - A at the wavefront's row slice, B at this thread's
  // colLocal column of the tile. promote-shared sees only loads feeding
  // shared stores and leaves this pre-staged form alone (idempotency).
  Op *kkLoop = b.emitLoopFor(c0, kRem, c1, /*iterArgs=*/{acc},
                             /*resultTypes=*/{Type::scalar(ScalarType::F32)});
  b.pushScope(*kkLoop->region);
  {
    const Value kk = b.arg(0);
    const Value acc2 = b.arg(1);
    const Value as = b.emitLoad(lds, {b.emitAdd(wfBase, kk)});
    const Value bsSlot =
        b.emitAdd(cLdsA, b.emitAdd(b.emitMul(kk, cBn), lane));
    const Value bs = b.emitLoad(lds, {bsSlot});
    b.emitYield({b.emitFma(as, bs, acc2)});
  }
  b.popScope();
  b.emitBarrier(); // compute reads done -> next tile overwrites LDS (2)

  b.emitYield({kkLoop->result(0)});
  b.popScope(); // k-loop

  b.emitStore(kLoop->result(0), c, {i, jc});
  b.emitYield({});
  b.popScope(); // jt-loop

  b.emitYield({});
  b.popScope(); // i-loop
}

} // namespace

namespace detail {

void registerGemmKernel() {
  registerKernel("gemm", gemmTemplate);
}

} // namespace detail

} // namespace kernelh
