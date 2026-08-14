// The scan kernel template (plan todo 29): inclusive prefix sum, f32, in the
// plan-pinned TWO-PASS Blelloch structure (Metis G38):
//
//   pass 1 ("scan"): per-workgroup UPSWEEP + DOWNSWEEP in static LDS, writing
//     each workgroup's block total to a device array.
//   pass 2 ("scan_totals"): a SEPARATE kernel launch (grid = 1 workgroup) that
//     scans the block totals and ADDS each block's exclusive offset to the
//     pass-1 partial sums.
//
// NO cooperative launch, NO decoupled lookback (plan Scope OUT).
//
// MAPPING (adapted to the ROCDL max_flat_workgroup_size = 256 cap - the
// plan's workgroup-1024 is not launchable, so W = 256 threads scan 256
// elements per workgroup; grid = ceil(n/256)): one element per thread in
// static LDS. The pass-2 scan therefore has N_blocks = ceil(n/256) totals
// (up to 4096 for n = 2^20), which EXCEEDS one 256-thread workgroup's LDS
// capacity to scan as a single tree - so pass 2 computes the block offsets
// with an EXACT 3-word (triple-float) expansion prefix carried through
// loop_for iter_args (each thread redundantly computes the full prefix, the
// softmax/layernorm pattern), then applies the offsets. This is the
// documented adaptation (see the e2e header): the plan's "single-workgroup
// scan of block totals" is honored (pass 2 IS a single workgroup); the
// 3-word expansion is what keeps it bit-exact.
//
// NUMERICS (the reason for the double-float machinery): the pinned tolerances
// {1024: 1e-4, 65536: 1e-3, 2^20: 5e-3} are *smaller than one fp32 ulp* of
// the large prefix sums (ulp(5e5) ~= 0.03), so the kernel's fp32 outputs must
// be EXACTLY correctly rounded relative to the fp64 reference. A plain fp32
// Blelloch is off by up to ~10 ulps at n = 2^20 (a plain tree's error grows
// with the magnitude of the partial sums). The fix used here:
//   - pass 1 scans in DOUBLE-FLOAT (Knuth TwoSum hi/lo words) in LDS, so the
//     per-workgroup partial sums are exact to ~2^-48 relative;
//   - pass 2 accumulates the block-total prefix as an EXACT 3-word expansion
//     (Shewchuk GrowExpansion), so the per-block offsets are exact;
//   - the final y = offset + seg rounding is a TwoSum-based combine that
//     consumes BOTH offset residual words and the segment's lo word, yielding
//     the correctly-rounded fp32 prefix (verified bit-exact vs the
//     round-to-fp32 fp64 reference at every pinned size and N=1000).
//
// The block total for the LAST (partially-filled) workgroup is garbage (the
// padded/clamped loads duplicate x[n-1]), but it is never consumed: pass 2's
// offset for block b is the exclusive prefix of totals[0..b), so only totals
// of blocks before b matter and the final block's total is unused. N=1000
// (not divisible by 256) is therefore handled correctly.
//
// Kernel arguments (memref-descriptor ABI, positional):
//   scan:         0 x tensor<{n}xf32, global>
//                 1 y tensor<{n}xf32, global>       (pass-1 inclusive hi)
//                 2 y_lo tensor<{n}xf32, global>    (pass-1 inclusive lo word)
//                 3 totals_hi tensor<{T}xf32, global>
//                 4 totals_lo tensor<{T}xf32, global>
//   scan_totals:  0 totals_hi tensor<{T}xf32, global>  (in: block totals hi)
//                 1 totals_lo tensor<{T}xf32, global>  (in: block totals lo)
//                 2 off_hi  tensor<{T}xf32, global>  (out: offset hi word)
//                 3 off_lo  tensor<{T}xf32, global>  (out: offset mid word)
//                 4 off_lo2 tensor<{T}xf32, global>  (out: offset low word)
//                 5 y     tensor<{n}xf32, global>   (in: seg hi, out: prefix)
//                 6 y_lo  tensor<{n}xf32, global>   (in: seg lo word)
//
// T = ceil(n/256). The default schedule (empty tile, map_blocks=false) is
// honored verbatim: no tiling pass fires, the runtime-bound loops are left
// alone, and the grid/block are set at launch time (grid = ceil(n/256),
// block = 256 for pass 1; grid = 1, block = 256 for pass 2).

#include "kernelh/dsl/program.hpp"

#include "kernels/kernels.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

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

// Workgroup scan width: 256 threads (the ROCDL-serialized kernel descriptor
// caps max_flat_workgroup_size at 256 - larger blocks are rejected at launch).
constexpr int64_t kW = 256;
// Number of Blelloch levels for kW elements.
constexpr int64_t kLevels = 8;

// ---------------------------------------------------------------------------
// Double-float primitives (all expressible with add/sub only - no fma, no
// conversions). A "double-float" (h, l) represents h + l with ~48 bits of
// mantissa; Knuth's TwoSum gives the exact rounding error of a single fp32
// add, and the DFP adds below fold the low words with a quick renormalization.
// ---------------------------------------------------------------------------

// TwoSum: returns (s, e) with s + e = a + b EXACTLY (s = fl(a + b)).
std::pair<Value, Value> twoSum(IRBuilder &b, Value a, Value c) {
  const Value s = b.emitAdd(a, c);
  const Value bb = b.emitSub(s, a);
  const Value e = b.emitAdd(b.emitSub(a, b.emitSub(s, bb)),
                            b.emitSub(c, bb));
  return {s, e};
}

// DFP add: (h1,l1) + (h2,l2) -> (hh, ll) accurate to ~eps^2 relative.
std::pair<Value, Value> dfAdd(IRBuilder &b, Value h1, Value l1, Value h2,
                              Value l2) {
  const auto [s, sErr] = twoSum(b, h1, h2);
  const Value e2 = b.emitAdd(b.emitAdd(sErr, l1), l2);
  const Value hh = b.emitAdd(s, e2);
  const Value ll = b.emitSub(e2, b.emitSub(hh, s));
  return {hh, ll};
}

// DFP add: (h,l) + v -> (hh, ll).
std::pair<Value, Value> dfAdd1(IRBuilder &b, Value h, Value l, Value v) {
  const auto [s, sErr] = twoSum(b, h, v);
  const Value e2 = b.emitAdd(sErr, l);
  const Value hh = b.emitAdd(s, e2);
  const Value ll = b.emitSub(e2, b.emitSub(hh, s));
  return {hh, ll};
}

// Shewchuk GrowExpansion step on a 3-word expansion: adds float v into
// (e0,e1,e2) keeping the 3 most significant words (the 4th word is < eps^3
// relative and is dropped - far below any rounding-relevant threshold).
std::tuple<Value, Value, Value> grow3(IRBuilder &b, Value e0, Value e1,
                                      Value e2, Value v) {
  const auto [a0, q] = twoSum(b, e0, v);
  const auto [b1, q2] = twoSum(b, e1, q);
  const auto [c2, q3] = twoSum(b, e2, q2);
  (void)q3;
  return {a0, b1, c2};
}

// Correctly-rounded combine: out = round(offset + seg) where offset =
// o0 + o1 + o2 (3-word expansion) and seg = (segH, segL) (double-float).
// The residual words are folded through TwoSum so the ONE final fp32 add
// sees the full precision value - this is what makes the rounding exact
// (a plain `o0 + seg` add would discard o1/o2/segL and land up to 1 ulp
// off at the largest magnitudes).
Value finalRound(IRBuilder &b, Value o0, Value o1, Value o2, Value segH,
                 Value segL, Value zeroF) {
  const auto [r1h, r1l] = dfAdd(b, segH, segL, o1, zeroF);
  const auto [rh, rl] = dfAdd(b, r1h, r1l, o2, zeroF);
  const auto [s, sErr] = twoSum(b, o0, rh);
  const Value e = b.emitAdd(sErr, rl);
  return b.emitAdd(s, e);
}

// ---------------------------------------------------------------------------
// Pass 1: per-workgroup two-pass (upsweep + downsweep) Blelloch in static
// LDS, in double-float. Writes y/y_lo (inclusive scan, correctly rounded to
// fp32 hi + residual lo) and the 2-word block totals.
// ---------------------------------------------------------------------------
void scanTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                  const ShapeMap &shape) {
  (void)attrs; // default (empty tile, map_blocks=false) schedule rides along

  const auto nIt = shape.find("n");
  if (nIt == shape.end()) {
    throw std::runtime_error(
        "scan template: shape attribute 'n' is required (add "
        ".shape({\"n\", N}) to the builder chain)");
  }
  const int64_t n = nIt->second;
  if (n <= 0) {
    throw std::runtime_error(
        "scan template: shape attribute 'n' must be positive");
  }
  const int64_t T = (n + kW - 1) / kW; // number of workgroups (block totals)

  Block &body = b.block();
  const Type vecTy =
      Type::tensor(TensorType{{n}, ScalarType::F32, MemSpace::Global});
  const Type totTy =
      Type::tensor(TensorType{{T}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{vecTy}); // 0 x
  body.args.push_back(BlockArg{vecTy}); // 1 y
  body.args.push_back(BlockArg{vecTy}); // 2 y_lo
  body.args.push_back(BlockArg{totTy}); // 3 totals_hi
  body.args.push_back(BlockArg{totTy}); // 4 totals_lo
  const Value x = Value::blockArg(&body, 0);
  const Value y = Value::blockArg(&body, 1);
  const Value yLo = Value::blockArg(&body, 2);
  const Value tHi = Value::blockArg(&body, 3);
  const Value tLo = Value::blockArg(&body, 4);

  // Constants.
  const Value zeroIdx = b.emitConst(int64_t{0}, ScalarType::Index);
  const Value oneIdx = b.emitConst(int64_t{1}, ScalarType::Index);
  const Value twoIdx = b.emitConst(int64_t{2}, ScalarType::Index);
  const Value eightIdx = b.emitConst(kLevels, ScalarType::Index);
  const Value k256Idx = b.emitConst(kW, ScalarType::Index);
  const Value k255Idx = b.emitConst(kW - 1, ScalarType::Index);
  const Value k128Idx = b.emitConst(kW / 2, ScalarType::Index);
  const Value nIdx = b.emitConst(n, ScalarType::Index);
  const Value nMinusOneIdx = b.emitConst(n - 1, ScalarType::Index);
  const Value zeroF = b.emitConst(0.0, ScalarType::F32);

  const Value bid = b.emitBlockId(IdDim::X);
  const Value tid = b.emitThreadId(IdDim::X);

  // Static LDS: ONE buffer of 3 x 256 f32 (hi scan words at [0,256), lo scan
  // words at [256,512), staged ORIGINAL padded segment at [512,768)). A
  // single buffer is REQUIRED: the emitter deduplicates memref.global LDS
  // symbols by memref type, so three tensor<256xf32, shared> allocs would all
  // map to one 256-slot buffer and the lo/orig stores would clobber the hi
  // words. The originals are kept in LDS so the inclusive output is
  // (exclusive scan) + original without any re-read of global x - the
  // promote-shared pass then sees a fully-staged, already-promoted form.
  const Value lds = b.emitAllocShared(
      TensorType{{3 * kW}, ScalarType::F32, MemSpace::Shared});
  const Value k512Idx = b.emitConst(2 * kW, ScalarType::Index);
  const auto idxLo = [&](Value i) { return b.emitAdd(i, k256Idx); };
  const auto idxOrig = [&](Value i) { return b.emitAdd(i, k512Idx); };

  // ---- stage the workgroup segment (padded: out-of-range slots duplicate
  // x[n-1]; the duplicated tail is never read as a real output and only
  // inflates the last block's total, which is unused).
  const Value blockStart = b.emitMul(bid, k256Idx);
  const Value g = b.emitAdd(blockStart, tid);
  const Value valid = b.emitCmp(CmpPred::Lt, g, nIdx);
  const Value gi = b.emitSelect(valid, g, nMinusOneIdx);
  const Value xv = b.emitLoad(x, {gi});
  b.emitStore(xv, lds, {tid});
  b.emitStore(zeroF, lds, {idxLo(tid)});
  b.emitStore(xv, lds, {idxOrig(tid)});
  b.emitBarrier();

  // ---- UPSWEEP: level d has stride = 2^(d+1), half = 2^d (carried as
  // iter_args); the "virtual thread" loop lets threads with tid < 256/stride
  // do the level's node adds (zero iterations for the rest - branch-free,
  // no clamps needed because t*stride + ... stays in [0, 256)).
  Op *up = b.emitLoopFor(zeroIdx, eightIdx, oneIdx,
                         /*iterArgs=*/{twoIdx, oneIdx},
                         /*resultTypes=*/{Type::scalar(ScalarType::Index),
                                          Type::scalar(ScalarType::Index)});
  b.pushScope(*up->region);
  {
    const Value stride = b.arg(1);
    const Value half = b.arg(2);
    const Value wdiv = b.emitDiv(k256Idx, stride);
    Op *vt = b.emitLoopFor(tid, wdiv, k256Idx, /*iterArgs=*/{},
                           /*resultTypes=*/{});
    b.pushScope(*vt->region);
    {
      const Value t = b.arg(0);
      const Value tS = b.emitMul(t, stride);
      const Value iA = b.emitAdd(tS, b.emitSub(stride, oneIdx));
      const Value iB = b.emitAdd(tS, b.emitSub(half, oneIdx));
      const auto [hh, ll] =
          dfAdd(b, b.emitLoad(lds, {iA}), b.emitLoad(lds, {idxLo(iA)}),
                b.emitLoad(lds, {iB}), b.emitLoad(lds, {idxLo(iB)}));
      b.emitStore(hh, lds, {iA});
      b.emitStore(ll, lds, {idxLo(iA)});
      b.emitYield({});
    }
    b.popScope();
    b.emitBarrier();
    b.emitYield({b.emitMul(stride, twoIdx), b.emitMul(half, twoIdx)});
  }
  b.popScope();

  // ---- seed the exclusive scan (Blelloch): a[n-1] = 0.
  b.emitStore(zeroF, lds, {k255Idx});
  b.emitStore(zeroF, lds, {idxLo(k255Idx)});
  b.emitBarrier();

  // ---- DOWNSWEEP: stride from 256 down to 2 (iter_args halve each level);
  // the standard a[i+half-1] = a[i+stride-1]; a[i+stride-1] += old a[i+half-1].
  Op *dn = b.emitLoopFor(zeroIdx, eightIdx, oneIdx,
                         /*iterArgs=*/{k256Idx, k128Idx},
                         /*resultTypes=*/{Type::scalar(ScalarType::Index),
                                          Type::scalar(ScalarType::Index)});
  b.pushScope(*dn->region);
  {
    const Value stride = b.arg(1);
    const Value half = b.arg(2);
    const Value wdiv = b.emitDiv(k256Idx, stride);
    Op *vt = b.emitLoopFor(tid, wdiv, k256Idx, /*iterArgs=*/{},
                           /*resultTypes=*/{});
    b.pushScope(*vt->region);
    {
      const Value t = b.arg(0);
      const Value tS = b.emitMul(t, stride);
      const Value iA = b.emitAdd(tS, b.emitSub(half, oneIdx));
      const Value iB = b.emitAdd(tS, b.emitSub(stride, oneIdx));
      const Value tvH = b.emitLoad(lds, {iA});
      const Value tvL = b.emitLoad(lds, {idxLo(iA)});
      const Value bh = b.emitLoad(lds, {iB});
      const Value bl = b.emitLoad(lds, {idxLo(iB)});
      b.emitStore(bh, lds, {iA});
      b.emitStore(bl, lds, {idxLo(iA)});
      const auto [hh, ll] = dfAdd(b, bh, bl, tvH, tvL);
      b.emitStore(hh, lds, {iB});
      b.emitStore(ll, lds, {idxLo(iB)});
      b.emitYield({});
    }
    b.popScope();
    b.emitBarrier();
    b.emitYield({b.emitDiv(stride, twoIdx), b.emitDiv(half, twoIdx)});
  }
  b.popScope();

  // ---- output: inclusive = (exclusive scan) + original element, in
  // double-float. The write loop is bounded by the block's REAL element count
  // (k = min(256, n - blockStart)) with lb = tid step 256, so threads past
  // the tail write NOTHING (no out-of-bounds, no clobber of the last real
  // element - the loop simply has zero iterations for them).
  const Value k = b.emitMin(k256Idx, b.emitSub(nIdx, blockStart));
  Op *outL = b.emitLoopFor(tid, k, k256Idx, /*iterArgs=*/{},
                           /*resultTypes=*/{});
  b.pushScope(*outL->region);
  {
    const Value w = b.arg(0);
    const Value g2 = b.emitAdd(blockStart, w);
    const auto [sh, sl] = dfAdd1(b, b.emitLoad(lds, {w}),
                                 b.emitLoad(lds, {idxLo(w)}),
                                 b.emitLoad(lds, {idxOrig(w)}));
    b.emitStore(sh, y, {g2});
    b.emitStore(sl, yLo, {g2});
    b.emitYield({});
  }
  b.popScope();

  // ---- block total = (exclusive[255]) + original padded last element;
  // written redundantly by every thread (identical values).
  const auto [ttH, ttL] =
      dfAdd1(b, b.emitLoad(lds, {k255Idx}), b.emitLoad(lds, {idxLo(k255Idx)}),
             b.emitLoad(lds, {idxOrig(k255Idx)}));
  b.emitStore(ttH, tHi, {bid});
  b.emitStore(ttL, tLo, {bid});
}

// ---------------------------------------------------------------------------
// Pass 2: scan the block totals + add offsets. Single workgroup (grid = 1,
// 256 threads). Each thread redundantly computes the EXACT 3-word expansion
// exclusive prefix of the totals (GrowExpansion through loop_for iter_args)
// into off_hi/off_lo/off_lo2, then applies the correctly-rounded combine to
// every element of the data blocks it owns (grid-strided over the T blocks).
// ---------------------------------------------------------------------------
void scanTotalsTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                        const ShapeMap &shape) {
  (void)attrs;

  const auto nIt = shape.find("n");
  if (nIt == shape.end()) {
    throw std::runtime_error(
        "scan_totals template: shape attribute 'n' is required (add "
        ".shape({\"n\", N}) to the builder chain)");
  }
  const int64_t n = nIt->second;
  if (n <= 0) {
    throw std::runtime_error(
        "scan_totals template: shape attribute 'n' must be positive");
  }
  const int64_t T = (n + kW - 1) / kW;

  Block &body = b.block();
  const Type totTy =
      Type::tensor(TensorType{{T}, ScalarType::F32, MemSpace::Global});
  const Type vecTy =
      Type::tensor(TensorType{{n}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{totTy}); // 0 totals_hi
  body.args.push_back(BlockArg{totTy}); // 1 totals_lo
  body.args.push_back(BlockArg{totTy}); // 2 off_hi
  body.args.push_back(BlockArg{totTy}); // 3 off_lo
  body.args.push_back(BlockArg{totTy}); // 4 off_lo2
  body.args.push_back(BlockArg{vecTy}); // 5 y
  body.args.push_back(BlockArg{vecTy}); // 6 y_lo
  const Value tHi = Value::blockArg(&body, 0);
  const Value tLo = Value::blockArg(&body, 1);
  const Value offHi = Value::blockArg(&body, 2);
  const Value offLo = Value::blockArg(&body, 3);
  const Value offLo2 = Value::blockArg(&body, 4);
  const Value y = Value::blockArg(&body, 5);
  const Value yLo = Value::blockArg(&body, 6);

  const Value zeroIdx = b.emitConst(int64_t{0}, ScalarType::Index);
  const Value oneIdx = b.emitConst(int64_t{1}, ScalarType::Index);
  const Value k256Idx = b.emitConst(kW, ScalarType::Index);
  const Value nIdx = b.emitConst(n, ScalarType::Index);
  const Value TIdx = b.emitConst(T, ScalarType::Index);
  const Value zeroF = b.emitConst(0.0, ScalarType::F32);
  const Value tid = b.emitThreadId(IdDim::X);

  // ---- Step A: exact 3-word expansion exclusive prefix of the totals.
  // Every thread runs the same serial loop and stores identical values
  // (redundant computation - the softmax/layernorm pattern). The totals are
  // read BEFORE their offset words are written (to separate off_* arrays, so
  // no in-place clobbering of the inputs).
  Op *pref = b.emitLoopFor(zeroIdx, TIdx, oneIdx,
                           /*iterArgs=*/{zeroF, zeroF, zeroF},
                           /*resultTypes=*/{Type::scalar(ScalarType::F32),
                                            Type::scalar(ScalarType::F32),
                                            Type::scalar(ScalarType::F32)});
  b.pushScope(*pref->region);
  {
    const Value j = b.arg(0);
    const Value e0 = b.arg(1);
    const Value e1 = b.arg(2);
    const Value e2 = b.arg(3);
    const Value thi = b.emitLoad(tHi, {j});
    const Value tlo = b.emitLoad(tLo, {j});
    b.emitStore(e0, offHi, {j});
    b.emitStore(e1, offLo, {j});
    b.emitStore(e2, offLo2, {j});
    const auto [a0, a1, a2] = grow3(b, e0, e1, e2, thi);
    const auto [b0, b1, b2] = grow3(b, a0, a1, a2, tlo);
    b.emitYield({b0, b1, b2});
  }
  b.popScope();
  b.emitBarrier();

  // ---- Step B: add each block's offset to its (up to 256) elements. Each
  // thread owns the blocks b = tid, tid + 256, ...; the inner loop is bounded
  // by the block's real element count k = min(256, n - b*256), so the tail
  // block's elements are exactly the real ones.
  Op *cb = b.emitLoopFor(tid, TIdx, k256Idx, /*iterArgs=*/{},
                         /*resultTypes=*/{});
  b.pushScope(*cb->region);
  {
    const Value bb = b.arg(0);
    const Value o0 = b.emitLoad(offHi, {bb});
    const Value o1 = b.emitLoad(offLo, {bb});
    const Value o2 = b.emitLoad(offLo2, {bb});
    const Value segStart = b.emitMul(bb, k256Idx);
    const Value kk = b.emitMin(k256Idx, b.emitSub(nIdx, segStart));
    Op *inner = b.emitLoopFor(zeroIdx, kk, oneIdx, /*iterArgs=*/{},
                              /*resultTypes=*/{});
    b.pushScope(*inner->region);
    {
      const Value i = b.arg(0);
      const Value gg = b.emitAdd(segStart, i);
      const Value out =
          finalRound(b, o0, o1, o2, b.emitLoad(y, {gg}), b.emitLoad(yLo, {gg}),
                     zeroF);
      b.emitStore(out, y, {gg});
      b.emitYield({});
    }
    b.popScope();
    b.emitYield({});
  }
  b.popScope();
}

} // namespace

namespace detail {

void registerScanKernel() {
  registerKernel("scan", scanTemplate);
  registerKernel("scan_totals", scanTotalsTemplate);
}

} // namespace detail

} // namespace kernelh
