// The attention kernel template (plan todo 31): NON-CAUSAL FlashAttention-lite,
// f32, fused tiled attention with ONLINE softmax rescaling across K/V tiles:
//
//   O[b,h,i,:] = softmax( Q[b,h,i,:] . K[b,h,:,:]^T / sqrt(d) ) . V[b,h,:,:]
//
//   (b in [0,batch), h in [0,heads), i in [0,seq); head_dim = d; the softmax
//   is over the seq axis t: P[t] = exp(S[t] - m) / l with S[t] =
//   (sum_d Q[b,h,i,d]*K[b,h,t,d]) / sqrt(d), l = sum_t P[t], and
//   O[b,h,i,j] = sum_t P[t] * V[b,h,t,j]).
//
// NO causal masking, NO dropout, NO MQA/GQA, NO WMMA/mfma: pure scalar VALU
// math on tiles staged through static LDS (the plan's pinned capstone -
// tiling + shared memory + reductions via loop_for iter_args together).
//
// ONLINE FORM (the pinned algorithm): the running max m, the (max-anchored)
// denominator l, and the per-output-element accumulator acc are carried as
// loop_for iter_args across K/V tiles. Within each tile every element does the
// branch-free rescale (the softmax.cpp pattern, extended with the accumulator):
//
//   m'  = max(m, S)
//   l'  = l * exp(m - m') + exp(S - m')
//   acc'= acc * exp(m - m') + exp(S - m') * V[t, tid]
//
// When S is not a new max, exp(m - m') = exp(0) = 1 (l and acc untouched apart
// from the new contribution); when it IS a new max, exp(m - m') rescales the
// old l/acc to the new max anchor. The seed m = -1e30 (a finite f32, safely
// below any S) makes the first element contribute exp(S - S) = 1 with a
// rescale factor exp(-1e30) = 0 - so every element is counted exactly once,
// branch-free, with NO -inf seed (the softmax.cpp lesson).
//
// MAPPING (pinned): a 3-D grid, one gpu block per query row:
//   grid = (seq, heads, batch), block = 64 threads (= head_dim; <= the ROCDL
//   max_flat_workgroup_size = 256 cap). block_id.z = batch, block_id.y = head,
//   and block_id.x = query row i via the TILE PASS: the DOMAIN loop over
//   query rows ([0, seq), the FIRST region op of the kernel body) is rewritten
//   by map_blocks with tile = {1} - base = block_id.x * 1, so grid.x = seq and
//   one block owns one query row (the todo-15/27 pattern). No boundary guards
//   (seq % 1 == 0, map_warps off - a 1-row tile cannot be wavefront-split).
//   Thread tid owns output element O[b,h,i,tid].
//
// TILES STAGED THROUGH STATIC LDS (the promote-shared spirit, done manually
// as the scan.cpp pattern - the emitter deduplicates memref.global LDS symbols
// BY TYPE, so separate K/V allocs would collide; ONE flat buffer with explicit
// offsets avoids that): Q row [64] + K tile [TILE x 64] + V tile [TILE x 64],
// staged per tile iteration with a barrier between the staging stores and the
// compute loads, and a barrier at the end of each tile iteration. The promote-
// shared pass sees only loads that already feed shared stores, so it leaves
// the pre-staged form alone (idempotency contract). TILE = 64 divides both
// pinned seq lengths.
//
// Kernel arguments (memref-descriptor ABI, positional):
//   0 Q tensor<{batch}x{heads}x{seq}x{head_dim}xf32, global>
//   1 K tensor<{batch}x{heads}x{seq}x{head_dim}xf32, global>
//   2 V tensor<{batch}x{heads}x{seq}x{head_dim}xf32, global>
//   3 O tensor<{batch}x{heads}x{seq}x{head_dim}xf32, global>

#include "kernelh/dsl/program.hpp"

#include "kernels/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kernelh {
namespace {

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::IdDim;
using kernelh::ir::IRBuilder;
using kernelh::ir::MathKind;
using kernelh::ir::MemSpace;
using kernelh::ir::Op;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;

// K/V tile extent over the seq axis. Divides both pinned seq lengths (512,
// 1024) exactly. The LDS footprint is (2*TILE + 1) * head_dim f32 =
// 32.25 KiB for head_dim = 64 - well inside the 64 KiB gfx1101 LDS budget.
constexpr int64_t kTile = 64;

void attentionTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                       const ShapeMap &shape) {
  const auto batchIt = shape.find("batch");
  const auto headsIt = shape.find("heads");
  const auto seqIt = shape.find("seq");
  const auto headDimIt = shape.find("head_dim");
  if (batchIt == shape.end() || headsIt == shape.end() ||
      seqIt == shape.end() || headDimIt == shape.end()) {
    throw std::runtime_error(
        "attention template: shape attributes 'batch', 'heads', 'seq' and "
        "'head_dim' are required "
        "(add .shape({\"batch\", B}, {\"heads\", H}, {\"seq\", S}, "
        "{\"head_dim\", D}) to the builder chain)");
  }
  const int64_t batch = batchIt->second;
  const int64_t heads = headsIt->second;
  const int64_t seq = seqIt->second;
  const int64_t headDim = headDimIt->second;
  if (batch <= 0 || heads <= 0 || seq <= 0 || headDim <= 0) {
    throw std::runtime_error(
        "attention template: shape attributes 'batch', 'heads', 'seq' and "
        "'head_dim' must be positive");
  }
  if (seq % kTile != 0) {
    throw std::runtime_error(
        "attention template: shape attribute 'seq' must be divisible by the "
        "K/V tile extent " + std::to_string(kTile) + " (got seq = " +
        std::to_string(seq) + ")");
  }
  if (headDim != 64) {
    throw std::runtime_error(
        "attention template: the kernel is pinned to head_dim = 64 (one "
        "thread per output element; got head_dim = " + std::to_string(headDim) +
        ")");
  }

  // Default schedule: one block per query row (see file header). Honored
  // only when the builder chain carried none - a DSL-supplied schedule wins.
  if (attrs.tile.empty()) {
    ir::ScheduleAttrs &mutableAttrs = const_cast<ir::ScheduleAttrs &>(attrs);
    mutableAttrs.tile = {1};
    mutableAttrs.map_blocks = true;
    mutableAttrs.map_warps = false;
    mutableAttrs.use_shared_memory = true;
  }

  Block &body = b.block();
  const Type t4 = Type::tensor(
      TensorType{{batch, heads, seq, headDim}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{t4}); // 0 Q
  body.args.push_back(BlockArg{t4}); // 1 K
  body.args.push_back(BlockArg{t4}); // 2 V
  body.args.push_back(BlockArg{t4}); // 3 O
  const Value q = Value::blockArg(&body, 0);
  const Value k = Value::blockArg(&body, 1);
  const Value v = Value::blockArg(&body, 2);
  const Value o = Value::blockArg(&body, 3);

  // ---- Index constants -----------------------------------------------------
  const Value zeroIdx = b.emitConst(int64_t{0}, ScalarType::Index);
  const Value oneIdx = b.emitConst(int64_t{1}, ScalarType::Index);
  const Value seqIdx = b.emitConst(seq, ScalarType::Index);
  const Value dIdx = b.emitConst(headDim, ScalarType::Index);
  const Value tileIdx = b.emitConst(kTile, ScalarType::Index);
  const Value tileDIdx = b.emitConst(kTile * headDim, ScalarType::Index); // V offset
  const Value qRowOff = b.emitConst(2 * kTile * headDim, ScalarType::Index); // Q row

  // ---- Float constants -----------------------------------------------------
  const Value zeroF = b.emitConst(0.0, ScalarType::F32);
  const Value seedM = b.emitConst(-1e30, ScalarType::F32); // finite -inf stand-in
  const Value invSqrtD =
      b.emitConst(1.0 / std::sqrt(static_cast<double>(headDim)), ScalarType::F32);

  // ---- Static LDS: ONE flat buffer (K tile, V tile, Q row) ----------------
  // The emitter deduplicates memref.global symbols BY TYPE (scan.cpp lesson):
  // separate tensor<{TILE}x{64}xf32, shared> allocs for K/V would collapse
  // into one buffer and clobber each other. A single flat buffer with
  // explicit offsets avoids that entirely.
  const Value lds = b.emitAllocShared(
      TensorType{{2 * kTile * headDim + headDim}, ScalarType::F32,
                 MemSpace::Shared});

  const Value tid = b.emitThreadId(IdDim::X); // output column
  const Value hIdx = b.emitBlockId(IdDim::Y); // head
  const Value bIdx = b.emitBlockId(IdDim::Z); // batch

  const auto kTileIndex = [&](Value t) {
    return b.emitAdd(b.emitMul(t, dIdx), tid);
  };
  const auto vTileIndex = [&](Value t) {
    return b.emitAdd(tileDIdx, kTileIndex(t));
  };

  // ---- DOMAIN loop over query rows i in [0, seq) ---------------------------
  // The FIRST region op of the kernel body: the tile pass (todo 15) rewrites
  // it with map_blocks + tile = {1} into block_id.x (one block per query row).
  Op *iLoop = b.emitLoopFor(zeroIdx, seqIdx, oneIdx, /*iterArgs=*/{},
                            /*resultTypes=*/{});
  b.pushScope(*iLoop->region);
  const Value i = b.arg(0);

  // ---- stage the Q row: qs[tid] = Q[b,h,i,tid] ----------------------------
  const Value qv = b.emitLoad(q, {bIdx, hIdx, i, tid});
  b.emitStore(qv, lds, {b.emitAdd(qRowOff, tid)});
  b.emitBarrier();

  // ---- ONLINE K/V tile loop over t0 in [0, seq) step TILE -----------------
  // iter_args (m, l, acc): the running max, the max-anchored denominator, and
  // this thread's output accumulator (output element tid). All threads run the
  // same m/l (S does not depend on tid); acc is per-thread.
  Op *tileLoop = b.emitLoopFor(
      zeroIdx, seqIdx, tileIdx, /*iterArgs=*/{seedM, zeroF, zeroF},
      /*resultTypes=*/{Type::scalar(ScalarType::F32),
                       Type::scalar(ScalarType::F32),
                       Type::scalar(ScalarType::F32)});
  b.pushScope(*tileLoop->region);
  const Value t0 = b.arg(0);
  const Value m = b.arg(1);
  const Value l = b.arg(2);
  const Value acc = b.arg(3);

  // ---- stage the K tile: Ks[t, tid] = K[b,h,t0+t,tid] ---------------------
  Op *stageK = b.emitLoopFor(zeroIdx, tileIdx, oneIdx, /*iterArgs=*/{},
                             /*resultTypes=*/{});
  b.pushScope(*stageK->region);
  {
    const Value t = b.arg(0);
    const Value gk = b.emitAdd(t0, t);
    const Value kv = b.emitLoad(k, {bIdx, hIdx, gk, tid});
    b.emitStore(kv, lds, {kTileIndex(t)});
    b.emitYield({});
  }
  b.popScope();

  // ---- stage the V tile: Vs[t, tid] = V[b,h,t0+t,tid] ---------------------
  Op *stageV = b.emitLoopFor(zeroIdx, tileIdx, oneIdx, /*iterArgs=*/{},
                             /*resultTypes=*/{});
  b.pushScope(*stageV->region);
  {
    const Value t = b.arg(0);
    const Value gk = b.emitAdd(t0, t);
    const Value vv = b.emitLoad(v, {bIdx, hIdx, gk, tid});
    b.emitStore(vv, lds, {vTileIndex(t)});
    b.emitYield({});
  }
  b.popScope();
  b.emitBarrier(); // staging stores -> compute loads

  // ---- per-element online update over the tile ----------------------------
  // iter_args (m, l, acc) seeded from the tile loop's iter args; the inner
  // results feed the tile loop's yield (the online rescale across tiles).
  Op *elemLoop = b.emitLoopFor(
      zeroIdx, tileIdx, oneIdx, /*iterArgs=*/{m, l, acc},
      /*resultTypes=*/{Type::scalar(ScalarType::F32),
                       Type::scalar(ScalarType::F32),
                       Type::scalar(ScalarType::F32)});
  b.pushScope(*elemLoop->region);
  {
    const Value t = b.arg(0);
    const Value mI = b.arg(1);
    const Value lI = b.arg(2);
    const Value accI = b.arg(3);

    // S = sum_d Q[b,h,i,d] * Ks[t,d]  (dot over head_dim), then * 1/sqrt(d).
    Op *dot = b.emitLoopFor(zeroIdx, dIdx, oneIdx, /*iterArgs=*/{zeroF},
                            /*resultTypes=*/{Type::scalar(ScalarType::F32)});
    b.pushScope(*dot->region);
    {
      const Value d = b.arg(0);
      const Value dot = b.arg(1);
      const Value qd = b.emitLoad(lds, {b.emitAdd(qRowOff, d)});
      const Value kd = b.emitLoad(lds, {b.emitAdd(b.emitMul(t, dIdx), d)});
      const Value prod = b.emitMul(qd, kd);
      b.emitYield({b.emitAdd(dot, prod)});
    }
    b.popScope();
    const Value sVal = b.emitMul(dot->result(0), invSqrtD);

    // Online rescale (branch-free, softmax.cpp pattern + accumulator):
    //   m'  = max(m, S)
    //   l'  = l * exp(m - m') + exp(S - m')
    //   acc'= acc * exp(m - m') + exp(S - m') * Vs[t, tid]
    const Value mNew = b.emitMax(mI, sVal);
    const Value scale = b.emitMath(MathKind::Exp, b.emitSub(mI, mNew));
    const Value lScaled = b.emitMul(lI, scale);
    const Value p = b.emitMath(MathKind::Exp, b.emitSub(sVal, mNew));
    const Value lNew = b.emitAdd(lScaled, p);
    const Value accScaled = b.emitMul(accI, scale);
    const Value vv = b.emitLoad(lds, {vTileIndex(t)});
    const Value accAdd = b.emitMul(p, vv);
    b.emitYield({mNew, lNew, b.emitAdd(accScaled, accAdd)});
  }
  b.popScope();
  b.emitBarrier(); // end of tile iteration: compute reads -> next staging

  b.emitYield({elemLoop->result(0), elemLoop->result(1), elemLoop->result(2)});
  b.popScope(); // tile loop

  // ---- write O[b,h,i,tid] = acc / l ---------------------------------------
  const Value out = b.emitDiv(tileLoop->result(2), tileLoop->result(1));
  b.emitStore(out, o, {bIdx, hIdx, i, tid});
  b.emitYield({});
  b.popScope(); // domain loop
}

} // namespace

namespace detail {

void registerAttentionKernel() {
  registerKernel("attention", attentionTemplate);
}

} // namespace detail

} // namespace kernelh
