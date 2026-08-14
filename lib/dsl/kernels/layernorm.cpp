// The layernorm kernel template (plan todo 28). Semantics: normalize the
// LAST dim of a {rows x cols} f32 tensor,
//
//   y[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
//
//   mean_r = (1/cols) * sum_c x[r, c]
//   var_r  = (1/cols) * sum_c (x[r, c] - mean_r)^2
//
// TWO-PASS, pinned by the plan (no Welford): pass 1 sums x -> mean; pass 2
// sums (x-mean)^2 -> var; pass 3 applies the affine normalization. eps is a
// compile-time constant (default 1e-5) guarding var=0 inputs (0/0 would be
// NaN). gamma/beta are learnable-shaped per-column vectors.
//
// MAPPING (pinned): one block per row - grid = rows, block = min(cols, 1024).
// block_id.x is the row index; each thread owns the columns {tid, tid+bdim,
// tid+2*bdim, ...} of its block's row (a grid-stride over the row). Every
// thread independently carries the two row reductions through its loop
// iter_args (serial 0..cols loops), so all threads in a block agree on
// mean/var without any cross-thread communication or shared memory - the
// redundant per-thread reduction is deliberate: simple, exact, and it
// exercises the loop-carried iter_args machinery the plan pins for this
// kernel. The write pass distributes columns over the block.
//
// Kernel arguments (memref-descriptor ABI, positional):
//   0 x     tensor<{rows}x{cols}xf32, global>
//   1 gamma tensor<{cols}xf32, global>
//   2 beta  tensor<{cols}xf32, global>
//   3 y     tensor<{rows}x{cols}xf32, global>
//
// The schedule attrs (map_blocks etc.) ride along on the emitted atom.kernel
// verbatim; the compile driver launches with the default (empty tile,
// map_blocks=false) schedule, so the tile pass leaves the 0..cols reduction
// loops untouched.

#include "kernelh/dsl/program.hpp"

#include "kernels/kernels.hpp"

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

void layernormTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                       const ShapeMap &shape) {
  (void)attrs; // schedule attrs are attached to the kernel container by emit()

  const auto rowsIt = shape.find("rows");
  const auto colsIt = shape.find("cols");
  if (rowsIt == shape.end() || colsIt == shape.end()) {
    throw std::runtime_error(
        "layernorm template: shape attributes 'rows' and 'cols' are required "
        "(add .shape({\"rows\", R}, {\"cols\", C}) to the builder chain)");
  }
  const int64_t rows = rowsIt->second;
  const int64_t cols = colsIt->second;
  if (rows <= 0 || cols <= 0) {
    throw std::runtime_error(
        "layernorm template: shape attributes 'rows' and 'cols' must be "
        "positive");
  }

  Block &body = b.block();
  const Type rowTy =
      Type::tensor(TensorType{{rows, cols}, ScalarType::F32, MemSpace::Global});
  const Type colTy =
      Type::tensor(TensorType{{cols}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{rowTy});   // x
  body.args.push_back(BlockArg{colTy});   // gamma
  body.args.push_back(BlockArg{colTy});   // beta
  body.args.push_back(BlockArg{rowTy});   // y
  const Value x = Value::blockArg(&body, 0);
  const Value gamma = Value::blockArg(&body, 1);
  const Value beta = Value::blockArg(&body, 2);
  const Value y = Value::blockArg(&body, 3);

  // Index plumbing.
  const Value zeroIdx = b.emitConst(int64_t{0}, ScalarType::Index);
  const Value colsIdx = b.emitConst(cols, ScalarType::Index);
  const Value oneIdx = b.emitConst(int64_t{1}, ScalarType::Index);
  const Value bid = b.emitBlockId(IdDim::X);   // row index
  const Value bdim = b.emitBlockDim(IdDim::X); // threads per block
  const Value tid = b.emitThreadId(IdDim::X);  // first owned column

  // Float constants.
  const Value zeroF = b.emitConst(0.0, ScalarType::F32);
  const Value colsF = b.emitConst(static_cast<double>(cols), ScalarType::F32);
  const Value eps = b.emitConst(1e-5, ScalarType::F32); // pinned eps default
  // ------------------------------------------------------------------ PASS 1
  // mean = (1/cols) * sum_c x[row, c]; the running sum is carried through the
  // loop's iter_args.
  Op *meanLoop =
      b.emitLoopFor(zeroIdx, colsIdx, oneIdx, /*iterArgs=*/{zeroF},
                    /*resultTypes=*/{Type::scalar(ScalarType::F32)});
  b.pushScope(*meanLoop->region);
  const Value c1 = b.arg(0);   // induction var over columns
  const Value sum1 = b.arg(1); // running sum (f32)
  const Value xv1 = b.emitLoad(x, {bid, c1});
  const Value sum2 = b.emitAdd(sum1, xv1);
  b.emitYield({sum2});
  b.popScope();
  const Value mean = b.emitDiv(meanLoop->result(), colsF);

  // ------------------------------------------------------------------ PASS 2
  // var = (1/cols) * sum_c (x[row, c] - mean)^2; iter_args again.
  Op *varLoop =
      b.emitLoopFor(zeroIdx, colsIdx, oneIdx, /*iterArgs=*/{zeroF},
                    /*resultTypes=*/{Type::scalar(ScalarType::F32)});
  b.pushScope(*varLoop->region);
  const Value c2 = b.arg(0);
  const Value sum3 = b.arg(1);
  const Value xv2 = b.emitLoad(x, {bid, c2});
  const Value diff = b.emitSub(xv2, mean);
  const Value diff2 = b.emitMul(diff, diff);
  const Value sum4 = b.emitAdd(sum3, diff2);
  b.emitYield({sum4});
  b.popScope();
  const Value var = b.emitDiv(varLoop->result(), colsF);
  const Value varEps = b.emitAdd(var, eps);
  const Value invStd = b.emitMath(MathKind::Rsqrt, varEps); // 1/sqrt(var+eps)

  // ------------------------------------------------------------------ PASS 3
  // y[row, c] = (x[row, c] - mean) * invStd * gamma[c] + beta[c] for the
  // columns this thread owns (grid-stride over the row).
  Op *writeLoop =
      b.emitLoopFor(tid, colsIdx, bdim, /*iterArgs=*/{}, /*resultTypes=*/{});
  b.pushScope(*writeLoop->region);
  const Value c3 = b.arg(0);
  const Value xv3 = b.emitLoad(x, {bid, c3});
  const Value centered = b.emitSub(xv3, mean);
  const Value scaled = b.emitMul(centered, invStd);
  const Value g = b.emitLoad(gamma, {c3});
  const Value bt = b.emitLoad(beta, {c3});
  const Value gv = b.emitMul(scaled, g);
  const Value out = b.emitAdd(gv, bt);
  b.emitStore(out, y, {bid, c3});
  b.emitYield({});
  b.popScope();
}

} // namespace

namespace detail {

void registerLayernormKernel() {
  registerKernel("layernorm", layernormTemplate);
}

} // namespace detail

} // namespace kernelh
