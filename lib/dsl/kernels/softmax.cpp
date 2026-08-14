// The softmax kernel template (plan todo 27): row-wise SAFE softmax,
//
//   y[r, c] = exp(x[r, c] - max_r) / sum_c exp(x[r, c] - max_r)
//
//   max_r = max_c x[r, c]
//
// in a SINGLE ONLINE PASS: the running max and the (max-anchored) running sum
// are carried through loop_for iter_args - exercising the loop-carried
// values machinery - and the running sum is RESCALED whenever a new max
// arrives:
//
//   m' = max(m, x_i)                       // new running max
//   s' = s * exp(m - m') + exp(x_i - m')   // rescale old sum, add element
//
// The formulation is BRANCH-FREE (no select/cmp): when x_i is not a new max,
// m - m' = 0 and the rescale factor is exp(0) = 1 (the running sum is
// untouched); when x_i IS a new max, the old sum is multiplied by
// exp(old_max - new_max) to re-anchor every previous contribution to the new
// max, and the new max element itself contributes exp(0) = 1. The running sum
// is therefore always relative to the CURRENT max, so no intermediate partial
// sum can overflow, and the MAX-SUBTRACTION makes the kernel STABLE: naive
// exp(x_i) overflows to inf for inputs ~+1e3, while exp(x_i - max_r) is
// always <= 1 - this is the stability case the e2e test pins.
//
// MAPPING (pinned): one block per row - grid = rows, block = min(cols, 256)
// (the ROCDL-serialized kernel descriptor caps max_flat_workgroup_size at
// 256; see the todo-28 layernorm lesson). The DOMAIN loop over rows is the
// first region op, so the tile pass (todo 15) rewrites it: the default
// schedule applied here is tile = {1} + map_blocks, which makes the rewrite
// emit base = block_id * 1 - exactly one row per gpu block (grid = rows), no
// boundary guards (rows % 1 == 0), map_warps left OFF (a one-row tile cannot
// be partitioned across wavefronts). Inside a block every thread
// independently runs the online reduction over the FULL row (redundant
// per-thread - simple, exact, no cross-thread communication, and it exercises
// the iter_args machinery), then the WRITE loop distributes the row's columns
// with a grid-stride (tid, tid+bdim, ...) so ANY workgroup size <= 256
// computes every element: cols <= block -> one column per thread, cols >
// block (e.g. the 4096-column and 8192-column shapes) -> several columns per
// thread via the grid-stride.
//
// Kernel arguments (memref-descriptor ABI, positional):
//   0 x tensor<{rows}x{cols}xf32, global>
//   1 y tensor<{rows}x{cols}xf32, global>

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

void softmaxTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                     const ShapeMap &shape) {
  const auto rowsIt = shape.find("rows");
  const auto colsIt = shape.find("cols");
  if (rowsIt == shape.end() || colsIt == shape.end()) {
    throw std::runtime_error(
        "softmax template: shape attributes 'rows' and 'cols' are required "
        "(add .shape({\"rows\", R}, {\"cols\", C}) to the builder chain)");
  }
  const int64_t rows = rowsIt->second;
  const int64_t cols = colsIt->second;
  if (rows <= 0 || cols <= 0) {
    throw std::runtime_error(
        "softmax template: shape attributes 'rows' and 'cols' must be "
        "positive");
  }

  // Default schedule: rows -> blocks (see file header). Honored only when the
  // builder chain carried none - a DSL-supplied schedule wins.
  if (attrs.tile.empty()) {
    ir::ScheduleAttrs &mutableAttrs = const_cast<ir::ScheduleAttrs &>(attrs);
    mutableAttrs.tile = {1};
    mutableAttrs.map_blocks = true;
    mutableAttrs.map_warps = false;
    mutableAttrs.use_shared_memory = false;
  }

  Block &body = b.block();
  const Type rowTy =
      Type::tensor(TensorType{{rows, cols}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{rowTy}); // x
  body.args.push_back(BlockArg{rowTy}); // y
  const Value x = Value::blockArg(&body, 0);
  const Value y = Value::blockArg(&body, 1);

  const Value zeroIdx = b.emitConst(int64_t{0}, ScalarType::Index);
  const Value rowsIdx = b.emitConst(rows, ScalarType::Index);
  const Value colsIdx = b.emitConst(cols, ScalarType::Index);
  const Value oneIdx = b.emitConst(int64_t{1}, ScalarType::Index);
  const Value zeroF = b.emitConst(0.0, ScalarType::F32);

  // r in [0, rows) - the DOMAIN loop (first region op; the tile pass
  // skeleton). After the map_blocks rewrite each gpu block owns exactly one
  // row (grid = rows).
  Op *rowLoop = b.emitLoopFor(zeroIdx, rowsIdx, oneIdx, /*iterArgs=*/{},
                              /*resultTypes=*/{});
  b.pushScope(*rowLoop->region);
  const Value r = b.arg(0);

  // Seed the online pass: running max = x[r, 0] (column 0), running sum = 0.
  // The loop then visits EVERY column 0..cols; on iteration 0 the seed max is
  // unchanged (m' = x[0]) and the element contributes exp(x[0] - m') =
  // exp(0) = 1, so every column is counted exactly once.
  const Value x0 = b.emitLoad(x, {r, zeroIdx});

  // ---- ONLINE single-pass reduction: iter_args (running_max, running_sum).
  // The rescale-on-new-max keeps the running sum anchored to the current max
  // (the classic safe-online form, here purely via max + exp, no branches).
  Op *online =
      b.emitLoopFor(zeroIdx, colsIdx, oneIdx, /*iterArgs=*/{x0, zeroF},
                    /*resultTypes=*/{Type::scalar(ScalarType::F32),
                                     Type::scalar(ScalarType::F32)});
  b.pushScope(*online->region);
  const Value c = b.arg(0); // column index
  const Value m = b.arg(1); // running max
  const Value s = b.arg(2); // running sum (anchored to `m`)
  const Value xv = b.emitLoad(x, {r, c});
  const Value mNew = b.emitMax(m, xv);
  const Value mDiff = b.emitSub(m, mNew); // old_max - new_max (<= 0)
  const Value scale = b.emitMath(MathKind::Exp, mDiff); // exp(m - m')
  const Value sScaled = b.emitMul(s, scale);            // rescale old sum
  const Value xDiff = b.emitSub(xv, mNew);              // x_i - max (<= 0)
  const Value addend = b.emitMath(MathKind::Exp, xDiff);
  const Value sNew = b.emitAdd(sScaled, addend);
  b.emitYield({mNew, sNew});
  b.popScope();
  const Value maxV = online->result(0);
  const Value sumV = online->result(1);

  // ---- WRITE loop: grid-stride over the row's columns
  // (c = tid, tid + bdim, ...). exp(x_i - max) can never overflow.
  const Value tid = b.emitThreadId(IdDim::X);
  const Value bdim = b.emitBlockDim(IdDim::X);
  Op *writeLoop =
      b.emitLoopFor(tid, colsIdx, bdim, /*iterArgs=*/{}, /*resultTypes=*/{});
  b.pushScope(*writeLoop->region);
  const Value cw = b.arg(0);
  const Value xw = b.emitLoad(x, {r, cw});
  const Value shifted = b.emitSub(xw, maxV); // x_i - max (<= 0, no overflow)
  const Value e = b.emitMath(MathKind::Exp, shifted);
  const Value out = b.emitDiv(e, sumV);
  b.emitStore(out, y, {r, cw});
  b.emitYield({});
  b.popScope();

  b.emitYield({});
  b.popScope();
}

} // namespace

namespace detail {

void registerSoftmaxKernel() {
  registerKernel("softmax", softmaxTemplate);
}

} // namespace detail

} // namespace kernelh
