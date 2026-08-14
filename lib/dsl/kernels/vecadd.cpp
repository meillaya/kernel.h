// The vecadd kernel template - the FIRST registry registration (plan
// todo 11). Semantics: c[i] = a[i] + b[i] for i in [0, n) with f32
// elements, grid-strided over blocks:
//
//   %bid   = atom.block_id x          // this block
//   %bdim  = atom.block_dim x         // threads per block
//   %gdim  = atom.grid_dim x          // blocks in the grid
//   %tid   = atom.thread_id x
//   %gid   = %bid * %bdim + %tid      // first global element of this thread
//   %stride = %bdim * %gdim           // grid-stride step
//   atom.loop_for %gid to %n step %stride {
//     x = load a[i]; y = load b[i]; c[i] = x + y
//   }
//
// Kernel arguments: three global f32 tensors of extent `n` (from the shape
// map's "n" attribute). The schedule attrs (tile/map_blocks/map_warps/...)
// ride along on the emitted atom.kernel verbatim; the template body itself
// is scalar and grid-stride (vectorized load_vec/store_vec forms land with
// the pass work in todos 16-17).

#include "kernelh/dsl/program.hpp"

#include "kernels/kernels.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernelh {
namespace {

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::IdDim;
using kernelh::ir::IRBuilder;
using kernelh::ir::MemSpace;
using kernelh::ir::Op;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;

void vecaddTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                    const ShapeMap &shape) {
  (void)attrs; // schedule attrs are attached to the kernel container by emit()

  const auto nIt = shape.find("n");
  if (nIt == shape.end()) {
    throw std::runtime_error(
        "vecadd template: shape attribute 'n' is required "
        "(add .shape({\"n\", <size>}) to the builder chain)");
  }
  const int64_t n = nIt->second;
  if (n <= 0) {
    throw std::runtime_error(
        "vecadd template: shape attribute 'n' must be positive");
  }

  Block &body = b.block();
  const Type argTy =
      Type::tensor(TensorType{{n}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{argTy}); // a
  body.args.push_back(BlockArg{argTy}); // b
  body.args.push_back(BlockArg{argTy}); // c
  const Value a = Value::blockArg(&body, 0);
  const Value bb = Value::blockArg(&body, 1);
  const Value c = Value::blockArg(&body, 2);

  const Value nv = b.emitConst(n, ScalarType::Index);
  const Value bid = b.emitBlockId(IdDim::X);
  const Value bdim = b.emitBlockDim(IdDim::X);
  const Value gdim = b.emitGridDim(IdDim::X);
  const Value tid = b.emitThreadId(IdDim::X);
  const Value base = b.emitMul(bid, bdim);
  const Value gid = b.emitAdd(base, tid);
  const Value stride = b.emitMul(bdim, gdim);

  Op *loop = b.emitLoopFor(gid, nv, stride, /*iterArgs=*/{},
                           /*resultTypes=*/{});
  b.pushScope(*loop->region);
  const Value i = b.arg(0);
  const Value x = b.emitLoad(a, {i});
  const Value y = b.emitLoad(bb, {i});
  const Value sum = b.emitAdd(x, y);
  b.emitStore(sum, c, {i});
  b.emitYield({});
  b.popScope();
}

} // namespace

namespace detail {

void registerVecaddKernel() {
  registerKernel("vecadd", vecaddTemplate);
}

} // namespace detail

} // namespace kernelh
