// The gemm kernel template (plan todo 26): C = A*B for f32 matrices
// (A MxK row-major, B KxN row-major, C MxN).
//
// The template emits the SERIAL TILE SKELETON the tile pass (todo 15)
// consumes - 3 nested loop_fors (i in [0,m), j in [0,n), k in [0,k)) where
// the OUTERMOST loop is the domain loop (`loop_for const-0 to const-m step
// const-1`, the FIRST region op of the kernel body). The k-loop carries the
// accumulator through iter_args (seeded from the f32 const 0.0):
//
//   i in [0,m)                       // domain loop -> the tile pass rewrites
//     j in [0,n)                     //   this into the block/tile loop
//       acc0 = 0.0
//       acc  = loop_for k in [0,k) iter_args(acc0):
//         acc' = fma(load A[i][k], load B[k][j], acc)
//       store acc, C[i][j]
//
// A zero seed (instead of acc0 = load C[i][j]) is REQUIRED for the boundary
// case: with map_blocks the tile pass clamps out-of-extent rows to element 0
// (select ok, gi, 0), so the last block's guard-clamped threads recompute
// row 0 CONCURRENTLY with block 0 - a load-C seed would then add the full
// product onto block 0's live accumulator (a read-modify-write race). Both
// writers compute the same value from zero, so last-writer-wins is harmless.
//
// SCHEDULE DEFAULTING: through the production `kernelh compile --kernel gemm`
// CLI the builder chain carries NO schedule (the CLI has no schedule flags),
// so this template applies a default tiled shared-memory schedule when the
// builder supplied none. The const reference is kernel->schedule (a non-const
// member), so mutating through it is well-defined; a DSL-chain schedule
// (program().tile(...).map_blocks()...) is honored verbatim.
//
// The default tile (BM=BN=BK=32) keeps the staged shared buffers at 12 KiB
// total (A[32,32] + B[32,32] + C[32,32]) - well inside the 64 KiB LDS budget
// of gfx1101 - while BM=32 divides every supported power-of-two extent and
// leaves 1000 % 32 != 0 so the boundary predicates fire on the 1000x1000x1000
// case. A small BM also maximizes the block count (grid = ceil(M/BM)), which
// keeps the device occupied for the big square sizes.

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
using kernelh::ir::IRBuilder;
using kernelh::ir::MemSpace;
using kernelh::ir::Op;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;

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
  const Value bm = Value::blockArg(&body, 1);
  const Value c = Value::blockArg(&body, 2);

  const Value c0 = b.emitConst(int64_t{0}, ScalarType::Index);
  const Value cm = b.emitConst(m, ScalarType::Index);
  const Value cn = b.emitConst(n, ScalarType::Index);
  const Value ck = b.emitConst(k, ScalarType::Index);
  const Value c1 = b.emitConst(int64_t{1}, ScalarType::Index);
  const Value zero = b.emitConst(0.0, ScalarType::F32);

  // i in [0, m) - the DOMAIN loop (first region op; the tile pass skeleton).
  Op *iLoop = b.emitLoopFor(c0, cm, c1, /*iterArgs=*/{}, /*resultTypes=*/{});
  b.pushScope(*iLoop->region);
  const Value i = b.arg(0);

  // j in [0, n)
  Op *jLoop = b.emitLoopFor(c0, cn, c1, {}, {});
  b.pushScope(*jLoop->region);
  const Value j = b.arg(0);

  // acc0 = 0.0; then accumulate over k with iter_args. A zero seed (rather
  // than acc0 = load C[i][j]) keeps the boundary case race-free: with the
  // load-C form, the tile pass's out-of-extent threads (guarded to row 0)
  // would read C[0][j] concurrently with block 0's own accumulation and add
  // the full product onto it. Both writers compute the same value from zero,
  // so last-writer-wins is harmless.
  const Value acc0 = zero;
  Op *kLoop = b.emitLoopFor(c0, ck, c1, {acc0},
                            {Type::scalar(ScalarType::F32)});
  b.pushScope(*kLoop->region);
  const Value kIdx = b.arg(0);
  const Value acc = b.arg(1);
  const Value aVal = b.emitLoad(a, {i, kIdx});
  const Value bVal = b.emitLoad(bm, {kIdx, j});
  const Value accNext = b.emitFma(aVal, bVal, acc);
  b.emitYield({accNext});
  b.popScope();

  b.emitStore(kLoop->result(0), c, {i, j});
  b.emitYield({});
  b.popScope();

  b.emitYield({});
  b.popScope();
}

} // namespace

namespace detail {

void registerGemmKernel() {
  registerKernel("gemm", gemmTemplate);
}

} // namespace detail

} // namespace kernelh
