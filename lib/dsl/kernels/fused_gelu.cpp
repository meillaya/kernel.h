// The fused_gelu kernel template (plan todo 30): an elementwise TWO-OP CHAIN
//
//   t      = a*x + b              (producer)
//   out    = gelu(t)              (consumer)
//   gelu(z) = 0.5*z*(1+tanh(c*(z + d*z^3)))     c = 0.7978845608, d = 0.044715
//
// expressed as TWO atom.loop_for nests over the SAME grid-stride domain, with
// the intermediate staged through a rank-1 global tensor `tmp`:
//
//   %p = atom.loop_for %gid to %n step %stride {   // producer
//     %t = atom.fma %a, %x[%i], %b
//     atom.store %t, %tmp[%i]
//   }
//   %c = atom.loop_for %gid to %n step %stride {   // consumer
//     %t = atom.load %tmp[%i]
//     ... gelu(t) via atom.mul/add/math(tanh) ...
//     atom.store %o, %out[%i]
//   }
//
// The fuse pass (todo 14) fires on the pair because every fire condition
// holds: (a) the two loops share handle-identical lb/ub/step (the same %gid /
// %n / %stride SSA definitions) and both bodies are pure elementwise; (b) the
// rank-1 intermediate %tmp has exactly ONE store (at the producer induction)
// and the consumer's loads (at the consumer induction) as its only uses; (c)
// no atom.barrier sits between the loops; (d) no atom.reduce boundaries
// exist. The result is ONE loop nest computing the whole chain pointwise -
// the "fusion-fired" differential the e2e test asserts.
//
// SCHEDULE (kept fuse-friendly on purpose): the template leaves the default
// schedule EMPTY (no tile/map_blocks/map_warps, vector_width=1, unroll=1,
// use_shared_memory=false). The domain loop is grid-stride (lb = runtime
// %gid, not const-0), so the tile pass (todo 15) leaves it alone (it only
// rewrites `loop_for const-0 to const-E step const-1`), and vectorize/unroll
// are width-1/factor-1 no-ops - every later pass is HARMLESS to the fused
// pair, so the .atom.final loop-nest count is exactly the fusion proof: 1
// loop fused vs 2 loops with --passes-no-fuse.
//
// Kernel arguments (memref-descriptor ABI, positional):
//   0 x   tensor<{n}xf32, global>   input
//   1 tmp tensor<{n}xf32, global>   rank-1 fuse intermediate (global, per
//                                   plan todo 30; eliminated by the fuse pass
//                                   whose dead-arg is verifier-legal)
//   2 out tensor<{n}xf32, global>   output
//
// The shape map takes either `n` (flat, e.g. n=1048576) or `rows`+`cols`
// (row-major 2D problem, e.g. rows=4096,cols=1024 -> n = rows*cols). Both
// emit the same rank-1 flat kernel; the tensors are always rank-1 so the
// fuse-required store[%i]/load[%i] induction correspondence holds.

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

void fusedGeluTemplate(IRBuilder &b, const ir::ScheduleAttrs &attrs,
                       const ShapeMap &shape) {
  (void)attrs; // schedule attrs are attached to the kernel container by emit();
               // the template deliberately leaves the default (empty) schedule
               // so tile/vectorize/unroll/promote-shared no-op (see header).

  // Resolve the total element count: `n` or the product of `rows`*`cols`.
  int64_t n = -1;
  const auto nIt = shape.find("n");
  if (nIt != shape.end()) {
    n = nIt->second;
  } else {
    const auto rowsIt = shape.find("rows");
    const auto colsIt = shape.find("cols");
    if (rowsIt == shape.end() || colsIt == shape.end()) {
      throw std::runtime_error(
          "fused_gelu template: shape requires 'n', or 'rows' AND 'cols' "
          "(add .shape({\"n\", <size>}) or .shape({\"rows\", R}, {\"cols\", C}) "
          "to the builder chain)");
    }
    n = rowsIt->second * colsIt->second;
  }
  if (n <= 0) {
    throw std::runtime_error(
        "fused_gelu template: shape element count must be positive");
  }

  Block &body = b.block();
  const Type argTy =
      Type::tensor(TensorType{{n}, ScalarType::F32, MemSpace::Global});
  body.args.push_back(BlockArg{argTy}); // x
  body.args.push_back(BlockArg{argTy}); // tmp (rank-1 fuse intermediate)
  body.args.push_back(BlockArg{argTy}); // out
  const Value x = Value::blockArg(&body, 0);
  const Value tmp = Value::blockArg(&body, 1);
  const Value out = Value::blockArg(&body, 2);

  // Chain constants (pinned in plan todo 30): t = a*x + b and the gelu
  // coefficients c = sqrt(2/pi) ~ 0.7978845608, d = 0.044715, plus the
  // 0.5 and the tanh shift.
  const Value nv = b.emitConst(n, ScalarType::Index);
  const Value a = b.emitConst(1.5, ScalarType::F32);          // a
  const Value bb = b.emitConst(0.5, ScalarType::F32);         // b
  const Value half = b.emitConst(0.5, ScalarType::F32);       // 0.5
  const Value one = b.emitConst(1.0, ScalarType::F32);        // 1.0
  const Value c = b.emitConst(0.7978845608, ScalarType::F32); // sqrt(2/pi)
  const Value d = b.emitConst(0.044715, ScalarType::F32);     // 0.044715

  // Grid-stride domain (vecadd pattern, todo 11): every thread visits
  // i = gid, gid+stride, ... The SAME %gid / %nv / %stride definitions feed
  // BOTH loops -> fuse fire condition (a) holds by handle identity.
  const Value bid = b.emitBlockId(IdDim::X);
  const Value bdim = b.emitBlockDim(IdDim::X);
  const Value gdim = b.emitGridDim(IdDim::X);
  const Value tid = b.emitThreadId(IdDim::X);
  const Value gid = b.emitAdd(b.emitMul(bid, bdim), tid);
  const Value stride = b.emitMul(bdim, gdim);

  // ---- PRODUCER loop: t = a*x + b, staged into tmp[%i].
  Op *producer =
      b.emitLoopFor(gid, nv, stride, /*iterArgs=*/{}, /*resultTypes=*/{});
  b.pushScope(*producer->region);
  const Value i = b.arg(0);
  const Value xi = b.emitLoad(x, {i});
  const Value t = b.emitFma(a, xi, bb);
  b.emitStore(t, tmp, {i});
  b.emitYield({});
  b.popScope();

  // ---- CONSUMER loop: out = 0.5*t*(1 + tanh(c*(t + d*t^3))).
  Op *consumer =
      b.emitLoopFor(gid, nv, stride, /*iterArgs=*/{}, /*resultTypes=*/{});
  b.pushScope(*consumer->region);
  const Value j = b.arg(0);
  const Value tj = b.emitLoad(tmp, {j});
  const Value t2 = b.emitMul(tj, tj);   // t^2
  const Value t3 = b.emitMul(t2, tj);   // t^3
  const Value dt3 = b.emitMul(d, t3);   // d*t^3
  const Value sum = b.emitAdd(tj, dt3); // t + d*t^3
  const Value cz = b.emitMul(c, sum);   // c*(t + d*t^3)
  const Value th = b.emitMath(MathKind::Tanh, cz); // tanh(...)
  const Value oneP = b.emitAdd(th, one);            // 1 + tanh(...)
  const Value halfT = b.emitMul(half, tj);          // 0.5*t
  const Value gelu = b.emitMul(halfT, oneP);        // 0.5*t*(1+tanh(...))
  b.emitStore(gelu, out, {j});
  b.emitYield({});
  b.popScope();
}

} // namespace

namespace detail {

void registerFusedGeluKernel() {
  registerKernel("fused_gelu", fusedGeluTemplate);
}

} // namespace detail

} // namespace kernelh
