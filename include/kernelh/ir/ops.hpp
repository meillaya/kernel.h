// kernel.h "atom" IR - operations (plan todo 9).
//
// EXACT op inventory (pinned by the plan; do not add ops here - later todos
// extend this set through the same file):
//
//   atom.const         scalar/tensor constant (tensor const = splat of one
//                      literal value).
//   atom.add/sub/mul/div/fma/max/min
//                      elementwise arithmetic; scalar or tensor operands of
//                      identical type. atom.div = float division for floats,
//                      SIGNED integer division for integers.
//   atom.cmp           comparison (pred lt/le/gt/ge/eq/ne) on two SCALARS of
//                      identical type; result is always i32, the boolean
//                      carrier (there is no i1 scalar in this IR - i32 0/1 is
//                      the documented boolean convention).
//   atom.select        %sel = atom.select %cond, %a, %b : T where %cond is an
//                      i32 scalar; %a/%b are scalar or tensor of type T.
//   atom.math          unary math {exp,log,sqrt,rsqrt,tanh}; f16/f32 only.
//   atom.load/store    tensor + one index value per tensor dim.
//   atom.load_vec/store_vec
//                      vector memory ops; the width comes from the vector
//                      type (vector<WxE>).
//   atom.alloc_shared  STATIC shared allocation; size computable at compile
//                      time from the tensor shape.
//   atom.barrier
//   atom.thread_id/block_id/block_dim/grid_dim {x,y,z} -> index
//   atom.loop_for      structured loop with iter_args + atom.yield (the
//                      loop-carried values REQUIRED for reductions).
//   atom.reduce        first-class reduction {sum,max} over one tensor dim
//                      -> scalar of the element type (lowered to loop_for in
//                      a later todo).
//   atom.copy          tile copy src -> dst, identical shape.
//
// An Op is a single tagged struct: kind plus a small set of payload fields.
// ops own their operands as Values and their region (loop_for body) as a
// unique_ptr<Block>; a Block owns its ops, so the whole Module is one tree.

#ifndef KERNELH_IR_OPS_HPP
#define KERNELH_IR_OPS_HPP

#include "kernelh/ir/value.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace kernelh::ir {

enum class OpKind : uint8_t {
  Const,
  Add,
  Sub,
  Mul,
  Div,
  Fma,
  Max,
  Min,
  Cmp,
  Select,
  Math,
  Load,
  Store,
  LoadVec,
  StoreVec,
  AllocShared,
  Barrier,
  ThreadId,
  BlockId,
  BlockDim,
  GridDim,
  LoopFor,
  Yield,
  Reduce,
  Copy,
};

// Op name WITHOUT the "atom." prefix ("add", "load_vec", ...).
const char *toString(OpKind kind);

struct Op {
  OpKind kind = OpKind::Barrier;

  // Common fields.
  std::vector<Value> operands;
  std::vector<Type> resultTypes;
  std::unique_ptr<Block> region; // loop_for body only

  // atom.const payload: EXACTLY ONE of intValue / floatValue is set,
  // depending on the result type (integer scalar types vs float element
  // types). Tensor consts are splats of one literal value.
  std::optional<int64_t> intValue;
  std::optional<double> floatValue;

  // Per-kind attributes.
  CmpPred pred = CmpPred::Lt;         // atom.cmp
  MathKind mathKind = MathKind::Exp;  // atom.math
  ReduceKind reduceKind = ReduceKind::Sum; // atom.reduce
  int64_t reduceDim = 0;              // atom.reduce (tensor dim index)
  IdDim idDim = IdDim::X;             // thread_id/block_id/block_dim/grid_dim

  // Convenience handles to this op's results.
  Value result() const { return Value::opResult(const_cast<Op *>(this), 0); }
  Value result(int64_t index) const {
    return Value::opResult(const_cast<Op *>(this), index);
  }
};

// Value::type() - resolved from the definition site. Requires Op complete.
inline Type Value::type() const {
  if (kind_ == Kind::OpResult && op_ != nullptr &&
      index_ >= 0 && static_cast<size_t>(index_) < op_->resultTypes.size()) {
    return op_->resultTypes[static_cast<size_t>(index_)];
  }
  if (kind_ == Kind::BlockArg && block_ != nullptr &&
      index_ >= 0 && static_cast<size_t>(index_) < block_->args.size()) {
    return block_->args[static_cast<size_t>(index_)].type;
  }
  return Type();
}

inline Block::~Block() = default;

// ---------------------------------------------------------------------------
// IR builders - convenience helpers used by tests, the corpus, the DSL (todo
// 11), and the passes. Each op-returning helper appends the op to `block` and
// returns its pointer. Body builders (loop_for) wire the region's parent
// pointers and set the body block arguments ([induction(index)] + iter-arg
// types).
// ---------------------------------------------------------------------------
namespace builder {

// Scalar/integer const: `value` is stored as the integer literal. `type`
// must be i32/i64/index (or a tensor of one of those for a splat const).
Op *makeConst(Block &block, int64_t value, Type type);
// Any other integral argument (e.g. a plain `0` literal) forwards to the
// int64_t overload so callers never hit the int/double ambiguity.
template <typename Int,
          std::enable_if_t<std::is_integral_v<Int> &&
                               !std::is_same_v<std::remove_cv_t<Int>, int64_t>,
                           int> = 0>
Op *makeConst(Block &block, Int value, Type type) {
  return makeConst(block, static_cast<int64_t>(value), type);
}
// Float const: `value` stored as double. `type` must be f16/f32 (scalar or
// tensor splat).
Op *makeConst(Block &block, double value, Type type);

// add/sub/mul/div/max/min. Result type == operand type.
Op *makeBin(Block &block, OpKind kind, Value a, Value b);
Op *makeFma(Block &block, Value a, Value b, Value c);
Op *makeCmp(Block &block, CmpPred pred, Value a, Value b);
Op *makeSelect(Block &block, Value cond, Value a, Value b);
Op *makeMath(Block &block, MathKind kind, Value a);

Op *makeLoad(Block &block, Value tensor, std::vector<Value> indices);
Op *makeStore(Block &block, Value value, Value tensor,
              std::vector<Value> indices);
Op *makeLoadVec(Block &block, Value tensor, std::vector<Value> indices,
                int64_t width);
Op *makeStoreVec(Block &block, Value vec, Value tensor,
                 std::vector<Value> indices);
Op *makeAllocShared(Block &block, const TensorType &type);
Op *makeBarrier(Block &block);
Op *makeId(Block &block, OpKind kind, IdDim dim); // ThreadId/BlockId/.../GridDim

// loop_for with `resultTypes` iter-arg types. The created region block's
// args are [induction(index)] followed by one arg per iter-arg type. Returns
// the op; callers fill op->region->ops.
Op *makeLoopFor(Block &block, Value lb, Value ub, Value step,
                std::vector<Value> iterArgs, std::vector<Type> resultTypes);
Op *makeYield(Block &block, std::vector<Value> values);
Op *makeReduce(Block &block, ReduceKind kind, Value tensor, int64_t dim);
Op *makeCopy(Block &block, Value src, Value dst);

} // namespace builder

} // namespace kernelh::ir

#endif // KERNELH_IR_OPS_HPP
