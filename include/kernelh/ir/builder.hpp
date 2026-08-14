// kernel.h "atom" IR - stateful programmatic builder (plan todo 11).
//
// The IR core owns types, ops, and the verifier; the low-level ir::builder::
// make* helpers in ops.hpp append single ops to a block. IRBuilder is the
// next layer of sugar: it tracks the current insertion block, keeps a stack
// of per-scope NAMED SSA values (mirroring the IR's structured-SSA dominance
// rules: names bound in an enclosing scope stay visible inside nested
// scopes), and exposes typed factory methods (emitConst, emitAdd, emitLoad,
// ...) so kernel templates (lib/dsl/kernels) can be written without touching
// Op payloads directly.
//
// Usage pattern (kernel template):
//   IRBuilder b(kernel.body);            // insertion block = kernel body
//   Value gid = b.named("gid", b.emitAdd(base, tid));  // or plain Values
//   Op *loop = b.emitLoopFor(gid, nv, step, {}, {});
//   b.pushScope(*loop->region);          // body: induction + iter args
//   Value i = b.arg(0);                  // induction variable
//   ...
//   b.emitYield({sum});
//   b.popScope();                        // back to the kernel body
//
// All emit* methods append to the CURRENT block (b.block()). The verifier is
// the authority on correctness - IRBuilder performs no type checking beyond
// what the make* helpers and the verifier enforce.

#ifndef KERNELH_IR_BUILDER_HPP
#define KERNELH_IR_BUILDER_HPP

#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/types.hpp"
#include "kernelh/ir/value.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace kernelh::ir {

class IRBuilder {
public:
  // Starts a builder on `block` (typically a kernel body block).
  explicit IRBuilder(Block &block);

  // The block ops are currently appended to (the innermost scope).
  Block &block() const;

  // The `index`-th argument of the CURRENT block (0 = loop induction var for
  // a loop body, i >= 1 = iter args; kernel body = kernel parameters).
  Value arg(int64_t index) const;

  // --- per-scope named values ----------------------------------------------
  // Looks up a name across the scope stack (innermost first). Returns
  // Value::none() when the name is not bound in any enclosing scope.
  Value lookup(std::string_view name) const;
  // Binds `name` to `value` in the CURRENT scope. Returns false (and does
  // NOT bind) when the name is already bound in the current scope.
  bool bind(std::string name, Value value);
  // Binds `name` to `value` and returns the value (one-shot bind + use).
  Value named(std::string name, Value value);

  // --- scope management ----------------------------------------------------
  // Enters a nested block (e.g. a loop_for body). The block must already be
  // wired into the IR tree (its parentBlock must be the current block).
  // Names bound after this call are scoped to the nested block; names from
  // enclosing scopes remain visible (dominance).
  void pushScope(Block &nested);
  // Leaves the innermost scope and restores the enclosing block. The root
  // scope (the block the builder was constructed on) cannot be popped.
  void popScope();

  // --- typed factories (thin sugar over ir::builder::make*) ----------------
  // Integer literal consts (element type i32/i64/index; tensor splats are
  // NOT produced here - use makeConst directly for those).
  Value emitConst(int64_t value, ScalarType type);
  // Float literal consts (element type f16/f32).
  Value emitConst(double value, ScalarType type);

  Value emitAdd(Value a, Value b);
  Value emitSub(Value a, Value b);
  Value emitMul(Value a, Value b);
  Value emitDiv(Value a, Value b);
  Value emitFma(Value a, Value b, Value c);
  Value emitMax(Value a, Value b);
  Value emitMin(Value a, Value b);
  Value emitCmp(CmpPred pred, Value a, Value b);
  Value emitSelect(Value cond, Value a, Value b);
  Value emitMath(MathKind kind, Value a);

  Value emitLoad(Value tensor, std::vector<Value> indices);
  void emitStore(Value value, Value tensor, std::vector<Value> indices);
  Value emitLoadVec(Value tensor, std::vector<Value> indices, int64_t width);
  void emitStoreVec(Value vec, Value tensor, std::vector<Value> indices);

  // Static shared-memory allocation (size comes from the tensor shape).
  Value emitAllocShared(const TensorType &type);
  void emitBarrier();

  Value emitThreadId(IdDim dim = IdDim::X);
  Value emitBlockId(IdDim dim = IdDim::X);
  Value emitBlockDim(IdDim dim = IdDim::X);
  Value emitGridDim(IdDim dim = IdDim::X);

  // Structured loop with iter_args; returns the loop op whose region is the
  // body block. The caller enters the body with pushScope(*loop->region),
  // emits body ops ending in emitYield, then popScope()s back.
  Op *emitLoopFor(Value lb, Value ub, Value step, std::vector<Value> iterArgs,
                  std::vector<Type> resultTypes);
  void emitYield(std::vector<Value> values);

  Value emitReduce(ReduceKind kind, Value tensor, int64_t dim);
  void emitCopy(Value src, Value dst);

private:
  struct Scope {
    Block *block;
    std::map<std::string, Value, std::less<>> names;
  };

  std::vector<Scope> scopes_;
};

} // namespace kernelh::ir

#endif // KERNELH_IR_BUILDER_HPP
