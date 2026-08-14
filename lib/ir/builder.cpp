// IRBuilder implementation (plan todo 11): thin sugar over the ir::builder::
// make* helpers in ops.cpp plus per-scope named-value tracking.

#include "kernelh/ir/builder.hpp"

#include <utility>

namespace kernelh::ir {

IRBuilder::IRBuilder(Block &block) { scopes_.push_back(Scope{&block, {}}); }

Block &IRBuilder::block() const { return *scopes_.back().block; }

Value IRBuilder::arg(int64_t index) const {
  return Value::blockArg(scopes_.back().block, index);
}

Value IRBuilder::lookup(std::string_view name) const {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    const auto found = it->names.find(name);
    if (found != it->names.end()) {
      return found->second;
    }
  }
  return Value::none();
}

bool IRBuilder::bind(std::string name, Value value) {
  auto &scope = scopes_.back();
  if (scope.names.count(name) != 0) {
    return false;
  }
  scope.names.emplace(std::move(name), value);
  return true;
}

Value IRBuilder::named(std::string name, Value value) {
  bind(std::move(name), value);
  return value;
}

void IRBuilder::pushScope(Block &nested) {
  if (nested.parentBlock != scopes_.back().block) {
    throw std::invalid_argument(
        "IRBuilder::pushScope: nested block's parentBlock does not match the "
        "current block (wire loop->region before entering it)");
  }
  scopes_.push_back(Scope{&nested, {}});
}

void IRBuilder::popScope() {
  if (scopes_.size() <= 1) {
    throw std::logic_error(
        "IRBuilder::popScope: cannot pop the root scope");
  }
  scopes_.pop_back();
}

// ---------------------------------------------------------------------------
// Typed factories
// ---------------------------------------------------------------------------

Value IRBuilder::emitConst(int64_t value, ScalarType type) {
  return builder::makeConst(block(), value, Type::scalar(type))->result();
}

Value IRBuilder::emitConst(double value, ScalarType type) {
  return builder::makeConst(block(), value, Type::scalar(type))->result();
}

Value IRBuilder::emitAdd(Value a, Value b) {
  return builder::makeBin(block(), OpKind::Add, a, b)->result();
}

Value IRBuilder::emitSub(Value a, Value b) {
  return builder::makeBin(block(), OpKind::Sub, a, b)->result();
}

Value IRBuilder::emitMul(Value a, Value b) {
  return builder::makeBin(block(), OpKind::Mul, a, b)->result();
}

Value IRBuilder::emitDiv(Value a, Value b) {
  return builder::makeBin(block(), OpKind::Div, a, b)->result();
}

Value IRBuilder::emitFma(Value a, Value b, Value c) {
  return builder::makeFma(block(), a, b, c)->result();
}

Value IRBuilder::emitMax(Value a, Value b) {
  return builder::makeBin(block(), OpKind::Max, a, b)->result();
}

Value IRBuilder::emitMin(Value a, Value b) {
  return builder::makeBin(block(), OpKind::Min, a, b)->result();
}

Value IRBuilder::emitCmp(CmpPred pred, Value a, Value b) {
  return builder::makeCmp(block(), pred, a, b)->result();
}

Value IRBuilder::emitSelect(Value cond, Value a, Value b) {
  return builder::makeSelect(block(), cond, a, b)->result();
}

Value IRBuilder::emitMath(MathKind kind, Value a) {
  return builder::makeMath(block(), kind, a)->result();
}

Value IRBuilder::emitLoad(Value tensor, std::vector<Value> indices) {
  return builder::makeLoad(block(), tensor, std::move(indices))->result();
}

void IRBuilder::emitStore(Value value, Value tensor,
                          std::vector<Value> indices) {
  builder::makeStore(block(), value, tensor, std::move(indices));
}

Value IRBuilder::emitLoadVec(Value tensor, std::vector<Value> indices,
                             int64_t width) {
  return builder::makeLoadVec(block(), tensor, std::move(indices), width)
      ->result();
}

void IRBuilder::emitStoreVec(Value vec, Value tensor,
                             std::vector<Value> indices) {
  builder::makeStoreVec(block(), vec, tensor, std::move(indices));
}

Value IRBuilder::emitAllocShared(const TensorType &type) {
  return builder::makeAllocShared(block(), type)->result();
}

void IRBuilder::emitBarrier() { builder::makeBarrier(block()); }

Value IRBuilder::emitThreadId(IdDim dim) {
  return builder::makeId(block(), OpKind::ThreadId, dim)->result();
}

Value IRBuilder::emitBlockId(IdDim dim) {
  return builder::makeId(block(), OpKind::BlockId, dim)->result();
}

Value IRBuilder::emitBlockDim(IdDim dim) {
  return builder::makeId(block(), OpKind::BlockDim, dim)->result();
}

Value IRBuilder::emitGridDim(IdDim dim) {
  return builder::makeId(block(), OpKind::GridDim, dim)->result();
}

Op *IRBuilder::emitLoopFor(Value lb, Value ub, Value step,
                           std::vector<Value> iterArgs,
                           std::vector<Type> resultTypes) {
  return builder::makeLoopFor(block(), lb, ub, step, std::move(iterArgs),
                              std::move(resultTypes));
}

void IRBuilder::emitYield(std::vector<Value> values) {
  builder::makeYield(block(), std::move(values));
}

Value IRBuilder::emitReduce(ReduceKind kind, Value tensor, int64_t dim) {
  return builder::makeReduce(block(), kind, tensor, dim)->result();
}

void IRBuilder::emitCopy(Value src, Value dst) {
  builder::makeCopy(block(), src, dst);
}

} // namespace kernelh::ir
