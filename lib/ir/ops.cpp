// Op definitions and builders for the atom IR (plan todo 9).

#include "kernelh/ir/ops.hpp"

#include <cassert>
#include <string>
#include <utility>

namespace kernelh::ir {

const char *toString(OpKind kind) {
  switch (kind) {
  case OpKind::Const:
    return "const";
  case OpKind::Add:
    return "add";
  case OpKind::Sub:
    return "sub";
  case OpKind::Mul:
    return "mul";
  case OpKind::Div:
    return "div";
  case OpKind::Fma:
    return "fma";
  case OpKind::Max:
    return "max";
  case OpKind::Min:
    return "min";
  case OpKind::Cmp:
    return "cmp";
  case OpKind::Select:
    return "select";
  case OpKind::Math:
    return "math";
  case OpKind::Load:
    return "load";
  case OpKind::Store:
    return "store";
  case OpKind::LoadVec:
    return "load_vec";
  case OpKind::StoreVec:
    return "store_vec";
  case OpKind::AllocShared:
    return "alloc_shared";
  case OpKind::Barrier:
    return "barrier";
  case OpKind::ThreadId:
    return "thread_id";
  case OpKind::BlockId:
    return "block_id";
  case OpKind::BlockDim:
    return "block_dim";
  case OpKind::GridDim:
    return "grid_dim";
  case OpKind::LoopFor:
    return "loop_for";
  case OpKind::Yield:
    return "yield";
  case OpKind::Reduce:
    return "reduce";
  case OpKind::Copy:
    return "copy";
  }
  return "?";
}

namespace {

// Appends a freshly built op to `block` and returns its stable pointer.
Op *appendOp(Block &block, std::unique_ptr<Op> op) {
  Op *raw = op.get();
  block.ops.push_back(std::move(op));
  return raw;
}

} // namespace

namespace builder {

Op *makeConst(Block &block, int64_t value, Type type) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Const;
  op->resultTypes.push_back(type);
  op->intValue = value;
  return appendOp(block, std::move(op));
}

Op *makeConst(Block &block, double value, Type type) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Const;
  op->resultTypes.push_back(type);
  op->floatValue = value;
  return appendOp(block, std::move(op));
}

Op *makeBin(Block &block, OpKind kind, Value a, Value b) {
  auto op = std::make_unique<Op>();
  op->kind = kind;
  op->operands = {a, b};
  op->resultTypes.push_back(a.type());
  return appendOp(block, std::move(op));
}

Op *makeFma(Block &block, Value a, Value b, Value c) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Fma;
  op->operands = {a, b, c};
  op->resultTypes.push_back(a.type());
  return appendOp(block, std::move(op));
}

Op *makeCmp(Block &block, CmpPred pred, Value a, Value b) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Cmp;
  op->pred = pred;
  op->operands = {a, b};
  // i32 is the documented boolean carrier of this IR.
  op->resultTypes.push_back(Type::scalar(ScalarType::I32));
  return appendOp(block, std::move(op));
}

Op *makeSelect(Block &block, Value cond, Value a, Value b) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Select;
  op->operands = {cond, a, b};
  op->resultTypes.push_back(a.type());
  return appendOp(block, std::move(op));
}

Op *makeMath(Block &block, MathKind kind, Value a) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Math;
  op->mathKind = kind;
  op->operands = {a};
  op->resultTypes.push_back(a.type());
  return appendOp(block, std::move(op));
}

Op *makeLoad(Block &block, Value tensor, std::vector<Value> indices) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Load;
  op->operands.push_back(tensor);
  for (Value i : indices) {
    op->operands.push_back(i);
  }
  op->resultTypes.push_back(Type::scalar(tensor.type().asTensor().elem));
  return appendOp(block, std::move(op));
}

Op *makeStore(Block &block, Value value, Value tensor,
              std::vector<Value> indices) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Store;
  op->operands.push_back(value);
  op->operands.push_back(tensor);
  for (Value i : indices) {
    op->operands.push_back(i);
  }
  return appendOp(block, std::move(op));
}

Op *makeLoadVec(Block &block, Value tensor, std::vector<Value> indices,
                int64_t width) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::LoadVec;
  op->operands.push_back(tensor);
  for (Value i : indices) {
    op->operands.push_back(i);
  }
  VectorType vt{width, tensor.type().asTensor().elem};
  op->resultTypes.push_back(Type::vector(vt));
  return appendOp(block, std::move(op));
}

Op *makeStoreVec(Block &block, Value vec, Value tensor,
                 std::vector<Value> indices) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::StoreVec;
  op->operands.push_back(vec);
  op->operands.push_back(tensor);
  for (Value i : indices) {
    op->operands.push_back(i);
  }
  return appendOp(block, std::move(op));
}

Op *makeAllocShared(Block &block, const TensorType &type) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::AllocShared;
  op->resultTypes.push_back(Type::tensor(type));
  return appendOp(block, std::move(op));
}

Op *makeBarrier(Block &block) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Barrier;
  return appendOp(block, std::move(op));
}

Op *makeId(Block &block, OpKind kind, IdDim dim) {
  auto op = std::make_unique<Op>();
  op->kind = kind;
  op->idDim = dim;
  op->resultTypes.push_back(Type::scalar(ScalarType::Index));
  return appendOp(block, std::move(op));
}

Op *makeLoopFor(Block &block, Value lb, Value ub, Value step,
                std::vector<Value> iterArgs, std::vector<Type> resultTypes) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::LoopFor;
  op->operands.push_back(lb);
  op->operands.push_back(ub);
  op->operands.push_back(step);
  for (Value arg : iterArgs) {
    op->operands.push_back(arg);
  }
  op->resultTypes = std::move(resultTypes);

  // Body block args: [induction(index)] + one per iter-arg type.
  auto body = std::make_unique<Block>();
  body->parentBlock = &block;
  body->parentOp = op.get();
  body->args.push_back(BlockArg{Type::scalar(ScalarType::Index)});
  for (const Type &t : op->resultTypes) {
    body->args.push_back(BlockArg{t});
  }
  op->region = std::move(body);
  return appendOp(block, std::move(op));
}

Op *makeYield(Block &block, std::vector<Value> values) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Yield;
  op->operands = std::move(values);
  return appendOp(block, std::move(op));
}

Op *makeReduce(Block &block, ReduceKind kind, Value tensor, int64_t dim) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Reduce;
  op->reduceKind = kind;
  op->reduceDim = dim;
  op->operands.push_back(tensor);
  op->resultTypes.push_back(Type::scalar(tensor.type().asTensor().elem));
  return appendOp(block, std::move(op));
}

Op *makeCopy(Block &block, Value src, Value dst) {
  auto op = std::make_unique<Op>();
  op->kind = OpKind::Copy;
  op->operands = {src, dst};
  return appendOp(block, std::move(op));
}

} // namespace builder

} // namespace kernelh::ir
