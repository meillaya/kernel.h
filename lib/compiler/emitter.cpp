//===- emitter.cpp - custom atom IR -> MLIR gpu dialect (plan todo 21) ----===//
//
// Two-stage device-only lowering:
//
//   Stage 1 (lowerToAtomDialect): structural translation of the custom atom
//     IR (todo 9) into the MLIR "atom" dialect (todo 20), 1:1 op-for-op.
//   Stage 2 (lowerAtomToGpu): the atom module -> a gpu.module whose entries
//     are gpu.funcs with the `kernel` unit attr, DYNAMIC-shape memref args
//     (memref-descriptor calling convention), shared memory as addrspace-3
//     memref.global symbols, and bodies lowered onto gpu/scf/vector/arith/
//     math/memref only. No gpu.launch, no host code, no kernel outlining -
//     the runtime owns the host side.
//
// emitGpuModule fuses the two stages and runs canonicalize + CSE on the
// result.
//
// This file is part of kernel.h.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "kernelh/compiler/emitter.hpp"

#include "kernelh/Dialect/Atom/IR/AtomDialect.h"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/types.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>
#include <string>

namespace kernelh::compiler {

namespace {

using kernelh::ir::Block;
using kernelh::ir::CmpPred;
using kernelh::ir::IdDim;
using kernelh::ir::Kernel;
using kernelh::ir::MathKind;
using kernelh::ir::MemSpace;
using kernelh::ir::Module;
using kernelh::ir::Op;
using kernelh::ir::OpKind;
using kernelh::ir::ReduceKind;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;

// ---------------------------------------------------------------------------
// Type mapping (pinned by todo 20): tensor<...E,global> -> memref<...xE>,
// tensor<...E,shared> -> memref<...xE, 3>, scalars/vectors map 1:1.
// ---------------------------------------------------------------------------

mlir::Type mapScalarType(mlir::MLIRContext &ctx, ScalarType type) {
  switch (type) {
  case ScalarType::F16:
    return mlir::Float16Type::get(&ctx);
  case ScalarType::F32:
    return mlir::Float32Type::get(&ctx);
  case ScalarType::I32:
    return mlir::IntegerType::get(&ctx, 32);
  case ScalarType::I64:
    return mlir::IntegerType::get(&ctx, 64);
  case ScalarType::Index:
    return mlir::IndexType::get(&ctx);
  }
  llvm_unreachable("unhandled scalar type");
}

mlir::Type mapElementType(mlir::MLIRContext &ctx, ScalarType type) {
  return mapScalarType(ctx, type);
}

mlir::VectorType mapVectorType(mlir::MLIRContext &ctx,
                               const kernelh::ir::VectorType &type) {
  return mlir::VectorType::get({type.width}, mapElementType(ctx, type.elem));
}

mlir::MemRefType mapTensorType(mlir::MLIRContext &ctx,
                               const TensorType &type) {
  return mlir::MemRefType::get(
      type.shape, mapElementType(ctx, type.elem),
      /*layout=*/mlir::AffineMap(),
      type.space == MemSpace::Shared ? 3 : 0);
}

mlir::Type mapValueType(mlir::MLIRContext &ctx, const Type &type) {
  if (type.isScalar()) {
    return mapScalarType(ctx, type.asScalar());
  }
  if (type.isVector()) {
    return mapVectorType(ctx, type.asVector());
  }
  if (type.isTensor()) {
    return mapTensorType(ctx, type.asTensor());
  }
  return {};
}

// A scalar-typed mlir::Type that is the element type of the atom value type
// `type` (the atom dialect const carries an element-typed literal even when
// the result is a memref splat).
mlir::Type mapLiteralType(mlir::MLIRContext &ctx, const Type &type) {
  if (type.isTensor()) {
    return mapElementType(ctx, type.asTensor().elem);
  }
  if (type.isVector()) {
    return mapElementType(ctx, type.asVector().elem);
  }
  return mapScalarType(ctx, type.asScalar());
}

// ---------------------------------------------------------------------------
// Shared lowering state.
// ---------------------------------------------------------------------------

} // namespace (anonymous helpers)

// Key for a custom-IR value in the SSA map: an op result or a block argument.
// Out of the anonymous namespace so the llvm::DenseMapInfo specialization
// below can name it.
struct ValueKey {
  const void *owner = nullptr;
  int64_t index = 0;
  uint8_t isOpResult = 1;

  bool operator==(const ValueKey &other) const {
    return owner == other.owner && index == other.index &&
           isOpResult == other.isOpResult;
  }
};

ValueKey keyOf(Value value) {
  if (value.kind() == Value::Kind::OpResult) {
    return ValueKey{value.op(), value.index(), 1};
  }
  return ValueKey{value.block(), value.index(), 0};
}

} // namespace kernelh::compiler

namespace llvm {
template <>
struct DenseMapInfo<kernelh::compiler::ValueKey> {
  using KeyT = kernelh::compiler::ValueKey;
  static KeyT getEmptyKey() { return KeyT{reinterpret_cast<const void *>(-1), -1, 0}; }
  static KeyT getTombstoneKey() {
    return KeyT{reinterpret_cast<const void *>(-2), -1, 0};
  }
  static unsigned getHashValue(const KeyT &key) {
    unsigned h = DenseMapInfo<const void *>::getHashValue(key.owner);
    h = h * 31 + static_cast<unsigned>(key.index);
    h = h * 31 + key.isOpResult;
    return h;
  }
  static bool isEqual(const KeyT &lhs, const KeyT &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace kernelh::compiler {
namespace {

mlir::Value lookupValue(const llvm::DenseMap<ValueKey, mlir::Value> &map,
                        Value value) {
  auto it = map.find(keyOf(value));
  if (it == map.end()) {
    return {};
  }
  return it->second;
}

void setValue(llvm::DenseMap<ValueKey, mlir::Value> &map, Value from,
              mlir::Value to) {
  map[keyOf(from)] = to;
}

// ---------------------------------------------------------------------------
// Stage 1: custom IR -> atom dialect.
// ---------------------------------------------------------------------------

class AtomEmitter {
public:
  explicit AtomEmitter(mlir::MLIRContext &ctx)
      : ctx_(ctx), builder_(&ctx) {}

  std::optional<std::string> error_;

  bool fail(const std::string &message) {
    if (!error_) {
      error_ = message;
    }
    return false;
  }

  mlir::Value mapValue(Value value) {
    mlir::Value mapped = lookupValue(valueMap_, value);
    if (!mapped) {
      fail("stage 1: operand references an unmapped value");
    }
    return mapped;
  }

  bool translateModule(const Module &module, mlir::ModuleOp target) {
    for (const auto &kernel : module.kernels) {
      if (!translateKernel(*kernel, target)) {
        return false;
      }
    }
    return true;
  }

  bool translateKernel(const Kernel &kernel, mlir::ModuleOp target) {
    mlir::Location loc = builder_.getUnknownLoc();

    mlir::ArrayAttr tileAttr;
    if (!kernel.schedule.tile.empty()) {
      tileAttr = builder_.getI64ArrayAttr(kernel.schedule.tile);
    }
    mlir::BoolAttr mapBlocksAttr;
    if (kernel.schedule.map_blocks) {
      mapBlocksAttr = builder_.getBoolAttr(true);
    }
    mlir::BoolAttr mapWarpsAttr;
    if (kernel.schedule.map_warps) {
      mapWarpsAttr = builder_.getBoolAttr(true);
    }
    mlir::BoolAttr useSharedAttr;
    if (kernel.schedule.use_shared_memory) {
      useSharedAttr = builder_.getBoolAttr(true);
    }
    mlir::IntegerAttr vectorWidthAttr;
    if (kernel.schedule.vector_width != 1) {
      vectorWidthAttr =
          builder_.getI64IntegerAttr(kernel.schedule.vector_width);
    }
    mlir::IntegerAttr unrollAttr;
    if (kernel.schedule.unroll != 1) {
      unrollAttr = builder_.getI64IntegerAttr(kernel.schedule.unroll);
    }
    mlir::DictionaryAttr shapeAttr;
    if (!kernel.shape.empty()) {
      mlir::NamedAttrList shapeAttrs;
      for (const auto &[name, value] : kernel.shape) {
        shapeAttrs.set(name, builder_.getI64IntegerAttr(value));
      }
      shapeAttr = shapeAttrs.getDictionary(&ctx_);
    }

    builder_.setInsertionPointToEnd(&target.getBodyRegion().front());
    auto kernelOp = builder_.create<mlir::kernelh::atom::KernelOp>(
        loc, kernel.name, tileAttr, mapBlocksAttr, mapWarpsAttr, useSharedAttr,
        vectorWidthAttr, unrollAttr, shapeAttr);

    auto *bodyBlock = new mlir::Block();
    kernelOp.getRegion().push_back(bodyBlock);
    mlir::SmallVector<mlir::Type> argTypes;
    argTypes.reserve(kernel.body.args.size());
    for (const auto &arg : kernel.body.args) {
      argTypes.push_back(mapValueType(ctx_, arg.type));
    }
    mlir::SmallVector<mlir::Location> argLocs(argTypes.size(), loc);
    bodyBlock->addArguments(argTypes, argLocs);
    for (size_t i = 0; i < kernel.body.args.size(); ++i) {
      setValue(valueMap_, Value::blockArg(const_cast<Block *>(&kernel.body), i),
               bodyBlock->getArgument(i));
    }

    builder_.setInsertionPointToStart(bodyBlock);
    return translateBlock(kernel.body, *bodyBlock);
  }

  bool translateBlock(const Block &block, mlir::Block &target) {
    for (const auto &op : block.ops) {
      if (!translateOp(*op, target)) {
        return false;
      }
    }
    return true;
  }

  bool translateOp(const Op &op, mlir::Block &target) {
    mlir::Location loc = builder_.getUnknownLoc();
    switch (op.kind) {
    case OpKind::Const: {
      mlir::Type resultType = mapValueType(ctx_, op.resultTypes[0]);
      mlir::Attribute literal;
      mlir::Type literalType = mapLiteralType(ctx_, op.resultTypes[0]);
      if (literalType.isFloat()) {
        double value = op.floatValue.value_or(0.0);
        literal = mlir::FloatAttr::get(literalType, value);
      } else if (literalType.isIndex()) {
        literal = mlir::IntegerAttr::get(literalType, op.intValue.value_or(0));
      } else {
        literal =
            mlir::IntegerAttr::get(literalType, op.intValue.value_or(0));
      }
      auto constOp =
          builder_.create<mlir::kernelh::atom::ConstOp>(loc, resultType, literal);
      setValue(valueMap_, op.result(), constOp.getResult());
      break;
    }
    case OpKind::Add:
    case OpKind::Sub:
    case OpKind::Mul:
    case OpKind::Div:
    case OpKind::Max:
    case OpKind::Min: {
      mlir::Type resultType = mapValueType(ctx_, op.resultTypes[0]);
      mlir::Value lhs = mapValue(op.operands[0]);
      mlir::Value rhs = mapValue(op.operands[1]);
      mlir::Value result;
      switch (op.kind) {
      case OpKind::Add:
        result =
            builder_.create<mlir::kernelh::atom::AddOp>(loc, resultType, lhs, rhs).getResult();
        break;
      case OpKind::Sub:
        result =
            builder_.create<mlir::kernelh::atom::SubOp>(loc, resultType, lhs, rhs).getResult();
        break;
      case OpKind::Mul:
        result =
            builder_.create<mlir::kernelh::atom::MulOp>(loc, resultType, lhs, rhs).getResult();
        break;
      case OpKind::Div:
        result =
            builder_.create<mlir::kernelh::atom::DivOp>(loc, resultType, lhs, rhs).getResult();
        break;
      case OpKind::Max:
        result =
            builder_.create<mlir::kernelh::atom::MaxOp>(loc, resultType, lhs, rhs).getResult();
        break;
      case OpKind::Min:
        result =
            builder_.create<mlir::kernelh::atom::MinOp>(loc, resultType, lhs, rhs).getResult();
        break;
      default:
        break;
      }
      setValue(valueMap_, op.result(), result);
      break;
    }
    case OpKind::Fma: {
      mlir::Type resultType = mapValueType(ctx_, op.resultTypes[0]);
      mlir::Value a = mapValue(op.operands[0]);
      mlir::Value b = mapValue(op.operands[1]);
      mlir::Value c = mapValue(op.operands[2]);
      auto fmaOp =
          builder_.create<mlir::kernelh::atom::FmaOp>(loc, resultType, a, b, c);
      setValue(valueMap_, op.result(), fmaOp.getResult());
      break;
    }
    case OpKind::Cmp: {
      mlir::Value lhs = mapValue(op.operands[0]);
      mlir::Value rhs = mapValue(op.operands[1]);
      mlir::kernelh::atom::CmpPred pred;
      switch (op.pred) {
      case CmpPred::Lt:
        pred = mlir::kernelh::atom::CmpPred::lt;
        break;
      case CmpPred::Le:
        pred = mlir::kernelh::atom::CmpPred::le;
        break;
      case CmpPred::Gt:
        pred = mlir::kernelh::atom::CmpPred::gt;
        break;
      case CmpPred::Ge:
        pred = mlir::kernelh::atom::CmpPred::ge;
        break;
      case CmpPred::Eq:
        pred = mlir::kernelh::atom::CmpPred::eq;
        break;
      case CmpPred::Ne:
        pred = mlir::kernelh::atom::CmpPred::ne;
        break;
      }
      auto cmpOp = builder_.create<mlir::kernelh::atom::CmpOp>(
          loc, builder_.getI32Type(), pred, lhs, rhs);
      setValue(valueMap_, op.result(), cmpOp.getResult());
      break;
    }
    case OpKind::Select: {
      mlir::Type resultType = mapValueType(ctx_, op.resultTypes[0]);
      mlir::Value cond = mapValue(op.operands[0]);
      mlir::Value onTrue = mapValue(op.operands[1]);
      mlir::Value onFalse = mapValue(op.operands[2]);
      auto selectOp = builder_.create<mlir::kernelh::atom::SelectOp>(
          loc, resultType, cond, onTrue, onFalse);
      setValue(valueMap_, op.result(), selectOp.getResult());
      break;
    }
    case OpKind::Math: {
      mlir::Type resultType = mapValueType(ctx_, op.resultTypes[0]);
      mlir::Value input = mapValue(op.operands[0]);
      mlir::kernelh::atom::MathKind kind;
      switch (op.mathKind) {
      case MathKind::Exp:
        kind = mlir::kernelh::atom::MathKind::exp;
        break;
      case MathKind::Log:
        kind = mlir::kernelh::atom::MathKind::log;
        break;
      case MathKind::Sqrt:
        kind = mlir::kernelh::atom::MathKind::sqrt;
        break;
      case MathKind::Rsqrt:
        kind = mlir::kernelh::atom::MathKind::rsqrt;
        break;
      case MathKind::Tanh:
        kind = mlir::kernelh::atom::MathKind::tanh;
        break;
      }
      auto mathOp =
          builder_.create<mlir::kernelh::atom::MathOp>(loc, resultType, kind, input);
      setValue(valueMap_, op.result(), mathOp.getResult());
      break;
    }
    case OpKind::Load: {
      mlir::Value memref = mapValue(op.operands[0]);
      mlir::SmallVector<mlir::Value> indices;
      for (size_t i = 1; i < op.operands.size(); ++i) {
        indices.push_back(mapValue(op.operands[i]));
      }
      auto loadOp = builder_.create<mlir::kernelh::atom::LoadOp>(
          loc, mapValueType(ctx_, op.resultTypes[0]), memref, indices);
      setValue(valueMap_, op.result(), loadOp.getResult());
      break;
    }
    case OpKind::Store: {
      mlir::Value value = mapValue(op.operands[0]);
      mlir::Value memref = mapValue(op.operands[1]);
      mlir::SmallVector<mlir::Value> indices;
      for (size_t i = 2; i < op.operands.size(); ++i) {
        indices.push_back(mapValue(op.operands[i]));
      }
      builder_.create<mlir::kernelh::atom::StoreOp>(loc, value, memref, indices);
      break;
    }
    case OpKind::LoadVec: {
      mlir::Value memref = mapValue(op.operands[0]);
      mlir::SmallVector<mlir::Value> indices;
      for (size_t i = 1; i < op.operands.size(); ++i) {
        indices.push_back(mapValue(op.operands[i]));
      }
      auto loadVecOp = builder_.create<mlir::kernelh::atom::LoadVecOp>(
          loc, mapValueType(ctx_, op.resultTypes[0]), memref, indices);
      setValue(valueMap_, op.result(), loadVecOp.getResult());
      break;
    }
    case OpKind::StoreVec: {
      mlir::Value value = mapValue(op.operands[0]);
      mlir::Value memref = mapValue(op.operands[1]);
      mlir::SmallVector<mlir::Value> indices;
      for (size_t i = 2; i < op.operands.size(); ++i) {
        indices.push_back(mapValue(op.operands[i]));
      }
      builder_.create<mlir::kernelh::atom::StoreVecOp>(loc, value, memref, indices);
      break;
    }
    case OpKind::AllocShared: {
      auto allocOp = builder_.create<mlir::kernelh::atom::AllocSharedOp>(
          loc, mapValueType(ctx_, op.resultTypes[0]));
      setValue(valueMap_, op.result(), allocOp.getResult());
      break;
    }
    case OpKind::Barrier: {
      builder_.create<mlir::kernelh::atom::BarrierOp>(loc);
      break;
    }
    case OpKind::ThreadId:
    case OpKind::BlockId:
    case OpKind::BlockDim:
    case OpKind::GridDim: {
      mlir::kernelh::atom::IdDim dim;
      switch (op.idDim) {
      case IdDim::X:
        dim = mlir::kernelh::atom::IdDim::x;
        break;
      case IdDim::Y:
        dim = mlir::kernelh::atom::IdDim::y;
        break;
      case IdDim::Z:
        dim = mlir::kernelh::atom::IdDim::z;
        break;
      }
      mlir::Value result;
      switch (op.kind) {
      case OpKind::ThreadId:
        result = builder_
                     .create<mlir::kernelh::atom::ThreadIdOp>(
                         loc, builder_.getIndexType(), dim)
                     .getResult();
        break;
      case OpKind::BlockId:
        result = builder_
                     .create<mlir::kernelh::atom::BlockIdOp>(
                         loc, builder_.getIndexType(), dim)
                     .getResult();
        break;
      case OpKind::BlockDim:
        result = builder_
                     .create<mlir::kernelh::atom::BlockDimOp>(
                         loc, builder_.getIndexType(), dim)
                     .getResult();
        break;
      case OpKind::GridDim:
        result = builder_
                     .create<mlir::kernelh::atom::GridDimOp>(
                         loc, builder_.getIndexType(), dim)
                     .getResult();
        break;
      default:
        break;
      }
      setValue(valueMap_, op.result(), result);
      break;
    }
    case OpKind::LoopFor: {
      mlir::Value lb = mapValue(op.operands[0]);
      mlir::Value ub = mapValue(op.operands[1]);
      mlir::Value step = mapValue(op.operands[2]);
      mlir::SmallVector<mlir::Value> iterArgs;
      for (size_t i = 3; i < op.operands.size(); ++i) {
        iterArgs.push_back(mapValue(op.operands[i]));
      }
      mlir::SmallVector<mlir::Type> resultTypes;
      for (const auto &type : op.resultTypes) {
        resultTypes.push_back(mapValueType(ctx_, type));
      }
      auto loopOp = builder_.create<mlir::kernelh::atom::LoopForOp>(
          loc, resultTypes, lb, ub, step, iterArgs);
      auto *bodyBlock = new mlir::Block();
      loopOp.getRegion().push_back(bodyBlock);
      mlir::SmallVector<mlir::Type> bodyArgTypes;
      bodyArgTypes.push_back(builder_.getIndexType());
      for (const auto &type : op.resultTypes) {
        bodyArgTypes.push_back(mapValueType(ctx_, type));
      }
      mlir::SmallVector<mlir::Location> bodyLocs(bodyArgTypes.size(), loc);
      bodyBlock->addArguments(bodyArgTypes, bodyLocs);
      const Block &customBody = *op.region;
      for (size_t i = 0; i < bodyArgTypes.size(); ++i) {
        setValue(valueMap_,
                 Value::blockArg(const_cast<Block *>(&customBody), i),
                 bodyBlock->getArgument(i));
      }
      mlir::OpBuilder::InsertPoint saved = builder_.saveInsertionPoint();
      builder_.setInsertionPointToStart(bodyBlock);
      if (!translateBlock(customBody, *bodyBlock)) {
        return false;
      }
      builder_.restoreInsertionPoint(saved);
      for (size_t i = 0; i < op.resultTypes.size(); ++i) {
        setValue(valueMap_, op.result(i), loopOp.getResult(i));
      }
      break;
    }
    case OpKind::Yield: {
      mlir::SmallVector<mlir::Value> values;
      for (const auto &operand : op.operands) {
        values.push_back(mapValue(operand));
      }
      builder_.create<mlir::kernelh::atom::YieldOp>(loc, values);
      break;
    }
    case OpKind::Reduce: {
      mlir::Value tensor = mapValue(op.operands[0]);
      mlir::kernelh::atom::ReduceKind kind =
          op.reduceKind == ReduceKind::Sum ? mlir::kernelh::atom::ReduceKind::sum
                                           : mlir::kernelh::atom::ReduceKind::max;
      auto reduceOp = builder_.create<mlir::kernelh::atom::ReduceOp>(
          loc, mapValueType(ctx_, op.resultTypes[0]), tensor, kind,
          static_cast<uint64_t>(op.reduceDim));
      setValue(valueMap_, op.result(), reduceOp.getResult());
      break;
    }
    case OpKind::Copy: {
      mlir::Value src = mapValue(op.operands[0]);
      mlir::Value dst = mapValue(op.operands[1]);
      builder_.create<mlir::kernelh::atom::CopyOp>(loc, src, dst);
      break;
    }
    }
    return true;
  }

private:
  mlir::MLIRContext &ctx_;
  mlir::OpBuilder builder_;
  llvm::DenseMap<ValueKey, mlir::Value> valueMap_;
};

// ---------------------------------------------------------------------------
// Stage 2: atom dialect -> gpu/scf/vector (device-only).
// ---------------------------------------------------------------------------

mlir::gpu::Dimension mapGpuDimension(mlir::kernelh::atom::IdDim dim) {
  switch (dim) {
  case mlir::kernelh::atom::IdDim::x:
    return mlir::gpu::Dimension::x;
  case mlir::kernelh::atom::IdDim::y:
    return mlir::gpu::Dimension::y;
  case mlir::kernelh::atom::IdDim::z:
    return mlir::gpu::Dimension::z;
  }
  llvm_unreachable("unhandled id dim");
}

class GpuEmitter {
public:
  explicit GpuEmitter(mlir::MLIRContext &ctx) : ctx_(ctx), builder_(&ctx) {}

  std::optional<std::string> error_;

  bool fail(const std::string &message) {
    if (!error_) {
      error_ = message;
    }
    return false;
  }

  mlir::Value mapValue(mlir::Value value) {
    auto it = valueMap_.find(value);
    if (it == valueMap_.end()) {
      fail("stage 2: operand references an unmapped value");
      return {};
    }
    return it->second;
  }

  bool translateKernels(mlir::ModuleOp atomModule, mlir::gpu::GPUModuleOp target) {
    for (auto &op : atomModule.getBodyRegion().getOps()) {
      auto kernelOp = llvm::dyn_cast<mlir::kernelh::atom::KernelOp>(op);
      if (!kernelOp) {
        continue;
      }
      if (!translateKernel(kernelOp, target)) {
        return false;
      }
    }
    return true;
  }

  // Returns the (kernel-local) symbol name for the shared buffer with the
  // given memref type, creating a memref.global symbol on first use.
  std::string sharedSymbolFor(mlir::kernelh::atom::KernelOp kernel,
                              mlir::MemRefType type) {
    auto it = sharedSymbols_.find(type);
    if (it != sharedSymbols_.end()) {
      return it->second;
    }
    std::string name = (kernel.getSymName() + "_shared_" +
                        std::to_string(sharedSymbols_.size()))
                           .str();
    sharedSymbols_[type] = name;
    return name;
  }

  bool translateKernel(mlir::kernelh::atom::KernelOp kernelOp,
                       mlir::gpu::GPUModuleOp target) {
    mlir::Location loc = builder_.getUnknownLoc();
    mlir::Block &atomBody = kernelOp.getBody().front();

    // Pass 1: collect the shared-buffer types so the addrspace-3 globals can
    // be emitted before the funcs.
    sharedSymbols_.clear();
    mlir::SmallVector<std::pair<std::string, mlir::MemRefType>> globals;
    kernelOp.walk([&](mlir::kernelh::atom::AllocSharedOp alloc) {
      auto type = mlir::cast<mlir::MemRefType>(alloc.getResult().getType());
      std::string name = sharedSymbolFor(kernelOp, type);
      bool known = false;
      for (const auto &g : globals) {
        if (g.first == name) {
          known = true;
          break;
        }
      }
      if (!known) {
        globals.emplace_back(name, type);
      }
    });

    builder_.setInsertionPointToEnd(&target.getBodyRegion().front());
    for (const auto &[name, type] : globals) {
      builder_.create<mlir::memref::GlobalOp>(loc, name,
                                              /*sym_visibility=*/nullptr, type,
                                              builder_.getUnitAttr(),
                                              /*constant=*/false,
                                              /*alignment=*/nullptr);
    }

    // Kernel ABI: every buffer argument is a dynamic-shape memref (descriptor
    // convention); scalars pass by value.
    mlir::SmallVector<mlir::Type> argTypes;
    for (mlir::BlockArgument atomArg : atomBody.getArguments()) {
      mlir::Type type = atomArg.getType();
      if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type)) {
        mlir::SmallVector<int64_t> shape(memrefType.getRank(),
                                         mlir::ShapedType::kDynamic);
        argTypes.push_back(
            mlir::MemRefType::get(shape, memrefType.getElementType()));
      } else {
        argTypes.push_back(type);
      }
    }
    mlir::FunctionType funcType = builder_.getFunctionType(argTypes, {});

    mlir::SmallVector<mlir::NamedAttribute> funcAttrs;
    funcAttrs.emplace_back(builder_.getStringAttr("kernel"),
                           builder_.getUnitAttr());
    builder_.setInsertionPointToEnd(&target.getBodyRegion().front());
    auto funcOp = builder_.create<mlir::gpu::GPUFuncOp>(
        loc, kernelOp.getSymName(), funcType,
        /*workgroupAttributions=*/mlir::TypeRange(),
        /*privateAttributions=*/mlir::TypeRange(), funcAttrs);

    // GPUFuncOp's function-interface build already created the entry block
    // with `argTypes` arguments.
    mlir::Block &funcBlock = funcOp.getBody().front();
    valueMap_.clear();
    for (size_t i = 0; i < atomBody.getNumArguments(); ++i) {
      valueMap_[atomBody.getArgument(i)] = funcBlock.getArgument(i);
    }

    builder_.setInsertionPointToStart(&funcBlock);
    if (!translateBlock(atomBody)) {
      return false;
    }
    builder_.create<mlir::gpu::ReturnOp>(loc);
    return true;
  }

  bool translateBlock(mlir::Block &block) {
    for (auto &op : block.getOperations()) {
      if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
        // atom.yield handled by the enclosing loop translation.
        continue;
      }
      if (!translateOp(&op)) {
        return false;
      }
    }
    return true;
  }

  // Creates `arith.constant` index `value`.
  mlir::Value indexConst(int64_t value) {
    return builder_.create<mlir::arith::ConstantOp>(
        builder_.getUnknownLoc(), builder_.getIndexType(),
        builder_.getIntegerAttr(builder_.getIndexType(), value));
  }

  // Loop bound for memref dim `d`: a constant when the shape is static,
  // memref.dim when dynamic.
  mlir::Value loopBound(mlir::Value memref, mlir::MemRefType type, int64_t d) {
    int64_t extent = type.getShape()[d];
    if (!mlir::ShapedType::isDynamic(extent)) {
      return indexConst(extent);
    }
    return builder_.create<mlir::memref::DimOp>(builder_.getUnknownLoc(), memref, d);
  }

  // Recursively emits nested scf.for loops over `dim` .. rank-1 with `acc` as
  // the loop-carried reduction accumulator, reducing `memref` element-wise
  // into it. Returns the final accumulated value.
  mlir::Value emitReduceNested(mlir::Value memref, mlir::MemRefType type,
                               mlir::kernelh::atom::ReduceKind kind, int64_t dim,
                               mlir::Value acc) {
    mlir::Location loc = builder_.getUnknownLoc();
    int64_t rank = type.getRank();
    if (dim >= rank) {
      mlir::Value loaded =
          builder_.create<mlir::memref::LoadOp>(loc, memref, indexStack_);
      return reduce(acc, loaded, type.getElementType(), kind);
    }
    mlir::Value lb = indexConst(0);
    mlir::Value ub = loopBound(memref, type, dim);
    mlir::Value step = indexConst(1);
    auto loopOp = builder_.create<mlir::scf::ForOp>(
        loc, lb, ub, step, acc,
        [&](mlir::OpBuilder &, mlir::Location l, mlir::Value iv,
            mlir::ValueRange its) {
          indexStack_.push_back(iv);
          mlir::Value carried =
              emitReduceNested(memref, type, kind, dim + 1, its[0]);
          indexStack_.pop_back();
          builder_.create<mlir::scf::YieldOp>(l, carried);
        });
    return loopOp.getResult(0);
  }

  mlir::Value reduce(mlir::Value lhs, mlir::Value rhs, mlir::Type elemType,
                     mlir::kernelh::atom::ReduceKind kind) {
    mlir::Location loc = builder_.getUnknownLoc();
    if (kind == mlir::kernelh::atom::ReduceKind::max) {
      if (elemType.isFloat()) {
        return builder_
            .create<mlir::arith::MaxNumFOp>(loc, lhs, rhs)
            .getResult();
      }
      return builder_
          .create<mlir::arith::MaxSIOp>(loc, lhs, rhs)
          .getResult();
    }
    if (elemType.isFloat()) {
      return builder_
          .create<mlir::arith::AddFOp>(loc, lhs, rhs,
                                       mlir::arith::FastMathFlagsAttr())
          .getResult();
    }
    return builder_
        .create<mlir::arith::AddIOp>(loc, lhs, rhs,
                                     mlir::arith::IntegerOverflowFlagsAttr())
        .getResult();
  }

  mlir::Value reduceIdentity(mlir::Type elemType,
                             mlir::kernelh::atom::ReduceKind kind) {
    mlir::Location loc = builder_.getUnknownLoc();
    if (kind == mlir::kernelh::atom::ReduceKind::sum) {
      if (elemType.isFloat()) {
        return builder_
            .create<mlir::arith::ConstantOp>(
                loc, elemType,
                mlir::FloatAttr::get(elemType, 0.0))
            .getResult();
      }
      return builder_
          .create<mlir::arith::ConstantOp>(
              loc, elemType, mlir::IntegerAttr::get(elemType, 0))
          .getResult();
    }
    if (elemType.isFloat()) {
      const llvm::fltSemantics &sem =
          llvm::cast<mlir::FloatType>(elemType).getFloatSemantics();
      double lowest = -llvm::APFloat::getLargest(sem).convertToDouble();
      return builder_
          .create<mlir::arith::ConstantOp>(loc, elemType,
                                           mlir::FloatAttr::get(elemType, lowest))
          .getResult();
    }
    int64_t lowest = std::numeric_limits<int64_t>::min();
    if (elemType.isIndex()) {
      lowest = std::numeric_limits<int64_t>::min();
    } else if (auto intType = llvm::dyn_cast<mlir::IntegerType>(elemType)) {
      if (intType.getWidth() == 32) {
        lowest = std::numeric_limits<int32_t>::min();
      }
    }
    return builder_
        .create<mlir::arith::ConstantOp>(
            loc, elemType, mlir::IntegerAttr::get(elemType, lowest))
        .getResult();
  }

  // Recursively emits nested scf.for loops copying src -> dst element-wise.
  void emitCopyNested(mlir::Value src, mlir::Value dst, mlir::MemRefType srcType,
                      mlir::MemRefType dstType, int64_t dim) {
    mlir::Location loc = builder_.getUnknownLoc();
    int64_t rank = srcType.getRank();
    if (dim >= rank) {
      mlir::Value loaded =
          builder_.create<mlir::memref::LoadOp>(loc, src, indexStack_);
      builder_.create<mlir::memref::StoreOp>(loc, loaded, dst, indexStack_);
      return;
    }
    (void)dstType;
    mlir::Value lb = indexConst(0);
    mlir::Value ub = loopBound(src, srcType, dim);
    mlir::Value step = indexConst(1);
    builder_.create<mlir::scf::ForOp>(
        loc, lb, ub, step, mlir::ValueRange(),
        [&](mlir::OpBuilder &, mlir::Location l, mlir::Value iv,
            mlir::ValueRange) {
          indexStack_.push_back(iv);
          emitCopyNested(src, dst, srcType, dstType, dim + 1);
          indexStack_.pop_back();
          builder_.create<mlir::scf::YieldOp>(l);
        });
  }

  bool translateOp(mlir::Operation *op) {
    mlir::Location loc = op->getLoc();
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (auto constOp = llvm::dyn_cast<mlir::kernelh::atom::ConstOp>(op)) {
      mlir::Type resultType = constOp.getResult().getType();
      if (llvm::isa<mlir::MemRefType>(resultType)) {
        // A memref-typed splat cannot lower to the gpu dialect surface (there
        // is no memref-typed pure value). Drop it when dead.
        if (constOp->use_empty()) {
          return true;
        }
        return fail("cannot lower atom.const splat into a buffer (result "
                    "memref<...> is used)");
      }
      auto constantOp = builder_.create<mlir::arith::ConstantOp>(
          loc, resultType, llvm::cast<mlir::TypedAttr>(constOp.getValueAttr()));
      valueMap_[constOp.getResult()] = constantOp.getResult();
      return true;
    }

    if (llvm::isa<mlir::kernelh::atom::AddOp, mlir::kernelh::atom::SubOp,
                  mlir::kernelh::atom::MulOp, mlir::kernelh::atom::DivOp,
                  mlir::kernelh::atom::MaxOp, mlir::kernelh::atom::MinOp,
                  mlir::kernelh::atom::FmaOp>(op)) {
      mlir::Type resultType = op->getResult(0).getType();
      if (llvm::isa<mlir::MemRefType>(resultType)) {
        return fail("cannot lower elementwise atom op with a memref result "
                    "(tensor-typed elementwise compute is unsupported)");
      }
      mlir::Value lhs = mapValue(op->getOperand(0));
      mlir::Value rhs = mapValue(op->getOperand(1));
      mlir::Type elemType = resultType;
      if (auto vectorType = llvm::dyn_cast<mlir::VectorType>(resultType)) {
        elemType = vectorType.getElementType();
      }
      mlir::Value result;
      if (llvm::isa<mlir::kernelh::atom::FmaOp>(op)) {
        mlir::Value c = mapValue(op->getOperand(2));
        if (!elemType.isFloat()) {
          return fail("cannot lower atom.fma with a non-floating-point result");
        }
        result = builder_
                     .create<mlir::math::FmaOp>(loc, resultType, lhs, rhs, c,
                                                mlir::arith::FastMathFlagsAttr())
                     .getResult();
      } else if (llvm::isa<mlir::kernelh::atom::AddOp>(op)) {
        result = elemType.isFloat()
                     ? builder_
                           .create<mlir::arith::AddFOp>(
                               loc, resultType, lhs, rhs,
                               mlir::arith::FastMathFlagsAttr())
                           .getResult()
                     : builder_
                           .create<mlir::arith::AddIOp>(
                               loc, resultType, lhs, rhs,
                               mlir::arith::IntegerOverflowFlagsAttr())
                           .getResult();
      } else if (llvm::isa<mlir::kernelh::atom::SubOp>(op)) {
        result = elemType.isFloat()
                     ? builder_
                           .create<mlir::arith::SubFOp>(
                               loc, resultType, lhs, rhs,
                               mlir::arith::FastMathFlagsAttr())
                           .getResult()
                     : builder_
                           .create<mlir::arith::SubIOp>(
                               loc, resultType, lhs, rhs,
                               mlir::arith::IntegerOverflowFlagsAttr())
                           .getResult();
      } else if (llvm::isa<mlir::kernelh::atom::MulOp>(op)) {
        result = elemType.isFloat()
                     ? builder_
                           .create<mlir::arith::MulFOp>(
                               loc, resultType, lhs, rhs,
                               mlir::arith::FastMathFlagsAttr())
                           .getResult()
                     : builder_
                           .create<mlir::arith::MulIOp>(
                               loc, resultType, lhs, rhs,
                               mlir::arith::IntegerOverflowFlagsAttr())
                           .getResult();
      } else if (llvm::isa<mlir::kernelh::atom::DivOp>(op)) {
        result = elemType.isFloat()
                     ? builder_
                           .create<mlir::arith::DivFOp>(
                               loc, resultType, lhs, rhs,
                               mlir::arith::FastMathFlagsAttr())
                           .getResult()
                     : builder_
                           .create<mlir::arith::DivSIOp>(loc, resultType, lhs, rhs)
                           .getResult();
      } else if (llvm::isa<mlir::kernelh::atom::MaxOp>(op)) {
        result = elemType.isFloat()
                     ? builder_
                           .create<mlir::arith::MaxNumFOp>(loc, resultType, lhs, rhs)
                           .getResult()
                     : builder_
                           .create<mlir::arith::MaxSIOp>(loc, resultType, lhs, rhs)
                           .getResult();
      } else if (llvm::isa<mlir::kernelh::atom::MinOp>(op)) {
        result = elemType.isFloat()
                     ? builder_
                           .create<mlir::arith::MinNumFOp>(loc, resultType, lhs, rhs)
                           .getResult()
                     : builder_
                           .create<mlir::arith::MinSIOp>(loc, resultType, lhs, rhs)
                           .getResult();
      }
      valueMap_[op->getResult(0)] = result;
      return true;
    }

    if (auto cmpOp = llvm::dyn_cast<mlir::kernelh::atom::CmpOp>(op)) {
      mlir::Value lhs = mapValue(cmpOp.getLhs());
      mlir::Value rhs = mapValue(cmpOp.getRhs());
      mlir::Type operandType = lhs.getType();
      mlir::Value condition;
      switch (cmpOp.getPred()) {
      case mlir::kernelh::atom::CmpPred::lt:
      case mlir::kernelh::atom::CmpPred::le:
      case mlir::kernelh::atom::CmpPred::gt:
      case mlir::kernelh::atom::CmpPred::ge:
      case mlir::kernelh::atom::CmpPred::eq:
      case mlir::kernelh::atom::CmpPred::ne: {
        if (operandType.isFloat()) {
          mlir::arith::CmpFPredicate predicate;
          switch (cmpOp.getPred()) {
          case mlir::kernelh::atom::CmpPred::lt:
            predicate = mlir::arith::CmpFPredicate::OLT;
            break;
          case mlir::kernelh::atom::CmpPred::le:
            predicate = mlir::arith::CmpFPredicate::OLE;
            break;
          case mlir::kernelh::atom::CmpPred::gt:
            predicate = mlir::arith::CmpFPredicate::OGT;
            break;
          case mlir::kernelh::atom::CmpPred::ge:
            predicate = mlir::arith::CmpFPredicate::OGE;
            break;
          case mlir::kernelh::atom::CmpPred::eq:
            predicate = mlir::arith::CmpFPredicate::OEQ;
            break;
          case mlir::kernelh::atom::CmpPred::ne:
            predicate = mlir::arith::CmpFPredicate::ONE;
            break;
          }
          condition = builder_
                          .create<mlir::arith::CmpFOp>(
                              loc, mlir::arith::CmpFPredicateAttr::get(&ctx_, predicate),
                              lhs, rhs, mlir::arith::FastMathFlagsAttr())
                          .getResult();
        } else {
          mlir::arith::CmpIPredicate predicate;
          switch (cmpOp.getPred()) {
          case mlir::kernelh::atom::CmpPred::lt:
            predicate = mlir::arith::CmpIPredicate::slt;
            break;
          case mlir::kernelh::atom::CmpPred::le:
            predicate = mlir::arith::CmpIPredicate::sle;
            break;
          case mlir::kernelh::atom::CmpPred::gt:
            predicate = mlir::arith::CmpIPredicate::sgt;
            break;
          case mlir::kernelh::atom::CmpPred::ge:
            predicate = mlir::arith::CmpIPredicate::sge;
            break;
          case mlir::kernelh::atom::CmpPred::eq:
            predicate = mlir::arith::CmpIPredicate::eq;
            break;
          case mlir::kernelh::atom::CmpPred::ne:
            predicate = mlir::arith::CmpIPredicate::ne;
            break;
          }
          condition = builder_
                          .create<mlir::arith::CmpIOp>(
                              loc, mlir::arith::CmpIPredicateAttr::get(&ctx_, predicate),
                              lhs, rhs)
                          .getResult();
        }
        break;
      }
      }
      // The custom IR cmp carries its boolean as i32 (there is no i1 scalar);
      // the lowered result is re-widened so select's i32 condition stays a
      // boolean carrier.
      mlir::Value carrier = builder_
                                .create<mlir::arith::ExtUIOp>(
                                    loc, builder_.getI32Type(), condition)
                                .getResult();
      valueMap_[cmpOp.getResult()] = carrier;
      return true;
    }

    if (auto selectOp = llvm::dyn_cast<mlir::kernelh::atom::SelectOp>(op)) {
      mlir::Value condition = mapValue(selectOp.getCondition());
      mlir::Value onTrue = mapValue(selectOp.getOnTrue());
      mlir::Value onFalse = mapValue(selectOp.getOnFalse());
      mlir::Value cond1 = builder_
                              .create<mlir::arith::TruncIOp>(
                                  loc, builder_.getI1Type(), condition,
                                  mlir::arith::IntegerOverflowFlagsAttr())
                              .getResult();
      auto lowered = builder_.create<mlir::arith::SelectOp>(
          loc, selectOp.getResult().getType(), cond1, onTrue, onFalse);
      valueMap_[selectOp.getResult()] = lowered.getResult();
      return true;
    }

    if (auto mathOp = llvm::dyn_cast<mlir::kernelh::atom::MathOp>(op)) {
      mlir::Value input = mapValue(mathOp.getInput());
      mlir::Value result;
      switch (mathOp.getKind()) {
      case mlir::kernelh::atom::MathKind::exp:
        result = builder_.create<mlir::math::ExpOp>(loc, input).getResult();
        break;
      case mlir::kernelh::atom::MathKind::log:
        result = builder_.create<mlir::math::LogOp>(loc, input).getResult();
        break;
      case mlir::kernelh::atom::MathKind::sqrt:
        result = builder_.create<mlir::math::SqrtOp>(loc, input).getResult();
        break;
      case mlir::kernelh::atom::MathKind::rsqrt:
        result = builder_.create<mlir::math::RsqrtOp>(loc, input).getResult();
        break;
      case mlir::kernelh::atom::MathKind::tanh:
        result = builder_.create<mlir::math::TanhOp>(loc, input).getResult();
        break;
      }
      valueMap_[mathOp.getResult()] = result;
      return true;
    }

    if (auto loadOp = llvm::dyn_cast<mlir::kernelh::atom::LoadOp>(op)) {
      mlir::Value memref = mapValue(loadOp.getMemref());
      mlir::SmallVector<mlir::Value> indices;
      for (mlir::Value index : loadOp.getIndices()) {
        indices.push_back(mapValue(index));
      }
      auto lowered = builder_.create<mlir::memref::LoadOp>(
          loc, loadOp.getResult().getType(), memref, indices);
      valueMap_[loadOp.getResult()] = lowered.getResult();
      return true;
    }

    if (auto storeOp = llvm::dyn_cast<mlir::kernelh::atom::StoreOp>(op)) {
      mlir::Value value = mapValue(storeOp.getValue());
      mlir::Value memref = mapValue(storeOp.getMemref());
      mlir::SmallVector<mlir::Value> indices;
      for (mlir::Value index : storeOp.getIndices()) {
        indices.push_back(mapValue(index));
      }
      builder_.create<mlir::memref::StoreOp>(loc, value, memref, indices);
      return true;
    }

    if (auto loadVecOp = llvm::dyn_cast<mlir::kernelh::atom::LoadVecOp>(op)) {
      mlir::Value memref = mapValue(loadVecOp.getMemref());
      mlir::SmallVector<mlir::Value> indices;
      for (mlir::Value index : loadVecOp.getIndices()) {
        indices.push_back(mapValue(index));
      }
      auto lowered = builder_.create<mlir::vector::LoadOp>(
          loc, loadVecOp.getResult().getType(), memref, indices);
      valueMap_[loadVecOp.getResult()] = lowered.getResult();
      return true;
    }

    if (auto storeVecOp = llvm::dyn_cast<mlir::kernelh::atom::StoreVecOp>(op)) {
      mlir::Value value = mapValue(storeVecOp.getValue());
      mlir::Value memref = mapValue(storeVecOp.getMemref());
      mlir::SmallVector<mlir::Value> indices;
      for (mlir::Value index : storeVecOp.getIndices()) {
        indices.push_back(mapValue(index));
      }
      builder_.create<mlir::vector::StoreOp>(loc, value, memref, indices);
      return true;
    }

    if (auto allocOp = llvm::dyn_cast<mlir::kernelh::atom::AllocSharedOp>(op)) {
      auto type = mlir::cast<mlir::MemRefType>(allocOp.getResult().getType());
      std::string name = sharedSymbols_.at(type);
      auto getGlobal = builder_.create<mlir::memref::GetGlobalOp>(loc, type, name);
      valueMap_[allocOp.getResult()] = getGlobal.getResult();
      return true;
    }

    if (llvm::isa<mlir::kernelh::atom::BarrierOp>(op)) {
      builder_.create<mlir::gpu::BarrierOp>(loc);
      return true;
    }

    if (auto threadIdOp = llvm::dyn_cast<mlir::kernelh::atom::ThreadIdOp>(op)) {
      mlir::Value result = builder_
                               .create<mlir::gpu::ThreadIdOp>(
                                   loc, builder_.getIndexType(),
                                   mapGpuDimension(threadIdOp.getDim()))
                               .getResult();
      valueMap_[threadIdOp.getResult()] = result;
      return true;
    }
    if (auto blockIdOp = llvm::dyn_cast<mlir::kernelh::atom::BlockIdOp>(op)) {
      mlir::Value result = builder_
                               .create<mlir::gpu::BlockIdOp>(
                                   loc, builder_.getIndexType(),
                                   mapGpuDimension(blockIdOp.getDim()))
                               .getResult();
      valueMap_[blockIdOp.getResult()] = result;
      return true;
    }
    if (auto blockDimOp = llvm::dyn_cast<mlir::kernelh::atom::BlockDimOp>(op)) {
      mlir::Value result = builder_
                               .create<mlir::gpu::BlockDimOp>(
                                   loc, builder_.getIndexType(),
                                   mapGpuDimension(blockDimOp.getDim()))
                               .getResult();
      valueMap_[blockDimOp.getResult()] = result;
      return true;
    }
    if (auto gridDimOp = llvm::dyn_cast<mlir::kernelh::atom::GridDimOp>(op)) {
      mlir::Value result = builder_
                               .create<mlir::gpu::GridDimOp>(
                                   loc, builder_.getIndexType(),
                                   mapGpuDimension(gridDimOp.getDim()))
                               .getResult();
      valueMap_[gridDimOp.getResult()] = result;
      return true;
    }

    if (auto loopOp = llvm::dyn_cast<mlir::kernelh::atom::LoopForOp>(op)) {
      mlir::Value lb = mapValue(loopOp.getLb());
      mlir::Value ub = mapValue(loopOp.getUb());
      mlir::Value step = mapValue(loopOp.getStep());
      mlir::SmallVector<mlir::Value> iterArgs;
      for (mlir::Value arg : loopOp.getIterArgs()) {
        iterArgs.push_back(mapValue(arg));
      }
      mlir::Block &atomBody = loopOp.getRegion().front();
      auto scfLoop = builder_.create<mlir::scf::ForOp>(
          loc, lb, ub, step, iterArgs,
          [&](mlir::OpBuilder &, mlir::Location l, mlir::Value iv,
              mlir::ValueRange its) {
            valueMap_[atomBody.getArgument(0)] = iv;
            for (size_t i = 0; i < its.size(); ++i) {
              valueMap_[atomBody.getArgument(i + 1)] = its[i];
            }
            for (auto &innerOp : atomBody.getOperations()) {
              if (auto yieldOp =
                      llvm::dyn_cast<mlir::kernelh::atom::YieldOp>(innerOp)) {
                mlir::SmallVector<mlir::Value> values;
                for (mlir::Value value : yieldOp.getValues()) {
                  values.push_back(mapValue(value));
                }
                builder_.create<mlir::scf::YieldOp>(l, values);
              } else {
                if (!translateOp(&innerOp)) {
                  return;
                }
              }
            }
          });
      if (error_) {
        return false;
      }
      for (size_t i = 0; i < loopOp.getNumResults(); ++i) {
        valueMap_[loopOp.getResult(i)] = scfLoop.getResult(i);
      }
      return true;
    }

    if (auto reduceOp = llvm::dyn_cast<mlir::kernelh::atom::ReduceOp>(op)) {
      mlir::Value tensor = mapValue(reduceOp.getTensor());
      auto type = mlir::cast<mlir::MemRefType>(tensor.getType());
      mlir::Value acc = reduceIdentity(type.getElementType(), reduceOp.getKind());
      indexStack_.clear();
      mlir::Value result =
          emitReduceNested(tensor, type, reduceOp.getKind(), 0, acc);
      valueMap_[reduceOp.getResult()] = result;
      return true;
    }

    if (auto copyOp = llvm::dyn_cast<mlir::kernelh::atom::CopyOp>(op)) {
      mlir::Value src = mapValue(copyOp.getSrc());
      mlir::Value dst = mapValue(copyOp.getDst());
      auto srcType = mlir::cast<mlir::MemRefType>(src.getType());
      auto dstType = mlir::cast<mlir::MemRefType>(dst.getType());
      indexStack_.clear();
      emitCopyNested(src, dst, srcType, dstType, 0);
      return true;
    }

    return fail("unsupported atom op for gpu lowering: " +
                op->getName().getStringRef().str());
  }

private:
  mlir::MLIRContext &ctx_;
  mlir::OpBuilder builder_;
  llvm::DenseMap<mlir::Value, mlir::Value> valueMap_;
  llvm::DenseMap<mlir::MemRefType, std::string> sharedSymbols_;
  llvm::SmallVector<mlir::Value> indexStack_;
};

// ---------------------------------------------------------------------------
// Dialect registration helper.
// ---------------------------------------------------------------------------

void ensureDialectsLoaded(mlir::MLIRContext &ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::kernelh::atom::AtomDialect, mlir::gpu::GPUDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::vector::VectorDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
}

} // namespace

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, std::string>
lowerToAtomDialect(const Module &module, mlir::MLIRContext &ctx) {
  ensureDialectsLoaded(ctx);

  mlir::OpBuilder builder(&ctx);
  auto target = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

  AtomEmitter emitter(ctx);
  if (!emitter.translateModule(module, target)) {
    return std::unexpected(emitter.error_.value_or("stage 1 failed"));
  }

  if (failed(mlir::verify(target))) {
    return std::unexpected("stage 1: atom dialect module failed verification");
  }
  return target;
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, std::string>
lowerAtomToGpu(mlir::ModuleOp atomModule, mlir::MLIRContext &ctx) {
  ensureDialectsLoaded(ctx);

  mlir::OpBuilder builder(&ctx);
  auto target = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(&target.getBodyRegion().front());
  auto gpuModule =
      builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(), "kernelh");

  GpuEmitter emitter(ctx);
  if (!emitter.translateKernels(atomModule, gpuModule)) {
    return std::unexpected(emitter.error_.value_or("stage 2 failed"));
  }

  if (failed(mlir::verify(gpuModule))) {
    return std::unexpected("stage 2: gpu module failed verification");
  }
  return target;
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, std::string>
emitGpuModule(const Module &module, mlir::MLIRContext &ctx) {
  auto atom = lowerToAtomDialect(module, ctx);
  if (!atom) {
    return std::unexpected(atom.error());
  }
  auto gpu = lowerAtomToGpu((*atom).get(), ctx);
  if (!gpu) {
    return std::unexpected(gpu.error());
  }

  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (failed(pm.run(**gpu))) {
    return std::unexpected("emitter cleanup (canonicalize + cse) failed");
  }
  return gpu;
}

} // namespace kernelh::compiler
