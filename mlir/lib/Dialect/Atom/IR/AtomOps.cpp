//===- AtomOps.cpp - kernel.h atom IR ops ------------------------*- C++ -*-===//
//
// Custom printer/parser/verifier implementations for ops declared in
// AtomOps.td. Ops that use the declarative `assemblyFormat` get their
// generated implementations from AtomOps.cpp.inc below; only the ops marked
// `hasCustomAssemblyFormat` / `hasVerifier` are implemented here.
//
// This file is part of kernel.h.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "kernelh/Dialect/Atom/IR/AtomDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/STLExtras.h"

#define GET_OP_CLASSES
#include "kernelh/Dialect/Atom/IR/AtomOps.cpp.inc"
#include "kernelh/Dialect/Atom/IR/AtomOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::kernelh::atom;

namespace mlir {
namespace kernelh {
namespace atom {

//===----------------------------------------------------------------------===//
// AtomConstOp
//===----------------------------------------------------------------------===//

// Parse `atom.const <literal> attr-dict : <result-type>`. The literal is
// parsed by hand (not via parseAttribute) so the trailing ": <result-type>"
// is never consumed as the literal's own type (the builtin literal parser
// greedily consumes an inline ": type"). Bare integer literals parse as
// signless i64 and floats as f64; the typed value attribute is rebuilt from
// the result element type below.
ParseResult ConstOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt intVal(64, 0);
  bool isFloat = false;
  double floatVal = 0.0;
  auto intRes = parser.parseOptionalInteger(intVal);
  if (intRes.has_value()) {
    if (failed(*intRes))
      return failure();
  } else if (succeeded(parser.parseFloat(floatVal))) {
    isFloat = true;
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected constant literal value (integer or "
                            "floating point)");
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();
  result.addTypes(resultType);

  // A memref/vector result is a splat of the literal; resolve the element
  // type used to type the value attribute.
  Type elemType = resultType;
  if (auto memrefType = llvm::dyn_cast<MemRefType>(resultType))
    elemType = memrefType.getElementType();
  else if (auto vectorType = llvm::dyn_cast<VectorType>(resultType))
    elemType = vectorType.getElementType();

  Attribute valueAttr;
  if (auto floatType = llvm::dyn_cast<FloatType>(elemType)) {
    double value = isFloat ? floatVal : static_cast<double>(intVal.getSExtValue());
    valueAttr = FloatAttr::get(floatType, value);
  } else if (elemType.isIndex()) {
    valueAttr = IntegerAttr::get(elemType, intVal);
  } else if (auto intType = llvm::dyn_cast<IntegerType>(elemType)) {
    // Range-check the literal against the target integer width so a
    // non-representable value is a clean parse error, not an assert.
    unsigned width = intType.getWidth();
    APInt truncated = intVal.trunc(width);
    bool inRange = intVal.isNegative()
                       ? (intVal == truncated.sext(64))
                       : (intVal == truncated.zext(64));
    if (!inRange) {
      return parser.emitError(parser.getCurrentLocation(),
                              "integer literal out of range for result type");
    }
    valueAttr = IntegerAttr::get(intType, truncated);
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "constant result type must be a scalar, memref, or "
                            "vector of a scalar");
  }
  result.addAttribute("value", valueAttr);
  return success();
}

void ConstOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printAttributeWithoutType(getValueAttr());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
  printer << " : ";
  printer.printType(getResult().getType());
}

LogicalResult ConstOp::verify() {
  Type elemType = getResult().getType();
  if (auto memrefType = llvm::dyn_cast<MemRefType>(elemType))
    elemType = memrefType.getElementType();
  else if (auto vectorType = llvm::dyn_cast<VectorType>(elemType))
    elemType = vectorType.getElementType();

  Attribute value = getValueAttr();
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(value)) {
    if (floatAttr.getType() != elemType) {
      return emitOpError("literal type ")
             << floatAttr.getType()
             << " does not match the result element type " << elemType;
    }
  } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value)) {
    if (intAttr.getType() != elemType) {
      return emitOpError("literal type ")
             << intAttr.getType()
             << " does not match the result element type " << elemType;
    }
  } else {
    return emitOpError("expected the value attribute to be an integer or "
                       "floating point literal");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AtomLoadOp / AtomStoreOp / AtomLoadVecOp / AtomStoreVecOp
//===----------------------------------------------------------------------===//

static LogicalResult checkIndices(Operation *op, MemRefType memrefType,
                                  unsigned numIndices) {
  if (numIndices != static_cast<unsigned>(memrefType.getRank())) {
    return op->emitOpError("expected ")
           << memrefType.getRank() << " indices, got " << numIndices;
  }
  return success();
}

LogicalResult LoadOp::verify() {
  auto memrefType = llvm::cast<MemRefType>(getMemref().getType());
  if (failed(checkIndices(*this, memrefType, getIndices().size())))
    return failure();
  if (getResult().getType() != memrefType.getElementType()) {
    return emitOpError("result type ")
           << getResult().getType()
           << " does not match the memref element type "
           << memrefType.getElementType();
  }
  return success();
}

LogicalResult StoreOp::verify() {
  auto memrefType = llvm::cast<MemRefType>(getMemref().getType());
  if (failed(checkIndices(*this, memrefType, getIndices().size())))
    return failure();
  if (getValue().getType() != memrefType.getElementType()) {
    return emitOpError("value type ")
           << getValue().getType()
           << " does not match the memref element type "
           << memrefType.getElementType();
  }
  return success();
}

static LogicalResult verifyVecAccess(Operation *op, MemRefType memrefType,
                                     VectorType vectorType,
                                     unsigned numIndices) {
  if (failed(checkIndices(op, memrefType, numIndices)))
    return failure();
  if (vectorType.getElementType() != memrefType.getElementType()) {
    return op->emitOpError("vector element type ")
           << vectorType.getElementType()
           << " does not match the memref element type "
           << memrefType.getElementType();
  }
  return success();
}

LogicalResult LoadVecOp::verify() {
  auto memrefType = llvm::cast<MemRefType>(getMemref().getType());
  auto vectorType = llvm::cast<VectorType>(getResult().getType());
  return verifyVecAccess(*this, memrefType, vectorType, getIndices().size());
}

LogicalResult StoreVecOp::verify() {
  auto memrefType = llvm::cast<MemRefType>(getMemref().getType());
  auto vectorType = llvm::cast<VectorType>(getValue().getType());
  return verifyVecAccess(*this, memrefType, vectorType, getIndices().size());
}

//===----------------------------------------------------------------------===//
// AtomReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  auto memrefType = llvm::cast<MemRefType>(getTensor().getType());
  int64_t rank = memrefType.getRank();
  if (getDim() >= static_cast<uint64_t>(rank)) {
    return emitOpError("reduction dimension ")
           << getDim() << " is out of range for memref of rank " << rank;
  }
  if (getResult().getType() != memrefType.getElementType()) {
    return emitOpError("result type ")
           << getResult().getType()
           << " does not match the memref element type "
           << memrefType.getElementType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AtomLoopForOp
//===----------------------------------------------------------------------===//

LogicalResult LoopForOp::verify() {
  Block *body = getBody();
  size_t numIterArgs = getIterArgs().size();
  if (body->getNumArguments() != 1 + numIterArgs) {
    return emitOpError("expected the region to have an induction argument "
                       "plus one argument per iter_arg, got ")
           << body->getNumArguments() << " arguments for " << numIterArgs
           << " iter_args";
  }
  if (!llvm::isa<IndexType>(body->getArgument(0).getType())) {
    return emitOpError("expected the induction argument to have 'index' "
                       "type, got ")
           << body->getArgument(0).getType();
  }
  for (auto [i, iter] : llvm::enumerate(getIterArgs())) {
    if (body->getArgument(i + 1).getType() != iter.getType()) {
      return emitOpError("region argument ")
             << (i + 1) << " type " << body->getArgument(i + 1).getType()
             << " does not match the iter_arg type " << iter.getType();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AtomKernelOp
//===----------------------------------------------------------------------===//

// Parse `atom.kernel @name(optional args) attributes {...} { region }`.
ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symName;
  if (parser.parseSymbolName(symName))
    return failure();
  result.addAttribute("sym_name", symName);

  // Optional function-style argument list: "(" %v0: type, %v1: type ")".
  SmallVector<OpAsmParser::Argument, 4> args;
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseArgumentList(args, OpAsmParser::Delimiter::None,
                                 /*allowType=*/true, /*allowAttrs=*/false) ||
        parser.parseRParen())
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, args, /*enableNameShadowing=*/true))
    return failure();
  if (body->empty())
    body->emplaceBlock();
  return success();
}

void KernelOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  if (getBody().getNumArguments() > 0) {
    printer << "(";
    llvm::interleaveComma(
        getBody().getArguments(), printer.getStream(),
        [&](BlockArgument arg) { printer.printRegionArgument(arg); });
    printer << ")";
  }
  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"sym_name"});
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

} // namespace atom
} // namespace kernelh
} // namespace mlir
