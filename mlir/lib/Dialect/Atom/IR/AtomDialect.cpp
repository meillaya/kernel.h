//===- AtomDialect.cpp - kernel.h atom IR MLIR dialect ----------*- C++ -*-===//
//
// Dialect registration and the shared-memref type predicate for the "atom"
// dialect (plan todo 20).
//
// This file is part of kernel.h.
//===----------------------------------------------------------------------===//

#include "kernelh/Dialect/Atom/IR/AtomDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "kernelh/Dialect/Atom/IR/AtomOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::kernelh::atom;

namespace mlir {
namespace kernelh {
namespace atom {

bool isSharedMemRefType(Type type) {
  auto memrefType = llvm::dyn_cast<MemRefType>(type);
  if (!memrefType)
    return false;
  if (!memrefType.getMemorySpace())
    return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memrefType.getMemorySpace()))
    return intAttr.getValue() == 3;
  return false;
}

void registerAtomDialect(DialectRegistry &registry) {
  registry.insert<AtomDialect>();
}

void AtomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "kernelh/Dialect/Atom/IR/AtomOps.cpp.inc"
  >();
}

} // namespace atom
} // namespace kernelh
} // namespace mlir
