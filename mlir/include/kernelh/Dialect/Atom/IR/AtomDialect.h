//===- AtomDialect.h - kernel.h atom IR MLIR dialect -----------*- C++ -*-===//
//
// The "atom" dialect (plan todo 20): the MLIR-side mirror of the custom
// structured-SSA atom IR (plan todo 9), used by the emitter (plan todo 21)
// as the bridge to the MLIR GPU dialect.
//
// This file is part of kernel.h.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef KERNELH_ATOM_DIALECT_H
#define KERNELH_ATOM_DIALECT_H

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "kernelh/Dialect/Atom/IR/AtomOpsDialect.h.inc"
#include "kernelh/Dialect/Atom/IR/AtomOpsEnums.h.inc"

namespace mlir {
namespace kernelh {
namespace atom {

/// Returns true if `type` is a memref in the shared memory space (3) - the
/// MLIR mirror of the custom IR's `tensor<...xf32, shared>` type.
bool isSharedMemRefType(::mlir::Type type);

/// Registers the atom dialect in `registry` (used by kernelh-opt).
void registerAtomDialect(::mlir::DialectRegistry &registry);

} // namespace atom
} // namespace kernelh
} // namespace mlir

#define GET_OP_CLASSES
#include "kernelh/Dialect/Atom/IR/AtomOps.h.inc"

#endif // KERNELH_ATOM_DIALECT_H
