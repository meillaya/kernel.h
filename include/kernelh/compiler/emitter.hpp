// kernel.h compiler - emitter (plan todo 21).
//
// Two-stage lowering from the custom "atom" IR (todo 9) to the MLIR GPU
// dialect surface, DEVICE-ONLY:
//
//   Stage 1 (lowerToAtomDialect): structural translation of a
//     kernelh::ir::Module into an mlir::ModuleOp of "atom" dialect ops
//     (todo 20), 1:1 with the custom op inventory.
//   Stage 2 (lowerAtomToGpu): the atom module -> a device-only gpu.module
//     containing one gpu.func per atom.kernel with the `kernel` unit attr
//     (required for .entry/.visible lowering), dynamic-shape memref args
//     (memref-descriptor ABI - NO bare-ptr convention), shared memory as
//     addrspace-3 global symbols (memref.global ... = uninitialized), and
//     lowered bodies using gpu/scf/vector/arith/math/memref ops only.
//
// emitGpuModule fuses both stages and runs canonicalize + CSE on the
// resulting gpu.module.
//
// The emitter lives in its own static library (kernelh_emitter, built from
// lib/compiler/emitter.cpp) so kernelh_compiler itself stays MLIR-free - the
// no-MLIR link of kernelh_weak_default_test (todo 23) is preserved.

#ifndef KERNELH_COMPILER_EMITTER_HPP
#define KERNELH_COMPILER_EMITTER_HPP

#include "kernelh/ir/module.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <expected>
#include <memory>
#include <string>

namespace kernelh::compiler {

// Stage 1: custom atom IR -> atom MLIR dialect. Returns a ModuleOp holding
// one atom.kernel op per custom kernel, or an error string on failure.
std::expected<mlir::OwningOpRef<mlir::ModuleOp>, std::string>
lowerToAtomDialect(const kernelh::ir::Module &module, mlir::MLIRContext &ctx);

// Stage 2: atom dialect module -> device-only gpu.module (gpu.func entries,
// dynamic memref args, addrspace-3 shared globals). `atomModule` must be the
// output of lowerToAtomDialect. Returns the gpu.module on success, or an
// error string describing the first unsupported construct.
std::expected<mlir::OwningOpRef<mlir::ModuleOp>, std::string>
lowerAtomToGpu(mlir::ModuleOp atomModule, mlir::MLIRContext &ctx);

// Fused entry point: custom IR -> gpu.module (stages 1+2) with a
// canonicalize + CSE cleanup run on the result. The returned module is a
// builtin.module containing a gpu.module with one gpu.func per kernel.
std::expected<mlir::OwningOpRef<mlir::ModuleOp>, std::string>
emitGpuModule(const kernelh::ir::Module &module, mlir::MLIRContext &ctx);

} // namespace kernelh::compiler

#endif // KERNELH_COMPILER_EMITTER_HPP
