// kernel.h compiler - vendor backends (plan todo 22).
//
// In-process vendor lowering of a device-only gpu.module (the emitter's
// emitGpuModule output, or a hand-written fixture like
// test/fixtures/w2_vecadd_gpu.mlir) into NVVM PTX/cubin and ROCDL
// HSACO/ISA artifacts. The pipelines are the EXACT pinned recipes from
// todos 7/8, now driven through mlir::PassManager instead of shelling out
// to mlir-opt:
//
//   NVVM  (target "nvptx-<sm>"):
//     gpu.module(convert-vector-to-llvm, convert-scf-to-cf,
//                convert-gpu-to-nvvm)
//     nvvm-attach-target{chip=<sm> O=3}
//     gpu-to-llvm
//     reconcile-unrealized-casts
//     gpu-module-to-binary{format=isa}                      (PTX text)
//     gpu-module-to-binary{format=bin toolkit=$CUDA_TOOLKIT_ROOT
//                          l=$CUDA_TOOLKIT_ROOT/nvvm/libdevice/libdevice.10.bc}
//                                                           (cubin blob)
//   ROCDL (target "amdgcn-<gfx>"):
//     gpu.module(convert-vector-to-llvm, convert-scf-to-cf,
//                convert-gpu-to-rocdl{chipset=<gfx> runtime=HIP})
//     rocdl-attach-target{chip=<gfx>}
//     gpu-to-llvm
//     reconcile-unrealized-casts
//     gpu-module-to-binary{format=bin toolkit=$ROCM_TOOLKIT_ROOT}   (HSACO)
//     gpu-module-to-binary{format=isa toolkit=$ROCM_TOOLKIT_ROOT}   (ISA text)
//
// Descriptor memref ABI throughout - NO bare-ptr options. The toolkit roots
// are read from the environment at RUNTIME (CUDA_TOOLKIT_ROOT /
// ROCM_TOOLKIT_ROOT, with a KERNELH_* override) and every artifact is
// validated by its content (ELF magic / mnemonics), never just exit codes.
//
// The chip string is validated against a small allowlist BEFORE any pipeline
// runs: a bogus chip (e.g. "gfx999") makes the ROCDL serializer SIGABRT with
// exit 134, so unknown chips are rejected up front with a clean error.
//
// Blob extraction: gpu-module-to-binary replaces the gpu.module with a
// gpu.binary op whose #gpu.object<...> attribute carries the payload. The
// in-memory StringAttr already holds the RAW bytes (MLIR escaping only
// happens when printing to text), so extraction copies them directly;
// decodeMlirEscapedString mirrors .omo/evidence/task-8-kernelh/extract_gpu_object.py
// for the text-round-tripped case.

#ifndef KERNELH_COMPILER_BACKENDS_HPP
#define KERNELH_COMPILER_BACKENDS_HPP

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <expected>
#include <string>
#include <vector>

namespace kernelh::compiler {

// A lowering target: either "nvptx-<sm_XX>" or "amdgcn-<gfxXXXX>". Construct
// via parseTargetSpec (which validates the chip against the allowlist).
struct TargetSpec {
  std::string target;

  // "nvptx-sm_80" etc.
  bool isNVPTX() const;
  // "amdgcn-gfx1101" etc.
  bool isROCDL() const;
  // The chip part ("sm_80" / "gfx1101"); empty for an unparseable target.
  std::string chip() const;
};

// Parses + validates a --compile-target value. Unknown prefixes or chips
// return a descriptive error naming the offending chip and listing the valid
// options (the pipeline never sees an invalid chip, so the ROCDL serializer's
// bogus-chip SIGABRT is avoided).
std::expected<TargetSpec, std::string> parseTargetSpec(std::string_view target);

// In-process vendor lowering. `format` is one of:
//   "ptx"   -> NVVM PTX text     (gpu-module-to-binary format=isa)
//   "cubin" -> NVVM cubin blob   (format=bin + libdevice link)
//   "hsaco" -> ROCDL HSACO blob  (format=bin + ROCm toolkit)
//   "isa"   -> ROCDL ISA text    (format=isa + ROCm toolkit; the toolkit is
//                                 still required for the serializer's ld.lld)
// The module must be a builtin.module containing a gpu.module. It is mutated
// in place (the gpu.module is replaced by a gpu.binary op).
// compileToBlob returns the raw extracted bytes; compileToText the decoded
// text payload. Errors name the failing stage / missing toolkit.
std::expected<std::vector<char>, std::string>
compileToBlob(mlir::OwningOpRef<mlir::ModuleOp> &module, const TargetSpec &spec,
              std::string_view format);

std::expected<std::string, std::string>
compileToText(mlir::OwningOpRef<mlir::ModuleOp> &module, const TargetSpec &spec,
              std::string_view format);

// Inverts MLIR AsmPrinter string escaping: \xx hex, \" and \\. Mirror of the
// `unescape` step in .omo/evidence/task-8-kernelh/extract_gpu_object.py.
// Needed only when a gpu.binary payload was round-tripped through TEXT (the
// in-memory StringAttr from gpu-module-to-binary already holds raw bytes).
std::string decodeMlirEscapedString(std::string_view escaped);

} // namespace kernelh::compiler

#endif // KERNELH_COMPILER_BACKENDS_HPP
