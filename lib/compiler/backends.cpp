// kernel.h compiler - vendor backends (plan todo 22).
//
// In-process NVVM (PTX/cubin) and ROCDL (HSACO/ISA) lowering of a
// device-only gpu.module. The pipelines are the EXACT pinned recipes from
// todos 7/8 translated 1:1 into mlir::PassManager calls:
//
//   NVVM:  gpu.module(convert-vector-to-llvm, convert-scf-to-cf,
//                     convert-gpu-to-nvvm) + nvvm-attach-target{chip O=3} +
//                     gpu-to-llvm + reconcile-unrealized-casts +
//                     gpu-module-to-binary{format=isa} | {format=bin toolkit
//                     l=libdevice}
//   ROCDL: gpu.module(convert-vector-to-llvm, convert-scf-to-cf,
//                     convert-gpu-to-rocdl{chipset runtime=HIP}) +
//                     rocdl-attach-target{chip} + gpu-to-llvm +
//                     reconcile-unrealized-casts + gpu-module-to-binary
//                     {format=bin toolkit} | {format=isa toolkit}
//
// NOT shelling out to mlir-opt, NOT bare-ptr call conventions, no "asm"
// format strings. The chip is validated against a small allowlist BEFORE any
// pipeline runs (a bogus chip aborts the ROCDL serializer with SIGABRT-134).
//
// Blob extraction mirrors .omo/evidence/task-8-kernelh/extract_gpu_object.py:
// the payload lives in the #gpu.object attribute of a gpu.binary op. In
// memory the StringAttr holds the RAW bytes (MLIR escaping is a print-time
// artifact), so extraction copies them directly; decodeMlirEscapedString
// exists for text-round-tripped payloads.

#include "kernelh/compiler/backends.hpp"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include <cstdlib>
#include <set>
#include <string>

namespace kernelh::compiler {
namespace {

// Registers every dialect the vendor pipelines touch plus the LLVMIR
// translation interfaces the serializers need. The compile entry points call
// this so callers do not need to pre-register dialects (the kernelh-opt mlir
// mode registers everything up front; the atom path only registers the
// emitter subset).
void ensureBackendDialects(mlir::MLIRContext &ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::gpu::GPUDialect, mlir::scf::SCFDialect,
                  mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                  mlir::index::IndexDialect, mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                  mlir::amdgpu::AMDGPUDialect>();
  // The ConvertToLLVMPatternInterface external models the GPU->LLVM passes
  // use to recursively convert nested func.func/vector/memref/arith ops.
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  mlir::NVVM::registerConvertGpuToNVVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);
}

// Small allowlists (todo 22 pin): sm_70..sm_90 for NVPTX, the RDNA/CDNA
// subset the pipeline gate proved for AMDGCN. NEVER grow these ad hoc.
const std::set<std::string> &nvptxChips() {
  static const std::set<std::string> chips = {
      "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"};
  return chips;
}
const std::set<std::string> &amdgcnChips() {
  static const std::set<std::string> chips = {
      "gfx90a", "gfx1030", "gfx1100", "gfx1101"};
  return chips;
}

std::string joinChips(const std::set<std::string> &chips) {
  std::string out;
  for (const auto &chip : chips) {
    if (!out.empty()) {
      out += ", ";
    }
    out += chip;
  }
  return out;
}

// Resolves a toolkit root from the environment: a KERNELH_<VAR> override wins
// over the plain <VAR> the devenv shell exports (CUDA_TOOLKIT_ROOT /
// ROCM_TOOLKIT_ROOT). A missing variable is a clean error naming it.
std::expected<std::string, std::string>
resolveToolkit(const char *envVar, std::string_view vendorName) {
  const std::string overrideVar = std::string("KERNELH_") + envVar;
  for (const std::string &var : {overrideVar, std::string(envVar)}) {
    if (const char *value = std::getenv(var.c_str());
        value != nullptr && *value != '\0') {
      return std::string(value);
    }
  }
  return std::unexpected("missing " + std::string(envVar) +
                         " environment variable: the " + std::string(vendorName) +
                         " toolkit root is required for this artifact");
}

// The todo-21 emitter stored the gpu.func kernel marker under the literal
// "kernel" attribute; the GPU dialect's kernel marker is "gpu.kernel"
// (GPUDialect::getKernelFuncAttrName), which is what isKernel() and the
// vendor lowering check. Without the rename the kernel lowers to a plain
// `.visible .func` instead of a launchable `.visible .entry`. The hand-written
// fixtures already carry `gpu.kernel`; this makes the emitter's output behave
// identically (kept in the backends so the emitter stays untouched).
void normalizeKernelMarkers(mlir::ModuleOp module) {
  module.walk([&](mlir::gpu::GPUFuncOp func) {
    if (func.isKernel()) {
      return;
    }
    mlir::Attribute kernelAttr = func->getAttr("kernel");
    if (!kernelAttr || !mlir::isa<mlir::UnitAttr>(kernelAttr)) {
      return;
    }
    func->removeAttr("kernel");
    func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), kernelAttr);
  });
}

// Runs `pm` on the module, capturing MLIR diagnostics into the returned error
// so callers can assert on them (e.g. the toolkit-path diagnostic).
std::expected<void, std::string> runWithDiagnostics(mlir::PassManager &pm,
                                                    mlir::ModuleOp module,
                                                    std::string_view stage) {
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(module->getContext(),
                                        [&](mlir::Diagnostic &diag) {
                                          diagnostics += diag.str();
                                          diagnostics += "\n";
                                          return mlir::success();
                                        });
  if (mlir::failed(pm.run(module))) {
    if (!diagnostics.empty()) {
      return std::unexpected("pipeline '" + std::string(stage) +
                             "' failed:\n" + diagnostics);
    }
    return std::unexpected("pipeline '" + std::string(stage) + "' failed");
  }
  return {};
}

// NVVM (todo 8 pin). `format` is "ptx" (format=isa) or "cubin"
// (format=bin + toolkit + libdevice link).
std::expected<void, std::string> runNVVMPipeline(mlir::ModuleOp module,
                                                 const std::string &chip,
                                                 std::string_view format) {
  mlir::MLIRContext *ctx = module->getContext();
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(*ctx);

  mlir::PassManager pm(ctx);
  auto &gpuPm = pm.nest<mlir::gpu::GPUModuleOp>();
  gpuPm.addPass(mlir::createConvertVectorToLLVMPass());
  gpuPm.addPass(mlir::createSCFToControlFlowPass());
  mlir::ConvertGpuOpsToNVVMOpsOptions nvvmOpts; // descriptor ABI: bare-ptr off
  gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvmOpts));

  mlir::GpuNVVMAttachTargetOptions attachOpts;
  attachOpts.chip = chip;
  attachOpts.optLevel = 3;
  pm.addPass(mlir::createGpuNVVMAttachTarget(attachOpts));

  mlir::GpuToLLVMConversionPassOptions gpuToLLVMOpts;
  pm.addPass(mlir::createGpuToLLVMConversionPass(gpuToLLVMOpts));

  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  mlir::GpuModuleToBinaryPassOptions binOpts;
  if (format == "cubin") {
    binOpts.compilationTarget = "bin";
    auto toolkit = resolveToolkit("CUDA_TOOLKIT_ROOT", "CUDA");
    if (!toolkit) {
      return std::unexpected(toolkit.error());
    }
    binOpts.toolkitPath = *toolkit;
    // l= linkFiles is the pass's OWN ListOption (Oracle B2): libdevice is
    // mandatory for math kernels - a cubin without it fails ptxas.
    binOpts.linkFiles.push_back(*toolkit + "/nvvm/libdevice/libdevice.10.bc");
  } else {
    binOpts.compilationTarget = "isa";
  }
  pm.addPass(mlir::createGpuModuleToBinaryPass(binOpts));

  return runWithDiagnostics(pm, module, "nvptx-" + chip);
}

// ROCDL (todo 7 pin). Both HSACO (format=bin) and ISA (format=isa) require
// the toolkit: the serializer invokes <toolkit>/llvm/bin/ld.lld even for
// assembly output.
std::expected<void, std::string> runROCDLPipeline(mlir::ModuleOp module,
                                                  const std::string &chip,
                                                  std::string_view format) {
  mlir::MLIRContext *ctx = module->getContext();
  mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(*ctx);

  mlir::PassManager pm(ctx);
  auto &gpuPm = pm.nest<mlir::gpu::GPUModuleOp>();
  gpuPm.addPass(mlir::createConvertVectorToLLVMPass());
  gpuPm.addPass(mlir::createSCFToControlFlowPass());
  gpuPm.addPass(mlir::createLowerGpuOpsToROCDLOpsPass(
      chip, mlir::kDeriveIndexBitwidthFromDataLayout,
      /*useBarePtrCallConv=*/false, mlir::gpu::amd::Runtime::HIP));

  mlir::GpuROCDLAttachTargetOptions attachOpts;
  attachOpts.chip = chip;
  pm.addPass(mlir::createGpuROCDLAttachTarget(attachOpts));

  mlir::GpuToLLVMConversionPassOptions gpuToLLVMOpts;
  pm.addPass(mlir::createGpuToLLVMConversionPass(gpuToLLVMOpts));

  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  auto toolkit = resolveToolkit("ROCM_TOOLKIT_ROOT", "ROCm");
  if (!toolkit) {
    return std::unexpected(toolkit.error());
  }
  mlir::GpuModuleToBinaryPassOptions binOpts;
  binOpts.toolkitPath = *toolkit;
  binOpts.compilationTarget =
      (format == "isa") ? "isa" : "bin";
  pm.addPass(mlir::createGpuModuleToBinaryPass(binOpts));

  return runWithDiagnostics(pm, module, "amdgcn-" + chip);
}

std::expected<void, std::string> runPipeline(mlir::ModuleOp module,
                                             const TargetSpec &spec,
                                             std::string_view format) {
  if (spec.isNVPTX()) {
    return runNVVMPipeline(module, spec.chip(), format);
  }
  if (spec.isROCDL()) {
    return runROCDLPipeline(module, spec.chip(), format);
  }
  return std::unexpected("invalid target \"" + spec.target + "\"");
}

// Extracts the payload out of the gpu.binary produced by gpu-module-to-binary.
// `wantText` selects the assembly (format=isa) vs binary (format=bin) object.
std::expected<std::vector<char>, std::string>
extractPayload(mlir::ModuleOp module, std::string_view format) {
  const bool wantText = (format == "ptx" || format == "isa");

  mlir::gpu::BinaryOp binOp;
  module.walk([&](mlir::gpu::BinaryOp op) {
    if (!binOp) {
      binOp = op;
    }
  });
  if (!binOp) {
    return std::unexpected(
        "no gpu.binary op found in the module after lowering (did the "
        "pipeline run?)");
  }

  mlir::gpu::ObjectAttr firstMatch;
  for (mlir::Attribute attr : binOp.getObjects()) {
    auto obj = mlir::dyn_cast<mlir::gpu::ObjectAttr>(attr);
    if (!obj) {
      continue;
    }
    if (!firstMatch) {
      firstMatch = obj;
    }
    const mlir::gpu::CompilationTarget kind = obj.getFormat();
    const bool isText = (kind == mlir::gpu::CompilationTarget::Assembly);
    const bool isBin = (kind == mlir::gpu::CompilationTarget::Binary ||
                        kind == mlir::gpu::CompilationTarget::Offload);
    if (wantText ? isText : isBin) {
      mlir::StringAttr payload = obj.getObject();
      return std::vector<char>(payload.getValue().begin(),
                               payload.getValue().end());
    }
  }

  // Fall back to the single object our pipelines always produce (guards
  // against surprising CompilationTarget spellings).
  if (firstMatch) {
    mlir::StringAttr payload = firstMatch.getObject();
    return std::vector<char>(payload.getValue().begin(),
                             payload.getValue().end());
  }
  return std::unexpected("gpu.binary op carries no #gpu.object payload");
}

} // namespace

bool TargetSpec::isNVPTX() const {
  return target.rfind("nvptx-", 0) == 0;
}

bool TargetSpec::isROCDL() const {
  return target.rfind("amdgcn-", 0) == 0;
}

std::string TargetSpec::chip() const {
  constexpr std::string_view kNvptxPrefix = "nvptx-";
  constexpr std::string_view kRocdlPrefix = "amdgcn-";
  if (target.rfind(kNvptxPrefix, 0) == 0) {
    return target.substr(kNvptxPrefix.size());
  }
  if (target.rfind(kRocdlPrefix, 0) == 0) {
    return target.substr(kRocdlPrefix.size());
  }
  return "";
}

std::expected<TargetSpec, std::string>
parseTargetSpec(std::string_view target) {
  constexpr std::string_view kNvptxPrefix = "nvptx-";
  constexpr std::string_view kRocdlPrefix = "amdgcn-";
  if (target.rfind(kNvptxPrefix, 0) == 0) {
    const std::string chip(target.substr(kNvptxPrefix.size()));
    if (nvptxChips().count(chip) == 0) {
      return std::unexpected("unknown NVPTX chip \"" + chip +
                             "\"; valid chips: " + joinChips(nvptxChips()));
    }
    return TargetSpec{std::string(target)};
  }
  if (target.rfind(kRocdlPrefix, 0) == 0) {
    const std::string chip(target.substr(kRocdlPrefix.size()));
    if (amdgcnChips().count(chip) == 0) {
      return std::unexpected("unknown AMDGCN chip \"" + chip +
                             "\"; valid chips: " + joinChips(amdgcnChips()));
    }
    return TargetSpec{std::string(target)};
  }
  return std::unexpected("unknown compile target \"" + std::string(target) +
                         "\"; expected \"nvptx-<sm_XX>\" or "
                         "\"amdgcn-<gfxXXXX>\"");
}

std::expected<std::vector<char>, std::string>
compileToBlob(mlir::OwningOpRef<mlir::ModuleOp> &module, const TargetSpec &spec,
              std::string_view format) {
  if (format == "cubin" && !spec.isNVPTX()) {
    return std::unexpected("format \"cubin\" requires an nvptx target (got \"" +
                           spec.target + "\")");
  }
  if (format == "hsaco" && !spec.isROCDL()) {
    return std::unexpected("format \"hsaco\" requires an amdgcn target (got \"" +
                           spec.target + "\")");
  }
  if (format != "cubin" && format != "hsaco") {
    return std::unexpected("compileToBlob: invalid format \"" +
                           std::string(format) +
                           "\" (expected \"cubin\" or \"hsaco\")");
  }
  ensureBackendDialects(*module->getContext());
  normalizeKernelMarkers(*module);
  auto pipeline = runPipeline(*module, spec, format);
  if (!pipeline) {
    return std::unexpected(pipeline.error());
  }
  return extractPayload(*module, format);
}std::expected<std::string, std::string>
compileToText(mlir::OwningOpRef<mlir::ModuleOp> &module, const TargetSpec &spec,
              std::string_view format) {
  if (format == "ptx" && !spec.isNVPTX()) {
    return std::unexpected("format \"ptx\" requires an nvptx target (got \"" +
                           spec.target + "\")");
  }
  if (format == "isa" && !spec.isROCDL()) {
    return std::unexpected("format \"isa\" requires an amdgcn target (got \"" +
                           spec.target + "\")");
  }
  if (format != "ptx" && format != "isa") {
    return std::unexpected("compileToText: invalid format \"" +
                           std::string(format) +
                           "\" (expected \"ptx\" or \"isa\")");
  }
  ensureBackendDialects(*module->getContext());
  normalizeKernelMarkers(*module);
  auto pipeline = runPipeline(*module, spec, format);
  if (!pipeline) {
    return std::unexpected(pipeline.error());
  }
  auto payload = extractPayload(*module, format);
  if (!payload) {
    return std::unexpected(payload.error());
  }
  return std::string(payload->begin(), payload->end());
}

std::string decodeMlirEscapedString(std::string_view escaped) {
  const auto hexValue = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  };

  std::string out;
  out.reserve(escaped.size());
  size_t i = 0;
  while (i < escaped.size()) {
    const char c = escaped[i];
    if (c != '\\' || i + 1 >= escaped.size()) {
      out.push_back(c);
      ++i;
      continue;
    }
    const char next = escaped[i + 1];
    if (next == '"') {
      out.push_back('"');
      i += 2;
    } else if (next == '\\') {
      out.push_back('\\');
      i += 2;
    } else if (i + 2 < escaped.size() && hexValue(next) >= 0 &&
               hexValue(escaped[i + 2]) >= 0) {
      out.push_back(static_cast<char>(hexValue(next) * 16 +
                                      hexValue(escaped[i + 2])));
      i += 3;
    } else {
      out.push_back(c);
      ++i;
    }
  }
  return out;
}

} // namespace kernelh::compiler
