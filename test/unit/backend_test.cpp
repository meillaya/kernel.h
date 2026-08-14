// Backend unit tests (plan todo 22): in-process NVVM (PTX/cubin) and ROCDL
// (HSACO/ISA) lowering + blob extraction. Assertions are content-based (ELF
// magic bytes, PTX/ISA text markers), never just exit codes: a wrong blob
// that "compiled" is caught by the magic-byte probe.
//
// The input is the todo-7/8 proven fixture test/fixtures/w2_vecadd_gpu.mlir
// (hand-written gpu.module) plus the todo-21 emitter's output (vecadd) -
// exercising the kernel-marker normalization shim.

#include "kernelh/compiler/backends.hpp"
#include "kernelh/compiler/emitter.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// A context with every dialect the fixture + the vendor pipelines touch
// registered (the fixture contains gpu/memref/arith/llvm ops; the pipelines
// create nvvm/rocdl/llvm ops).
mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::gpu::GPUDialect, mlir::scf::SCFDialect,
                  mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                  mlir::index::IndexDialect, mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                  mlir::amdgpu::AMDGPUDialect>();
  return mlir::MLIRContext(registry);
}

// Reads the todo-7 fixture (a proven gpu.module that lowers to a working
// HSACO / PTX at sm_80).
const std::string &w2Fixture() {
  static const std::string text = [] {
    std::ifstream in(std::string(KERNELH_SOURCE_DIR) +
                     "/test/fixtures/w2_vecadd_gpu.mlir");
    REQUIRE(in.good());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
  }();
  return text;
}

// The todo-21 emitter path on a small atom kernel (exercises the backends
// consuming the emitter's exact output, incl. its literal `kernel` attr).
const std::string &vecaddAtom() {
  static const std::string text = R"atom(
atom.kernel @vecadd(%v0: tensor<1024xf32, global>, %v1: tensor<1024xf32, global>) {
  tile = [256]
  map_blocks = true
  map_warps = false
  use_shared_memory = false
  vector_width = 1
  unroll = 1
  shape = {n = 1048576}
  %v2 = atom.const 0 : index
  %v3 = atom.block_id x : index
  %v4 = atom.block_dim x : index
  %v5 = atom.thread_id x : index
  %v6 = atom.mul %v3, %v4 : index
  %v7 = atom.add %v6, %v5 : index
  %v8 = atom.load %v0[%v7] : tensor<1024xf32, global>
  %v9 = atom.load %v1[%v7] : tensor<1024xf32, global>
  %v10 = atom.add %v8, %v9 : f32
  atom.store %v10, %v1[%v7] : tensor<1024xf32, global>
}
)atom";
  return text;
}

mlir::OwningOpRef<mlir::ModuleOp> parseFixture(mlir::MLIRContext &ctx) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(w2Fixture(), &ctx);
  REQUIRE(static_cast<bool>(module));
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> emitterVecadd(mlir::MLIRContext &ctx) {
  auto parsed = kernelh::ir::parseModule(vecaddAtom());
  REQUIRE(parsed.has_value());
  REQUIRE(kernelh::ir::verifyModule(*parsed).empty());
  auto gpu = kernelh::compiler::emitGpuModule(*parsed, ctx);
  REQUIRE(gpu.has_value());
  return std::move(*gpu);
}

const std::vector<char> kElfMagic = {
    static_cast<char>(0x7f), 'E', 'L', 'F'};

bool startsWith(const std::vector<char> &bytes, const std::vector<char> &prefix) {
  return bytes.size() >= prefix.size() &&
         std::equal(prefix.begin(), prefix.end(), bytes.begin());
}

// RAII restore for a process environment variable.
class EnvGuard {
public:
  explicit EnvGuard(const char *name) : name_(name) {
    if (const char *value = std::getenv(name)) {
      saved_ = value;
      hadValue_ = true;
    }
  }
  ~EnvGuard() {
    if (hadValue_) {
      ::setenv(name_.c_str(), saved_.c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  std::string saved_;
  bool hadValue_ = false;
};

} // namespace

TEST_CASE("backend_chip_validation", "[unit][backend]") {
  CHECK(kernelh::compiler::parseTargetSpec("nvptx-sm_80").has_value());
  CHECK(kernelh::compiler::parseTargetSpec("amdgcn-gfx1101").has_value());
  CHECK_FALSE(kernelh::compiler::parseTargetSpec("nvptx-sm_999").has_value());
  CHECK_FALSE(kernelh::compiler::parseTargetSpec("amdgcn-gfx999").has_value());
  CHECK_FALSE(kernelh::compiler::parseTargetSpec("sm_80").has_value());
  CHECK_FALSE(kernelh::compiler::parseTargetSpec("").has_value());
  CHECK_FALSE(kernelh::compiler::parseTargetSpec("bogus").has_value());

  auto badNv = kernelh::compiler::parseTargetSpec("nvptx-sm_999");
  REQUIRE_FALSE(badNv.has_value());
  CHECK(badNv.error().find("sm_999") != std::string::npos);
  CHECK(badNv.error().find("sm_80") != std::string::npos);

  auto badAmd = kernelh::compiler::parseTargetSpec("amdgcn-gfx999");
  REQUIRE_FALSE(badAmd.has_value());
  CHECK(badAmd.error().find("gfx999") != std::string::npos);
  CHECK(badAmd.error().find("gfx1101") != std::string::npos);
}

TEST_CASE("backend_rocdl_hsaco_elf_magic", "[unit][backend]") {
  mlir::MLIRContext ctx = makeContext();
  auto module = parseFixture(ctx);
  auto spec = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
  REQUIRE(spec.has_value());
  auto blob = kernelh::compiler::compileToBlob(module, *spec, "hsaco");
  REQUIRE(blob.has_value());
  CHECK(blob->size() > 0);
  INFO("first bytes: " << std::string(blob->begin(), blob->begin() + 8));
  CHECK(startsWith(*blob, kElfMagic));
}

TEST_CASE("backend_rocdl_isa_mnemonics", "[unit][backend]") {
  mlir::MLIRContext ctx = makeContext();
  auto module = parseFixture(ctx);
  auto spec = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
  REQUIRE(spec.has_value());
  auto text = kernelh::compiler::compileToText(module, *spec, "isa");
  REQUIRE(text.has_value());
  // Loose amdgcn mnemonic probes (the fixture exercises LDS + global paths).
  const bool hasMnemonic = (text->find("s_waitcnt") != std::string::npos) ||
                           (text->find("v_add") != std::string::npos) ||
                           (text->find("ds_") != std::string::npos);
  CHECK(hasMnemonic);
}

TEST_CASE("backend_nvvm_ptx_markers", "[unit][backend]") {
  mlir::MLIRContext ctx = makeContext();
  auto module = parseFixture(ctx);
  auto spec = kernelh::compiler::parseTargetSpec("nvptx-sm_80");
  REQUIRE(spec.has_value());
  auto ptx = kernelh::compiler::compileToText(module, *spec, "ptx");
  REQUIRE(ptx.has_value());
  CHECK(ptx->find(".version") != std::string::npos);
  CHECK(ptx->find(".target sm_80") != std::string::npos);
  CHECK(ptx->find(".visible .entry xchg_kernel") != std::string::npos);
}

TEST_CASE("backend_nvvm_cubin_elf_magic", "[unit][backend]") {
  mlir::MLIRContext ctx = makeContext();
  auto module = parseFixture(ctx);
  auto spec = kernelh::compiler::parseTargetSpec("nvptx-sm_80");
  REQUIRE(spec.has_value());
  auto cubin = kernelh::compiler::compileToBlob(module, *spec, "cubin");
  REQUIRE(cubin.has_value());
  CHECK(startsWith(*cubin, kElfMagic));
}

TEST_CASE("backend_emitter_output_lowers_to_entry", "[unit][backend]") {
  // The todo-21 emitter stores the kernel marker as a literal `kernel` attr;
  // the backends normalize it to `gpu.kernel` so the vendor lowering emits a
  // launchable `.visible .entry` (and the ROCDL side a real kernel entry).
  mlir::MLIRContext ctx = makeContext();
  {
    auto module = emitterVecadd(ctx);
    auto spec = kernelh::compiler::parseTargetSpec("nvptx-sm_80");
    REQUIRE(spec.has_value());
    auto ptx = kernelh::compiler::compileToText(module, *spec, "ptx");
    REQUIRE(ptx.has_value());
    CHECK(ptx->find(".visible .entry vecadd(") != std::string::npos);
    CHECK(ptx->find(".target sm_80") != std::string::npos);
  }
  {
    auto module = emitterVecadd(ctx);
    auto spec = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
    REQUIRE(spec.has_value());
    auto hsaco = kernelh::compiler::compileToBlob(module, *spec, "hsaco");
    REQUIRE(hsaco.has_value());
    CHECK(startsWith(*hsaco, kElfMagic));
  }
}

TEST_CASE("backend_hsaco_deterministic", "[unit][backend]") {
  // Byte-for-byte determinism across two independent pipeline runs.
  std::vector<char> first;
  std::vector<char> second;
  {
    mlir::MLIRContext ctx = makeContext();
    auto module = parseFixture(ctx);
    auto spec = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
    REQUIRE(spec.has_value());
    auto blob = kernelh::compiler::compileToBlob(module, *spec, "hsaco");
    REQUIRE(blob.has_value());
    first = *blob;
  }
  {
    mlir::MLIRContext ctx = makeContext();
    auto module = parseFixture(ctx);
    auto spec = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
    REQUIRE(spec.has_value());
    auto blob = kernelh::compiler::compileToBlob(module, *spec, "hsaco");
    REQUIRE(blob.has_value());
    second = *blob;
  }
  CHECK(first.size() == second.size());
  CHECK(std::equal(first.begin(), first.end(), second.begin()));
}

TEST_CASE("backend_format_target_mismatch", "[unit][backend]") {
  mlir::MLIRContext ctx = makeContext();
  auto module = parseFixture(ctx);
  auto amd = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
  REQUIRE(amd.has_value());
  auto wrongBlob = kernelh::compiler::compileToBlob(module, *amd, "cubin");
  REQUIRE_FALSE(wrongBlob.has_value());
  CHECK(wrongBlob.error().find("cubin") != std::string::npos);

  auto nv = kernelh::compiler::parseTargetSpec("nvptx-sm_80");
  REQUIRE(nv.has_value());
  auto wrongText = kernelh::compiler::compileToText(module, *nv, "isa");
  REQUIRE_FALSE(wrongText.has_value());
  CHECK(wrongText.error().find("isa") != std::string::npos);
}

TEST_CASE("backend_missing_toolkit_diagnostic", "[unit][backend]") {
  EnvGuard guard("ROCM_TOOLKIT_ROOT");
  ::unsetenv("ROCM_TOOLKIT_ROOT");
  mlir::MLIRContext ctx = makeContext();
  auto module = parseFixture(ctx);
  auto spec = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
  REQUIRE(spec.has_value());
  auto blob = kernelh::compiler::compileToBlob(module, *spec, "hsaco");
  REQUIRE_FALSE(blob.has_value());
  CHECK(blob.error().find("ROCM_TOOLKIT_ROOT") != std::string::npos);
}

TEST_CASE("backend_empty_toolkit_diagnostic", "[unit][backend]") {
  // Empty-dir toolkit: the serializer's ld.lld invocation must fail with a
  // diagnostic mentioning the ROCm bitcode path (not a silent success).
  llvm::SmallString<128> dir;
  const auto ec = llvm::sys::fs::createUniqueDirectory(
      "/tmp/kernelh-empty-toolkit", dir);
  REQUIRE(!ec);
  EnvGuard guard("ROCM_TOOLKIT_ROOT");
  ::setenv("ROCM_TOOLKIT_ROOT", dir.c_str(), 1);
  mlir::MLIRContext ctx = makeContext();
  auto module = parseFixture(ctx);
  auto spec = kernelh::compiler::parseTargetSpec("amdgcn-gfx1101");
  REQUIRE(spec.has_value());
  auto blob = kernelh::compiler::compileToBlob(module, *spec, "hsaco");
  llvm::sys::fs::remove(dir.c_str());
  REQUIRE_FALSE(blob.has_value());
  CHECK(blob.error().find("bitcode") != std::string::npos);
}

TEST_CASE("backend_escape_decoder", "[unit][backend]") {
  using kernelh::compiler::decodeMlirEscapedString;
  CHECK(decodeMlirEscapedString(R"(\\")") == std::string("\\\""));
  CHECK(decodeMlirEscapedString("\\7FELF") == std::string("\x7f", 1) + "ELF");
  CHECK(decodeMlirEscapedString("a\\01b") == std::string("a\x01", 2) + "b");
  CHECK(decodeMlirEscapedString("plain") == "plain");
}
