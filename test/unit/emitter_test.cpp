// Emitter unit tests (plan todo 21): custom atom IR -> MLIR atom dialect
// (stage 1) -> device-only gpu.module (stage 2). Structural assertions on the
// printed MLIR output: dynamic memref args, gpu.func kernel attr, addrspace-3
// shared globals, gpu.barrier, scf.for iter_args.

#include "kernelh/compiler/emitter.hpp"

#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <llvm/Support/raw_ostream.h>

#include <string>

namespace {

// Parses + verifies `text`, runs emitGpuModule, returns the printed module.
std::string emitText(const std::string &text) {
  auto parsed = kernelh::ir::parseModule(text);
  REQUIRE(parsed.has_value());
  const kernelh::ir::Diagnostics diags = kernelh::ir::verifyModule(*parsed);
  INFO("verify: " << [&]() {
    std::string all;
    for (const auto &d : diags) {
      all += d.toString() + "\n";
    }
    return all;
  }());
  REQUIRE(diags.empty());

  mlir::MLIRContext ctx;
  auto gpu = kernelh::compiler::emitGpuModule(*parsed, ctx);
  REQUIRE(gpu.has_value());
  std::string out;
  llvm::raw_string_ostream os(out);
  (**gpu).print(os);
  return os.str();
}

// Parses + verifies `text`, runs stage 1 only, returns the printed module.
std::string atomText(const std::string &text) {
  auto parsed = kernelh::ir::parseModule(text);
  REQUIRE(parsed.has_value());
  REQUIRE(kernelh::ir::verifyModule(*parsed).empty());

  mlir::MLIRContext ctx;
  auto atom = kernelh::compiler::lowerToAtomDialect(*parsed, ctx);
  REQUIRE(atom.has_value());
  std::string out;
  llvm::raw_string_ostream os(out);
  (**atom).print(os);
  return os.str();
}

const char *kVecadd = R"atom(
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

const char *kSharedLoop = R"atom(
atom.kernel @shl(%v0: tensor<64xf32, global>, %v1: tensor<64xf32, global>) {
  tile = [64]
  map_blocks = true
  map_warps = false
  use_shared_memory = true
  vector_width = 1
  unroll = 1
  shape = {n = 64}
  %v2 = atom.const 0 : index
  %v3 = atom.const 64 : index
  %v4 = atom.const 1 : index
  %v5 = atom.thread_id x : index
  %v6 = atom.block_id x : index
  %v7 = atom.block_dim x : index
  %v8 = atom.mul %v6, %v7 : index
  %v9 = atom.add %v8, %v5 : index
  %v10 = atom.alloc_shared : tensor<64xf32, shared>
  %v11 = atom.load %v0[%v9] : tensor<64xf32, global>
  atom.store %v11, %v10[%v5] : tensor<64xf32, shared>
  atom.barrier
  %v12 = atom.load %v10[%v5] : tensor<64xf32, shared>
  %v13 = atom.loop_for %v2 to %v3 step %v4 iter_args(%v12) -> (f32) {
    ^bb0(%v14: index, %v15: f32):
      %v16 = atom.load %v10[%v14] : tensor<64xf32, shared>
      %v17 = atom.add %v16, %v15 : f32
      atom.yield %v17 : f32
  }
  atom.store %v13, %v1[%v9] : tensor<64xf32, global>
}
)atom";

} // namespace

TEST_CASE("emitter_stage1_roundtrip", "[unit][emitter]") {
  const std::string text = atomText(kVecadd);
  INFO(text);
  CHECK(text.find("module {") != std::string::npos);
  CHECK(text.find("atom.kernel @vecadd(") != std::string::npos);
  CHECK(text.find("memref<1024xf32>") != std::string::npos);
  CHECK(text.find("atom.add") != std::string::npos);
  CHECK(text.find("atom.load") != std::string::npos);
  CHECK(text.find("atom.store") != std::string::npos);
  CHECK(text.find("map_blocks = true") != std::string::npos);
  CHECK(text.find("tile = [256]") != std::string::npos);
}

TEST_CASE("emitter_gpu_func_dynamic_args", "[unit][emitter]") {
  const std::string text = emitText(kVecadd);
  INFO(text);
  CHECK(text.find("gpu.module @kernelh") != std::string::npos);
  // Kernel entry: DYNAMIC memref args (descriptor ABI) + the kernel attr.
  CHECK(text.find("gpu.func @vecadd(%arg0: memref<?xf32>, %arg1: memref<?xf32>)"
                  " attributes {kernel} {") != std::string::npos);
  CHECK(text.find("gpu.thread_id") != std::string::npos);
  CHECK(text.find("gpu.block_id") != std::string::npos);
  CHECK(text.find("arith.muli") != std::string::npos);
  CHECK(text.find("memref.load") != std::string::npos);
  CHECK(text.find("arith.addf") != std::string::npos);
  CHECK(text.find("gpu.return") != std::string::npos);
}

TEST_CASE("emitter_shared_memory_global", "[unit][emitter]") {
  const std::string text = emitText(kSharedLoop);
  INFO(text);
  // Shared alloc -> addrspace-3 memref.global (the proven Oracle B4
  // mechanism), never a function-local alloc.
  CHECK(text.find("memref.global @shl_shared_0 : memref<64xf32, 3> = "
                  "uninitialized") != std::string::npos);
  CHECK(text.find("memref.get_global @shl_shared_0 : memref<64xf32, 3>") !=
        std::string::npos);
  CHECK(text.find("memref.alloc") == std::string::npos);
}

TEST_CASE("emitter_barrier", "[unit][emitter]") {
  const std::string text = emitText(kSharedLoop);
  INFO(text);
  CHECK(text.find("gpu.barrier") != std::string::npos);
  CHECK(text.find("atom.barrier") == std::string::npos);
}

TEST_CASE("emitter_loop_iter_args", "[unit][emitter]") {
  const std::string text = emitText(kSharedLoop);
  INFO(text);
  // atom.loop_for with iter_args -> scf.for with iter_args + scf.yield.
  CHECK(text.find("scf.for %") != std::string::npos);
  CHECK(text.find("iter_args(") != std::string::npos);
  CHECK(text.find("-> (f32) {") != std::string::npos);
  CHECK(text.find("scf.yield") != std::string::npos);
  CHECK(text.find("atom.loop_for") == std::string::npos);
}

TEST_CASE("emitter_unsupported_splat_diagnostic", "[unit][emitter]") {
  const std::string text = R"atom(
atom.kernel @bad(%v0: tensor<16xf32, global>) {
  tile = [16]
  map_blocks = false
  map_warps = false
  use_shared_memory = false
  vector_width = 1
  unroll = 1
  shape = {}
  %v1 = atom.const 0 : index
  %v2 = atom.const 0.0 : f32
  %v3 = atom.const 1.0 : tensor<16xf32, global>
  atom.store %v2, %v3[%v1] : tensor<16xf32, global>
}
)atom";
  auto parsed = kernelh::ir::parseModule(text);
  REQUIRE(parsed.has_value());
  REQUIRE(kernelh::ir::verifyModule(*parsed).empty());

  mlir::MLIRContext ctx;
  auto gpu = kernelh::compiler::emitGpuModule(*parsed, ctx);
  REQUIRE(!gpu.has_value());
  CHECK(gpu.error().find("cannot lower atom.const splat into a buffer") !=
        std::string::npos);
}
