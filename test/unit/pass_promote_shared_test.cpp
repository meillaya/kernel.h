// promote-shared pass unit tests (plan todo 16): static shared-memory
// promotion with the pinned barrier rule. Passes run through the GLOBAL
// registry (self-registered by lib/passes/*.cpp at static init).
//
// The COPY-mode path is exercised through the REAL pipeline
// (tile,promote-shared) on the serial GEMM tile skeleton from the lit
// fixture (test/fixtures/passes/promote_shared.atom): the tile pass rewrites
// the domain loop into the block-tile loop (block_id offsets + boundary
// atom.select guards), and promote-shared stages the tile loads - whose
// tensors ARE the tiles (A: 64x32, B: 32x64) - with the whole-tile
// atom.copy global->shared. The STORE-mode path (full-size tensors) is
// exercised with promote-shared alone on a hand-built direct-body tiled
// form.

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::IdDim;
using kernelh::ir::Kernel;
using kernelh::ir::MemSpace;
using kernelh::ir::Module;
using kernelh::ir::Op;
using kernelh::ir::OpKind;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;
namespace builder = kernelh::ir::builder;
using kernelh::ir::printModule;
using kernelh::ir::runPasses;
using kernelh::ir::verifyModule;

namespace {

Type f32() { return Type::scalar(ScalarType::F32); }
Type index() { return Type::scalar(ScalarType::Index); }
Type tensor(std::vector<int64_t> shape) {
  return Type::tensor(TensorType{std::move(shape), ScalarType::F32,
                                 MemSpace::Global});
}

// The serial GEMM tile skeleton the tile pass consumes (the contract in
// lib/passes/tiling.cpp): the outermost M-loop is the domain loop
// (0 to extent, step 1) feeding the body's loads; N- and K-loops nest below.
Module buildSerialGemm(const std::string &name, std::vector<int64_t> tile,
                       int64_t extent, int64_t kExtent,
                       std::vector<int64_t> aShape, std::vector<int64_t> bShape,
                       std::vector<int64_t> cShape, bool useShared = true) {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  kernel->schedule.tile = tile;
  kernel->schedule.map_blocks = true;
  kernel->schedule.map_warps = true;
  kernel->schedule.use_shared_memory = useShared;
  kernel->shape["k"] = kExtent;
  kernel->shape["m"] = extent;
  kernel->shape["n"] = extent;

  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensor(aShape)});
  b.args.push_back(BlockArg{tensor(bShape)});
  b.args.push_back(BlockArg{tensor(cShape)});
  const Value a = Value::blockArg(&b, 0);
  const Value bb = Value::blockArg(&b, 1);
  const Value c = Value::blockArg(&b, 2);

  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value cE = builder::makeConst(b, extent, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value cK = builder::makeConst(b, kExtent, index())->result();

  Op *mLoop = builder::makeLoopFor(b, c0, cE, c1, {}, {});
  Block &mBody = *mLoop->region;
  const Value m = Value::blockArg(&mBody, 0);
  const Value f0 = builder::makeConst(mBody, 0.0, f32())->result();

  Op *nLoop = builder::makeLoopFor(mBody, c0, cE, c1, {}, {});
  Block &nBody = *nLoop->region;
  const Value n = Value::blockArg(&nBody, 0);

  Op *kLoop = builder::makeLoopFor(nBody, c0, cK, c1, {f0}, {f32()});
  Block &kBody = *kLoop->region;
  const Value k = Value::blockArg(&kBody, 0);
  const Value acc = Value::blockArg(&kBody, 1);
  const Value la = builder::makeLoad(kBody, a, {m, k})->result();
  const Value lb = builder::makeLoad(kBody, bb, {k, n})->result();
  const Value f = builder::makeFma(kBody, la, lb, acc)->result();
  builder::makeYield(kBody, {f});

  builder::makeStore(nBody, kLoop->result(), c, {m, n});
  builder::makeYield(nBody, {});
  builder::makeYield(mBody, {});

  module.kernels.push_back(std::move(kernel));
  return module;
}

// Copy-mode pipeline fixture: one-tile problem (extent 64 = BM = BN, K = 32
// = BK) with tile-shaped tensors, so promotion fires atom.copy.
Module buildCopyModeGemm(bool useShared = true) {
  return buildSerialGemm("promote_gemm", {64, 64, 32}, 64, 32, {64, 32},
                         {32, 64}, {64, 64}, useShared);
}

// Overflow pipeline fixture: tile [256, 256, 64] needs 256x64 + 64x256 f32
// tiles = 131072 B > the 65536-byte gfx1101 LDS budget.
Module buildOverflowGemm() {
  return buildSerialGemm("shmem_overflow", {256, 256, 64}, 256, 64, {256, 64},
                         {64, 256}, {256, 256});
}

// Hand-built DIRECT-BODY tiled GEMM with full-size tensors (1024x1024): the
// store-mode path (per-element staging, block_id offsets in the load
// indices). Mirrors the tile pass output shape for a full-size GEMM.
Module buildStoreModeGemm() {
  Module module;
  auto kernel = std::make_unique<Kernel>("promote_gemm_full");
  kernel->schedule.tile = {64, 64, 32};
  kernel->schedule.map_blocks = true;
  kernel->schedule.map_warps = true;
  kernel->schedule.use_shared_memory = true;
  kernel->shape["k"] = 1024;
  kernel->shape["m"] = 1024;
  kernel->shape["n"] = 1024;

  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensor({1024, 1024})});
  b.args.push_back(BlockArg{tensor({1024, 1024})});
  b.args.push_back(BlockArg{tensor({1024, 1024})});
  const Value a = Value::blockArg(&b, 0);
  const Value bb = Value::blockArg(&b, 1);
  const Value c = Value::blockArg(&b, 2);

  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value cK = builder::makeConst(b, 1024, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value cT = builder::makeConst(b, 64, index())->result();
  const Value f0 = builder::makeConst(b, 0.0, f32())->result();
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value bidx = builder::makeId(b, OpKind::BlockId, IdDim::X)->result();
  const Value bidy = builder::makeId(b, OpKind::BlockId, IdDim::Y)->result();
  const Value blockRow = builder::makeBin(b, OpKind::Mul, bidx, cT)->result();
  const Value row = builder::makeBin(b, OpKind::Add, blockRow, tid)->result();
  const Value blockCol = builder::makeBin(b, OpKind::Mul, bidy, cT)->result();
  const Value col = builder::makeBin(b, OpKind::Add, blockCol, tid)->result();

  Op *loop = builder::makeLoopFor(b, c0, cK, c1, {f0}, {f32()});
  Block &body = *loop->region;
  const Value k = Value::blockArg(&body, 0);
  const Value acc = Value::blockArg(&body, 1);
  const Value la = builder::makeLoad(body, a, {row, k})->result();
  const Value lb = builder::makeLoad(body, bb, {k, col})->result();
  const Value f = builder::makeFma(body, la, lb, acc)->result();
  builder::makeYield(body, {f});

  builder::makeStore(b, loop->result(), c, {row, col});

  module.kernels.push_back(std::move(kernel));
  return module;
}

// Runs a pipeline through the global registry, requiring success.
std::string run(Module &module, const std::string &pipeline) {
  const auto result = runPasses(module, pipeline);
  REQUIRE(result.has_value());
  return printModule(module);
}

size_t count(const std::string &text, const std::string &needle) {
  size_t n = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++n;
    pos += needle.size();
  }
  return n;
}

} // namespace

TEST_CASE("pass_promote_shared_fires_only_with_use_shared", "[unit][pass]") {
  // use_shared_memory = true: promotion fires (2 static buffers + 2
  // whole-tile copies + 2 barriers per the pinned rule).
  {
    Module module = buildCopyModeGemm(true);
    const std::string out = run(module, "tile,promote-shared");
    CHECK(count(out, "atom.alloc_shared") == 2);
    CHECK(count(out, "atom.copy") == 2);
    CHECK(count(out, "atom.barrier") == 2);
  }
  // use_shared_memory = false: promote-shared is a documented no-op - the
  // tiled output is identical to running tile alone (only the schedule attr
  // line differs, so compare with the attr normalized out).
  {
    Module withShared = buildCopyModeGemm(true);
    (void)run(withShared, "tile");
    const std::string tiledAlone = printModule(withShared);

    Module withoutShared = buildCopyModeGemm(false);
    const std::string out = run(withoutShared, "tile,promote-shared");
    CHECK(count(out, "atom.alloc_shared") == 0);
    CHECK(count(out, "atom.copy") == 0);
    // Same body modulo the use_shared_memory attr: strip both attr lines and
    // compare the rest.
    auto stripAttr = [](const std::string &text) {
      std::string t = text;
      const size_t pos = t.find("  use_shared_memory = ");
      if (pos != std::string::npos) {
        t.erase(pos, t.find('\n', pos) - pos + 1);
      }
      return t;
    };
    CHECK(stripAttr(out) == stripAttr(tiledAlone));
  }
  // Per-kernel gating: in a two-kernel module only the use_shared kernel is
  // promoted.
  {
    Module module = buildCopyModeGemm(true);
    module.kernels.push_back(std::make_unique<Kernel>("plain_gemm"));
    const std::string out = run(module, "tile,promote-shared");
    CHECK(count(out, "atom.alloc_shared") == 2);
    CHECK(count(out, "atom.copy") == 2);
    CHECK(verifyModule(module).empty());
  }
}

TEST_CASE("pass_promote_shared_static_sizes", "[unit][pass]") {
  Module module = buildCopyModeGemm();
  const std::string out = run(module, "tile,promote-shared");
  // The buffers are static (compile-time) and sized from the tile dims x
  // element size: 64x32xf32 = 8192 B and 32x64xf32 = 8192 B.
  CHECK(count(out, "atom.alloc_shared : tensor<64x32xf32, shared>") == 1);
  CHECK(count(out, "atom.alloc_shared : tensor<32x64xf32, shared>") == 1);
  // The staged shared total (the verifier's element-size table) fits the
  // gfx1101 LDS budget and the promoted IR verifies clean.
  CHECK(kernelh::ir::kernelTotalSharedBytes(*module.kernels[0]) == 16384);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_promote_shared_barrier_count_pinned_rule", "[unit][pass]") {
  Module module = buildCopyModeGemm();
  const std::string out = run(module, "tile,promote-shared");
  // Pinned rule (G20) on this 2-stage fixture (store stage: 2 whole-tile
  // copies; compute stage): exactly ONE barrier after the last shared store
  // and before the first shared load, and ONE at the end of the tile
  // iteration -> 2 barriers total.
  CHECK(count(out, "atom.barrier") == 2);
  // ORDER assertion: copies (shared stores) < first barrier < shared loads
  // < compute < second barrier < yield.
  const size_t firstCopy = out.find("atom.copy");
  const size_t firstBarrier = out.find("atom.barrier");
  const size_t sharedLoad = out.find("atom.load");
  const size_t fma = out.find("atom.fma");
  const size_t secondBarrier = out.rfind("atom.barrier");
  // The LAST yield is the tile loop's (the K-loop yields earlier inside).
  const size_t yieldPos = out.rfind("atom.yield");
  CHECK(firstCopy != std::string::npos);
  CHECK(firstBarrier != std::string::npos);
  CHECK(firstCopy < firstBarrier);
  CHECK(firstBarrier < sharedLoad);
  CHECK(sharedLoad < fma);
  CHECK(fma < secondBarrier);
  CHECK(secondBarrier < yieldPos);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_promote_shared_store_mode_full_tensors", "[unit][pass]") {
  // Full-size tensors: the store stage keeps the global loads and stores
  // each loaded value into the tile-sized buffer at decomposed indices;
  // compute reads shared.
  Module module = buildStoreModeGemm();
  const std::string out = run(module, "promote-shared");
  CHECK(count(out, "atom.alloc_shared : tensor<64x32xf32, shared>") == 1);
  CHECK(count(out, "atom.alloc_shared : tensor<32x64xf32, shared>") == 1);
  CHECK(count(out, "atom.copy") == 0); // no whole-tile copy in store mode
  // 2 global tile loads (kept, moved to the store stage) + 2 shared loads in
  // the compute.
  CHECK(count(out, "atom.load") == 4);
  // 2 shared stores + the C store.
  CHECK(count(out, "atom.store") == 3);
  CHECK(count(out, "atom.barrier") == 2);
  // ORDER: the store stage (global loads + shared stores) precedes the first
  // barrier, the shared loads follow it, and the second barrier ends the
  // tile iteration. The first load AFTER the barrier is a shared load (the
  // moved global loads all precede it).
  const size_t firstBarrier = out.find("atom.barrier");
  CHECK(firstBarrier != std::string::npos);
  const std::string prefix = out.substr(0, firstBarrier);
  CHECK(prefix.find("atom.store") != std::string::npos);
  CHECK(prefix.find("atom.load") != std::string::npos);
  CHECK(prefix.find("atom.fma") == std::string::npos);
  const size_t sharedLoad = out.find("atom.load", firstBarrier);
  CHECK(sharedLoad < out.find("atom.fma"));
  CHECK(verifyModule(module).empty());

  // Idempotency: a second run is a no-op (the global loads now feed only
  // shared stores, which the planner detects and skips).
  const std::string once = out;
  const auto result = runPasses(module, "promote-shared");
  REQUIRE(result.has_value());
  CHECK(printModule(module) == once);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_promote_shared_overflow_rejected", "[unit][pass]") {
  // The tile config [256, 256, 64] needs 131072 B > the 65536-byte gfx1101
  // LDS budget. The pass returns a std::expected error naming the tile
  // config (no crash); the module is left at the tiled, unpromoted state.
  Module tiled = buildOverflowGemm();
  (void)run(tiled, "tile");
  const std::string tiledAlone = printModule(tiled);

  Module module = buildOverflowGemm();
  const auto result = runPasses(module, "tile,promote-shared");
  REQUIRE_FALSE(result.has_value());
  const std::string &message = result.error();
  CHECK(message.find("shared memory usage 131072 bytes") != std::string::npos);
  CHECK(message.find("tile config [256, 256, 64]") != std::string::npos);
  CHECK(message.find("65536-byte LDS limit") != std::string::npos);
  CHECK(message.find("gfx1101") != std::string::npos);
  CHECK(printModule(module) == tiledAlone);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_promote_shared_promoted_ir_verifies_clean", "[unit][pass]") {
  Module copyMode = buildCopyModeGemm();
  (void)run(copyMode, "tile,promote-shared");
  CHECK(verifyModule(copyMode).empty());

  Module storeMode = buildStoreModeGemm();
  (void)run(storeMode, "promote-shared");
  CHECK(verifyModule(storeMode).empty());
}
