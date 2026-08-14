// Tiling pass unit tests (plan todo 15): the tile pass consumes the kernel's
// schedule attrs (tile = [BM, BN, BK], map_blocks, map_warps) and rewrites
// the iteration domain into a grid-level block loop (block_id-derived
// offsets) + intra-block tile loops, with boundary predicates when the tile
// does not divide the problem extent. Passes run through the GLOBAL registry
// (self-registered by lib/passes/*.cpp at static init).

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <string>

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

using kernelh::test::buildCorpusModule;

namespace {

Type f32() { return Type::scalar(ScalarType::F32); }
Type index() { return Type::scalar(ScalarType::Index); }
Type tensorOf(std::vector<int64_t> shape) {
  return Type::tensor(TensorType{std::move(shape), ScalarType::F32,
                                 MemSpace::Global});
}

// A SERIAL GEMM SKELETON - the shape the tile pass recognizes: the kernel
// body's outermost loop is the DOMAIN LOOP (const 0 to const `extent`, step
// const 1) whose induction feeds the loads/stores of a nested N/K loop
// structure (C[i][j] = sum_k A[i][k]*B[k][j]). `warps` selects map_warps.
Module makeSerialGemm(int64_t extent, int64_t tile, bool warps,
                      const std::string &name = "tiled") {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  kernel->schedule.tile = {tile, 64, 32};
  kernel->schedule.map_blocks = true;
  kernel->schedule.map_warps = warps;
  kernel->shape["m"] = extent;
  kernel->shape["n"] = extent;
  kernel->shape["k"] = extent;

  Block &b = kernel->body;
  const Type t = tensorOf({extent, extent});
  b.args.push_back(BlockArg{t});
  b.args.push_back(BlockArg{t});
  b.args.push_back(BlockArg{t});
  const Value a = Value::blockArg(&b, 0);
  const Value bb = Value::blockArg(&b, 1);
  const Value c = Value::blockArg(&b, 2);

  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value cE = builder::makeConst(b, extent, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();

  // Domain (M) loop.
  Op *mloop = builder::makeLoopFor(b, c0, cE, c1, {}, {});
  Block &mb = *mloop->region;
  const Value mi = Value::blockArg(&mb, 0);
  const Value acc0 = builder::makeConst(mb, 0.0, f32())->result();
  // N loop.
  Op *nloop = builder::makeLoopFor(mb, c0, cE, c1, {}, {});
  Block &nb = *nloop->region;
  const Value ni = Value::blockArg(&nb, 0);
  // K loop with a loop-carried accumulator.
  Op *kloop = builder::makeLoopFor(nb, c0, cE, c1, {acc0}, {f32()});
  Block &kb = *kloop->region;
  const Value ki = Value::blockArg(&kb, 0);
  const Value acc = Value::blockArg(&kb, 1);
  const Value la = builder::makeLoad(kb, a, {mi, ki})->result();
  const Value lb = builder::makeLoad(kb, bb, {ki, ni})->result();
  const Value f = builder::makeFma(kb, la, lb, acc)->result();
  builder::makeYield(kb, {f});
  builder::makeStore(nb, kloop->result(), c, {mi, ni});
  builder::makeYield(nb, {});
  builder::makeYield(mb, {});

  module.kernels.push_back(std::move(kernel));
  return module;
}

// The tiled domain loop (first region op of the kernel body).
const Op *findDomainLoop(const Module &module, size_t kernelIndex = 0) {
  for (const auto &op : module.kernels[kernelIndex]->body.ops) {
    if (op->region != nullptr && op->kind == OpKind::LoopFor) {
      return op.get();
    }
  }
  return nullptr;
}

bool isIndexConst(const Value &value, int64_t expected) {
  if (value.kind() != Value::Kind::OpResult || value.op() == nullptr ||
      value.op()->kind != OpKind::Const) {
    return false;
  }
  return value.op()->intValue.has_value() && *value.op()->intValue == expected;
}

std::string runTile(Module &module) {
  const auto result = runPasses(module, "tile");
  REQUIRE(result.has_value());
  return printModule(module);
}

bool contains(const std::string &text, const std::string &needle) {
  return text.find(needle) != std::string::npos;
}

size_t countOf(const std::string &text, const std::string &needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

} // namespace

TEST_CASE("pass_tiling_map_blocks_produces_block_id_offsets", "[unit][pass]") {
  // Divisible extent, no warps: the domain loop becomes a tile loop over
  // [0, 64) with base = block_id * 64; no guards are needed.
  Module module = makeSerialGemm(1024, 64, /*warps=*/false);
  const std::string out = runTile(module);

  CHECK(contains(out, "atom.block_id x : index"));
  CHECK(contains(out, "atom.mul %v")); // base = block_id * 64
  CHECK(contains(out, "atom.const 64 : index"));
  // Divisible extent + no warps -> no boundary predicates.
  CHECK_FALSE(contains(out, "atom.cmp"));
  CHECK_FALSE(contains(out, "atom.select"));

  // The tiled loop's upper bound is the tile extent, not the problem extent.
  const Op *loop = findDomainLoop(module);
  REQUIRE(loop != nullptr);
  CHECK(isIndexConst(loop->operands[0], 0));
  CHECK(isIndexConst(loop->operands[1], 64));
  CHECK(isIndexConst(loop->operands[2], 1));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_map_warps_partitions_by_wavefront", "[unit][pass]") {
  Module module = makeSerialGemm(1024, 64, /*warps=*/true);
  const std::string out = runTile(module);

  // Wavefront index = thread_id / wavefront_size; wavefront_size comes from
  // the device-limits table (gfx1101 -> 32), so the const 32 appears.
  CHECK(contains(out, "atom.thread_id x : index"));
  CHECK(contains(out, "atom.const 32 : index"));
  CHECK(contains(out, "atom.div"));
  CHECK(contains(out, "atom.block_dim x : index"));
  CHECK(countOf(out, "atom.div") == 3); // wf, W = ceil(bd/32), trip = ceil(64/W)

  // W is runtime, so the in-tile row is always guarded (cmp + select).
  CHECK(contains(out, "atom.cmp lt"));
  CHECK(contains(out, "atom.select"));

  // The tiled loop's upper bound is the runtime trip count (a div result).
  const Op *loop = findDomainLoop(module);
  REQUIRE(loop != nullptr);
  CHECK(loop->operands[1].kind() == Value::Kind::OpResult);
  CHECK(loop->operands[1].op()->kind == OpKind::Div);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_nondivisible_produces_boundary_predicates",
          "[unit][pass]") {
  // 100 % 32 != 0: the extent guard must exist and every load/store index
  // derived from the block_id-based global index must be select-guarded, so
  // no access can go out of bounds.
  Module module = makeSerialGemm(100, 32, /*warps=*/false);
  const std::string out = runTile(module);

  CHECK(contains(out, "atom.const 100 : index")); // extent still referenced
  CHECK(contains(out, "atom.cmp lt"));
  CHECK(contains(out, "atom.select"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_warps_and_nondiv_combine_guards", "[unit][pass]") {
  // Both guards apply (tile guard + extent guard) and combine via atom.min
  // (the i32 boolean convention for AND).
  Module module = makeSerialGemm(100, 32, /*warps=*/true);
  const std::string out = runTile(module);
  CHECK(contains(out, "atom.min"));
  CHECK(countOf(out, "atom.cmp") == 2);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_divisible_extent_emits_no_guards", "[unit][pass]") {
  // Without map_warps a divisible extent needs no boundary predicates at
  // all: the tile loop covers exactly [base, base + T) with base + T <= E.
  Module module = makeSerialGemm(1024, 64, /*warps=*/false);
  const std::string out = runTile(module);
  CHECK_FALSE(contains(out, "atom.cmp"));
  CHECK_FALSE(contains(out, "atom.select"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_empty_tile_is_noop", "[unit][pass]") {
  Module module = makeSerialGemm(1024, 64, /*warps=*/false);
  module.kernels[0]->schedule.tile.clear();
  const std::string before = printModule(module);
  const std::string out = runTile(module);
  CHECK(out == before);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_map_blocks_false_is_noop", "[unit][pass]") {
  Module module = makeSerialGemm(1024, 64, /*warps=*/true);
  module.kernels[0]->schedule.map_blocks = false;
  const std::string before = printModule(module);
  const std::string out = runTile(module);
  CHECK(out == before);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_no_domain_loop_is_noop", "[unit][pass]") {
  Module module;
  auto kernel = std::make_unique<Kernel>("noloop");
  kernel->schedule.tile = {64, 64, 32};
  kernel->schedule.map_blocks = true;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensorOf({8, 8})});
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  builder::makeLoad(b, a, {c0, c0}); // no region op anywhere
  module.kernels.push_back(std::move(kernel));

  const std::string before = printModule(module);
  const std::string out = runTile(module);
  CHECK(out == before);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_nonpositive_tile_is_diagnostic_error", "[unit][pass]") {
  Module module = makeSerialGemm(1024, 64, /*warps=*/false);
  module.kernels[0]->schedule.tile = {0, 64, 32};
  const std::string before = printModule(module);
  const auto result = runPasses(module, "tile");
  REQUIRE_FALSE(result.has_value());
  CHECK(contains(result.error(), "non-positive tile extent 0"));
  CHECK(contains(result.error(), "tile sizes must be positive"));
  // The pass aborts before mutating anything.
  CHECK(printModule(module) == before);

  // Negative tile extents too.
  Module negative = makeSerialGemm(1024, 64, /*warps=*/false);
  negative.kernels[0]->schedule.tile = {-64, 64, 32};
  const auto negResult = runPasses(negative, "tile");
  REQUIRE_FALSE(negResult.has_value());
  CHECK(contains(negResult.error(), "non-positive tile extent -64"));
}

TEST_CASE("pass_tiling_iter_arg_results_are_rewritten", "[unit][pass]") {
  // A domain loop WITH iter args whose result is used after the loop: the
  // uses must resolve to the new tiled loop (SSA stays consistent).
  Module module;
  auto kernel = std::make_unique<Kernel>("tiled_acc");
  kernel->schedule.tile = {128};
  kernel->schedule.map_blocks = true;
  kernel->shape["n"] = 1024;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensorOf({1024})});
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value cE = builder::makeConst(b, 1024, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value acc0 = builder::makeConst(b, 0.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, c0, cE, c1, {acc0}, {f32()});
  Block &lb = *loop->region;
  const Value acc = Value::blockArg(&lb, 1);
  const Value one = builder::makeConst(lb, 1.0, f32())->result();
  const Value sum = builder::makeBin(lb, OpKind::Add, acc, one)->result();
  builder::makeYield(lb, {sum});
  builder::makeStore(b, loop->result(), a, {c0}); // uses the loop result
  module.kernels.push_back(std::move(kernel));

  const std::string out = runTile(module);
  CHECK(contains(out, "atom.store"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_mixed_module_tiles_only_eligible_kernels",
          "[unit][pass]") {
  Module module = makeSerialGemm(1024, 64, /*warps=*/false, "tile_me");
  auto other = std::make_unique<Kernel>("leave_me");
  other->schedule.tile = {64, 64, 32};
  other->schedule.map_blocks = false; // ineligible: no-op
  Block &b = other->body;
  b.args.push_back(BlockArg{tensorOf({8, 8})});
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c8 = builder::makeConst(b, 8, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  Op *loop = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  Block &lb = *loop->region;
  const Value ind = Value::blockArg(&lb, 0);
  const Value v = builder::makeLoad(lb, a, {ind, c0})->result();
  builder::makeStore(lb, v, a, {ind, c0});
  builder::makeYield(lb, {});
  module.kernels.push_back(std::move(other));

  const std::string out = runTile(module);
  // Exactly one block_id: the eligible kernel only.
  CHECK(countOf(out, "atom.block_id") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_tiling_corpus_verifies_clean", "[unit][pass]") {
  // The pass must be robust on every corpus kernel shape (grid-form bodies,
  // iter-arg domain loops, shared memory, reductions) and never break
  // verifiability.
  Module module = buildCorpusModule();
  const auto result = runPasses(module, "tile");
  REQUIRE(result.has_value());
  CHECK(verifyModule(module).empty());
  CHECK(module.kernels.size() == 5);
}
