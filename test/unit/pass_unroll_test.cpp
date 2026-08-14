// Unroll pass unit tests (plan todo 18): the pinned full-unroll rule (trip
// count <= 8 AND unroll factor >= trip removes the loop), partial unroll by
// the factor with a cleanup loop for the remainder, the iter_args carry
// chain replicated across copies (same count/order, values forwarded), the
// documented non-actions (factor 1 no-op, non-constant trip counts left
// alone), the invalid-factor diagnostic, and nested loops (innermost first).
// All outputs must verify clean. Passes run through the GLOBAL registry
// (self-registered by lib/passes/*.cpp at static init).

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
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
Type globalTensorOf(int64_t n) {
  return Type::tensor(TensorType{{n}, ScalarType::F32, MemSpace::Global});
}

// A kernel with a sum loop over [0, trip) step 1 and TWO iter args
// (acc0, acc1): the body adds the loaded element to BOTH accumulators and
// yields them, so the carry chain has a definite count and order to assert.
// The two results are stored to out0/out1[0].
struct SumLoop {
  Module module;
  Value acc0;  // initial iter arg 0 (const 0.0)
  Value acc1;  // initial iter arg 1 (const 1.0)
  Value c0;    // lb const 0
  Value cTrip; // ub const trip
  Value out0;  // kernel arg 1
  Value out1;  // kernel arg 2
};

SumLoop makeSumLoopModule(int64_t trip, int64_t unroll) {
  SumLoop built;
  Module &module = built.module;
  auto kernel = std::make_unique<Kernel>("unroll_k");
  kernel->schedule.unroll = unroll;
  kernel->shape["n"] = trip;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensorOf(trip)}); // in
  b.args.push_back(BlockArg{globalTensorOf(trip)}); // out0
  b.args.push_back(BlockArg{globalTensorOf(trip)}); // out1
  const Value in = Value::blockArg(&b, 0);
  built.out0 = Value::blockArg(&b, 1);
  built.out1 = Value::blockArg(&b, 2);
  built.c0 = builder::makeConst(b, 0, index())->result();
  built.cTrip = builder::makeConst(b, trip, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  built.acc0 = builder::makeConst(b, 0.0, f32())->result();
  built.acc1 = builder::makeConst(b, 1.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, built.c0, built.cTrip, c1,
                                  {built.acc0, built.acc1}, {f32(), f32()});
  Block &body = *loop->region;
  const Value ind = Value::blockArg(&body, 0);
  const Value a0 = Value::blockArg(&body, 1);
  const Value a1 = Value::blockArg(&body, 2);
  const Value v = builder::makeLoad(body, in, {ind})->result();
  const Value s0 = builder::makeBin(body, OpKind::Add, v, a0)->result();
  const Value s1 = builder::makeBin(body, OpKind::Add, v, a1)->result();
  builder::makeYield(body, {s0, s1});
  builder::makeStore(b, loop->result(0), built.out0, {built.c0});
  builder::makeStore(b, loop->result(1), built.out1, {built.c0});
  module.kernels.push_back(std::move(kernel));
  return built;
}

// Top-level ops of one kind in a block, in textual order.
std::vector<const Op *> opsInBlock(const Block &block, OpKind kind) {
  std::vector<const Op *> out;
  for (const auto &op : block.ops) {
    if (op->kind == kind) {
      out.push_back(op.get());
    }
  }
  return out;
}

// All ops of one kind across the whole block tree (depth-first, pre-order).
std::vector<const Op *> opsEverywhere(const Block &block, OpKind kind) {
  std::vector<const Op *> out;
  for (const auto &op : block.ops) {
    if (op->kind == kind) {
      out.push_back(op.get());
    }
    if (op->region != nullptr) {
      const std::vector<const Op *> nested = opsEverywhere(*op->region, kind);
      out.insert(out.end(), nested.begin(), nested.end());
    }
  }
  return out;
}

const Op *onlyLoopOf(Block &block) {
  const std::vector<const Op *> loops = opsInBlock(block, OpKind::LoopFor);
  REQUIRE(loops.size() == 1);
  return loops.front();
}

std::string unroll(Module &module) {
  const auto result = runPasses(module, "unroll");
  REQUIRE(result.has_value());
  return printModule(module);
}

bool contains(const std::string &text, const std::string &needle) {
  return text.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("pass_unroll_full_removes_loop_keeps_carry_chain", "[unit][pass]") {
  // trip 4, unroll 4: the pinned rule (trip <= 8 AND unroll >= trip) fully
  // unrolls - the loop disappears and the body appears 4x. The two iter args
  // are forwarded across the copies in order (chain per result index).
  SumLoop built = makeSumLoopModule(4, 4);
  Module &module = built.module;
  const std::string out = unroll(module);
  CHECK_FALSE(contains(out, "atom.loop_for"));
  CHECK_FALSE(contains(out, "atom.yield")); // copies consumed, loop gone

  Block &b = module.kernels.front()->body;
  const std::vector<const Op *> loads = opsInBlock(b, OpKind::Load);
  const std::vector<const Op *> adds = opsInBlock(b, OpKind::Add);
  const std::vector<const Op *> stores = opsInBlock(b, OpKind::Store);
  REQUIRE(loads.size() == 4);
  REQUIRE(adds.size() == 8);
  REQUIRE(stores.size() == 2);

  // Induction constants: copy 0 reuses the original lb (%v2 = const 0);
  // copies 1..3 load at new consts 1, 2, 3.
  CHECK(loads[0]->operands[1] == built.c0);
  for (size_t i = 1; i < loads.size(); ++i) {
    CHECK(loads[i]->operands[1].op()->intValue.has_value());
    CHECK(*loads[i]->operands[1].op()->intValue ==
          static_cast<int64_t>(i));
  }

  // Carry chain, per result index: the adds alternate chains
  // (add[0] = acc0 copy0, add[1] = acc1 copy0, add[2] = acc0 copy1, ...).
  // Each copy's chain element consumes the previous copy's yield value.
  CHECK(adds[0]->operands[1] == built.acc0);
  CHECK(adds[1]->operands[1] == built.acc1);
  for (size_t i = 1; i < 4; ++i) {
    CHECK(adds[2 * i]->operands[1] == adds[2 * (i - 1)]->result());
    CHECK(adds[2 * i + 1]->operands[1] == adds[2 * (i - 1) + 1]->result());
  }
  // The stores consume the LAST copy's yield values (same 2 results, same
  // order as the loop's results).
  CHECK(stores[0]->operands[0] == adds[6]->result());
  CHECK(stores[1]->operands[0] == adds[7]->result());
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_unroll_partial_cleanup_loop", "[unit][pass]") {
  // trip 10, unroll 4: main loop [0, 8] with a 4-copy body (2 trips = 4+4)
  // plus a cleanup loop [8, 10] running the ORIGINAL body for the leftover
  // 2 iterations. The cleanup loop's iter args ARE the main loop's results.
  SumLoop built = makeSumLoopModule(10, 4);
  Module &module = built.module;
  const std::string out = unroll(module);
  CHECK(contains(out, "atom.loop_for"));
  CHECK(contains(out, "atom.yield"));

  Block &b = module.kernels.front()->body;
  const std::vector<const Op *> loops = opsInBlock(b, OpKind::LoopFor);
  REQUIRE(loops.size() == 2);
  const Op &main = *loops[0];
  const Op &cleanup = *loops[1];

  // Main loop: [0, const 8] with the original iter args.
  CHECK(main.operands[0] == built.c0);
  CHECK(main.operands[1].op()->intValue.has_value());
  CHECK(*main.operands[1].op()->intValue == 8);
  CHECK(main.operands[2].op()->intValue.has_value());
  CHECK(*main.operands[2].op()->intValue == 1);
  CHECK(main.operands[3] == built.acc0);
  CHECK(main.operands[4] == built.acc1);
  REQUIRE(main.region != nullptr);
  Block &mainBody = *main.region;

  // Main body: 4 copies (4 loads, 8 f32 adds) ending with one yield whose
  // operands are the last copy's values; copy i loads at ind + i.
  const std::vector<const Op *> loads = opsInBlock(mainBody, OpKind::Load);
  const std::vector<const Op *> adds = opsInBlock(mainBody, OpKind::Add);
  REQUIRE(loads.size() == 4);
  REQUIRE(adds.size() == 11); // 8 f32 sums + 3 index induction offsets
  const Value mainInd = Value::blockArg(&mainBody, 0);
  CHECK(loads[0]->operands[1] == mainInd);
  for (size_t i = 1; i < loads.size(); ++i) {
    const Op *offset = loads[i]->operands[1].op();
    REQUIRE(offset != nullptr);
    CHECK(offset->kind == OpKind::Add); // ind + const(i)
    CHECK(offset->resultTypes[0].isScalar());
    CHECK(offset->resultTypes[0].asScalar() == ScalarType::Index);
  }
  // f32 adds only: chain per result index as in the full-unroll test.
  std::vector<const Op *> f32adds;
  for (const Op *add : adds) {
    if (add->resultTypes[0].isScalar() &&
        add->resultTypes[0].asScalar() == ScalarType::F32) {
      f32adds.push_back(add);
    }
  }
  REQUIRE(f32adds.size() == 8);
  CHECK(f32adds[0]->operands[1] == built.acc0);
  CHECK(f32adds[1]->operands[1] == built.acc1);
  for (size_t i = 1; i < 4; ++i) {
    CHECK(f32adds[2 * i]->operands[1] == f32adds[2 * (i - 1)]->result());
    CHECK(f32adds[2 * i + 1]->operands[1] ==
          f32adds[2 * (i - 1) + 1]->result());
  }
  const Op *mainYield = mainBody.ops.back().get();
  REQUIRE(mainYield->kind == OpKind::Yield);
  CHECK(mainYield->operands[0] == f32adds[6]->result());
  CHECK(mainYield->operands[1] == f32adds[7]->result());

  // Cleanup loop: [const 8, 10], iter args = the main loop's results (the
  // carry chain hops from the main loop into the cleanup loop).
  CHECK(cleanup.operands[0] == main.operands[1]);
  CHECK(cleanup.operands[1] == built.cTrip);
  CHECK(cleanup.operands[3] == main.result(0));
  CHECK(cleanup.operands[4] == main.result(1));
  REQUIRE(cleanup.region != nullptr);
  Block &cleanupBody = *cleanup.region;
  // Original body replicated once: 1 load, and the carry chain restarts
  // from the cleanup loop's OWN iter args (identity clone of the incoming
  // values), with the yield carrying the add results.
  CHECK(opsInBlock(cleanupBody, OpKind::Load).size() == 1);
  const Op *cleanupYield = cleanupBody.ops.back().get();
  REQUIRE(cleanupYield->kind == OpKind::Yield);
  const std::vector<const Op *> cadds = opsInBlock(cleanupBody, OpKind::Add);
  REQUIRE(cadds.size() == 2);
  CHECK(cadds[0]->operands[1] == Value::blockArg(&cleanupBody, 1));
  CHECK(cadds[1]->operands[1] == Value::blockArg(&cleanupBody, 2));
  CHECK(cleanupYield->operands[0] == cadds[0]->result());
  CHECK(cleanupYield->operands[1] == cadds[1]->result());

  // The stores consume the cleanup loop's results.
  const std::vector<const Op *> stores = opsInBlock(b, OpKind::Store);
  REQUIRE(stores.size() == 2);
  CHECK(stores[0]->operands[0] == cleanup.result(0));
  CHECK(stores[1]->operands[0] == cleanup.result(1));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_unroll_partial_divisible_no_cleanup", "[unit][pass]") {
  // trip 8, unroll 4: 8 % 4 == 0, so no cleanup loop is needed - one loop
  // with a 4-copy body keeps its original bounds.
  SumLoop built = makeSumLoopModule(8, 4);
  Module &module = built.module;
  unroll(module);
  Block &b = module.kernels.front()->body;
  const Op &loop = *onlyLoopOf(b);
  CHECK(loop.operands[0] == built.c0);
  CHECK(loop.operands[1] == built.cTrip);
  CHECK(loop.operands[3] == built.acc0);
  CHECK(loop.operands[4] == built.acc1);
  CHECK(opsInBlock(*loop.region, OpKind::Load).size() == 4);
  const Op *yield = loop.region->ops.back().get();
  REQUIRE(yield->kind == OpKind::Yield);
  // The f32 adds (excluding the index induction-offset adds) carry the two
  // chains; the yield threads the last copy's values as the loop results.
  std::vector<const Op *> f32adds;
  for (const Op *add : opsInBlock(*loop.region, OpKind::Add)) {
    if (add->resultTypes[0].isScalar() &&
        add->resultTypes[0].asScalar() == ScalarType::F32) {
      f32adds.push_back(add);
    }
  }
  REQUIRE(f32adds.size() == 8);
  CHECK(yield->operands.size() == 2);
  CHECK(yield->operands[0].op() == f32adds[6]);
  CHECK(yield->operands[1].op() == f32adds[7]);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_unroll_trip_le_8_but_factor_smaller_is_partial",
          "[unit][pass]") {
  // trip 8, unroll 2: the trip count is <= 8 but the factor is NOT >= the
  // trip count, so the pinned rule falls through to partial unrolling
  // (factor 2, no remainder): one loop with a 2-copy body.
  SumLoop built = makeSumLoopModule(8, 2);
  Module &module = built.module;
  unroll(module);
  Block &b = module.kernels.front()->body;
  const Op &loop = *onlyLoopOf(b);
  CHECK(loop.operands[1] == built.cTrip);
  CHECK(opsInBlock(*loop.region, OpKind::Load).size() == 2);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_unroll_nonconstant_trip_left_alone", "[unit][pass]") {
  // A loop whose lb is a runtime value (atom.block_id) has no constant trip
  // count: the pass must leave it alone (no dynamic unrolling) and the
  // kernel must remain verifier-clean.
  Module module;
  auto kernel = std::make_unique<Kernel>("unroll_dyn");
  kernel->schedule.unroll = 4;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensorOf(4)});
  b.args.push_back(BlockArg{globalTensorOf(4)});
  const Value in = Value::blockArg(&b, 0);
  const Value out = Value::blockArg(&b, 1);
  const Value lb = builder::makeId(b, OpKind::BlockId, kernelh::ir::IdDim::X)
                       ->result();
  const Value ub = builder::makeConst(b, 4, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value acc0 = builder::makeConst(b, 0.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, lb, ub, c1, {acc0}, {f32()});
  Block &body = *loop->region;
  const Value ind = Value::blockArg(&body, 0);
  const Value v = builder::makeLoad(body, in, {ind})->result();
  builder::makeYield(body, {builder::makeBin(body, OpKind::Add, v,
                                             Value::blockArg(&body, 1))
                                ->result()});
  builder::makeStore(b, loop->result(0), out, {c0});
  module.kernels.push_back(std::move(kernel));

  const std::string before = printModule(module);
  const auto result = runPasses(module, "unroll");
  REQUIRE(result.has_value());
  CHECK(printModule(module) == before);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_unroll_factor1_noop", "[unit][pass]") {
  SumLoop built = makeSumLoopModule(4, 1);
  Module &module = built.module;
  const std::string before = printModule(module);
  const auto result = runPasses(module, "unroll");
  REQUIRE(result.has_value());
  CHECK(printModule(module) == before);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_unroll_invalid_factor_diagnostic", "[unit][pass]") {
  for (const int64_t factor : {0, -2}) {
    SumLoop built = makeSumLoopModule(4, factor);
    Module &module = built.module;
    const std::string before = printModule(module);
    const auto result = runPasses(module, "unroll");
    REQUIRE_FALSE(result.has_value());
    CHECK(contains(result.error(), "unroll factor"));
    CHECK(printModule(module) == before);
  }
}

TEST_CASE("pass_unroll_nested_loops_innermost_first", "[unit][pass]") {
  // Outer loop [0, 8] step 1, inner loop [0, 4] step 1 with one f32 iter
  // arg, kernel unroll = 4. The inner loop is unrolled FIRST (bottom-up):
  // trip 4 <= 8 and 4 >= 4 -> FULLY unrolled into 4 copies inside the outer
  // body. The outer loop (trip 8, 4 >= 8 false) is then PARTIALLY unrolled:
  // one loop with a 4-copy body, each copy carrying the inner-unrolled
  // form. No inner loop_for survives.
  Module module;
  auto kernel = std::make_unique<Kernel>("unroll_nested");
  kernel->schedule.unroll = 4;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensorOf(8)});
  b.args.push_back(BlockArg{globalTensorOf(8)});
  const Value in = Value::blockArg(&b, 0);
  const Value out = Value::blockArg(&b, 1);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c8 = builder::makeConst(b, 8, index())->result();
  const Value c4 = builder::makeConst(b, 4, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value acc0 = builder::makeConst(b, 0.0, f32())->result();
  Op *outer = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  Block &ob = *outer->region;
  const Value oi = Value::blockArg(&ob, 0);
  Op *inner = builder::makeLoopFor(ob, c0, c4, c1, {acc0}, {f32()});
  Block &ib = *inner->region;
  const Value ii = Value::blockArg(&ib, 0);
  const Value v = builder::makeLoad(ib, in, {ii})->result();
  builder::makeYield(ib, {builder::makeBin(ib, OpKind::Add, v,
                                           Value::blockArg(&ib, 1))
                              ->result()});
  builder::makeStore(ob, inner->result(0), out, {oi});
  builder::makeYield(ob, {});
  module.kernels.push_back(std::move(kernel));

  unroll(module);
  const Block &kb = module.kernels.front()->body;
  const std::vector<const Op *> allLoops = opsEverywhere(kb, OpKind::LoopFor);
  REQUIRE(allLoops.size() == 1); // only the outer main loop remains
  const Op &main = *allLoops.front();
  // 4 outer copies, each containing the inner-unrolled 4 loads.
  CHECK(opsInBlock(*main.region, OpKind::Load).size() == 16);
  CHECK(opsEverywhere(kb, OpKind::Store).size() == 4);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_unroll_corpus_stays_clean", "[unit][pass]") {
  // The full corpus (mixed kernels incl. nested loops with iter args) has
  // unroll = 1 by default: the pass is a no-op and everything verifies.
  Module module = buildCorpusModule();
  const auto result = runPasses(module, "unroll");
  REQUIRE(result.has_value());
  CHECK(verifyModule(module).empty());
}
