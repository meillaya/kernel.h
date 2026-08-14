// Vectorize pass unit tests (plan todo 17): width-N atom.load_vec/atom.store_vec
// rewriting plus pointwise-compute widening, the pinned tile-extent
// divisibility diagnostic, the vector_width=1 no-op, and the conservative
// non-widening of chains this IR cannot widen (scalar consts / atom.math) -
// all with verifier-clean output. Passes run through the GLOBAL registry
// (self-registered by lib/passes/*.cpp at static init).

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::CmpPred;
using kernelh::ir::Kernel;
using kernelh::ir::MemSpace;
using kernelh::ir::Module;
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
Type globalTensor8() {
  return Type::tensor(TensorType{{8}, ScalarType::F32, MemSpace::Global});
}

// A kernel with three tensor<8xf32, global> args and the contiguous chain
//   x = load a[0]; y = load b[0]; m = x * y; s = m + x; store s -> c[0]
// plus the given schedule (tile extents + vector width).
Module makeVecKernel(std::vector<int64_t> tile, int64_t width) {
  Module module;
  auto kernel = std::make_unique<Kernel>("vec");
  kernel->schedule.tile = std::move(tile);
  kernel->schedule.vector_width = width;
  kernel->shape["n"] = 8;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensor8()});
  b.args.push_back(BlockArg{globalTensor8()});
  b.args.push_back(BlockArg{globalTensor8()});
  const Value a = Value::blockArg(&b, 0);
  const Value c = Value::blockArg(&b, 2);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value x = builder::makeLoad(b, a, {c0})->result();
  const Value y = builder::makeLoad(b, Value::blockArg(&b, 1), {c0})->result();
  const Value m = builder::makeBin(b, OpKind::Mul, x, y)->result();
  const Value s = builder::makeBin(b, OpKind::Add, m, x)->result();
  builder::makeStore(b, s, c, {c0});
  module.kernels.push_back(std::move(kernel));
  return module;
}

std::string vectorize(Module &module) {
  const auto result = runPasses(module, "vectorize");
  REQUIRE(result.has_value());
  return printModule(module);
}

bool contains(const std::string &text, const std::string &needle) {
  return text.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("pass_vectorize_width4_emits_vec_ops_and_widens_compute",
          "[unit][pass]") {
  Module module = makeVecKernel({8}, 4);
  const std::string before = printModule(module);
  const std::string out = vectorize(module);
  CHECK(out != before);
  // All three loads become width-4 vector loads, the store a vector store,
  // and the pointwise mul/add widen to vector<4xf32>.
  CHECK(contains(out,
                 "%v4 = atom.load_vec %v0[%v3] : tensor<8xf32, global>, "
                 "vector<4xf32>"));
  CHECK(contains(out,
                 "%v5 = atom.load_vec %v1[%v3] : tensor<8xf32, global>, "
                 "vector<4xf32>"));
  CHECK(contains(out, "%v6 = atom.mul %v4, %v5 : vector<4xf32>"));
  CHECK(contains(out, "%v7 = atom.add %v6, %v4 : vector<4xf32>"));
  CHECK(contains(out,
                 "atom.store_vec %v7, %v2[%v3] : tensor<8xf32, global>, "
                 "vector<4xf32>"));
  // No scalar loads/stores remain (the trailing-space boundary keeps the
  // load_vec/store_vec lines from matching).
  CHECK_FALSE(contains(out, "atom.load "));
  CHECK_FALSE(contains(out, "atom.store "));
  // The vectorized kernel must stay verifier-clean.
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_vectorize_nondiv_extent_diagnostic", "[unit][pass]") {
  Module module = makeVecKernel({6}, 4);
  const std::string before = printModule(module);
  const auto result = runPasses(module, "vectorize");
  // std::expected error - the diagnostic names the offending extent - and no
  // crash, no partial rewrite.
  REQUIRE_FALSE(result.has_value());
  CHECK(contains(result.error(), "6"));
  CHECK(contains(result.error(), "divisib"));
  CHECK(printModule(module) == before);
}

TEST_CASE("pass_vectorize_width1_is_noop", "[unit][pass]") {
  Module module = makeVecKernel({8}, 1);
  const std::string before = printModule(module);
  const auto result = runPasses(module, "vectorize");
  REQUIRE(result.has_value());
  CHECK(printModule(module) == before);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_vectorize_invalid_width_diagnostic", "[unit][pass]") {
  Module module = makeVecKernel({8}, 0);
  const auto result = runPasses(module, "vectorize");
  REQUIRE_FALSE(result.has_value());
  CHECK(contains(result.error(), "vector width"));
}

TEST_CASE("pass_vectorize_shared_tensor_loads", "[unit][pass]") {
  // Loads from shared tensors vectorize too (the pass covers global AND
  // shared); the store back to global becomes store_vec.
  Module module;
  auto kernel = std::make_unique<Kernel>("vec_shared");
  kernel->schedule.tile = {8};
  kernel->schedule.vector_width = 4;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensor8()});
  const Value out = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value sh = builder::makeAllocShared(
                       b, TensorType{{8}, ScalarType::F32, MemSpace::Shared})
                       ->result();
  const Value x = builder::makeLoad(b, sh, {c0})->result();
  builder::makeStore(b, x, out, {c0});
  module.kernels.push_back(std::move(kernel));

  const std::string outText = vectorize(module);
  CHECK(contains(outText,
                 "%v3 = atom.load_vec %v2[%v1] : tensor<8xf32, shared>, "
                 "vector<4xf32>"));
  CHECK(contains(outText,
                 "atom.store_vec %v3, %v0[%v1] : tensor<8xf32, global>, "
                 "vector<4xf32>"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_vectorize_select_arms_widen", "[unit][pass]") {
  // select with two widened loads as arms: the condition stays an i32 scalar,
  // both arms and the result widen to vector<4xf32>.
  Module module;
  auto kernel = std::make_unique<Kernel>("vec_select");
  kernel->schedule.tile = {8};
  kernel->schedule.vector_width = 4;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensor8()});
  b.args.push_back(BlockArg{globalTensor8()});
  b.args.push_back(BlockArg{globalTensor8()});
  const Value a = Value::blockArg(&b, 0);
  const Value c = Value::blockArg(&b, 2);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value lt = builder::makeCmp(b, CmpPred::Lt, c0, c1)->result();
  const Value x = builder::makeLoad(b, a, {c0})->result();
  const Value y = builder::makeLoad(b, Value::blockArg(&b, 1), {c0})->result();
  const Value sel = builder::makeSelect(b, lt, x, y)->result();
  builder::makeStore(b, sel, c, {c0});
  module.kernels.push_back(std::move(kernel));

  const std::string outText = vectorize(module);
  CHECK(contains(outText, "%v8 = atom.select %v5, %v6, %v7 : vector<4xf32>"));
  CHECK(contains(outText, "atom.store_vec %v8, %v2[%v3]"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_vectorize_scalar_const_blocks_widening", "[unit][pass]") {
  // add(load, const): there is no vector splat const in this IR, so the
  // whole chain must stay scalar - the pass must NOT emit a mixed-typed op.
  Module module;
  auto kernel = std::make_unique<Kernel>("vec_cst");
  kernel->schedule.tile = {8};
  kernel->schedule.vector_width = 4;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensor8()});
  b.args.push_back(BlockArg{globalTensor8()});
  const Value a = Value::blockArg(&b, 0);
  const Value out = Value::blockArg(&b, 1);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1.0, f32())->result();
  const Value x = builder::makeLoad(b, a, {c0})->result();
  const Value s = builder::makeBin(b, OpKind::Add, x, one)->result();
  builder::makeStore(b, s, out, {c0});
  module.kernels.push_back(std::move(kernel));

  const std::string outText = vectorize(module);
  CHECK(contains(outText, "atom.load "));
  CHECK_FALSE(contains(outText, "vector<4xf32>"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_vectorize_math_blocks_widening", "[unit][pass]") {
  // atom.math has no vector form in this IR (and must never be created); the
  // load feeding it stays scalar and the kernel remains verifier-clean.
  Module module;
  auto kernel = std::make_unique<Kernel>("vec_math");
  kernel->schedule.tile = {8};
  kernel->schedule.vector_width = 4;
  Block &b = kernel->body;
  b.args.push_back(BlockArg{globalTensor8()});
  b.args.push_back(BlockArg{globalTensor8()});
  const Value a = Value::blockArg(&b, 0);
  const Value out = Value::blockArg(&b, 1);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value x = builder::makeLoad(b, a, {c0})->result();
  const Value sq = builder::makeMath(b, kernelh::ir::MathKind::Sqrt, x)->result();
  builder::makeStore(b, sq, out, {c0});
  module.kernels.push_back(std::move(kernel));

  const std::string outText = vectorize(module);
  CHECK(contains(outText, "atom.load "));
  CHECK_FALSE(contains(outText, "vector<4xf32>"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_vectorize_corpus_stays_clean", "[unit][pass]") {
  // The full corpus (mixed kernels incl. pre-existing load_vec/store_vec,
  // loop-carried values, math chains): the pass succeeds, only the legal
  // global->shared chains widen, and everything verifies clean.
  Module module = buildCorpusModule();
  const auto result = runPasses(module, "vectorize");
  REQUIRE(result.has_value());
  CHECK(verifyModule(module).empty());
  // gemm (tile {64,64,32}, width 4) and scan (tile {256}, width 4) both
  // widen their load-global -> store-shared staging chains.
  const std::string out = printModule(module);
  CHECK(contains(out, "atom.load_vec"));
  CHECK(contains(out, "atom.store_vec"));
}
