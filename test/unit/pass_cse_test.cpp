// CSE pass unit tests (plan todo 13): value-numbered common-subexpression
// elimination within block scopes. Passes run through the GLOBAL registry
// (self-registered by lib/passes/*.cpp at static init).

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::CmpPred;
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
Type tensorArg() {
  return Type::tensor(TensorType{{8}, ScalarType::F32, MemSpace::Global});
}

// A kernel whose body takes one global tensor argument (tensor<8xf32>).
Module makeTensorKernel(const std::string &name) {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  kernel->body.args.push_back(BlockArg{tensorArg()});
  module.kernels.push_back(std::move(kernel));
  return module;
}

std::string cse(Module &module) {
  const auto result = runPasses(module, "cse");
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

TEST_CASE("pass_cse_same_block_merges", "[unit][pass]") {
  Module module = makeTensorKernel("cse1");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c1 = builder::makeConst(b, 1.0, f32())->result();
  const Value x = builder::makeLoad(b, a, {c0})->result();
  const Value y1 = builder::makeBin(b, OpKind::Add, x, c1)->result();
  builder::makeBin(b, OpKind::Add, x, c1); // duplicate -> reuses y1
  builder::makeStore(b, y1, a, {c0});

  const std::string out = cse(module);
  CHECK(count(out, "atom.add") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_does_not_merge_loads", "[unit][pass]") {
  Module module = makeTensorKernel("cse_loads");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value x1 = builder::makeLoad(b, a, {c0})->result();
  const Value x2 = builder::makeLoad(b, a, {c0})->result(); // identical load
  builder::makeStore(b, x1, a, {c0});
  builder::makeStore(b, x2, a, {c0});

  const std::string out = cse(module);
  // Memory reads are excluded: an intervening store/barrier can change the
  // value, so both loads must survive.
  CHECK(count(out, "atom.load") == 2);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_does_not_merge_alloc_shared", "[unit][pass]") {
  Module module = makeTensorKernel("cse_alloc");
  Block &b = module.kernels[0]->body;
  builder::makeAllocShared(b, TensorType{{4}, ScalarType::F32, MemSpace::Shared});
  builder::makeAllocShared(b, TensorType{{4}, ScalarType::F32, MemSpace::Shared});

  const std::string out = cse(module);
  CHECK(count(out, "atom.alloc_shared") == 2);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_does_not_merge_barrier", "[unit][pass]") {
  Module module = makeTensorKernel("cse_barrier");
  Block &b = module.kernels[0]->body;
  builder::makeBarrier(b);
  builder::makeBarrier(b);

  const std::string out = cse(module);
  CHECK(count(out, "atom.barrier") == 2);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_distinct_kinds_not_merged", "[unit][pass]") {
  Module module = makeTensorKernel("cse_kinds");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c1 = builder::makeConst(b, 1.0, f32())->result();
  const Value x = builder::makeLoad(b, a, {c0})->result();
  const Value s1 = builder::makeBin(b, OpKind::Sub, x, c1)->result();
  builder::makeBin(b, OpKind::Add, x, c1); // add is NOT the same op as sub
  builder::makeStore(b, s1, a, {c0});

  const std::string out = cse(module);
  CHECK(count(out, "atom.sub") == 1);
  CHECK(count(out, "atom.add") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_merges_identical_consts", "[unit][pass]") {
  Module module = makeTensorKernel("cse_const");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c5a = builder::makeConst(b, 5.0, f32())->result();
  const Value c5b = builder::makeConst(b, 5.0, f32())->result();
  builder::makeStore(b, c5a, a, {c0});
  builder::makeStore(b, c5b, a, {c0}); // c5b's use rewritten to c5a

  const std::string out = cse(module);
  CHECK(count(out, "atom.const 5 : f32") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_merges_id_ops", "[unit][pass]") {
  Module module = makeTensorKernel("cse_id");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value t1 = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value t2 = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value x1 = builder::makeLoad(b, a, {t1})->result();
  const Value x2 = builder::makeLoad(b, a, {t2})->result();
  builder::makeStore(b, x1, a, {c0});
  builder::makeStore(b, x2, a, {c0});

  const std::string out = cse(module);
  CHECK(count(out, "atom.thread_id") == 1);
  CHECK(count(out, "atom.load") == 2); // loads still not merged
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_nested_scope_reuses_outer_value", "[unit][pass]") {
  Module module = makeTensorKernel("cse_nested");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c8 = builder::makeConst(b, 8, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value x = builder::makeLoad(b, a, {tid})->result();
  const Value y = builder::makeBin(b, OpKind::Mul, x, x)->result();
  builder::makeBin(b, OpKind::Mul, x, x); // dup in the parent scope

  // The loop body re-computes mul(x, x) with the OUTER `x`: the nested table
  // (a copy of the parent's) must map it to the outer `y`.
  Op *loop = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  const Value w = builder::makeBin(*loop->region, OpKind::Mul, x, x)->result();
  builder::makeStore(*loop->region, w, a, {tid});
  builder::makeYield(*loop->region, {});

  builder::makeStore(b, y, a, {tid});

  const std::string out = cse(module);
  // The parent dup AND the loop-body dup both collapse to the first mul.
  CHECK(count(out, "atom.mul") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_cse_shadow_avoiding_numbering_roundtrip", "[unit][pass]") {
  Module module = makeTensorKernel("cse_shadow");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c8 = builder::makeConst(b, 8, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();

  Op *loop = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  // The body re-defines const 0; under shadow-avoiding numbering the body's
  // %vN differs from the parent's. CSE must merge it into the outer const and
  // the result must still round-trip byte-identically.
  const Value bodyC0 = builder::makeConst(*loop->region, 0, index())->result();
  const Value bodyX = builder::makeLoad(*loop->region, a, {bodyC0})->result();
  builder::makeStore(*loop->region, bodyX, a, {bodyC0});
  builder::makeYield(*loop->region, {});

  const std::string out = cse(module);
  CHECK(count(out, "atom.const 0 : index") == 1);
  CHECK(verifyModule(module).empty());
  // print -> parse -> print must be byte-identical after CSE.
  const auto reparsed = kernelh::ir::parseModule(out);
  REQUIRE(reparsed.has_value());
  CHECK(printModule(*reparsed) == out);
  CHECK(verifyModule(*reparsed).empty());
}

TEST_CASE("pass_cse_value_defined_after_loop_not_reachable", "[unit][pass]") {
  Module module = makeTensorKernel("cse_no_leak");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c8 = builder::makeConst(b, 8, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();

  Op *loop = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  const Value bodyX = builder::makeLoad(*loop->region, a, {tid})->result();
  builder::makeStore(*loop->region, bodyX, a, {tid});
  builder::makeYield(*loop->region, {});

  // Defined AFTER the loop: must NOT leak into the loop body table. The body
  // has no duplicate of it, so nothing to merge - the module stays valid.
  const Value x = builder::makeLoad(b, a, {tid})->result();
  builder::makeStore(b, x, a, {tid});

  const std::string out = cse(module);
  CHECK(count(out, "atom.load") == 2);
  CHECK(verifyModule(module).empty());
}
