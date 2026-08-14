// Canonicalize pass unit tests (plan todo 13): constant folding, identity
// elision, and dead-op elimination. Passes run through the GLOBAL registry
// (self-registered by lib/passes/*.cpp at static init).

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
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

using kernelh::test::buildCorpusModule;

namespace {

Type f32() { return Type::scalar(ScalarType::F32); }
Type i32() { return Type::scalar(ScalarType::I32); }
Type i64Type() { return Type::scalar(ScalarType::I64); }
Type index() { return Type::scalar(ScalarType::Index); }
Type tensorOf(ScalarType elem) {
  return Type::tensor(TensorType{{8}, elem, MemSpace::Global});
}

// A kernel whose body takes one scalar argument of `argType`.
Module makeScalarKernel(const std::string &name, Type argType) {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  kernel->body.args.push_back(BlockArg{argType});
  module.kernels.push_back(std::move(kernel));
  return module;
}

// A kernel whose body takes one global tensor argument (tensor<8x<elem>>).
Module makeTensorKernel(const std::string &name,
                        ScalarType elem = ScalarType::F32) {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  kernel->body.args.push_back(BlockArg{tensorOf(elem)});
  module.kernels.push_back(std::move(kernel));
  return module;
}

// A kernel whose body takes a global tensor plus an index scalar argument
// (index values cannot be stored, so index-typed results must be consumed as
// load/store indices).
Module makeTensorIndexKernel(const std::string &name) {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  kernel->body.args.push_back(BlockArg{tensorOf(ScalarType::F32)});
  kernel->body.args.push_back(BlockArg{index()});
  module.kernels.push_back(std::move(kernel));
  return module;
}

// Stores `value` into the kernel's first tensor argument at index 0, keeping
// the value (and its producers) alive across DCE.
void keepAlive(Block &b, const Value &value) {
  const Value c0 = builder::makeConst(b, 0, index())->result();
  builder::makeStore(b, value, Value::blockArg(&b, 0), {c0});
}

std::string canonicalize(Module &module) {
  const auto result = runPasses(module, "canonicalize");
  REQUIRE(result.has_value());
  return printModule(module);
}

bool contains(const std::string &text, const std::string &needle) {
  return text.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("pass_canonicalize_folds_int_add_chain", "[unit][pass]") {
  Module module = makeTensorKernel("fold_int", ScalarType::I32);
  Block &b = module.kernels[0]->body;
  const Value c2 = builder::makeConst(b, 2, i32())->result();
  const Value c3 = builder::makeConst(b, 3, i32())->result();
  const Value sum = builder::makeBin(b, OpKind::Add, c2, c3)->result();
  const Value c10 = builder::makeConst(b, 10, i32())->result();
  const Value prod = builder::makeBin(b, OpKind::Mul, sum, c10)->result();
  keepAlive(b, sum);
  keepAlive(b, prod);

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.const 5 : i32"));
  CHECK(contains(out, "atom.const 50 : i32"));
  CHECK_FALSE(contains(out, "atom.add"));
  CHECK_FALSE(contains(out, "atom.mul"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_folds_float_mul", "[unit][pass]") {
  Module module = makeTensorKernel("fold_float", ScalarType::F32);
  Block &b = module.kernels[0]->body;
  const Value c = builder::makeConst(b, 1.5, f32())->result();
  const Value d = builder::makeConst(b, 2.5, f32())->result();
  const Value prod = builder::makeBin(b, OpKind::Mul, c, d)->result();
  keepAlive(b, prod);

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.const 3.75 : f32"));
  CHECK_FALSE(contains(out, "atom.mul"));
}

TEST_CASE("pass_canonicalize_int_div_zero_not_folded", "[unit][pass]") {
  Module module = makeTensorKernel("div0", ScalarType::I32);
  Block &b = module.kernels[0]->body;
  const Value c7 = builder::makeConst(b, 7, i32())->result();
  const Value c0 = builder::makeConst(b, 0, i32())->result();
  const Value q = builder::makeBin(b, OpKind::Div, c7, c0)->result();
  keepAlive(b, q);

  const std::string out = canonicalize(module);
  // The div must survive as the op (never folded away by an eager 7/0).
  CHECK(contains(out, "atom.div"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_int64_overflow_not_folded", "[unit][pass]") {
  Module module = makeTensorKernel("ovf_i64", ScalarType::I64);
  Block &b = module.kernels[0]->body;
  const Value max = builder::makeConst(b, INT64_MAX, i64Type())->result();
  const Value c2 = builder::makeConst(b, 2, i64Type())->result();
  const Value prod = builder::makeBin(b, OpKind::Mul, max, c2)->result();
  keepAlive(b, prod);

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.mul"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_i32_result_must_fit_int32", "[unit][pass]") {
  Module module = makeTensorKernel("ovf_i32", ScalarType::I32);
  Block &b = module.kernels[0]->body;
  const Value max = builder::makeConst(b, 2147483647LL, i32())->result();
  const Value c1 = builder::makeConst(b, 1, i32())->result();
  const Value sum = builder::makeBin(b, OpKind::Add, max, c1)->result();
  keepAlive(b, sum);

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.add"));
  CHECK_FALSE(contains(out, "atom.const 2147483648"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_cmp_folds", "[unit][pass]") {
  Module module = makeTensorKernel("fold_cmp", ScalarType::I32);
  Block &b = module.kernels[0]->body;
  const Value c2 = builder::makeConst(b, 2, i32())->result();
  const Value c3 = builder::makeConst(b, 3, i32())->result();
  const Value lt = builder::makeCmp(b, CmpPred::Lt, c2, c3)->result();
  keepAlive(b, lt);

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.const 1 : i32"));
  CHECK_FALSE(contains(out, "atom.cmp"));
}

TEST_CASE("pass_canonicalize_identity_elision", "[unit][pass]") {
  Module module = makeScalarKernel("identity", f32());
  Block &b = module.kernels[0]->body;
  const Value x = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0.0, f32())->result();
  const Value c1 = builder::makeConst(b, 1.0, f32())->result();
  builder::makeBin(b, OpKind::Add, x, c0); // x + 0 -> x
  builder::makeBin(b, OpKind::Mul, x, c1); // x * 1 -> x
  builder::makeBin(b, OpKind::Sub, x, c0); // x - 0 -> x
  builder::makeBin(b, OpKind::Div, x, c1); // x / 1 -> x

  const std::string out = canonicalize(module);
  CHECK_FALSE(contains(out, "atom.add"));
  CHECK_FALSE(contains(out, "atom.mul"));
  CHECK_FALSE(contains(out, "atom.sub"));
  CHECK_FALSE(contains(out, "atom.div"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_int_mul_zero_folds_float_survives",
          "[unit][pass]") {
  Module module = makeTensorKernel("mul0");
  Block &b = module.kernels[0]->body;
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value x = builder::makeLoad(b, Value::blockArg(&b, 0), {c0})->result();
  const Value zf = builder::makeConst(b, 0.0, f32())->result();
  const Value m1 = builder::makeBin(b, OpKind::Mul, x, zf)->result();
  const Value m2 = builder::makeBin(b, OpKind::Mul, zf, x)->result();
  keepAlive(b, m1);
  keepAlive(b, m2);

  const std::string out = canonicalize(module);
  // Both float x*0 forms must SURVIVE (IEEE 0*x is NaN for inf/NaN x).
  CHECK(contains(out, "atom.mul"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_int_mul_zero_folds", "[unit][pass]") {
  Module module = makeTensorIndexKernel("mul0i");
  Block &b = module.kernels[0]->body;
  const Value x = Value::blockArg(&b, 1); // index
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value m = builder::makeBin(b, OpKind::Mul, x, c0)->result(); // const 0
  const Value v = builder::makeLoad(b, Value::blockArg(&b, 0), {c0})->result();
  builder::makeStore(b, v, Value::blockArg(&b, 0), {m});

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.const 0 : index"));
  CHECK_FALSE(contains(out, "atom.mul"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_elision_rewrites_uses", "[unit][pass]") {
  Module module = makeTensorKernel("elide_use");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value idx = builder::makeBin(b, OpKind::Add, tid, c0)->result(); // tid+0
  const Value x = builder::makeLoad(b, a, {idx})->result();
  builder::makeStore(b, x, a, {idx}); // uses the elided value as index

  const std::string out = canonicalize(module);
  CHECK_FALSE(contains(out, "atom.add"));
  // The load/store now index directly with the thread_id value.
  CHECK(contains(out, "atom.load"));
  CHECK(contains(out, "atom.store"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_dce_removes_dead_pure_ops", "[unit][pass]") {
  Module module = makeTensorKernel("dce");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c1 = builder::makeConst(b, 1.0, f32())->result();
  builder::makeConst(b, 5, i32()); // dead const
  const Value x = builder::makeLoad(b, a, {c0})->result();
  builder::makeBin(b, OpKind::Mul, x, x); // dead
  builder::makeBin(b, OpKind::Add, x, c1); // dead
  builder::makeStore(b, x, a, {c0}); // live: keeps x and c0 alive

  const std::string out = canonicalize(module);
  CHECK_FALSE(contains(out, "atom.const 5 : i32"));
  CHECK_FALSE(contains(out, "atom.mul"));
  CHECK_FALSE(contains(out, "atom.add"));
  CHECK(contains(out, "atom.load"));
  CHECK(contains(out, "atom.store"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_dce_keeps_side_effects", "[unit][pass]") {
  Module module = makeTensorKernel("dce_side");
  Block &b = module.kernels[0]->body;
  const Value a = Value::blockArg(&b, 0);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value x = builder::makeLoad(b, a, {c0})->result();
  builder::makeBarrier(b);
  builder::makeStore(b, x, a, {c0});

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.barrier"));
  CHECK(contains(out, "atom.store"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_keeps_unused_loop_for", "[unit][pass]") {
  Module module = makeTensorKernel("dce_loop");
  Block &b = module.kernels[0]->body;
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c8 = builder::makeConst(b, 8, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value a = Value::blockArg(&b, 0);
  const Value x = builder::makeLoad(b, a, {c0})->result();
  Op *loop = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  builder::makeYield(*loop->region, {});
  builder::makeStore(b, x, a, {c0});

  const std::string out = canonicalize(module);
  CHECK(contains(out, "atom.loop_for"));
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_canonicalize_corpus_stays_valid", "[unit][pass]") {
  Module module = buildCorpusModule();
  const std::string before = printModule(module);
  const auto result = runPasses(module, "canonicalize");
  REQUIRE(result.has_value());
  CHECK(verifyModule(module).empty());
  // Every kernel survives (pass never drops a kernel).
  CHECK(module.kernels.size() == 5);
  const std::string after = printModule(module);
  CHECK_FALSE(before.empty());
  CHECK_FALSE(after.empty());
}
