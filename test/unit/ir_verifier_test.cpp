// Structural-verifier unit tests for the atom IR (plan todo 9).
//
// Accept cases: every op kind, plus the corpus kernels. Reject cases: SSA
// dominance violations, per-op type mismatches, barrier placement, shared
// memory overflow (naming the device's LDS limit from the DeviceLimits
// table), loop_for/yield consistency, invalid kernel attributes, and i32
// const overflow. verifyModule() must report ALL violations, never
// early-exit. The shared-memory budget is device-dependent: the default
// overload verifies against gfx1101 (65536 bytes); the explicit-limits
// overload proves sm_80 (49152 bytes) rejects what gfx1101 accepts.

#include "ir_corpus.hpp"

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/ir/module.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::CmpPred;
using kernelh::ir::Diagnostics;
using kernelh::ir::IdDim;
using kernelh::ir::Kernel;
using kernelh::ir::MathKind;
using kernelh::ir::MemSpace;
using kernelh::ir::Module;
using kernelh::ir::Op;
using kernelh::ir::OpKind;
using kernelh::ir::ReduceKind;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;
namespace builder = kernelh::ir::builder;
using kernelh::ir::verifyModule;

using kernelh::test::buildCorpusModule;
using kernelh::test::buildGemm;
using kernelh::test::buildMinimal;
using kernelh::test::buildScan;
using kernelh::test::buildSoftmax;
using kernelh::test::buildVecadd;

namespace {

// Builds a kernel with a single block and returns it (kernel body).
std::unique_ptr<Kernel> makeKernel(std::string name) {
  return std::make_unique<Kernel>(std::move(name));
}

// Errors only (warnings excluded) from a module verified against `limits`.
std::vector<std::string> errorMessages(const Module &module,
                                       const kernelh::DeviceLimits &limits) {
  std::vector<std::string> messages;
  for (const kernelh::ir::Diagnostic &d : verifyModule(module, limits)) {
    if (d.severity == kernelh::ir::Diagnostic::Severity::Error) {
      messages.push_back(d.message);
    }
  }
  return messages;
}

// Errors only under the default gfx1101 device limits.
std::vector<std::string> errorMessages(const Module &module) {
  return errorMessages(module, kernelh::device_limits_for("gfx1101"));
}

bool hasMessage(const Module &module, const std::string &needle,
                const kernelh::DeviceLimits &limits) {
  for (const std::string &message : errorMessages(module, limits)) {
    if (message.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool hasMessage(const Module &module, const std::string &needle) {
  return hasMessage(module, needle, kernelh::device_limits_for("gfx1101"));
}

Type f32() { return Type::scalar(ScalarType::F32); }
Type i32() { return Type::scalar(ScalarType::I32); }
Type index() { return Type::scalar(ScalarType::Index); }
Type globalTensor(std::vector<int64_t> shape, ScalarType elem = ScalarType::F32) {
  return Type::tensor(TensorType{std::move(shape), elem, MemSpace::Global});
}
Type sharedTensor(std::vector<int64_t> shape, ScalarType elem = ScalarType::F32) {
  return Type::tensor(TensorType{std::move(shape), elem, MemSpace::Shared});
}

} // namespace

// ===========================================================================
// ACCEPT: corpus kernels
// ===========================================================================

TEST_CASE("ir_verify_accept_corpus_all_five", "[unit][ir]") {
  Module module = buildCorpusModule();
  const Diagnostics diags = verifyModule(module);
  REQUIRE(diags.empty());
}

TEST_CASE("ir_verify_accept_minimal", "[unit][ir]") {
  const Diagnostics diags = verifyModule(buildMinimal());
  REQUIRE(diags.empty());
}

TEST_CASE("ir_verify_accept_vecadd", "[unit][ir]") {
  const Diagnostics diags = verifyModule(buildVecadd());
  REQUIRE(diags.empty());
}

TEST_CASE("ir_verify_accept_gemm", "[unit][ir]") {
  const Diagnostics diags = verifyModule(buildGemm());
  REQUIRE(diags.empty());
}

TEST_CASE("ir_verify_accept_softmax", "[unit][ir]") {
  const Diagnostics diags = verifyModule(buildSoftmax());
  REQUIRE(diags.empty());
}

TEST_CASE("ir_verify_accept_scan", "[unit][ir]") {
  const Diagnostics diags = verifyModule(buildScan());
  REQUIRE(diags.empty());
}

// ===========================================================================
// ACCEPT: every op kind
// ===========================================================================

TEST_CASE("ir_verify_accept_const_scalar_and_tensor", "[unit][ir]") {
  Module module;
  auto k = makeKernel("consts");
  Block &b = k->body;
  builder::makeConst(b, 42, i32());
  builder::makeConst(b, 42, Type::scalar(ScalarType::I64));
  builder::makeConst(b, 42, index());
  builder::makeConst(b, -7, i32());
  builder::makeConst(b, 1.5, f32());
  builder::makeConst(b, 2.0, Type::scalar(ScalarType::F16));
  builder::makeConst(b, 1, globalTensor({4, 4}, ScalarType::I32));
  builder::makeConst(b, 0.5, globalTensor({8}));
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_all_scalar_arith_ops", "[unit][ir]") {
  Module module;
  auto k = makeKernel("arith");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 2.0, f32())->result();
  builder::makeBin(b, OpKind::Add, a, c);
  builder::makeBin(b, OpKind::Sub, a, c);
  builder::makeBin(b, OpKind::Mul, a, c);
  builder::makeBin(b, OpKind::Div, a, c);
  builder::makeBin(b, OpKind::Max, a, c);
  builder::makeBin(b, OpKind::Min, a, c);
  builder::makeFma(b, a, c, c);
  // Integer arithmetic too.
  const Value ai = builder::makeConst(b, 1, i32())->result();
  const Value ci = builder::makeConst(b, 2, i32())->result();
  builder::makeBin(b, OpKind::Div, ai, ci);
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_arith_on_tensors", "[unit][ir]") {
  Module module;
  auto k = makeKernel("tensor_arith");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  const Value t = Value::blockArg(&b, 0);
  const Value u = builder::makeConst(b, 0.5, globalTensor({16, 16}))->result();
  builder::makeBin(b, OpKind::Add, t, u);
  builder::makeBin(b, OpKind::Mul, t, u);
  builder::makeFma(b, t, u, u);
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_cmp_select", "[unit][ir]") {
  Module module;
  auto k = makeKernel("cmp_select");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 2.0, f32())->result();
  const Value lt = builder::makeCmp(b, CmpPred::Lt, a, c)->result();
  const Value ge = builder::makeCmp(b, CmpPred::Ge, c, a)->result();
  builder::makeSelect(b, lt, a, c);
  builder::makeSelect(b, ge, c, a);
  // Index comparisons (tiling predicates come from indices later).
  const Value z = builder::makeConst(b, 0, index())->result();
  builder::makeCmp(b, CmpPred::Eq, z, z);
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_all_math_ops", "[unit][ir]") {
  Module module;
  auto k = makeKernel("math");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 0.5, f32())->result();
  builder::makeMath(b, MathKind::Exp, a);
  builder::makeMath(b, MathKind::Log, a);
  builder::makeMath(b, MathKind::Sqrt, a);
  builder::makeMath(b, MathKind::Rsqrt, a);
  builder::makeMath(b, MathKind::Tanh, a);
  const Value h = builder::makeConst(b, 0.5, Type::scalar(ScalarType::F16))->result();
  builder::makeMath(b, MathKind::Exp, h);
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_load_store", "[unit][ir]") {
  Module module;
  auto k = makeKernel("load_store");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  const Value t = Value::blockArg(&b, 0);
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  const Value loaded = builder::makeLoad(b, t, {z, one})->result();
  builder::makeStore(b, loaded, t, {z, one});
  builder::makeStore(b, loaded, t, {one, z});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_load_vec_store_vec", "[unit][ir]") {
  Module module;
  auto k = makeKernel("vec_mem");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({1024})});
  const Value t = Value::blockArg(&b, 0);
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value v = builder::makeLoadVec(b, t, {z}, 4)->result();
  builder::makeStoreVec(b, v, t, {z});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_alloc_shared_and_barrier", "[unit][ir]") {
  Module module;
  auto k = makeKernel("shared");
  Block &b = k->body;
  builder::makeAllocShared(b, TensorType{{64, 64}, ScalarType::F32, MemSpace::Shared});
  builder::makeBarrier(b);
  builder::makeAllocShared(b, TensorType{{64, 16}, ScalarType::F32, MemSpace::Shared});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_all_id_ops", "[unit][ir]") {
  Module module;
  auto k = makeKernel("ids");
  Block &b = k->body;
  for (const IdDim dim : {IdDim::X, IdDim::Y, IdDim::Z}) {
    builder::makeId(b, OpKind::ThreadId, dim);
    builder::makeId(b, OpKind::BlockId, dim);
    builder::makeId(b, OpKind::BlockDim, dim);
    builder::makeId(b, OpKind::GridDim, dim);
  }
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_loop_for_with_iter_args", "[unit][ir]") {
  Module module;
  auto k = makeKernel("loop");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  const Value init = builder::makeConst(b, 0.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {init}, {f32()});
  Block &body = *loop->region;
  const Value ind = Value::blockArg(&body, 0);
  const Value acc = Value::blockArg(&body, 1);
  const Value sum = builder::makeBin(body, OpKind::Add, acc, init)->result();
  builder::makeYield(body, {sum});
  (void)ind;
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_loop_for_zero_iter_args", "[unit][ir]") {
  Module module;
  auto k = makeKernel("loop0");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &body = *loop->region;
  const Value ind = Value::blockArg(&body, 0);
  builder::makeBin(body, OpKind::Add, ind, z);
  builder::makeYield(body, {});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_barrier_in_loop_body", "[unit][ir]") {
  // Barrier placement rule: barriers are allowed in kernel-body blocks and
  // loop_for body blocks (the only two block kinds in v1).
  Module module;
  auto k = makeKernel("barrier_loop");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &body = *loop->region;
  builder::makeBarrier(body);
  builder::makeYield(body, {});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_reduce_sum_and_max", "[unit][ir]") {
  Module module;
  auto k = makeKernel("reduce");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  const Value t = Value::blockArg(&b, 0);
  builder::makeReduce(b, ReduceKind::Sum, t, 0);
  builder::makeReduce(b, ReduceKind::Max, t, 1);
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_copy", "[unit][ir]") {
  Module module;
  auto k = makeKernel("copy");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  const Value src = Value::blockArg(&b, 0);
  const Value dst = Value::blockArg(&b, 1);
  builder::makeCopy(b, src, dst);
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_empty_kernel", "[unit][ir]") {
  Module module;
  auto k = makeKernel("empty");
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_accept_kernel_args", "[unit][ir]") {
  Module module;
  auto k = makeKernel("args");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({1024})});
  b.args.push_back(BlockArg{f32()}); // scalar kernel arg allowed
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

// ===========================================================================
// REJECT: SSA dominance
// ===========================================================================

TEST_CASE("ir_verify_reject_dominance_use_before_def", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_dom");
  Block &b = k->body;
  Op *first = builder::makeBin(b, OpKind::Add, Value::none(), Value::none());
  const Value c = builder::makeConst(b, 1.0, f32())->result();
  // Mutate the earlier op to use the LATER definition: use-before-def.
  first->operands = {c, c};
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "SSA dominance violation"));
}

TEST_CASE("ir_verify_reject_dominance_inner_to_outer", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_dom2");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &body = *loop->region;
  const Value inner = builder::makeConst(body, 2.0, f32())->result();
  builder::makeYield(body, {});
  // Use the inner value in the OUTER block: not dominated.
  builder::makeBin(b, OpKind::Add, inner, inner);
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "SSA dominance violation"));
}

TEST_CASE("ir_verify_reject_loop_self_result_use", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_self");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  const Value init = builder::makeConst(b, 0.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {init}, {f32()});
  Block &body = *loop->region;
  const Value acc = Value::blockArg(&body, 1);
  // Reference the loop's own result inside its body: not dominated.
  builder::makeBin(body, OpKind::Add, acc, loop->result());
  builder::makeYield(body, {acc});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "SSA dominance violation"));
}

TEST_CASE("ir_verify_reject_use_of_value_from_sibling_scope", "[unit][ir]") {
  // A value defined inside one loop body cannot be used inside another loop
  // body (sibling scopes).
  Module module;
  auto k = makeKernel("bad_sibling");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  Op *loopA = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &bodyA = *loopA->region;
  const Value inner = builder::makeConst(bodyA, 2.0, f32())->result();
  builder::makeYield(bodyA, {});
  Op *loopB = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &bodyB = *loopB->region;
  builder::makeBin(bodyB, OpKind::Add, inner, inner);
  builder::makeYield(bodyB, {});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "SSA dominance violation"));
}

TEST_CASE("ir_verify_reject_invalid_null_operand", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_null");
  Block &b = k->body;
  builder::makeBin(b, OpKind::Add, Value::none(), Value::none());
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "invalid (null) operand"));
}

// ===========================================================================
// REJECT: per-op type rules
// ===========================================================================

TEST_CASE("ir_verify_reject_type_mismatch_binary", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_type");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 1, i32())->result();
  Op *op = builder::makeBin(b, OpKind::Add, a, c);
  op->resultTypes = {f32()}; // operands still mismatch
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "type mismatch"));
}

TEST_CASE("ir_verify_reject_type_mismatch_binary_result", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_result_type");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 2.0, f32())->result();
  Op *op = builder::makeBin(b, OpKind::Add, a, c);
  op->resultTypes = {i32()};
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "type mismatch"));
}

TEST_CASE("ir_verify_reject_type_mismatch_fma", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_fma");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 1, i32())->result();
  builder::makeFma(b, a, c, a);
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "type mismatch"));
}

TEST_CASE("ir_verify_reject_cmp_type_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_cmp");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 1, i32())->result();
  builder::makeCmp(b, CmpPred::Lt, a, c);
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "type mismatch"));
}

TEST_CASE("ir_verify_reject_cmp_result_not_i32", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_cmp_res");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 2.0, f32())->result();
  Op *op = builder::makeCmp(b, CmpPred::Lt, a, c);
  op->resultTypes = {f32()};
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "boolean carrier"));
}

TEST_CASE("ir_verify_reject_select_cond_not_i32", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_select");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 2.0, f32())->result();
  builder::makeSelect(b, a, a, c); // cond is f32, not i32
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "condition must be an i32"));
}

TEST_CASE("ir_verify_reject_math_on_i32", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_math");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1, i32())->result();
  builder::makeMath(b, MathKind::Exp, a);
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "f16/f32"));
}

TEST_CASE("ir_verify_reject_load_index_count_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_load_idx");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  const Value t = Value::blockArg(&b, 0);
  const Value z = builder::makeConst(b, 0, index())->result();
  builder::makeLoad(b, t, {z}); // 2-D tensor, 1 index
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "index count"));
}

TEST_CASE("ir_verify_reject_load_index_not_index", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_load_ty");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16})});
  const Value t = Value::blockArg(&b, 0);
  const Value f = builder::makeConst(b, 0.0, f32())->result();
  builder::makeLoad(b, t, {f});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "index"));
}

TEST_CASE("ir_verify_reject_store_value_type_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_store");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16})});
  const Value t = Value::blockArg(&b, 0);
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value wrong = builder::makeConst(b, 0, i32())->result();
  builder::makeStore(b, wrong, t, {z}); // f32 tensor, i32 value
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "element type"));
}

TEST_CASE("ir_verify_reject_load_vec_elem_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_vec_elem");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16})});
  const Value t = Value::blockArg(&b, 0);
  const Value z = builder::makeConst(b, 0, index())->result();
  Op *op = builder::makeLoadVec(b, t, {z}, 4);
  op->resultTypes = {Type::vector(kernelh::ir::VectorType{4, ScalarType::I32})};
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "element type"));
}

TEST_CASE("ir_verify_reject_alloc_shared_wrong_space", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_alloc");
  Block &b = k->body;
  builder::makeAllocShared(b, TensorType{{64}, ScalarType::F32, MemSpace::Global});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "shared memory space"));
}

TEST_CASE("ir_verify_reject_tensor_zero_extent", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_extent");
  Block &b = k->body;
  builder::makeConst(b, 0.0, globalTensor({0}));
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "invalid type"));
}

TEST_CASE("ir_verify_reject_vector_zero_width", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_width");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16})});
  const Value t = Value::blockArg(&b, 0);
  const Value z = builder::makeConst(b, 0, index())->result();
  builder::makeLoadVec(b, t, {z}, 0);
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "invalid type"));
}

TEST_CASE("ir_verify_reject_reduce_dim_out_of_range", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_reduce_dim");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  const Value t = Value::blockArg(&b, 0);
  builder::makeReduce(b, ReduceKind::Sum, t, 5);
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "out of range"));
}

TEST_CASE("ir_verify_reject_copy_shape_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_copy");
  Block &b = k->body;
  b.args.push_back(BlockArg{globalTensor({16, 16})});
  b.args.push_back(BlockArg{globalTensor({16})});
  const Value src = Value::blockArg(&b, 0);
  const Value dst = Value::blockArg(&b, 1);
  builder::makeCopy(b, src, dst);
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "identical shape"));
}

TEST_CASE("ir_verify_reject_kernel_arg_shared_tensor", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_arg");
  Block &b = k->body;
  b.args.push_back(BlockArg{sharedTensor({16})});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "cannot be a shared-space tensor"));
}

// ===========================================================================
// REJECT: loop_for / yield consistency
// ===========================================================================

TEST_CASE("ir_verify_reject_loop_iter_yield_count_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_loop_yield");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  const Value init = builder::makeConst(b, 0.0, f32())->result();
  const Value init2 = builder::makeConst(b, 0.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {init, init2}, {f32(), f32()});
  Block &body = *loop->region;
  const Value acc = Value::blockArg(&body, 1);
  builder::makeYield(body, {acc}); // 1 operand for 2 iter args
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "iter_args"));
}

TEST_CASE("ir_verify_reject_loop_body_args_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_loop_args");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  const Value init = builder::makeConst(b, 0.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {init}, {f32()});
  Block &body = *loop->region;
  body.args.clear(); // body must have [index, f32]
  body.args.push_back(BlockArg{index()});
  const Value acc = Value::blockArg(&body, 1);
  builder::makeYield(body, {acc});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "block argument count"));
}

TEST_CASE("ir_verify_reject_loop_body_arg_type_mismatch", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_loop_arg_type");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  const Value init = builder::makeConst(b, 0.0, f32())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {init}, {f32()});
  Block &body = *loop->region;
  body.args[1].type = i32(); // iter arg must be f32
  const Value acc = Value::blockArg(&body, 1);
  builder::makeYield(body, {acc});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "type mismatch"));
}

TEST_CASE("ir_verify_reject_loop_missing_yield", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_no_yield");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  builder::makeLoopFor(b, z, one, one, {}, {});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "terminate with an atom.yield"));
}

TEST_CASE("ir_verify_reject_yield_outside_loop", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_yield_top");
  Block &b = k->body;
  builder::makeYield(b, {});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "inside an atom.loop_for body"));
}

TEST_CASE("ir_verify_reject_yield_not_last", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_yield_mid");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &body = *loop->region;
  builder::makeYield(body, {});
  builder::makeConst(body, 1.0, f32()); // op after yield
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "last operation of its block"));
}

TEST_CASE("ir_verify_reject_loop_bound_not_index", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_bound");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value f = builder::makeConst(b, 1.0, f32())->result();
  builder::makeLoopFor(b, z, f, z, {}, {});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "index"));
}

// ===========================================================================
// REJECT: kernel attributes
// ===========================================================================

TEST_CASE("ir_verify_reject_tile_zero", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_tile");
  k->schedule.tile = {0, 128};
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "tile size must be positive"));
}

TEST_CASE("ir_verify_reject_tile_negative", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_tile_neg");
  k->schedule.tile = {-1};
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "tile size must be positive"));
}

TEST_CASE("ir_verify_reject_vector_width_zero", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_vw");
  k->schedule.vector_width = 0;
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "vector_width must be >= 1"));
}

TEST_CASE("ir_verify_reject_unroll_zero", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_unroll");
  k->schedule.unroll = 0;
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "unroll must be >= 1"));
}

TEST_CASE("ir_verify_reject_shape_value_nonpositive", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_shape");
  k->shape["m"] = -4;
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "must be positive"));
}

// ===========================================================================
// REJECT: shared memory overflow (names the device's LDS limit from the
// DeviceLimits table; default overload verifies against gfx1101 = 65536)
// ===========================================================================

TEST_CASE("ir_verify_reject_shared_overflow_names_limit", "[unit][ir]") {
  Module module;
  auto k = makeKernel("overflow");
  Block &b = k->body;
  // 128x128 f32 = 65536 bytes alone equals gfx1101's budget; two tiles
  // (131072 bytes) blow past it.
  builder::makeAllocShared(b, TensorType{{128, 128}, ScalarType::F32, MemSpace::Shared});
  builder::makeAllocShared(b, TensorType{{128, 128}, ScalarType::F32, MemSpace::Shared});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "shared memory usage"));
  CHECK(hasMessage(module, "65536"));
  CHECK(hasMessage(module, "gfx1101"));
}

TEST_CASE("ir_verify_accept_shared_exactly_at_limit", "[unit][ir]") {
  Module module;
  auto k = makeKernel("at_limit");
  Block &b = k->body;
  // 128x128 f32 = 65536 bytes = exactly gfx1101's LDS budget.
  builder::makeAllocShared(b, TensorType{{128, 128}, ScalarType::F32, MemSpace::Shared});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_reject_shared_overflow_counts_loop_scopes", "[unit][ir]") {
  // alloc_shared inside a loop body still counts toward the workgroup budget.
  Module module;
  auto k = makeKernel("overflow_loop");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &body = *loop->region;
  builder::makeAllocShared(body, TensorType{{128, 128}, ScalarType::F32, MemSpace::Shared});
  builder::makeAllocShared(body, TensorType{{128, 128}, ScalarType::F32, MemSpace::Shared});
  builder::makeYield(body, {});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "65536"));
}

// ===========================================================================
// Device-limits switchover (todo 10a): the same kernel discriminates per
// target - gfx1101's 65536-byte budget accepts 51200 bytes, sm_80's
// 49152-byte budget rejects it.
// ===========================================================================

TEST_CASE("ir_verify_shared_51200_accept_gfx1101_reject_sm80", "[unit][ir]") {
  const kernelh::DeviceLimits &gfx1101 = kernelh::device_limits_for("gfx1101");
  const kernelh::DeviceLimits &sm80 = kernelh::device_limits_for("sm_80");
  // Guard the discrimination itself: if the table values drift, this test
  // fails loudly instead of silently passing with both targets agreeing.
  REQUIRE(51200 <= gfx1101.lds_bytes_per_workgroup);
  REQUIRE(51200 > sm80.lds_bytes_per_workgroup);
  Module module;
  auto k = makeKernel("shared_51200");
  Block &b = k->body;
  // 12800 f32 elements = 51200 bytes of static shared memory.
  builder::makeAllocShared(b, TensorType{{12800}, ScalarType::F32, MemSpace::Shared});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module, gfx1101).empty());
  CHECK_FALSE(verifyModule(module, sm80).empty());
  CHECK(hasMessage(module, "51200", sm80));
  CHECK(hasMessage(module, "49152", sm80));
  CHECK(hasMessage(module, "sm_80", sm80));
}

TEST_CASE("ir_verify_shared_49152_accept_gfx1101", "[unit][ir]") {
  // 12288 f32 elements = 49152 bytes (sm_80's exact budget) is still within
  // gfx1101's larger 65536-byte budget.
  const kernelh::DeviceLimits &gfx1101 = kernelh::device_limits_for("gfx1101");
  REQUIRE(49152 <= gfx1101.lds_bytes_per_workgroup);
  Module module;
  auto k = makeKernel("shared_49152");
  Block &b = k->body;
  builder::makeAllocShared(b, TensorType{{12288}, ScalarType::F32, MemSpace::Shared});
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module, gfx1101).empty());
}

// ===========================================================================
// REJECT: const value range
// ===========================================================================

TEST_CASE("ir_verify_reject_const_i32_overflow", "[unit][ir]") {
  Module module;
  auto k = makeKernel("big_i32");
  Block &b = k->body;
  const int64_t tooBig = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  builder::makeConst(b, tooBig, i32());
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "does not fit i32"));
}

TEST_CASE("ir_verify_accept_const_i32_edges", "[unit][ir]") {
  Module module;
  auto k = makeKernel("i32_edges");
  Block &b = k->body;
  builder::makeConst(b, std::numeric_limits<int32_t>::min(), i32());
  builder::makeConst(b, std::numeric_limits<int32_t>::max(), i32());
  builder::makeConst(b, static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
                     Type::scalar(ScalarType::I64)); // fits i64
  module.kernels.push_back(std::move(k));
  REQUIRE(verifyModule(module).empty());
}

TEST_CASE("ir_verify_reject_const_int_for_float_type", "[unit][ir]") {
  // A const with an integer payload on a float element type is malformed
  // (the parser can only build well-formed consts; a mutated op is caught).
  Module module;
  auto k = makeKernel("bad_const");
  Block &b = k->body;
  Op *op = builder::makeConst(b, 1.5, f32());
  op->intValue = 7;
  op->floatValue.reset();
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "float literal"));
}

// ===========================================================================
// REJECT: nested block parent pointers / all-violations reporting
// ===========================================================================

TEST_CASE("ir_verify_reject_inconsistent_region_parent", "[unit][ir]") {
  Module module;
  auto k = makeKernel("bad_parent");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, index())->result();
  const Value one = builder::makeConst(b, 1, index())->result();
  Op *loop = builder::makeLoopFor(b, z, one, one, {}, {});
  loop->region->parentBlock = nullptr; // stale pointer
  builder::makeYield(*loop->region, {});
  module.kernels.push_back(std::move(k));
  CHECK(hasMessage(module, "inconsistent block nesting"));
}

TEST_CASE("ir_verify_reports_all_violations_not_first_only", "[unit][ir]") {
  Module module;
  auto k = makeKernel("many_bad");
  Block &b = k->body;
  const Value a = builder::makeConst(b, 1.0, f32())->result();
  const Value c = builder::makeConst(b, 1, i32())->result();
  builder::makeBin(b, OpKind::Add, a, c);   // type mismatch
  builder::makeYield(b, {a});               // yield outside loop
  k->schedule.vector_width = 0;             // bad attr
  module.kernels.push_back(std::move(k));
  const std::vector<std::string> messages = errorMessages(module);
  REQUIRE(messages.size() >= 3);
  bool sawType = false, sawYield = false, sawVw = false;
  for (const std::string &m : messages) {
    sawType |= m.find("type mismatch") != std::string::npos;
    sawYield |= m.find("inside an atom.loop_for body") != std::string::npos;
    sawVw |= m.find("vector_width") != std::string::npos;
  }
  CHECK(sawType);
  CHECK(sawYield);
  CHECK(sawVw);
}

TEST_CASE("ir_verify_is_module_valid", "[unit][ir]") {
  CHECK(kernelh::ir::isModuleValid(buildMinimal()));
  CHECK_FALSE(kernelh::ir::isModuleValid([] {
    Module module;
    auto k = makeKernel("bad");
    k->schedule.vector_width = 0;
    module.kernels.push_back(std::move(k));
    return module;
  }()));
}
