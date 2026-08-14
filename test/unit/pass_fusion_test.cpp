// Fuse pass unit tests (plan todo 14): elementwise producer-consumer chain
// fusion into one loop nest. Passes run through the GLOBAL registry
// (self-registered by lib/passes/*.cpp at static init).
//
// Positive: a two-loop chain (and a three-loop chain) collapses to ONE loop
// and the intermediate tensor is eliminated. Negatives: a multi-use
// intermediate, a barrier between the loops, a reduction boundary (an
// atom.reduce operand or a reduction-style consumer loop), a mismatched
// iteration domain, and a producer store at a non-induction index all block
// fusion. Every fused output must re-verify clean.

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <string>

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
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
using kernelh::ir::printModule;
using kernelh::ir::runPasses;
using kernelh::ir::verifyModule;

namespace {

Type f32() { return Type::scalar(ScalarType::F32); }
Type index() { return Type::scalar(ScalarType::Index); }
Type tensorOf(ScalarType elem) {
  return Type::tensor(TensorType{{8}, elem, MemSpace::Global});
}
TensorType shared8() {
  return TensorType{{8}, ScalarType::F32, MemSpace::Shared};
}

// Builds the two-loop chain  out = tanh(a*2 + b)  through an intermediate
// shared tensor, plus optional negative-case variations.
struct ChainOpts {
  bool barrier = false;       // atom.barrier between the two loops
  bool extraBodyLoad = false; // second use of the intermediate in the body
  bool reduceOnT = false;     // the intermediate feeds an atom.reduce
  bool redLoopConsumer = false; // consumer is a reduction-style loop
  bool diffDomain = false;    // consumer iterates a different ub
  bool nonMatchIndex = false; // producer stores at a const index
};

Module buildChain(const std::string &name, const ChainOpts &opts = {}) {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensorOf(ScalarType::F32)}); // a
  b.args.push_back(BlockArg{tensorOf(ScalarType::F32)}); // b
  b.args.push_back(BlockArg{tensorOf(ScalarType::F32)}); // out
  const Value a = Value::blockArg(&b, 0);
  const Value bv = Value::blockArg(&b, 1);
  const Value out = Value::blockArg(&b, 2);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value ub = builder::makeConst(b, 8, index())->result();
  // The consumer iterates a DIFFERENT ub value (a distinct SSA definition)
  // when diffDomain is set; otherwise both loops share the same ub handle.
  const Value ubC =
      opts.diffDomain ? builder::makeConst(b, 4, index())->result() : ub;
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value t =
      builder::makeAllocShared(b, shared8())->result();

  Op *p = builder::makeLoopFor(b, c0, ub, c1, {}, {});
  Block &pb = *p->region;
  const Value pi = Value::blockArg(&pb, 0);
  const Value x = builder::makeLoad(pb, a, {pi})->result();
  const Value c2 = builder::makeConst(pb, 2.0, f32())->result();
  const Value prod = builder::makeBin(pb, OpKind::Mul, x, c2)->result();
  const Value y = builder::makeLoad(pb, bv, {pi})->result();
  const Value sum = builder::makeBin(pb, OpKind::Add, prod, y)->result();
  builder::makeStore(pb, sum, t, {opts.nonMatchIndex ? c0 : pi});
  builder::makeYield(pb, {});

  if (opts.barrier) {
    builder::makeBarrier(b);
  }

  if (opts.redLoopConsumer) {
    const Value init = builder::makeConst(b, 0.0, f32())->result();
    Op *rc = builder::makeLoopFor(b, c0, ubC, c1, {init}, {f32()});
    Block &rb = *rc->region;
    const Value ri = Value::blockArg(&rb, 0);
    const Value acc = Value::blockArg(&rb, 1);
    const Value tv = builder::makeLoad(rb, t, {ri})->result();
    const Value ns = builder::makeBin(rb, OpKind::Add, tv, acc)->result();
    builder::makeYield(rb, {ns});
  } else {
    Op *c = builder::makeLoopFor(b, c0, ubC, c1, {}, {});
    Block &cb = *c->region;
    const Value ci = Value::blockArg(&cb, 0);
    const Value tv = builder::makeLoad(cb, t, {ci})->result();
    const Value g = builder::makeMath(cb, MathKind::Tanh, tv)->result();
    builder::makeStore(cb, g, out, {ci});
    builder::makeYield(cb, {});
  }

  if (opts.extraBodyLoad) {
    builder::makeLoad(b, t, {c0}); // dead load: a second use of the chain's
                                   // intermediate, outside the consumer chain
  }
  if (opts.reduceOnT) {
    builder::makeReduce(b, ReduceKind::Sum, t, 0); // t feeds a reduction
  }

  module.kernels.push_back(std::move(kernel));
  return module;
}

// Three-loop chain: t1 = a*2 (loop 1), t2 = t1 + b (loop 2),
// out = tanh(t2) (loop 3). The fixpoint must collapse all three loops.
Module buildThreeLoopChain(const std::string &name) {
  Module module;
  auto kernel = std::make_unique<Kernel>(name);
  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensorOf(ScalarType::F32)});
  b.args.push_back(BlockArg{tensorOf(ScalarType::F32)});
  b.args.push_back(BlockArg{tensorOf(ScalarType::F32)});
  const Value a = Value::blockArg(&b, 0);
  const Value bv = Value::blockArg(&b, 1);
  const Value out = Value::blockArg(&b, 2);
  const Value c0 = builder::makeConst(b, 0, index())->result();
  const Value c8 = builder::makeConst(b, 8, index())->result();
  const Value c1 = builder::makeConst(b, 1, index())->result();
  const Value t1 = builder::makeAllocShared(b, shared8())->result();
  const Value t2 = builder::makeAllocShared(b, shared8())->result();

  Op *l1 = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  Block &b1 = *l1->region;
  const Value i1 = Value::blockArg(&b1, 0);
  const Value x = builder::makeLoad(b1, a, {i1})->result();
  const Value c2 = builder::makeConst(b1, 2.0, f32())->result();
  const Value prod = builder::makeBin(b1, OpKind::Mul, x, c2)->result();
  builder::makeStore(b1, prod, t1, {i1});
  builder::makeYield(b1, {});

  Op *l2 = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  Block &b2 = *l2->region;
  const Value i2 = Value::blockArg(&b2, 0);
  const Value tv1 = builder::makeLoad(b2, t1, {i2})->result();
  const Value y = builder::makeLoad(b2, bv, {i2})->result();
  const Value sum = builder::makeBin(b2, OpKind::Add, tv1, y)->result();
  builder::makeStore(b2, sum, t2, {i2});
  builder::makeYield(b2, {});

  Op *l3 = builder::makeLoopFor(b, c0, c8, c1, {}, {});
  Block &b3 = *l3->region;
  const Value i3 = Value::blockArg(&b3, 0);
  const Value tv2 = builder::makeLoad(b3, t2, {i3})->result();
  const Value g = builder::makeMath(b3, MathKind::Tanh, tv2)->result();
  builder::makeStore(b3, g, out, {i3});
  builder::makeYield(b3, {});

  module.kernels.push_back(std::move(kernel));
  return module;
}

std::string fuse(Module &module) {
  const auto result = runPasses(module, "fuse");
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

TEST_CASE("pass_fuse_chain_fuses", "[unit][pass]") {
  Module module = buildChain("fuse1");
  const std::string out = fuse(module);
  // One loop nest computes the whole chain; the intermediate tensor and its
  // alloc_shared are eliminated; only the final store remains.
  CHECK(count(out, "atom.loop_for") == 1);
  CHECK(count(out, "atom.alloc_shared") == 0);
  CHECK(count(out, "atom.store") == 1);
  CHECK(count(out, "atom.load") == 2); // a and b only
  CHECK(out.find("atom.math tanh") != std::string::npos);
  CHECK(verifyModule(module).empty());
  // print -> parse -> print round-trips byte-identically.
  const auto reparsed = kernelh::ir::parseModule(out);
  REQUIRE(reparsed.has_value());
  CHECK(printModule(*reparsed) == out);
  CHECK(verifyModule(*reparsed).empty());
}

TEST_CASE("pass_fuse_multi_use_blocks", "[unit][pass]") {
  Module module = buildChain("fuse_mu", ChainOpts{.extraBodyLoad = true});
  const std::string out = fuse(module);
  // The intermediate has a use outside the consumer chain: no fusion. Four
  // loads survive (a, b, the consumer's load, and the extra body load).
  CHECK(count(out, "atom.loop_for") == 2);
  CHECK(count(out, "atom.alloc_shared") == 1);
  CHECK(count(out, "atom.load") == 4);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_fuse_barrier_blocks", "[unit][pass]") {
  Module module = buildChain("fuse_bar", ChainOpts{.barrier = true});
  const std::string out = fuse(module);
  // A barrier between producer and consumer is a memory-ordering point.
  CHECK(count(out, "atom.loop_for") == 2);
  CHECK(count(out, "atom.alloc_shared") == 1);
  CHECK(count(out, "atom.barrier") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_fuse_reduce_operand_blocks", "[unit][pass]") {
  Module module = buildChain("fuse_red", ChainOpts{.reduceOnT = true});
  const std::string out = fuse(module);
  // The intermediate feeds an atom.reduce: fusion never crosses reductions.
  CHECK(count(out, "atom.loop_for") == 2);
  CHECK(count(out, "atom.alloc_shared") == 1);
  CHECK(count(out, "atom.reduce") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_fuse_reduction_loop_blocks", "[unit][pass]") {
  Module module =
      buildChain("fuse_redloop", ChainOpts{.redLoopConsumer = true});
  const std::string out = fuse(module);
  // A reduction-style consumer (non-empty iter_args) is not elementwise:
  // the producer stays, the reduction stays.
  CHECK(count(out, "atom.loop_for") == 2);
  CHECK(count(out, "atom.alloc_shared") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_fuse_different_domain_blocks", "[unit][pass]") {
  Module module = buildChain("fuse_dom", ChainOpts{.diffDomain = true});
  const std::string out = fuse(module);
  // Different iteration bounds: the load/store element correspondence is
  // undefined, so the pair must not fuse.
  CHECK(count(out, "atom.loop_for") == 2);
  CHECK(count(out, "atom.alloc_shared") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_fuse_nonmatching_index_blocks", "[unit][pass]") {
  Module module = buildChain("fuse_idx", ChainOpts{.nonMatchIndex = true});
  const std::string out = fuse(module);
  // The producer stores at a constant index, not the induction var: the
  // store[i] <-> load[i] correspondence cannot be proven, so no fusion.
  CHECK(count(out, "atom.loop_for") == 2);
  CHECK(count(out, "atom.alloc_shared") == 1);
  CHECK(verifyModule(module).empty());
}

TEST_CASE("pass_fuse_three_loop_chain_collapses", "[unit][pass]") {
  Module module = buildThreeLoopChain("fuse3");
  const std::string out = fuse(module);
  // t1 = a*2; t2 = t1 + b; out = tanh(t2): all three loops fuse into one.
  CHECK(count(out, "atom.loop_for") == 1);
  CHECK(count(out, "atom.alloc_shared") == 0);
  CHECK(count(out, "atom.store") == 1);
  CHECK(count(out, "atom.load") == 2);
  CHECK(verifyModule(module).empty());
}
