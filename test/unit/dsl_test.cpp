// DSL unit tests (plan todo 11): fluent schedule builder + kernel template
// registry + the vecadd template. Test-case names carry the "dsl_" prefix so
// `ctest -R dsl_` selects exactly these.
//
// Coverage:
//   dsl_chain_produces_exact_schedule_attrs - the builder chain accumulates
//     exactly the expected ScheduleAttrs / shape map / kernel name
//   dsl_vecadd_emits_verifiable_kernel - emit() through the registry yields a
//     module that VERIFIES CLEAN (empty diagnostics) and contains one
//     atom.kernel @vecadd with the schedule attrs
//   dsl_emit_round_trip - print(parse(print(M))) == print(M)
//   dsl_unknown_kernel_lists_registry - .kernel("nope").emit() throws an
//     error naming the registered kernels (vecadd)
//   dsl_registry_custom_template - registerKernel() + IRBuilder API surface
//     (shared alloc, barrier, loop_for with iter_args)
//   dsl_irbuilder_named_values_and_scopes - IRBuilder name/scope semantics

#include "kernelh/dsl/program.hpp"
#include "kernelh/ir/builder.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Emits the canonical vecadd chain used by several tests below.
kernelh::EmittedKernel emitVecadd() {
  return kernelh::program()
      .tile(128, 128, 32)
      .map_blocks()
      .map_warps()
      .use_shared_memory()
      .vectorize(4)
      .unroll(2)
      .kernel("vecadd")
      .shape({{"n", 1024}})
      .emit();
}

} // namespace

TEST_CASE("dsl_chain_produces_exact_schedule_attrs", "[dsl]") {
  kernelh::ProgramBuilder p = kernelh::program()
                                  .tile(128, 128, 32)
                                  .map_blocks()
                                  .map_warps()
                                  .use_shared_memory()
                                  .vectorize(4)
                                  .unroll(2)
                                  .kernel("vecadd")
                                  .shape({{"n", 1024}});

  const kernelh::ir::ScheduleAttrs &s = p.schedule();
  CHECK(s.tile == (std::vector<int64_t>{128, 128, 32}));
  CHECK(s.map_blocks);
  CHECK(s.map_warps);
  CHECK(s.use_shared_memory);
  CHECK(s.vector_width == 4);
  CHECK(s.unroll == 2);
  CHECK(p.kernelName() == "vecadd");
  CHECK(p.shapeMap() == (kernelh::ShapeMap{{"n", 1024}}));

  // Variadic tile: 1-D and 2-D forms.
  CHECK(kernelh::program().tile(256).schedule().tile ==
        (std::vector<int64_t>{256}));
  CHECK(kernelh::program().tile(16, 32).schedule().tile ==
        (std::vector<int64_t>{16, 32}));

  // Defaults on a fresh builder. NB: bind to the builder object, not to
  // program().schedule() - a member of a temporary dangles (no extension).
  const kernelh::ProgramBuilder fresh = kernelh::program();
  const kernelh::ir::ScheduleAttrs &def = fresh.schedule();
  CHECK(def.tile.empty());
  CHECK_FALSE(def.map_blocks);
  CHECK_FALSE(def.map_warps);
  CHECK_FALSE(def.use_shared_memory);
  CHECK(def.vector_width == 1);
  CHECK(def.unroll == 1);
}

TEST_CASE("dsl_vecadd_emits_verifiable_kernel", "[dsl]") {
  const kernelh::EmittedKernel k = emitVecadd();

  // VERIFIES CLEAN - empty diagnostics, not just exit-0.
  const kernelh::ir::Diagnostics diags = kernelh::ir::verifyModule(k.module);
  for (const kernelh::ir::Diagnostic &d : diags) {
    INFO(d.toString());
  }
  CHECK(diags.empty());

  CHECK(k.module.size() == 1);
  const kernelh::ir::Kernel &kernel = *k.module.kernels[0];
  CHECK(kernel.name == "vecadd");
  CHECK(kernel.body.args.size() == 3); // a, b, c

  // Schedule + shape attrs ride along on the emitted kernel.
  const kernelh::ir::ScheduleAttrs &s = kernel.schedule;
  CHECK(s.tile == (std::vector<int64_t>{128, 128, 32}));
  CHECK(s.map_blocks);
  CHECK(s.map_warps);
  CHECK(s.use_shared_memory);
  CHECK(s.vector_width == 4);
  CHECK(s.unroll == 2);
  CHECK(kernel.shape == (kernelh::ShapeMap{{"n", 1024}}));

  // Printed text: kernel + attrs + grid-stride ingredients.
  const std::string &text = k.atomText;
  CHECK(text.find("atom.kernel @vecadd") != std::string::npos);
  CHECK(text.find("tile = [128, 128, 32]") != std::string::npos);
  CHECK(text.find("map_blocks = true") != std::string::npos);
  CHECK(text.find("map_warps = true") != std::string::npos);
  CHECK(text.find("use_shared_memory = true") != std::string::npos);
  CHECK(text.find("vector_width = 4") != std::string::npos);
  CHECK(text.find("unroll = 2") != std::string::npos);
  CHECK(text.find("shape = {n = 1024}") != std::string::npos);
  CHECK(text.find("atom.block_id x : index") != std::string::npos);
  CHECK(text.find("atom.block_dim x : index") != std::string::npos);
  CHECK(text.find("atom.grid_dim x : index") != std::string::npos);
  CHECK(text.find("atom.thread_id x : index") != std::string::npos);
  CHECK(text.find("atom.loop_for") != std::string::npos);
  CHECK(text.find("atom.load") != std::string::npos);
  CHECK(text.find("atom.store") != std::string::npos);
}

TEST_CASE("dsl_emit_round_trip", "[dsl]") {
  const kernelh::EmittedKernel k = emitVecadd();

  const auto parsed = kernelh::ir::parseModule(k.atomText);
  if (!parsed) {
    INFO(parsed.error());
  }
  REQUIRE(parsed.has_value());
  const std::string reparsed = kernelh::ir::printModule(*parsed);
  CHECK(reparsed == k.atomText); // print(parse(print(M))) == print(M)

  const auto again = kernelh::ir::parseModule(reparsed);
  if (!again) {
    INFO(again.error());
  }
  REQUIRE(again.has_value());
  CHECK(kernelh::ir::printModule(*again) == k.atomText); // stable across gens
}

TEST_CASE("dsl_unknown_kernel_lists_registry", "[dsl]") {
  kernelh::ProgramBuilder p = kernelh::program().kernel("nope");
  try {
    (void)p.emit();
    FAIL("emit() must throw for an unknown kernel name");
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("nope") != std::string::npos);
    CHECK(msg.find("vecadd") != std::string::npos);
  }
}

TEST_CASE("dsl_registry_custom_template", "[dsl]") {
  // A user-supplied template exercising the IRBuilder API surface: consts,
  // id ops, shared alloc + barrier, loop_for with iter_args + yield,
  // load/store.
  kernelh::registerKernel(
      "dsl_probe",
      [](kernelh::ir::IRBuilder &b, const kernelh::ir::ScheduleAttrs &,
         const kernelh::ShapeMap &) {
        kernelh::ir::Block &body = b.block();
        const kernelh::ir::Type argTy = kernelh::ir::Type::tensor(
            kernelh::ir::TensorType{{16}, kernelh::ir::ScalarType::F32,
                                    kernelh::ir::MemSpace::Global});
        body.args.push_back(kernelh::ir::BlockArg{argTy});
        const kernelh::ir::Value in =
            kernelh::ir::Value::blockArg(&body, 0);

        const kernelh::ir::Value zero =
            b.emitConst(int64_t{0}, kernelh::ir::ScalarType::Index);
        const kernelh::ir::Value sixteen =
            b.emitConst(int64_t{16}, kernelh::ir::ScalarType::Index);
        const kernelh::ir::Value one =
            b.emitConst(int64_t{1}, kernelh::ir::ScalarType::Index);
        const kernelh::ir::Value tid = b.emitThreadId();
        const kernelh::ir::Value f0 =
            b.emitConst(0.0, kernelh::ir::ScalarType::F32);

        const kernelh::ir::Value shared = b.emitAllocShared(
            kernelh::ir::TensorType{{16}, kernelh::ir::ScalarType::F32,
                                    kernelh::ir::MemSpace::Shared});
        b.emitStore(f0, shared, {tid});
        b.emitBarrier();

        kernelh::ir::Op *loop = b.emitLoopFor(
            zero, sixteen, one, {f0},
            {kernelh::ir::Type::scalar(kernelh::ir::ScalarType::F32)});
        b.pushScope(*loop->region);
        const kernelh::ir::Value i = b.arg(0);
        const kernelh::ir::Value acc = b.arg(1);
        const kernelh::ir::Value sv = b.emitLoad(shared, {i});
        const kernelh::ir::Value sum = b.emitAdd(sv, acc);
        b.emitYield({sum});
        b.popScope();

        const kernelh::ir::Value total = loop->result();
        b.emitStore(total, in, {tid});
      });

  const kernelh::EmittedKernel k =
      kernelh::program().kernel("dsl_probe").emit();
  const kernelh::ir::Diagnostics diags = kernelh::ir::verifyModule(k.module);
  for (const kernelh::ir::Diagnostic &d : diags) {
    INFO(d.toString());
  }
  CHECK(diags.empty());
  CHECK(k.module.size() == 1);
  CHECK(k.module.kernels[0]->name == "dsl_probe");
  CHECK(k.atomText.find("atom.kernel @dsl_probe") != std::string::npos);
  CHECK(k.atomText.find("atom.alloc_shared") != std::string::npos);
  CHECK(k.atomText.find("atom.barrier") != std::string::npos);

  // The registry exposes built-in and user templates.
  const std::vector<std::string> names = kernelh::registeredKernels();
  CHECK(std::find(names.begin(), names.end(), "vecadd") != names.end());
  CHECK(std::find(names.begin(), names.end(), "dsl_probe") != names.end());
}

TEST_CASE("dsl_irbuilder_named_values_and_scopes", "[dsl]") {
  kernelh::ir::Module module;
  auto kernel = std::make_unique<kernelh::ir::Kernel>("probe2");
  kernelh::ir::Block &body = kernel->body;
  kernelh::ir::IRBuilder b(body);

  b.bind("zero", b.emitConst(int64_t{0}, kernelh::ir::ScalarType::Index));
  CHECK(b.lookup("zero").valid());
  CHECK_FALSE(b.lookup("absent").valid());
  // A duplicate bind in the SAME scope is rejected.
  CHECK_FALSE(b.bind("zero", kernelh::ir::Value::none()));

  const kernelh::ir::Value two =
      b.named("two", b.emitConst(int64_t{2}, kernelh::ir::ScalarType::Index));
  CHECK(two == b.lookup("two"));

  // Nested scope: outer names stay visible; inner names are scoped away.
  kernelh::ir::Op *loop = b.emitLoopFor(two, two, two, {}, {});
  b.pushScope(*loop->region);
  CHECK(b.lookup("zero").valid()); // dominance: outer visible inside
  b.bind("inner", b.arg(0));
  CHECK(b.lookup("inner").valid());
  b.popScope();
  CHECK_FALSE(b.lookup("inner").valid()); // released on scope exit

  // Scope hygiene errors are caught loudly.
  kernelh::ir::Block stray;
  CHECK_THROWS_AS(b.pushScope(stray), std::invalid_argument);
  CHECK_THROWS_AS(b.popScope(), std::logic_error); // root cannot be popped

  module.kernels.push_back(std::move(kernel));
  // The empty loop body is not verifiable; this case only tests name/scope
  // semantics, so no verification is performed here.
}
