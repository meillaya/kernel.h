// Printer unit tests for the atom IR (plan todo 9).
//
// Goldens are pinned as exact strings; the corpus kernels must also print
// byte-for-byte equal to the committed .atom fixtures under test/fixtures
// (KERNELH_SOURCE_DIR is baked in by CMake).

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::CmpPred;
using kernelh::ir::IdDim;
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

using kernelh::test::buildCorpusModule;
using kernelh::test::buildGemm;
using kernelh::test::buildMinimal;
using kernelh::test::buildScan;
using kernelh::test::buildSoftmax;
using kernelh::test::buildVecadd;

namespace {

std::string readFixture(const std::string &name) {
  std::ifstream in(std::string(KERNELH_SOURCE_DIR) + "/test/fixtures/" + name);
  if (!in) {
    return "<unreadable fixture " + name + ">";
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

} // namespace

TEST_CASE("ir_printer_minimal_golden", "[unit][ir]") {
  const std::string expected =
      "atom.kernel @minimal {\n"
      "  tile = []\n"
      "  map_blocks = false\n"
      "  map_warps = false\n"
      "  use_shared_memory = false\n"
      "  vector_width = 1\n"
      "  unroll = 1\n"
      "  shape = {}\n"
      "  %v0 = atom.const 0.5 : f32\n"
      "  %v1 = atom.const 0.25 : f32\n"
      "  %v2 = atom.add %v0, %v1 : f32\n"
      "}\n";
  CHECK(printModule(buildMinimal()) == expected);
}

TEST_CASE("ir_printer_loop_layout_golden", "[unit][ir]") {
  Module module;
  auto k = std::make_unique<Kernel>("loopk");
  k->schedule.tile = {8};
  k->shape["n"] = 64;
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, Type::scalar(ScalarType::Index))->result();
  const Value one = builder::makeConst(b, 1, Type::scalar(ScalarType::Index))->result();
  const Value init = builder::makeConst(b, 0.0, Type::scalar(ScalarType::F32))->result();
  kernelh::ir::Op *loop = builder::makeLoopFor(
      b, z, one, one, {init}, {Type::scalar(ScalarType::F32)});
  Block &body = *loop->region;
  const Value acc = Value::blockArg(&body, 1);
  builder::makeBin(body, OpKind::Add, acc, init);
  builder::makeYield(body, {acc});
  module.kernels.push_back(std::move(k));

  const std::string expected =
      "atom.kernel @loopk {\n"
      "  tile = [8]\n"
      "  map_blocks = false\n"
      "  map_warps = false\n"
      "  use_shared_memory = false\n"
      "  vector_width = 1\n"
      "  unroll = 1\n"
      "  shape = {n = 64}\n"
      "  %v0 = atom.const 0 : index\n"
      "  %v1 = atom.const 1 : index\n"
      "  %v2 = atom.const 0 : f32\n"
      "  %v3 = atom.loop_for %v0 to %v1 step %v1 iter_args(%v2) -> (f32) {\n"
      "    ^bb0(%v4: index, %v5: f32):\n"
      "      %v6 = atom.add %v5, %v2 : f32\n"
      "      atom.yield %v5 : f32\n"
      "  }\n"
      "}\n";
  CHECK(printModule(module) == expected);
}

TEST_CASE("ir_printer_kernel_args_golden", "[unit][ir]") {
  Module module;
  auto k = std::make_unique<Kernel>("argk");
  Block &b = k->body;
  b.args.push_back(BlockArg{
      Type::tensor(TensorType{{1024}, ScalarType::F32, MemSpace::Global})});
  builder::makeConst(b, 1, Type::scalar(ScalarType::Index));
  module.kernels.push_back(std::move(k));

  const std::string expected =
      "atom.kernel @argk(%v0: tensor<1024xf32, global>) {\n"
      "  tile = []\n"
      "  map_blocks = false\n"
      "  map_warps = false\n"
      "  use_shared_memory = false\n"
      "  vector_width = 1\n"
      "  unroll = 1\n"
      "  shape = {}\n"
      "  %v1 = atom.const 1 : index\n"
      "}\n";
  CHECK(printModule(module) == expected);
}

TEST_CASE("ir_printer_value_numbering_per_scope", "[unit][ir]") {
  // Every nested scope restarts its %vN numbering at 0.
  Module module;
  auto k = std::make_unique<Kernel>("nested");
  Block &b = k->body;
  const Value z = builder::makeConst(b, 0, Type::scalar(ScalarType::Index))->result();
  const Value one = builder::makeConst(b, 1, Type::scalar(ScalarType::Index))->result();
  kernelh::ir::Op *outer = builder::makeLoopFor(b, z, one, one, {}, {});
  Block &outerBody = *outer->region;
  const Value oi = Value::blockArg(&outerBody, 0);
  kernelh::ir::Op *inner = builder::makeLoopFor(outerBody, z, oi, one, {}, {});
  Block &innerBody = *inner->region;
  builder::makeConst(innerBody, 1.0, Type::scalar(ScalarType::F32));
  builder::makeYield(innerBody, {});
  builder::makeYield(outerBody, {});
  module.kernels.push_back(std::move(k));

  const std::string text = printModule(module);
  // Kernel body scope: %v0..%v1 (consts), then the result-less outer loop.
  CHECK(text.find("  %v0 = atom.const 0 : index") != std::string::npos);
  CHECK(text.find("  %v1 = atom.const 1 : index") != std::string::npos);
  CHECK(text.find("  atom.loop_for %v0 to %v1 step %v1 iter_args() -> () {") !=
        std::string::npos);
  // The outer loop body restarts its counter but SKIPS the enclosing-scope
  // names %v0/%v1: induction lands on %v2.
  CHECK(text.find("    ^bb0(%v2: index):") != std::string::npos);
  CHECK(text.find("      atom.loop_for %v0 to %v2 step %v1 iter_args() -> () {") !=
        std::string::npos);
  // The inner loop body skips %v0..%v2 too: induction %v3, const %v4.
  CHECK(text.find("        ^bb0(%v3: index):") != std::string::npos);
  CHECK(text.find("          %v4 = atom.const 1 : f32") != std::string::npos);
}

TEST_CASE("ir_printer_deterministic", "[unit][ir]") {
  const Module module = buildCorpusModule();
  const std::string first = printModule(module);
  const std::string second = printModule(module);
  CHECK(first == second);
}

TEST_CASE("ir_printer_float_shortest_round_trip", "[unit][ir]") {
  Module module;
  auto k = std::make_unique<Kernel>("floats");
  Block &b = k->body;
  builder::makeConst(b, 1.0, Type::scalar(ScalarType::F32));   // "1"
  builder::makeConst(b, -0.5, Type::scalar(ScalarType::F32));  // "-0.5"
  builder::makeConst(b, 3.141592653589793, Type::scalar(ScalarType::F32));
  builder::makeConst(b, 1.0e20, Type::scalar(ScalarType::F32)); // "1e+20"
  module.kernels.push_back(std::move(k));

  const std::string text = printModule(module);
  CHECK(text.find("%v0 = atom.const 1 : f32") != std::string::npos);
  CHECK(text.find("%v1 = atom.const -0.5 : f32") != std::string::npos);
  CHECK(text.find("%v2 = atom.const 3.141592653589793 : f32") !=
        std::string::npos);
  CHECK(text.find("%v3 = atom.const 1e+20 : f32") != std::string::npos);
}

TEST_CASE("ir_printer_attrs_fixed_order", "[unit][ir]") {
  Module module;
  auto k = std::make_unique<Kernel>("attrs");
  k->schedule.tile = {128, 128, 32};
  k->schedule.map_blocks = true;
  k->schedule.map_warps = true;
  k->schedule.use_shared_memory = true;
  k->schedule.vector_width = 4;
  k->schedule.unroll = 2;
  k->shape["k"] = 1024;
  k->shape["m"] = 512;
  k->shape["n"] = 1024;
  module.kernels.push_back(std::move(k));

  const std::string text = printModule(module);
  const std::string expected =
      "atom.kernel @attrs {\n"
      "  tile = [128, 128, 32]\n"
      "  map_blocks = true\n"
      "  map_warps = true\n"
      "  use_shared_memory = true\n"
      "  vector_width = 4\n"
      "  unroll = 2\n"
      "  shape = {k = 1024, m = 512, n = 1024}\n"
      "}\n";
  CHECK(text == expected);
}

TEST_CASE("ir_printer_shape_map_sorted", "[unit][ir]") {
  // std::map iteration = key-sorted output regardless of insertion order.
  Module module;
  auto k = std::make_unique<Kernel>("sorted");
  k->shape["zeta"] = 3;
  k->shape["alpha"] = 1;
  k->shape["m"] = 2;
  module.kernels.push_back(std::move(k));
  CHECK(printModule(module).find("shape = {alpha = 1, m = 2, zeta = 3}") !=
        std::string::npos);
}

TEST_CASE("ir_printer_corpus_matches_fixtures", "[unit][ir]") {
  CHECK(printModule(buildMinimal()) == readFixture("minimal.atom"));
  CHECK(printModule(buildVecadd()) == readFixture("vecadd.atom"));
  CHECK(printModule(buildGemm()) == readFixture("gemm.atom"));
  CHECK(printModule(buildSoftmax()) == readFixture("softmax.atom"));
  CHECK(printModule(buildScan()) == readFixture("scan.atom"));
}

TEST_CASE("ir_printer_multikernel_concatenation", "[unit][ir]") {
  const Module module = buildCorpusModule();
  const std::string text = printModule(module);
  // Kernels appear in insertion order, each as its own block, no wrapper.
  const std::string minimal = "atom.kernel @minimal {";
  const std::string vecadd = "atom.kernel @vecadd(";
  const std::string gemm = "atom.kernel @gemm(";
  const std::string softmax = "atom.kernel @softmax(";
  const std::string scan = "atom.kernel @scan(";
  CHECK(text.find(minimal) < text.find(vecadd));
  CHECK(text.find(vecadd) < text.find(gemm));
  CHECK(text.find(gemm) < text.find(softmax));
  CHECK(text.find(softmax) < text.find(scan));
}
