// Parser unit tests for the atom IR (plan todo 9).
//
// Round-trip guarantee: print(parse(print(M))) == print(M) on the corpus.
// Malformed input must fail with an error carrying line/column info.

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using kernelh::ir::Module;
using kernelh::ir::parseModule;
using kernelh::ir::printModule;
using kernelh::ir::verifyModule;

using kernelh::test::buildCorpusModule;
using kernelh::test::buildGemm;
using kernelh::test::buildMinimal;
using kernelh::test::buildScan;
using kernelh::test::buildSoftmax;
using kernelh::test::buildVecadd;

namespace {

// Asserts the round-trip property for a single module.
void checkRoundTrip(const Module &module) {
  const std::string text = printModule(module);
  auto parsed = parseModule(text);
  REQUIRE(parsed.has_value());
  const std::string reparsed = printModule(*parsed);
  CHECK(reparsed == text);
}

bool parseFails(std::string_view text, const std::string &needle) {
  auto parsed = parseModule(text);
  if (parsed.has_value()) {
    return false;
  }
  return parsed.error().find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("ir_parser_roundtrip_minimal", "[unit][ir]") {
  checkRoundTrip(buildMinimal());
}

TEST_CASE("ir_parser_roundtrip_vecadd", "[unit][ir]") {
  checkRoundTrip(buildVecadd());
}

TEST_CASE("ir_parser_roundtrip_gemm", "[unit][ir]") {
  checkRoundTrip(buildGemm());
}

TEST_CASE("ir_parser_roundtrip_softmax", "[unit][ir]") {
  checkRoundTrip(buildSoftmax());
}

TEST_CASE("ir_parser_roundtrip_scan", "[unit][ir]") {
  checkRoundTrip(buildScan());
}

TEST_CASE("ir_parser_roundtrip_all_corpus", "[unit][ir]") {
  checkRoundTrip(buildCorpusModule());
}

TEST_CASE("ir_parser_parsed_corpus_verifies", "[unit][ir]") {
  auto parsed = parseModule(printModule(buildCorpusModule()));
  REQUIRE(parsed.has_value());
  CHECK(verifyModule(*parsed).empty());
}

TEST_CASE("ir_parser_parses_printed_loop", "[unit][ir]") {
  const std::string text =
      "atom.kernel @k {\n"
      "  tile = []\n"
      "  map_blocks = false\n"
      "  map_warps = false\n"
      "  use_shared_memory = false\n"
      "  vector_width = 1\n"
      "  unroll = 1\n"
      "  shape = {}\n"
      "  %v0 = atom.const 0 : index\n"
      "  %v1 = atom.const 8 : index\n"
      "  %v2 = atom.const 1 : index\n"
      "  %v3 = atom.const 0 : index\n"
      "  %v4 = atom.loop_for %v0 to %v1 step %v2 iter_args(%v3) -> (index) {\n"
      "    ^bb0(%v5: index, %v6: index):\n"
      "      %v7 = atom.add %v5, %v6 : index\n"
      "      atom.yield %v7 : index\n"
      "  }\n"
      "}\n";
  auto parsed = parseModule(text);
  REQUIRE(parsed.has_value());
  CHECK(verifyModule(*parsed).empty());
}

TEST_CASE("ir_parser_empty_module", "[unit][ir]") {
  auto parsed = parseModule("");
  REQUIRE(parsed.has_value());
  CHECK(parsed->empty());
  CHECK(printModule(*parsed) == "");
}

TEST_CASE("ir_parser_accepts_comments", "[unit][ir]") {
  auto parsed = parseModule("// a comment\natom.kernel @c {\n"
                            "  tile = []\n"
                            "  map_blocks = false\n"
                            "  map_warps = false\n"
                            "  use_shared_memory = false\n"
                            "  vector_width = 1\n"
                            "  unroll = 1\n"
                            "  shape = {}\n"
                            "  %v0 = atom.const 1 : index\n"
                            "}\n");
  REQUIRE(parsed.has_value());
  CHECK(parsed->size() == 1);
}

TEST_CASE("ir_parser_multiple_kernels", "[unit][ir]") {
  auto parsed = parseModule(printModule(buildCorpusModule()));
  REQUIRE(parsed.has_value());
  CHECK(parsed->size() == 5);
}

// ---------------------------------------------------------------------------
// Error cases (line/col reporting)
// ---------------------------------------------------------------------------

TEST_CASE("ir_parser_error_truncated_op", "[unit][ir]") {
  const std::string text =
      "atom.kernel @k {\n"
      "  tile = []\n"
      "  map_blocks = false\n"
      "  map_warps = false\n"
      "  use_shared_memory = false\n"
      "  vector_width = 1\n"
      "  unroll = 1\n"
      "  shape = {}\n"
      "  %v0 = atom.add %v1\n"; // truncated: no type, EOF
  const auto parsed = parseModule(text);
  REQUIRE_FALSE(parsed.has_value());
  CHECK(parsed.error().find("line") != std::string::npos);
  CHECK(parsed.error().find("col") != std::string::npos);
}

TEST_CASE("ir_parser_error_unknown_op", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "  %v0 = atom.frobnicate 1 : f32\n"
                   "}\n",
                   "unknown atom operation"));
}

TEST_CASE("ir_parser_error_use_before_def_dominance", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "  %v0 = atom.add %v1, %v1 : f32\n"
                   "  %v1 = atom.const 1.0 : f32\n"
                   "}\n",
                   "dominance"));
}

TEST_CASE("ir_parser_error_undefined_name", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "  %v0 = atom.add %v99, %v99 : f32\n"
                   "}\n",
                   "undefined value"));
}

TEST_CASE("ir_parser_error_duplicate_name", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "  %v0 = atom.const 1 : index\n"
                   "  %v0 = atom.const 2 : index\n"
                   "}\n",
                   "duplicate value name"));
}

TEST_CASE("ir_parser_error_bad_type", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "  %v0 = atom.const 1 : u64\n"
                   "}\n",
                   "unknown type"));
}

TEST_CASE("ir_parser_error_invalid_integer_literal", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "  %v0 = atom.const 1.5 : i32\n"
                   "}\n",
                   "invalid integer literal"));
}

TEST_CASE("ir_parser_error_missing_brace", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "  %v0 = atom.const 1 : index\n", // missing '}'
                   "unexpected end of input"));
}

TEST_CASE("ir_parser_error_unknown_attr", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  bogus_attr = 1\n"
                   "  shape = {}\n"
                   "}\n",
                   "unknown kernel attribute"));
}

TEST_CASE("ir_parser_error_garbage_after_module", "[unit][ir]") {
  CHECK(parseFails("atom.kernel @k {\n"
                   "  tile = []\n"
                   "  map_blocks = false\n"
                   "  map_warps = false\n"
                   "  use_shared_memory = false\n"
                   "  vector_width = 1\n"
                   "  unroll = 1\n"
                   "  shape = {}\n"
                   "}\n"
                   "garbage",
                   "expected 'atom'"));
}

TEST_CASE("ir_parser_error_line_col_reported", "[unit][ir]") {
  // The offending token is at a specific line/column; the error must name it.
  const std::string text =
      "atom.kernel @k {\n"
      "  tile = []\n"
      "  map_blocks = false\n"
      "  map_warps = false\n"
      "  use_shared_memory = false\n"
      "  vector_width = 1\n"
      "  unroll = 1\n"
      "  shape = {}\n"
      "  %v0 = atom.const 1 : f32\n"
      "  %v1 = atom.bogus %v0\n" // line 10
      "}\n";
  const auto parsed = parseModule(text);
  REQUIRE_FALSE(parsed.has_value());
  CHECK(parsed.error().find("line 10") != std::string::npos);
}

TEST_CASE("ir_parser_error_rejects_negative_shape_ok_syntax", "[unit][ir]") {
  // Negative ints are lexed correctly in attrs and rejected by the verifier,
  // not the parser.
  const std::string text =
      "atom.kernel @k {\n"
      "  tile = []\n"
      "  map_blocks = false\n"
      "  map_warps = false\n"
      "  use_shared_memory = false\n"
      "  vector_width = 1\n"
      "  unroll = 1\n"
      "  shape = {n = -4}\n"
      "  %v0 = atom.const -1 : i32\n"
      "}\n";
  auto parsed = parseModule(text);
  REQUIRE(parsed.has_value());
  REQUIRE_FALSE(verifyModule(*parsed).empty());
}
