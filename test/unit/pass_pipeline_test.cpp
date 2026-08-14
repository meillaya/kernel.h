// Default pipeline unit tests (plan todo 19): the pinned order literal, the
// registered-pass precondition for every pinned name, an end-to-end
// runDefaultPipeline over the fused_gemm_hl fixture that verifies clean, and
// the unknown-pass error path (a pipeline with a missing pass name fails with
// a clean std::expected error naming it - the same error the default-pipeline
// runner surfaces if a pass file ever failed to self-register).
//
// The order vector is pinned by Metis G3 (canonicalize, cse, fuse, tile,
// promote-shared, vectorize, unroll) - the test asserts it verbatim, and the
// lit pipeline_order.test locks the observable per-stage markers.

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/verifier.hpp"
#include "kernelh/passes/pipeline.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string readFixture(const char *name) {
  std::ifstream in(std::string(KERNELH_SOURCE_DIR) + "/test/fixtures/passes/" +
                   name);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

} // namespace

TEST_CASE("pass_pipeline_default_order_literal", "[unit][pass]") {
  const std::vector<std::string> order = kernelh::passes::defaultPipelineOrder();
  const std::vector<std::string> pinned = {"canonicalize", "cse", "fuse",
                                           "tile", "promote-shared",
                                           "vectorize", "unroll"};
  REQUIRE(order.size() == pinned.size());
  CHECK(order == pinned);

  // The runner executes through the GLOBAL registry: every pinned name must
  // actually be registered, or runDefaultPipeline would fail at runtime.
  const std::vector<std::string> registered = kernelh::ir::registeredPasses();
  for (const std::string &name : order) {
    CHECK(std::find(registered.begin(), registered.end(), name) !=
          registered.end());
  }
}

TEST_CASE("pass_pipeline_run_default_end_to_end", "[unit][pass]") {
  auto parsed = kernelh::ir::parseModule(readFixture("fused_gemm_hl.atom"));
  REQUIRE(parsed.has_value());
  kernelh::ir::Module &module = *parsed;
  const auto result = kernelh::passes::runDefaultPipeline(module);
  REQUIRE(result.has_value());
  CHECK(kernelh::ir::verifyModule(module).empty());
}

TEST_CASE("pass_pipeline_unknown_pass_error", "[unit][pass]") {
  auto parsed = kernelh::ir::parseModule(readFixture("fused_gemm_hl.atom"));
  REQUIRE(parsed.has_value());
  const auto result =
      kernelh::ir::runPasses(*parsed, "canonicalize,no-such-pass,unroll");
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().find("unknown pass 'no-such-pass'") !=
        std::string::npos);
}
