// lib/passes/pipeline.cpp - the DEFAULT pass pipeline definition (plan todo
// 19).
//
// The canonical default pipeline order (Metis G3 pin - do not reorder):
//
//   canonicalize, cse, fuse, tile, promote-shared, vectorize, unroll
//
// This file is NOT a pass: it defines the order vector and a runner over the
// GLOBAL pass registry (the same registry the pass .cpp files in this
// directory self-register into - see lib/passes/CMakeLists.txt for the
// self-registration contract). It is picked up by the existing GLOB with zero
// CMake edits; it registers nothing and never appears in --run-passes
// listings.
//
// Pipeline order is set HERE (the single source; todos 24/34 reuse it), never
// by the individual passes - each pass is order-independent and the runner
// simply executes the pinned sequence. An unknown pass name (e.g. a pass
// file that failed to self-register) surfaces as a std::expected error naming
// the missing pass, exactly like --run-passes.

#include "kernelh/passes/pipeline.hpp"

#include "kernelh/ir/pass_manager.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace kernelh::passes {

std::vector<std::string> defaultPipelineOrder() {
  // Metis G3 pin - do not reorder. The order-lock tests (unit + lit) assert
  // this literal verbatim.
  return {"canonicalize", "cse", "fuse", "tile",
          "promote-shared", "vectorize", "unroll"};
}

std::expected<void, std::string> runDefaultPipeline(
    kernelh::ir::Module &module, std::string_view dumpDir) {
  std::string pipeline;
  const std::vector<std::string> order = defaultPipelineOrder();
  for (size_t i = 0; i < order.size(); ++i) {
    if (i != 0) {
      pipeline += ",";
    }
    pipeline += order[i];
  }
  return kernelh::ir::runPasses(module, pipeline, dumpDir);
}

} // namespace kernelh::passes
