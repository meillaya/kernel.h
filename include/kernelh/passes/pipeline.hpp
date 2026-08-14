// kernel.h "atom" IR - default pass pipeline definition (plan todo 19).
//
// The canonical default pipeline order (Metis G3 pin - do not reorder):
//
//   canonicalize, cse, fuse, tile, promote-shared, vectorize, unroll
//
// This header is the SINGLE source of order: lib/passes/pipeline.cpp returns
// it verbatim, the pipeline-order lock tests assert it literally, and todos
// 24/34 (kernelh compile / autotune) reuse it for the production path. The
// implementation lives in lib/passes/ (kernelh_passes) but is NOT itself a
// pass: it defines no PassFn and self-registers nothing.

#ifndef KERNELH_PASSES_PIPELINE_HPP
#define KERNELH_PASSES_PIPELINE_HPP

#include "kernelh/ir/module.hpp"

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace kernelh::passes {

// The canonical default pipeline order (Metis G3 pin - do not reorder):
// canonicalize, cse, fuse, tile, promote-shared, vectorize, unroll.
std::vector<std::string> defaultPipelineOrder();

// Runs the default pipeline through the GLOBAL pass registry in exactly the
// pinned order. Every registered pass must exist; an unknown name is a
// std::expected error (same semantics and error text as --run-passes).
// When `dumpDir` is non-empty the per-pass dumps are written: 00-input.atom
// plus 01-canonicalize.atom ... 07-unroll.atom (one per pass, zero-padded
// index - the dump names prove the order).
std::expected<void, std::string> runDefaultPipeline(kernelh::ir::Module &module,
                                                    std::string_view dumpDir =
                                                        "");

} // namespace kernelh::passes

#endif // KERNELH_PASSES_PIPELINE_HPP
