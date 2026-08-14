// kernel.h "atom" IR - MLIR-like text printer (plan todo 9).
//
// Determinism requirements (later todos FileCheck the output):
//   - value names are "%vN" with per-scope numbering (block args first, then
//     op results in textual order; each scope restarts at 0),
//   - kernel attributes print in a FIXED order (tile, map_blocks, map_warps,
//     use_shared_memory, vector_width, unroll, shape),
//   - the shape map prints key-sorted (std::map iteration order),
//   - float constants print via std::to_chars shortest round-trip form.
//
// Format summary (stable contract for todos 13-19 FileCheck tests):
//   atom.kernel @name(%v0: tensor<16x16xf32, global>) {
//     tile = [128, 128, 32]
//     map_blocks = true
//     map_warps = false
//     use_shared_memory = true
//     vector_width = 4
//     unroll = 1
//     shape = {m = 1024, n = 1024}
//     %v2 = atom.const 0 : index
//     %v3 = atom.loop_for %v2 to %v4 step %v5 iter_args(%v6) -> (f32) {
//       ^bb0(%v7: index, %v8: f32):
//         %v9 = atom.add %v7, %v8 : index
//         atom.yield %v9 : f32
//     }
//   }
// There is no module wrapper: a .atom file is a concatenation of kernel
// blocks. Empty modules print as the empty string.

#ifndef KERNELH_IR_PRINTER_HPP
#define KERNELH_IR_PRINTER_HPP

#include "kernelh/ir/module.hpp"

#include <string>

namespace kernelh::ir {

// Prints the module as deterministic ".atom" text.
std::string printModule(const Module &module);

} // namespace kernelh::ir

#endif // KERNELH_IR_PRINTER_HPP
