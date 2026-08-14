// kernel.h "atom" IR - module / kernel containers (plan todo 9).
//
// A Module holds one or more KernelOps. Each kernel carries:
//   - its schedule attributes (tile, map_blocks, map_warps,
//     use_shared_memory, vector_width, unroll) - exactly the shape the DSL
//     builder produces,
//   - shape attributes (std::map<string, i64>, sorted for deterministic
//     printing),
//   - a body block whose block ARGUMENTS are the kernel parameters (global
//     tensors / scalars passed at launch).

#ifndef KERNELH_IR_MODULE_HPP
#define KERNELH_IR_MODULE_HPP

#include "kernelh/ir/ops.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace kernelh::ir {

// Kernel schedule attributes (pinned field set). vector_width and unroll
// default to 1; tile may be empty (no tiling).
struct ScheduleAttrs {
  std::vector<int64_t> tile;
  bool map_blocks = false;
  bool map_warps = false;
  bool use_shared_memory = false;
  int64_t vector_width = 1;
  int64_t unroll = 1;
};

struct Kernel {
  std::string name;
  ScheduleAttrs schedule;
  // Problem-dimension metadata (e.g. {m, n, k}); kept for the emitter.
  std::map<std::string, int64_t> shape;
  Block body; // block args = kernel parameters

  explicit Kernel(std::string kernelName) : name(std::move(kernelName)) {}
};

struct Module {
  std::vector<std::unique_ptr<Kernel>> kernels;

  bool empty() const { return kernels.empty(); }
  // Number of kernels in the module.
  size_t size() const { return kernels.size(); }
};

// Total static shared-memory (LDS) bytes referenced by every atom.alloc_shared
// in the kernel across ALL scopes. Size is compile-time (shape * elem size).
int64_t kernelTotalSharedBytes(const Kernel &kernel);

} // namespace kernelh::ir

#endif // KERNELH_IR_MODULE_HPP
