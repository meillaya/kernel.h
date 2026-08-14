// kernel.h fluent DSL: schedule builder + kernel template registry (plan
// todo 11). API is EXACTLY the user sketch:
//
//   auto k = kernelh::program()
//                .tile(128, 128, 32)          // variadic int64 tile sizes
//                .map_blocks()
//                .map_warps()
//                .use_shared_memory()
//                .vectorize(4)
//                .unroll(2)
//                .kernel("vecadd")            // named template in the registry
//                .shape({{"n", 1024}})        // problem-dimension metadata
//                .emit();                     // -> EmittedKernel
//
// There is NO general compute-expression language (plan Metis G12): each
// kernel name selects a C++ template function (KernelTemplate) in the
// registry that emits the kernel's atom IR with an ir::IRBuilder, using the
// accumulated schedule attributes and shape map. Registry entries land as
// kernels are implemented (vecadd registers NOW; gemm/softmax/layernorm/
// scan/fused_gelu/attention arrive in todos 25-31 via registerKernel).
//
// emit() invokes the selected template, builds a kernelh::ir::Module, and
// returns it together with the deterministic ".atom" text from the todo-9
// printer (print(parse(print(M))) == print(M) holds). Unknown kernel names
// throw std::runtime_error listing the registered names.
//
// All API lives in namespace kernelh (kernelh::program() per the sketch);
// the ir namespace is kernelh::ir.

#ifndef KERNELH_DSL_PROGRAM_HPP
#define KERNELH_DSL_PROGRAM_HPP

#include "kernelh/ir/builder.hpp"
#include "kernelh/ir/module.hpp"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace kernelh {

// Problem-dimension metadata attached to a kernel (key -> value, e.g.
// {"n", 1024}); printed key-sorted in the ".atom" text.
using ShapeMap = std::map<std::string, int64_t>;

// A kernel template: emits the kernel's atom IR ops into `builder`'s block
// (a fresh kernel body), reading schedule attrs and the shape map. `schedule`
// is the accumulated ScheduleAttrs from the builder chain; the template may
// honor or ignore individual fields, but the attrs ride along on the emitted
// atom.kernel verbatim.
using KernelTemplate = std::function<void(ir::IRBuilder &,
                                          const ir::ScheduleAttrs &,
                                          const ShapeMap &)>;

// ---------------------------------------------------------------------------
// Kernel template registry (implemented in lib/dsl/registry.cpp)
// ---------------------------------------------------------------------------

// Registers (or replaces) the template for `name`. Built-in templates are
// registered lazily on first emit() - explicit registration is only needed
// for user-supplied templates. Duplicate names replace the earlier entry.
void registerKernel(std::string name, KernelTemplate fn);

// All registered kernel names, sorted.
std::vector<std::string> registeredKernels();

// ---------------------------------------------------------------------------
// Emit result
// ---------------------------------------------------------------------------

struct EmittedKernel {
  ir::Module module;     // the built kernel(s); verify with ir::verifyModule
  std::string atomText;  // printModule(module) - the ".atom" text
};

// ---------------------------------------------------------------------------
// Fluent schedule builder
// ---------------------------------------------------------------------------

class ProgramBuilder {
public:
  // `.tile(BM, BN, BK)` - variadic int64 tile sizes (also 1-D, 2-D, ...).
  template <typename... Ints>
    requires(std::is_integral_v<Ints> && ...)
  ProgramBuilder &tile(Ints... dims) {
    schedule_.tile = {static_cast<int64_t>(dims)...};
    return *this;
  }

  ProgramBuilder &map_blocks() {
    schedule_.map_blocks = true;
    return *this;
  }
  ProgramBuilder &map_warps() {
    schedule_.map_warps = true;
    return *this;
  }
  ProgramBuilder &use_shared_memory() {
    schedule_.use_shared_memory = true;
    return *this;
  }
  ProgramBuilder &vectorize(int64_t width) {
    schedule_.vector_width = width;
    return *this;
  }
  ProgramBuilder &unroll(int64_t factor) {
    schedule_.unroll = factor;
    return *this;
  }
  ProgramBuilder &kernel(std::string name) {
    kernelName_ = std::move(name);
    return *this;
  }

  // `.shape({{"n", 1024}, ...})` - problem dims as an initializer list of
  // key/value pairs (entries merge into the shape map).
  ProgramBuilder &shape(
      std::initializer_list<std::pair<const std::string, int64_t>> entries) {
    shape_.insert(entries.begin(), entries.end());
    return *this;
  }
  ProgramBuilder &shape(ShapeMap entries) {
    shape_ = std::move(entries);
    return *this;
  }

  // Selects the kernel template by name and builds the module. Throws
  // std::runtime_error for an unknown kernel name (listing the registered
  // names) or when the selected template rejects the shape map.
  EmittedKernel emit();

  // --- introspection (tests, CLI) ------------------------------------------
  const ir::ScheduleAttrs &schedule() const { return schedule_; }
  const std::string &kernelName() const { return kernelName_; }
  const ShapeMap &shapeMap() const { return shape_; }

private:
  ir::ScheduleAttrs schedule_;
  std::string kernelName_;
  ShapeMap shape_;
};

// Entry point: `kernelh::program()` starts a fresh builder chain.
inline ProgramBuilder program() { return ProgramBuilder(); }

} // namespace kernelh

#endif // KERNELH_DSL_PROGRAM_HPP
