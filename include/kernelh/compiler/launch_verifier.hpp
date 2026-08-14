// Pre-launch launch-config verifier (plan todo 32).
//
// DISTINCT from todo 9's structural IR verifier (kernelh/ir/verifier.hpp):
// that one checks the atom IR's SSA/type/barrier structure, this one checks a
// concrete LAUNCH CONFIG (workgroup size, grid shape, static shared bytes)
// against the per-device hardware limits table (plan todo 10,
// kernelh/compiler/device_limits.hpp) BEFORE a kernel is launched. It runs at
// launch time in the runtime (lib/runtime/kernel_launch.cpp) and is also
// consumed by the autotuner (plan todo 34) to prune candidate configs.
//
// All violations are collected and returned in ONE diagnostic string (one
// violation per line) - never an early exit on the first failure.
//
// OCCUPANCY HOOK CONTRACT (weak symbol, mirrors the todo-10
// query_cu_count_runtime() contract in device_limits.hpp)
// ----------------------------------------------------------
// When a loaded kernel function handle (hipFunction_t / CUfunction, passed as
// void*) is available, the verifier prefers the vendor occupancy API
// (hipOccupancyMaxActiveBlocksPerMultiprocessor). lib/compiler never links
// HIP/ROCm, so the API call lives behind the weak symbol
// query_occupancy_max_blocks_runtime(), whose default definition here returns
// -1 ("no GPU runtime linked / function unusable"). The todo-23 runtime
// library provides a STRONG definition (lib/runtime/query_occupancy.cpp) that
// calls the HIP API; the override is resolved at FINAL LINK time. When the
// hook returns -1 the verifier falls back to the analytic estimate
// (threads/wavefront vs the table's waves_per_cu).
//
// LAUNCH CONFIG TYPE NOTE: grid dimensions are stored as int (a grid extent
// beyond INT_MAX is already beyond every known target's grid limit).

#ifndef KERNELH_COMPILER_LAUNCH_VERIFIER_HPP
#define KERNELH_COMPILER_LAUNCH_VERIFIER_HPP

#include "kernelh/compiler/device_limits.hpp"

#include <expected>
#include <string>

namespace kernelh::compiler {

// A concrete launch configuration, as submitted to the runtime.
struct LaunchConfig {
  int threads_per_workgroup = 0; // block/workgroup thread count
  int grid_x = 0;                // grid extents {x, y, z}
  int grid_y = 0;
  int grid_z = 0;
  size_t static_shared_bytes = 0; // static shared/LDS bytes for the workgroup
};

// Weak hook - default returns -1 ("occupancy API unavailable"). Plan todo 23
// supplies the strong override calling hipOccupancyMaxActiveBlocksPerMultiprocessor;
// see the contract at the top of this header.
//
//   function: loaded kernel entry (hipFunction_t / CUfunction as void*), may
//             be nullptr (fall back to analytic); garbage pointers are the
//             caller's responsibility (the runtime only passes real handles).
//   block_size: workgroup thread count the occupancy API is asked about.
//   shared_bytes: static + dynamic shared bytes for the workgroup.
// Returns the number of resident blocks per multiprocessor (>= 0), or -1 when
// the API is unavailable or failed (caller falls back to the analytic
// estimate). Never throws.
int query_occupancy_max_blocks_runtime(void *function, int block_size,
                                       size_t shared_bytes)
    __attribute__((weak));

// Pre-launch device-limit + occupancy checks. Returns std::nullopt (success)
// when the config is launchable, otherwise std::unexpected carrying a
// diagnostic that lists EVERY violation on its own line.
class LaunchVerifier {
public:
  // Basic form: analytic occupancy only (no loaded kernel function).
  std::expected<void, std::string> verify(const LaunchConfig &config,
                                          const kernelh::DeviceLimits &limits) const;

  // Form with a loaded kernel function handle (hipFunction_t / CUfunction as
  // void*): occupancy goes through the weak query_occupancy_max_blocks_runtime()
  // hook (vendor occupancy API when available) instead of the analytic
  // estimate. `function` may be nullptr - then this behaves like the basic
  // form.
  std::expected<void, std::string> verify(const LaunchConfig &config,
                                          const kernelh::DeviceLimits &limits,
                                          void *function) const;

private:
  std::expected<void, std::string> verify_impl(const LaunchConfig &config,
                                               const kernelh::DeviceLimits &limits,
                                               void *function) const;
};

} // namespace kernelh::compiler

#endif // KERNELH_COMPILER_LAUNCH_VERIFIER_HPP
