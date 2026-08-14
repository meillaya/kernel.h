// Occupancy report generator (plan todo 36).
//
// Produces a per-kernel+config occupancy report from the todo-33 resources
// (vgpr_count / lds_bytes) and the todo-10 device-limits table (Oracle B5:
// waves_per_cu and reg_file_per_cu come ONLY from the device table):
//
//   VGPR-limited waves     = floor(reg_file_per_cu / (vgpr_count x
//                            wavefront_size)) capped by waves_per_cu
//   LDS-limited blocks     = floor(lds_bytes_per_workgroup / lds_bytes)
//   thread-limited blocks  = floor(max_threads_per_workgroup / threads)
//   achieved_waves         = min of the three available bounds
//   occupancy_pct          = achieved_waves / waves_per_cu x 100
//   limiter                = whichever bound achieved_waves came from
//                            (vgpr | lds | threads)
//
// The formula is pinned EXACTLY as Oracle B5 states it (the plan's unit tests
// hand-compute these values - see test/unit/occupancy_test.cpp); a bound whose
// inputs are unknown (vgpr_count <= 0, lds_bytes <= 0, threads <= 0) is "not
// limiting" and excluded from the min.
//
// MEASURED OCCUPANCY: when a loaded kernel function handle (hipFunction_t /
// CUfunction as void*) is provided, the report ALSO calls the todo-32 weak
// hook query_occupancy_max_blocks_runtime() (the launch-verifier contract) ->
// hipOccupancyMaxActiveBlocksPerMultiprocessor. On the local ROCm 7.2.3 box
// that API returns hipErrorInvalidDeviceFunction (98) for EVERY kernel
// (recorded in learnings todo 32), so the hook returns -1 and the report
// carries measured_pct = 0 with a measured_note - the graceful fallback the
// [gpu] validation test documents. Cross-target (nvptx-sm_80) occupancy is
// ANALYTIC only (no local NVIDIA card; Metis G30).
//
// kernelh_compiler stays MLIR-free and HIP-free: this file uses only the
// device-limits table + the weak occupancy hook (defined in launch_verifier.cpp
// in this same library). The vendor API call itself lives in the todo-23
// runtime (lib/runtime/query_occupancy.cpp, strong override).
//
// Consumers: `kernelh bench --occupancy` (todo 36) and the todo-37
// `kernelh explore` occupancy section (exposed here; the explore wiring lands
// in todo 37 - Oracle R2-B2).

#ifndef KERNELH_COMPILER_OCCUPANCY_HPP
#define KERNELH_COMPILER_OCCUPANCY_HPP

#include "kernelh/compiler/device_limits.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace kernelh::compiler {

// Which resource bound produced achieved_waves.
enum class OccupancyLimiter {
  kNone,   // no bound could be computed (all inputs unknown)
  kVgpr,   // register-file bound (VGPR-limited waves)
  kLds,    // shared-memory bound (LDS-limited blocks)
  kThreads // workgroup-size bound (thread-limited blocks)
};

// The inputs to the occupancy computation (todo-36 contract: kernel name +
// config/target + todo-33 resources + todo-10 DeviceLimits).
struct OccupancyInput {
  std::string kernel;              // kernel name
  std::string target;              // device-limits token ("gfx1101" / "sm_80")
  std::string config;              // human-readable config (e.g. "m=512,...")
  int vgpr_count = 0;              // todo-33 resources.json vgpr_count
  long lds_bytes = 0;              // todo-33 resources.json lds_bytes
  int threads_per_workgroup = 0;   // launch workgroup size (block)
};

// One per-kernel+config occupancy report (the occupancy.{json,md} payload).
struct OccupancyReport {
  std::string kernel;
  std::string target;
  std::string config;
  int vgpr_count = 0;
  long lds_bytes = 0;
  int threads_per_workgroup = 0;

  // The three Oracle B5 bounds (0 when the bound is not computable).
  int vgpr_limited_waves = 0;    // floor(reg_file_per_cu/(vgpr x wavefront))
  int lds_limited_blocks = 0;    // floor(lds_bytes_per_workgroup / lds_bytes)
  int thread_limited_blocks = 0; // floor(max_threads_per_workgroup / threads)

  // Theoretical occupancy: min of the three available bounds + percentage.
  int achieved_waves = 0;        // min(...); the report's "waves per CU"
  double theoretical_pct = 0.0;  // achieved_waves / waves_per_cu x 100
  OccupancyLimiter limiter = OccupancyLimiter::kNone;

  // Measured occupancy via the todo-32 weak hook (when a function handle was
  // supplied AND the vendor API worked). measured_blocks = -1 means the API
  // was unavailable/failed; measured_pct is then 0 and measured_note says why.
  int measured_blocks = -1;
  double measured_pct = 0.0; // blocks x waves_per_block / waves_per_cu x 100
  std::string measured_note;
};

// Computes the occupancy report. `function` is the loaded kernel handle
// (hipFunction_t / CUfunction as void*); pass nullptr for analytic-only. The
// analytic part never fails (bounds with unknown inputs are skipped); measured
// occupancy degrades gracefully when the todo-32 hook returns -1.
OccupancyReport occupancy_report(const OccupancyInput &input,
                                 const DeviceLimits &limits,
                                 void *function = nullptr);

// Hand-rolled writers (the project has no JSON dependency; see
// resource_reporter.cpp for the same pattern).
//
// write_occupancy_json writes a SINGLE report as a flat object with the
// top-level fields {theoretical_waves, theoretical_pct, limiter,
// measured_pct, ...} - the todo-36 acceptance shape (jq
// '.theoretical_pct >= 0 and .theoretical_pct <= 100' applies to it).
bool write_occupancy_json(const std::filesystem::path &file,
                          const OccupancyReport &report);

// Suite-mode form: several reports as {"reports": [ {...}, ... ]}.
bool write_occupancy_json_all(const std::filesystem::path &file,
                              const std::vector<OccupancyReport> &reports);

// Markdown report: the pinned table (kernel, config, vgpr, lds, theoretical %,
// measured %, limiter) plus a notes section. `measured %` shows "n/a" when the
// vendor API was unavailable.
bool write_occupancy_markdown(const std::filesystem::path &file,
                              const std::vector<OccupancyReport> &reports);

} // namespace kernelh::compiler

#endif // KERNELH_COMPILER_OCCUPANCY_HPP
