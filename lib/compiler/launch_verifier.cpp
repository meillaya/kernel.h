// Launch-config verifier implementation (plan todo 32) - see
// include/kernelh/compiler/launch_verifier.hpp for the API contract and the
// occupancy weak-hook contract.
//
// All checks collect their violations and return them in ONE diagnostic (one
// line per violation); a config is launchable only when every check passes.

#include "kernelh/compiler/launch_verifier.hpp"

#include <string>
#include <vector>

namespace kernelh::compiler {

namespace {

constexpr const char *kNameFallback = "(unknown device)";

// Human-readable target name (the table name always carries the target token,
// e.g. "AMD Navi 32 (gfx1101)"; never null in the shipped tables).
const char *name_of(const kernelh::DeviceLimits &limits) {
  return limits.name != nullptr ? limits.name : kNameFallback;
}

// Joins `violations`, one per line, into the returned diagnostic.
std::string join_violations(const std::vector<std::string> &violations) {
  std::string out;
  for (size_t i = 0; i < violations.size(); ++i) {
    if (i > 0) {
      out += '\n';
    }
    out += violations[i];
  }
  return out;
}

// Analytic occupancy estimate: wavefronts needed per workgroup
// (ceil(threads / wavefront_size)) vs the target's per-CU wavefront budget
// (waves_per_cu). Returns a violation line when occupancy would be 0, else "".
// Unknown budgets (0 fields) mean "cannot judge" - the check is skipped rather
// than fabricating a violation.
std::string analytic_occupancy_violation(const LaunchConfig &config,
                                         const kernelh::DeviceLimits &limits) {
  if (limits.waves_per_cu <= 0 || limits.wavefront_size <= 0 ||
      config.threads_per_workgroup <= 0) {
    return "";
  }
  const long waves_needed =
      (static_cast<long>(config.threads_per_workgroup) +
       limits.wavefront_size - 1) /
      limits.wavefront_size;
  if (waves_needed > limits.waves_per_cu) {
    return "occupancy would be 0 on target '" + std::string(name_of(limits)) +
           "': threads per workgroup " +
           std::to_string(config.threads_per_workgroup) + " needs " +
           std::to_string(waves_needed) +
           " wavefronts, exceeding the target's per-CU budget of " +
           std::to_string(limits.waves_per_cu) +
           " wavefronts (waves_per_cu * wavefront_size)";
  }
  return "";
}

// Occupancy check: the vendor occupancy API (via the weak
// query_occupancy_max_blocks_runtime hook) when a loaded function handle is
// available, else the analytic estimate. The hook returns -1 when the runtime
// is not linked or the query failed; both paths judge per-multiprocessor
// residency (a workgroup needs at least one resident block per CU).
std::string occupancy_violation(const LaunchConfig &config,
                                const kernelh::DeviceLimits &limits,
                                void *function) {
  if (function != nullptr && config.threads_per_workgroup > 0) {
    const int blocks = query_occupancy_max_blocks_runtime(
        function, config.threads_per_workgroup, config.static_shared_bytes);
    if (blocks >= 0) {
      // blocks * threads (resident threads per multiprocessor as reported by
      // the API) never exceeds the per-CU thread budget
      // (waves_per_cu * wavefront_size) - the API already respects the
      // hardware - so "occupancy would be 0" is exactly blocks == 0.
      if (blocks == 0) {
        return "occupancy would be 0 on target '" +
               std::string(name_of(limits)) +
               "': hipOccupancyMaxActiveBlocksPerMultiprocessor reported 0 "
               "resident blocks per multiprocessor for threads=" +
               std::to_string(config.threads_per_workgroup) + ", shared=" +
               std::to_string(config.static_shared_bytes) + " bytes";
      }
      return "";
    }
  }
  return analytic_occupancy_violation(config, limits);
}

} // namespace

std::expected<void, std::string>
LaunchVerifier::verify(const LaunchConfig &config,
                       const kernelh::DeviceLimits &limits) const {
  return verify_impl(config, limits, nullptr);
}

std::expected<void, std::string>
LaunchVerifier::verify(const LaunchConfig &config,
                       const kernelh::DeviceLimits &limits,
                       void *function) const {
  return verify_impl(config, limits, function);
}

std::expected<void, std::string>
LaunchVerifier::verify_impl(const LaunchConfig &config,
                            const kernelh::DeviceLimits &limits,
                            void *function) const {
  std::vector<std::string> violations;

  // 1. Static shared bytes vs the LDS budget per workgroup.
  if (limits.lds_bytes_per_workgroup > 0 &&
      config.static_shared_bytes >
          static_cast<size_t>(limits.lds_bytes_per_workgroup)) {
    violations.push_back(
        "shared memory " + std::to_string(config.static_shared_bytes) +
        " bytes exceeds the " +
        std::to_string(limits.lds_bytes_per_workgroup) +
        "-byte LDS limit for target '" + std::string(name_of(limits)) + "'");
  }

  // 2. Workgroup size vs the per-workgroup thread limit.
  if (limits.max_threads_per_workgroup > 0 &&
      config.threads_per_workgroup > limits.max_threads_per_workgroup) {
    violations.push_back(
        "threads per workgroup " +
        std::to_string(config.threads_per_workgroup) + " exceeds the max " +
        std::to_string(limits.max_threads_per_workgroup));
  }

  // 3. Workgroup size must be a multiple of the wavefront/warp size.
  if (limits.wavefront_size > 1 && config.threads_per_workgroup > 0 &&
      config.threads_per_workgroup % limits.wavefront_size != 0) {
    violations.push_back(
        "threads per workgroup " +
        std::to_string(config.threads_per_workgroup) +
        " is not a multiple of the wavefront size " +
        std::to_string(limits.wavefront_size));
  }

  // 4. Grid extents within the target's grid limits. One line names the whole
  //    tuple; dimensions with an unknown (0) limit are skipped.
  const long grid[3] = {config.grid_x, config.grid_y, config.grid_z};
  bool grid_exceeds = false;
  for (int i = 0; i < 3; ++i) {
    if (limits.max_grid_dim[i] > 0 && grid[i] > limits.max_grid_dim[i]) {
      grid_exceeds = true;
      break;
    }
  }
  if (grid_exceeds) {
    violations.push_back("grid dim (" + std::to_string(grid[0]) + "," +
                         std::to_string(grid[1]) + "," +
                         std::to_string(grid[2]) +
                         ") exceeds the target grid limits");
  }

  // 5. Occupancy must be > 0 (vendor API when a function is available).
  const std::string occ = occupancy_violation(config, limits, function);
  if (!occ.empty()) {
    violations.push_back(occ);
  }

  if (violations.empty()) {
    return {};
  }
  return std::unexpected(join_violations(violations));
}

// Weak default: no GPU runtime is linked into lib/compiler, so the occupancy
// API is unavailable. Plan todo 23's runtime library provides the strong
// override (contract documented in launch_verifier.hpp).
__attribute__((weak)) int query_occupancy_max_blocks_runtime(
    void * /*function*/, int /*block_size*/, size_t /*shared_bytes*/) {
  return -1;
}

} // namespace kernelh::compiler
