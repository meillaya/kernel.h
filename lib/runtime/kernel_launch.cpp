// KernelLaunch implementation (plan todo 23).
//
// Every launch is pre-verified against the device-limits table (plan todo 32,
// kernelh::compiler::LaunchVerifier): a config that violates ANY limit
// (static shared > LDS, threads > max, threads % wavefront != 0, grid beyond
// the target's extents, occupancy would be 0) is REFUSED with the full
// diagnostic instead of being submitted to the driver.

#include "kernelh/runtime/kernel_launch.hpp"

#include "kernelh/compiler/launch_verifier.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

namespace kernelh::runtime {

namespace {

// Target token per backend kind: HIP is the local gfx1101 execution device
// (todo 6 gate), CUDA is the cross-only sm_80 target (todo 8).
const char *target_token_for(BackendKind kind) {
  return kind == BackendKind::kCuda ? "sm_80" : "gfx1101";
}

// DeviceLimits for launch verification: the target's table entry (grid
// extents + waves_per_cu, which DeviceInfo does not carry) overridden with the
// LIVE DeviceInfo fields (LDS, wavefront, max threads, CU count) queried from
// the backend. Should the table lookup ever fail, the fallback limits carry
// 0 for the table-only fields and the verifier skips those checks (it only
// judges known limits).
kernelh::DeviceLimits resolve_limits(Backend &backend) {
  kernelh::DeviceLimits limits{};
  try {
    limits = kernelh::device_limits_for(target_token_for(backend.kind()));
  } catch (const std::invalid_argument &) {
    limits = kernelh::DeviceLimits{};
    limits.name = "(unknown device)"; // static literal - never dangles
  }
  auto info = backend.device_info();
  if (info) {
    limits.lds_bytes_per_workgroup = info->lds_bytes_per_workgroup;
    limits.wavefront_size = info->wavefront_size;
    limits.max_threads_per_workgroup = info->max_threads_per_workgroup;
    limits.cu_count = info->cu_count;
  }
  return limits;
}

// LaunchConfig from the KernelLaunch state. Grid extents beyond INT_MAX are
// clamped: no known target admits a grid that large (the x-limit IS INT_MAX),
// so clamping saturates the check; the driver rejects anything beyond anyway.
kernelh::compiler::LaunchConfig make_launch_config(const LaunchDims &grid,
                                                   const LaunchDims &block,
                                                   size_t shared_bytes) {
  kernelh::compiler::LaunchConfig config;
  config.grid_x = static_cast<int>(std::min<unsigned>(grid.x, INT_MAX));
  config.grid_y = static_cast<int>(std::min<unsigned>(grid.y, INT_MAX));
  config.grid_z = static_cast<int>(std::min<unsigned>(grid.z, INT_MAX));
  const long long threads = static_cast<long long>(block.x) * block.y * block.z;
  config.threads_per_workgroup =
      static_cast<int>(std::min<long long>(threads, INT_MAX));
  config.static_shared_bytes = shared_bytes;
  return config;
}

} // namespace

KernelLaunch::KernelLaunch(std::shared_ptr<Backend> backend)
    : backend_(std::move(backend)) {}

KernelLaunch &KernelLaunch::grid(unsigned x, unsigned y, unsigned z) {
  grid_ = {x, y, z};
  return *this;
}

KernelLaunch &KernelLaunch::block(unsigned x, unsigned y, unsigned z) {
  block_ = {x, y, z};
  return *this;
}

KernelLaunch &KernelLaunch::shared_mem_bytes(size_t bytes) {
  shared_ = bytes;
  return *this;
}

KernelLaunch &KernelLaunch::stream(StreamHandle stream) {
  stream_ = stream;
  return *this;
}

KernelLaunch &KernelLaunch::memref_arg(const void *dev_ptr,
                                       const std::vector<int64_t> &shape) {
  args_.add_memref(dev_ptr, shape);
  return *this;
}

Result<void> KernelLaunch::launch(KernelFunction function) const {
  if (!backend_) {
    return std::unexpected(Error{1, "KernelLaunch::launch: null backend"});
  }
  if (function.handle == nullptr) {
    return std::unexpected(
        Error{1, "KernelLaunch::launch: null kernel function handle"});
  }
  if (args_.empty()) {
    return std::unexpected(
        Error{1, "KernelLaunch::launch: no kernel arguments packed"});
  }

  // Pre-launch verification (plan todo 32): refuse the launch with the FULL
  // diagnostic when the config violates any device limit. Limits are resolved
  // from the live device once and cached (launch is const).
  if (!limits_cache_) {
    limits_cache_ = std::make_shared<kernelh::DeviceLimits>(
        resolve_limits(*backend_));
  }
  const kernelh::compiler::LaunchConfig config =
      make_launch_config(grid_, block_, shared_);
  kernelh::compiler::LaunchVerifier verifier;
  auto verified = verifier.verify(config, *limits_cache_, function.handle);
  if (!verified) {
    return std::unexpected(
        Error{1, "KernelLaunch::launch refused: " + verified.error()});
  }

  void **params = args_.params();
  return backend_->launch_kernel(function.handle, grid_, block_, shared_,
                                 stream_, params);
}

} // namespace kernelh::runtime
