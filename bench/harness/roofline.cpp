#include "roofline.hpp"

#include "metrics.hpp"
#include "refs_loader.hpp"
#include "timer.hpp"
#include "util.hpp"

#include "kernelh/runtime/buffer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace kernelh::bench {

namespace {

// A loaded refs kernel ready for repeated raw-ABI launches. The module must
// outlive the launches, so it is held by optional.
struct RefKernel {
  std::optional<kernelh::runtime::GpuModule> module;
  kernelh::runtime::KernelFunction fn;
};

// Loads the roofline COF and resolves `symbol`. "" on success.
std::string loadRoofline(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                         const char *symbol, RefKernel &out) {
  auto loaded = load_refs_module(backend, "roofline.o");
  if (!loaded) {
    return loaded.error().message;
  }
  auto fn = loaded->get_function(symbol);
  if (!fn) {
    return std::string("get_function(\"") + symbol +
           "\") failed: " + fn.error().message;
  }
  out.module.emplace(std::move(*loaded));
  out.fn = *fn;
  return "";
}

// Times `warmup`+`iterations` launches of the raw-ABI kernel. -1 on failure.
double timeRaw(const std::shared_ptr<kernelh::runtime::Backend> &backend,
               const RefKernel &ref, const kernelh::runtime::LaunchDims &grid,
               const kernelh::runtime::LaunchDims &block, void *const *params) {
  const auto launch = [&]() -> kernelh::runtime::Result<void> {
    return backend->launch_kernel(ref.fn.handle, grid, block,
                                  /*shared_bytes=*/0, /*stream=*/nullptr,
                                  params);
  };
  return time_launch_median(backend, launch, /*warmup=*/5, /*iterations=*/20);
}

} // namespace

std::optional<BenchRow> run_roofline_triad(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  // 2^28 f32 elements per buffer = 1 GiB; 3 GiB moved per launch.
  constexpr long kElems = 1L << 28;
  constexpr unsigned kGrid = 512;
  constexpr unsigned kBlock = 256;

  RefKernel ref;
  error = loadRoofline(backend, "roofline_triad", ref);
  if (!error.empty()) {
    return std::nullopt;
  }
  const size_t bytes = static_cast<size_t>(kElems) * sizeof(float);
  auto a = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto b = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto c = kernelh::runtime::Buffer::allocate(backend, bytes);
  if (!a || !b || !c) {
    error = "buffer allocation failed (3 GiB)";
    return std::nullopt;
  }
  void *da = a->device_ptr();
  void *db = b->device_ptr();
  void *dc = c->device_ptr();
  int n = static_cast<int>(kElems);
  void *params[4] = {&da, &db, &dc, &n};

  const double ms = timeRaw(backend, ref, {kGrid, 1, 1}, {kBlock, 1, 1}, params);
  if (ms < 0.0) {
    error = "timing failed";
    return std::nullopt;
  }
  BenchRow row;
  row.kernel = spec.kernel;
  row.shape = spec.shapeText;
  row.backend = spec.backend;
  row.ms_median = ms;
  row.gflops_or_gbps = gbps_from_ms(roofline_triad_bytes(kElems), ms);
  row.timestamp = now_timestamp();
  row.device = deviceName;
  return row;
}

std::optional<BenchRow> run_roofline_fma(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  // 2^22 f32 elements (16 MiB, tiny on purpose - pure compute). One element
  // per thread (grid x block >= n), 8192 FMAs each -> 2^22 * 2^13 = 2^35 FMAs
  // -> 2^36 flops, ~2 ms at 35 TFLOPs (well above launch overhead).
  constexpr long kElems = 1L << 22;
  constexpr int kIters = 8192; // FMAs per thread (kernel splits into 8 chains)
  constexpr unsigned kBlock = 256;
  constexpr unsigned kGrid =
      static_cast<unsigned>((kElems + kBlock - 1) / kBlock);
  const long long kTotalFmas = kElems * static_cast<long long>(kIters);

  RefKernel ref;
  error = loadRoofline(backend, "roofline_fma", ref);
  if (!error.empty()) {
    return std::nullopt;
  }
  const size_t bytes = static_cast<size_t>(kElems) * sizeof(float);
  auto c = kernelh::runtime::Buffer::allocate(backend, bytes);
  if (!c) {
    error = "buffer allocation failed (16 MiB)";
    return std::nullopt;
  }
  void *dc = c->device_ptr();
  int n = static_cast<int>(kElems);
  int iters = kIters;
  void *params[3] = {&dc, &n, &iters};

  const double ms = timeRaw(backend, ref, {kGrid, 1, 1}, {kBlock, 1, 1}, params);
  if (ms < 0.0) {
    error = "timing failed";
    return std::nullopt;
  }
  BenchRow row;
  row.kernel = spec.kernel;
  row.shape = spec.shapeText;
  row.backend = spec.backend;
  row.ms_median = ms;
  row.gflops_or_gbps = gflops_from_ms(roofline_fma_flops(kTotalFmas), ms);
  row.timestamp = now_timestamp();
  row.device = deviceName;
  return row;
}

} // namespace kernelh::bench
