#include "hipref_backend.hpp"

#include "metrics.hpp"
#include "refs_loader.hpp"
#include "shape_text.hpp"
#include "timer.hpp"
#include "util.hpp"

#include "kernelh/runtime/buffer.hpp"

#include <optional>
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

// Loads the named COF artifact and resolves `symbol`. "" on success.
std::string loadRef(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                    const std::string &cof, const char *symbol, RefKernel &out) {
  auto loaded = load_refs_module(backend, cof);
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

// Times TWO raw-ABI launches (the scan two-pass) as one measurement.
double timeRawPair(
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const RefKernel &ref1, const RefKernel &ref2,
    const kernelh::runtime::LaunchDims &grid1,
    const kernelh::runtime::LaunchDims &block1, void *const *params1,
    const kernelh::runtime::LaunchDims &grid2,
    const kernelh::runtime::LaunchDims &block2, void *const *params2) {
  const auto launch = [&]() -> kernelh::runtime::Result<void> {
    auto r1 = backend->launch_kernel(ref1.fn.handle, grid1, block1,
                                     /*shared_bytes=*/0, /*stream=*/nullptr,
                                     params1);
    if (!r1) {
      return r1;
    }
    return backend->launch_kernel(ref2.fn.handle, grid2, block2,
                                  /*shared_bytes=*/0, /*stream=*/nullptr,
                                  params2);
  };
  return time_launch_median(backend, launch, /*warmup=*/5, /*iterations=*/20);
}

// The hipref variant suffix: "hipref-vec" -> "vec4", else "naive".
std::string variantSuffix(const std::string &backend) {
  return backend == "hipref-vec" ? "vec4" : "naive";
}

BenchRow makeRow(const RowSpec &spec, const std::string &deviceName, double ms,
                 double rate) {
  BenchRow row;
  row.kernel = spec.kernel;
  row.shape = spec.shapeText;
  row.backend = spec.backend;
  row.ms_median = ms;
  row.gflops_or_gbps = rate;
  row.timestamp = now_timestamp();
  row.device = deviceName;
  return row;
}

std::optional<BenchRow> runSoftmax(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long rows = shape.at("rows"), cols = shape.at("cols");
  RefKernel ref;
  error = loadRef(backend, "softmax.o",
                  ("softmax_" + variantSuffix(spec.backend)).c_str(), ref);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  if (!x || !y) {
    error = "softmax buffer allocation failed";
    return std::nullopt;
  }
  void *dx = x->device_ptr();
  void *dy = y->device_ptr();
  int irows = static_cast<int>(rows);
  int icols = static_cast<int>(cols);
  void *params[4] = {&dx, &dy, &irows, &icols};
  const double ms =
      timeRaw(backend, ref, {static_cast<unsigned>(rows), 1, 1},
              {static_cast<unsigned>(std::min(cols, 256L)), 1, 1}, params);
  if (ms < 0.0) {
    error = "softmax timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms,
                 gbps_from_ms(elementwise_bytes(rows, cols), ms));
}

std::optional<BenchRow> runLayernorm(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long rows = shape.at("rows"), cols = shape.at("cols");
  RefKernel ref;
  error = loadRef(backend, "layernorm.o",
                  ("layernorm_" + variantSuffix(spec.backend)).c_str(), ref);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  auto gamma = kernelh::runtime::Buffer::allocate(backend, cols * 4);
  auto beta = kernelh::runtime::Buffer::allocate(backend, cols * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  if (!x || !gamma || !beta || !y) {
    error = "layernorm buffer allocation failed";
    return std::nullopt;
  }
  void *dx = x->device_ptr();
  void *dg = gamma->device_ptr();
  void *db = beta->device_ptr();
  void *dy = y->device_ptr();
  int irows = static_cast<int>(rows);
  int icols = static_cast<int>(cols);
  void *params[6] = {&dx, &dg, &db, &dy, &irows, &icols};
  const double ms = timeRaw(backend, ref,
                            {static_cast<unsigned>(rows), 1, 1}, {256u, 1, 1},
                            params);
  if (ms < 0.0) {
    error = "layernorm timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms,
                 gbps_from_ms(layernorm_bytes(rows, cols), ms));
}

std::optional<BenchRow> runFusedGelu(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long n = shape.at("n");
  RefKernel ref;
  error = loadRef(backend, "fused_gelu.o",
                  ("fused_gelu_" + variantSuffix(spec.backend)).c_str(), ref);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, n * 4);
  if (!x || !y) {
    error = "fused_gelu buffer allocation failed";
    return std::nullopt;
  }
  void *dx = x->device_ptr();
  void *dy = y->device_ptr();
  int in = static_cast<int>(n);
  void *params[3] = {&dx, &dy, &in};
  const double ms =
      timeRaw(backend, ref,
              {static_cast<unsigned>((n + 255) / 256), 1, 1}, {256u, 1, 1},
              params);
  if (ms < 0.0) {
    error = "fused_gelu timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms, gbps_from_ms(fused_gelu_bytes(n), ms));
}

std::optional<BenchRow> runScan(const RowSpec &spec,
                                const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                const std::string &deviceName,
                                std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long n = shape.at("n");
  const long t = (n + 255) / 256;
  const std::string suffix = variantSuffix(spec.backend);
  RefKernel ref1, ref2;
  error = loadRef(backend, "scan.o", ("scan_pass1_" + suffix).c_str(), ref1);
  if (!error.empty()) {
    return std::nullopt;
  }
  error = loadRef(backend, "scan.o", ("scan_totals_" + suffix).c_str(), ref2);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto totals = kernelh::runtime::Buffer::allocate(backend, t * 4);
  if (!x || !y || !totals) {
    error = "scan buffer allocation failed";
    return std::nullopt;
  }
  void *dx = x->device_ptr();
  void *dy = y->device_ptr();
  void *dt = totals->device_ptr();
  int in = static_cast<int>(n);
  int it = static_cast<int>(t);
  void *params1[5] = {&dx, &dy, &dt, &in, &it};
  void *params2[4] = {&dt, &dy, &it, &in};
  const double ms = timeRawPair(
      backend, ref1, ref2, {static_cast<unsigned>(t), 1, 1}, {256u, 1, 1},
      params1, {1u, 1, 1}, {256u, 1, 1}, params2);
  if (ms < 0.0) {
    error = "scan timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms, gbps_from_ms(scan_bytes(n), ms));
}

std::optional<BenchRow> runGemm(const RowSpec &spec,
                                const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                const std::string &deviceName,
                                std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long m = shape.at("m"), n = shape.at("n"), k = shape.at("k");
  RefKernel ref;
  error = loadRef(backend, "gemm.o", "gemm_tiled", ref);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto a = kernelh::runtime::Buffer::allocate(backend, m * k * 4);
  auto b = kernelh::runtime::Buffer::allocate(backend, k * n * 4);
  auto c = kernelh::runtime::Buffer::allocate(backend, m * n * 4);
  if (!a || !b || !c) {
    error = "gemm buffer allocation failed";
    return std::nullopt;
  }
  void *da = a->device_ptr();
  void *db = b->device_ptr();
  void *dc = c->device_ptr();
  int im = static_cast<int>(m);
  int in = static_cast<int>(n);
  int ik = static_cast<int>(k);
  void *params[6] = {&da, &db, &dc, &im, &in, &ik};
  const double ms =
      timeRaw(backend, ref,
              {static_cast<unsigned>(n / 32), static_cast<unsigned>(m / 32), 1},
              {32u, 8u, 1}, params);
  if (ms < 0.0) {
    error = "gemm timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms, gflops_from_ms(gemm_flops(m, n, k), ms));
}

} // namespace

std::optional<BenchRow> run_hipref_row(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  if (spec.backend != "hipref-naive" && spec.backend != "hipref-vec") {
    error = "hipref backend must be 'hipref-naive' or 'hipref-vec'";
    return std::nullopt;
  }
  if (spec.kernel == "gemm" && spec.backend == "hipref-vec") {
    error = "gemm has no hipref-vec variant (only the naive tiled ref)";
    return std::nullopt;
  }
  if (spec.kernel == "softmax") {
    return runSoftmax(spec, backend, deviceName, error);
  }
  if (spec.kernel == "layernorm") {
    return runLayernorm(spec, backend, deviceName, error);
  }
  if (spec.kernel == "scan") {
    return runScan(spec, backend, deviceName, error);
  }
  if (spec.kernel == "fused_gelu") {
    return runFusedGelu(spec, backend, deviceName, error);
  }
  if (spec.kernel == "gemm") {
    return runGemm(spec, backend, deviceName, error);
  }
  error = "hipref backend has no runner for kernel '" + spec.kernel + "'";
  return std::nullopt;
}

} // namespace kernelh::bench
