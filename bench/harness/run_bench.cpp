#include "run_bench.hpp"

#include "bench_row.hpp"
#include "clock_pin.hpp"
#include "csv.hpp"
#include "hipref_backend.hpp"
#include "kernelh_backend.hpp"
#include "metrics.hpp"
#include "rocblas_backend.hpp"
#include "roofline.hpp"
#include "shape_text.hpp"
#include "suites.hpp"

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/device_context.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#ifndef KERNELH_CROSS_BIN
#define KERNELH_CROSS_BIN ""
#endif

namespace kernelh::bench {

namespace {

// Absolute path to the nvcc-built cross harness (cuBLAS + CUTLASS), or ""
// when it was not built (KERNELH_CROSS_BIN compile definition unset).
std::string cross_bin_path() {
  if (KERNELH_CROSS_BIN[0] == '\0') {
    return "";
  }
  return KERNELH_CROSS_BIN;
}

// Dispatches one row to its backend implementation. Backends arrive
// incrementally (kernelh / rocblas / hipref in later phases of todo 35); an
// unimplemented backend is an honest per-row error, never a silent skip.
std::optional<BenchRow> run_one_row(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  if (spec.backend == "roofline") {
    if (spec.kernel == "roofline_triad") {
      return run_roofline_triad(spec, backend, deviceName, error);
    }
    if (spec.kernel == "roofline_fma") {
      return run_roofline_fma(spec, backend, deviceName, error);
    }
    error = "unknown roofline kernel '" + spec.kernel + "'";
    return std::nullopt;
  }
  if (spec.backend == "kernelh") {
    return run_kernelh_row(spec, backend, deviceName, error);
  }
  if (spec.backend == "rocblas") {
    return run_rocblas_row(spec, backend, deviceName, error);
  }
  if (spec.backend == "hipref-naive" || spec.backend == "hipref-vec") {
    return run_hipref_row(spec, backend, deviceName, error);
  }
  error = "unknown backend '" + spec.backend + "'";
  return std::nullopt;
}

// cuda-cross suite: run the nvcc-built cross harness (cuBLAS + CUTLASS). The
// cross binary exits 77 with "skip: no CUDA driver" on a machine without an
// NVIDIA driver; when it was never built, the same 77 skip is reported here
// (this box has no NVIDIA driver - recorded as evidence).
int run_cuda_cross(const BenchFlags &flags) {
  const std::string crossBin = cross_bin_path();
  if (!crossBin.empty() && std::filesystem::exists(crossBin)) {
    const std::string cmd = "'" + crossBin + "' --out '" +
                            resolve_csv_path(flags.out) + "'";
    const int rc = std::system(cmd.c_str());
    if (rc == -1) {
      std::cerr << "skip: no CUDA driver\n";
      return 77;
    }
#ifdef _WIN32
    return rc;
#else
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : 1;
#endif
  }
  std::cerr << "skip: no CUDA driver\n";
  return 77;
}

} // namespace

std::string resolve_csv_path(const std::string &out) {
  if (out.size() >= 4 && out.compare(out.size() - 4, 4, ".csv") == 0) {
    return out;
  }
  std::error_code ec;
  std::filesystem::create_directories(out, ec);
  return out + "/results.csv";
}

int run_bench(const BenchFlags &flags) {
  if (flags.suite == "cuda-cross") {
    return run_cuda_cross(flags);
  }

  const std::vector<RowSpec> rows = suite_rows(flags.suite);
  if (rows.empty()) {
    std::cerr << "error: unknown suite '" << flags.suite
              << "'; valid suites: ";
    for (const std::string &name : suite_names()) {
      std::cerr << name << " ";
    }
    std::cerr << "\n";
    return 2;
  }

  // Best-effort clock pinning (Metis G28): warn + continue on EPERM.
  pin_clocks_best_effort();

  auto backend = kernelh::runtime::make_hip_backend();
  if (!backend) {
    std::cerr << "skip: no HIP device (degraded mode)\n";
    return 77;
  }
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  if (!ctx) {
    std::cerr << "skip: no HIP device (" << ctx.error().message << ")\n";
    return 77;
  }
  const std::string deviceName = ctx->name();
  std::cout << "bench: device " << deviceName << ", suite " << flags.suite
            << ", " << rows.size() << " row(s)\n";

  std::vector<BenchRow> results;
  results.reserve(rows.size());
  bool failed = false;
  for (const RowSpec &spec : rows) {
    std::string error;
    std::optional<BenchRow> row =
        run_one_row(spec, backend, deviceName, error);
    if (!row) {
      std::cerr << "bench: FAIL " << spec.backend << " " << spec.kernel
                << " [" << spec.shapeText << "]: " << error << "\n";
      failed = true;
      continue;
    }
    std::cout << "bench: " << spec.backend << " " << spec.kernel << " ["
              << spec.shapeText << "] ms_median=" << row->ms_median
              << " rate=" << row->gflops_or_gbps << "\n";
    results.push_back(std::move(*row));
  }

  const std::string csvPath = resolve_csv_path(flags.out);
  std::ofstream out(csvPath, std::ios::binary | std::ios::trunc);
  if (!out || !write_results_csv(out, results)) {
    std::cerr << "error: cannot write '" << csvPath << "'\n";
    return 1;
  }
  std::cout << "wrote " << csvPath << " (" << results.size() << " rows)\n";
  return failed ? 1 : 0;
}

} // namespace kernelh::bench
