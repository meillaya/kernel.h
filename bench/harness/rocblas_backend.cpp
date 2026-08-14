#include "rocblas_backend.hpp"

#include "metrics.hpp"
#include "shape_text.hpp"
#include "timer.hpp"
#include "util.hpp"

#include "kernelh/runtime/buffer.hpp"

#ifdef KERNELH_HAVE_ROCBLAS
#include <rocblas/rocblas.h>
#endif

#include <optional>
#include <string>
#include <vector>

namespace kernelh::bench {

#ifdef KERNELH_HAVE_ROCBLAS

namespace {

// One rocblas GEMM timing: allocate A/B/C, then warmup-5 + median-of-20
// rocblas_gemm_ex launches with the pinned algorithm. The handle is created
// per row (cheap; avoids cross-row state).
struct RocblasGemm {
  rocblas_handle handle = nullptr;
  bool ok = false;
  std::string error;

  ~RocblasGemm() {
    if (handle) {
      rocblas_destroy_handle(handle);
    }
  }
};

std::string rocblasErrorMessage(rocblas_status status) {
  return std::string("rocblas_gemm_ex failed with status ") +
         std::to_string(static_cast<int>(status));
}

} // namespace

std::optional<BenchRow> run_rocblas_row(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  if (spec.kernel != "gemm") {
    error = "rocblas backend only benchmarks gemm (got '" + spec.kernel + "')";
    return std::nullopt;
  }
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long m = shape.at("m"), n = shape.at("n"), k = shape.at("k");

  RocblasGemm gemm;
  const rocblas_status createStatus = rocblas_create_handle(&gemm.handle);
  if (createStatus != rocblas_status_success) {
    error = "rocblas_create_handle failed with status " +
            std::to_string(static_cast<int>(createStatus));
    return std::nullopt;
  }

  auto a = kernelh::runtime::Buffer::allocate(backend, m * k * 4);
  auto b = kernelh::runtime::Buffer::allocate(backend, k * n * 4);
  auto c = kernelh::runtime::Buffer::allocate(backend, m * n * 4);
  if (!a || !b || !c) {
    error = "rocblas buffer allocation failed";
    return std::nullopt;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  void *da = a->device_ptr();
  void *db = b->device_ptr();
  void *dc = c->device_ptr();
  const rocblas_int rm = static_cast<rocblas_int>(m);
  const rocblas_int rn = static_cast<rocblas_int>(n);
  const rocblas_int rk = static_cast<rocblas_int>(k);

  const auto oneLaunch = [&]() -> kernelh::runtime::Result<void> {
    const rocblas_status status = rocblas_gemm_ex(
        gemm.handle, rocblas_operation_none, rocblas_operation_none, rm, rn,
        rk, &alpha, da, rocblas_datatype_f32_r, rm, db, rocblas_datatype_f32_r,
        rk, &beta, dc, rocblas_datatype_f32_r, rm, dc, rocblas_datatype_f32_r,
        rm, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0);
    if (status != rocblas_status_success) {
      return std::unexpected(
          kernelh::runtime::Error{1, rocblasErrorMessage(status)});
    }
    return {};
  };

  const double ms = time_launch_median(backend, oneLaunch, /*warmup=*/5,
                                       /*iterations=*/20);
  if (ms < 0.0) {
    error = "rocblas gemm timing failed";
    return std::nullopt;
  }
  BenchRow row;
  row.kernel = spec.kernel;
  row.shape = spec.shapeText;
  row.backend = spec.backend;
  row.ms_median = ms;
  row.gflops_or_gbps = gflops_from_ms(gemm_flops(m, n, k), ms);
  row.timestamp = now_timestamp();
  row.device = deviceName;
  return row;
}

#else // !KERNELH_HAVE_ROCBLAS

std::optional<BenchRow> run_rocblas_row(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  (void)spec;
  (void)backend;
  (void)deviceName;
  error = "rocblas library not available in this build (KERNELH_HAVE_ROCBLAS "
          "unset)";
  return std::nullopt;
}

#endif // KERNELH_HAVE_ROCBLAS

} // namespace kernelh::bench
