// rocblas backend: the LOCAL GEMM baseline (plan todo 35, backend (b)).
//
// GEMM rows are timed through rocblas_gemm_ex with rocblas_gemm_algo_standard
// (fp32, C = A*B, alpha=1/beta=0). The rocblas handle uses the default stream
// (stream 0), which is the same stream the backend's EventTimer records on,
// so the warmup-5 + median-of-20 methodology applies to the vendor baseline
// exactly as it does to the kernelh/hipref backends.
//
// The rocblas library is linked only when present in the build env
// (KERNELH_HAVE_ROCBLAS); without it the backend reports an honest per-row
// error instead of a silent skip.

#ifndef KERNELH_BENCH_ROCBLAS_BACKEND_HPP
#define KERNELH_BENCH_ROCBLAS_BACKEND_HPP

#include "bench_row.hpp"
#include "kernelh/runtime/backend.hpp"

#include <memory>
#include <optional>
#include <string>

namespace kernelh::bench {

// Runs the rocblas backend for `spec` (spec.backend must be "rocblas";
// only GEMM rows are supported - rocblas has no elementwise entry points).
// Returns the measured row, or nullopt with `error` set on any failure.
std::optional<BenchRow> run_rocblas_row(
    const RowSpec &spec, const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_ROCBLAS_BACKEND_HPP
