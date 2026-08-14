// Benchmark suite matrix (plan todo 35).
//
// Suites bundle {kernel x shape x backend} rows:
//   roofline   - the two measured machine-ceiling microbenches (Oracle N8),
//                consumed by todo 38 charts.
//   kernelh    - our compiled kernels (todo 25-31 templates) for the pinned
//                GEMM / softmax / layernorm / scan / fused_gelu / attention
//                shapes.
//   rocblas    - the LOCAL GEMM baseline (rocblas_gemm_ex).
//   hipref     - hand-written HIP reference kernels in bench/refs
//                (hipref-naive + hipref-vec variant labels).
//   cuda-cross - cuBLAS + CUTLASS (handled by the separate nvcc-built cross
//                harness; not a row list here).
//   all        - the union of every local suite.
//
// GEMM shapes are the SQUARE sizes {512,1024,2048,4096,8192} (M=K=N) - the
// plan's "{512,1024,2048,4096,8192}^3" read as the square diagonal (the full
// 125-combo cartesian product would exceed the session's practical runtime;
// recorded in scripts/README.md).

#ifndef KERNELH_BENCH_SUITES_HPP
#define KERNELH_BENCH_SUITES_HPP

#include "bench_row.hpp"

#include <string>
#include <vector>

namespace kernelh::bench {

// Rows for a suite name; empty when the name is unknown OR the suite has no
// local rows (cuda-cross is executed by the cross binary, not by row lists).
std::vector<RowSpec> suite_rows(const std::string &name);

// Valid suite names (for usage/errors), sorted.
std::vector<std::string> suite_names();

} // namespace kernelh::bench

#endif // KERNELH_BENCH_SUITES_HPP
