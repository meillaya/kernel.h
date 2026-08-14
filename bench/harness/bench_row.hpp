// Benchmark row + suite-matrix spec (plan todo 35).
//
// The G32 CSV schema (pinned by the plan, consumed by todo 36 occupancy and
// todo 38 charts):
//
//   kernel,shape,backend,ms_median,gflops_or_gbps,timestamp,device
//
// - gflops_or_gbps is a SINGLE column carrying either GFLOPS (compute-bound
//   kernels: gemm, attention, roofline_fma) or effective GB/s (bandwidth
//   kernels: softmax/layernorm/scan/fused_gelu, roofline_triad). The per-row
//   semantics are documented in scripts/README.md (todo 38 roofline AI
//   consumption).
// - shape is a "k=v,k=v,..." text (same convention as the tune CSV).
// - backend carries the variant label directly: kernelh, rocblas,
//   hipref-naive, hipref-vec, cublas, cutlass, roofline.

#ifndef KERNELH_BENCH_BENCH_ROW_HPP
#define KERNELH_BENCH_BENCH_ROW_HPP

#include <string>

namespace kernelh::bench {

// One CSV row (G32 schema).
struct BenchRow {
  std::string kernel;
  std::string shape;
  std::string backend;
  double ms_median = 0.0;
  double gflops_or_gbps = 0.0;
  std::string timestamp;
  std::string device;
};

// One {kernel x shape x backend} cell of a suite matrix. `shapeText` is the
// G32 shape column value; backends decode the k=v pairs they need.
struct RowSpec {
  std::string kernel;
  std::string shapeText;
  std::string backend;
};

} // namespace kernelh::bench

#endif // KERNELH_BENCH_BENCH_ROW_HPP
