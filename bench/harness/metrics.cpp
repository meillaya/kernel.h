#include "metrics.hpp"

namespace kernelh::bench {

double gflops_from_ms(double flops, double ms) {
  // flops / (ms * 1e-3 s) / 1e9.
  return flops * 1000.0 / ms / 1e9;
}

double gbps_from_ms(double bytes, double ms) {
  return bytes * 1000.0 / ms / 1e9;
}

double gemm_flops(long m, long n, long k) {
  return 2.0 * static_cast<double>(m) * static_cast<double>(n) *
         static_cast<double>(k);
}

double gemm_bytes(long m, long n, long k) {
  return (static_cast<double>(m) * k + static_cast<double>(k) * n +
          static_cast<double>(m) * n) *
         4.0;
}

double elementwise_bytes(long rows, long cols) {
  return 2.0 * static_cast<double>(rows) * static_cast<double>(cols) * 4.0;
}

double layernorm_bytes(long rows, long cols) {
  return 2.0 * static_cast<double>(rows) * static_cast<double>(cols) * 4.0 +
         2.0 * static_cast<double>(cols) * 4.0;
}

double scan_bytes(long n) { return 2.0 * static_cast<double>(n) * 4.0; }

double fused_gelu_bytes(long n) {
  return 3.0 * static_cast<double>(n) * 4.0;
}

double attention_flops(long batch, long heads, long seq, long head_dim) {
  return 4.0 * static_cast<double>(batch) * static_cast<double>(heads) *
         static_cast<double>(seq) * static_cast<double>(seq) *
         static_cast<double>(head_dim);
}

double roofline_triad_bytes(long n) {
  return 3.0 * static_cast<double>(n) * 4.0;
}

double roofline_fma_flops(long long fmas) {
  return 2.0 * static_cast<double>(fmas);
}

} // namespace kernelh::bench
