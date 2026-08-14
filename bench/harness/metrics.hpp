// Metric formulas for the gflops_or_gbps CSV column (plan todo 35).
//
// Formulas are pure functions so unit tests assert hand-computed values. The
// per-kernel FLOPs / bytes-moved derivations are documented in
// scripts/README.md (consumed by todo 38 for the roofline arithmetic
// intensity). All rates are GIGA units: GFLOPS = flops / s / 1e9,
// GB/s = bytes / s / 1e9 (decimal gigabytes, 1 GB = 1e9 bytes).

#ifndef KERNELH_BENCH_METRICS_HPP
#define KERNELH_BENCH_METRICS_HPP

namespace kernelh::bench {

// flops executed in `ms` milliseconds -> GFLOPS.
double gflops_from_ms(double flops, double ms);

// bytes moved in `ms` milliseconds -> effective GB/s (bytes / s / 1e9).
double gbps_from_ms(double bytes, double ms);

// --- Per-kernel FLOPs / bytes-moved formulas (fp32 everywhere) ---

// GEMM C = A*B: 2 flops per MAC over M*N*K.
double gemm_flops(long m, long n, long k);
// GEMM bytes moved: read A (M*K*4) + read B (K*N*4) + write C (M*N*4).
double gemm_bytes(long m, long n, long k);

// softmax: read x (rows*cols*4) + write y (rows*cols*4).
double elementwise_bytes(long rows, long cols);

// layernorm: read x + gamma + beta, write y.
double layernorm_bytes(long rows, long cols);

// scan: read x + write y (the two-pass intermediate traffic is an
// implementation detail and is NOT counted; documented in scripts/README).
double scan_bytes(long n);

// fused_gelu: x read, tmp read+write, y write.
double fused_gelu_bytes(long n);

// Attention (seq s, heads h, batch b, head_dim d): QK^T = 2*b*h*s^2*d MACs,
// weighted sum PV = 2*b*h*s^2*d MACs.
double attention_flops(long batch, long heads, long seq, long head_dim);

// roofline_triad over `n` f32 elements per buffer: read a + read b + write c.
double roofline_triad_bytes(long n);

// roofline_fma: 2 flops per FMA; `fmas` is the total across all threads.
double roofline_fma_flops(long long fmas);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_METRICS_HPP
