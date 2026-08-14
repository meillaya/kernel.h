# Benchmark methodology and metric formulas (plan todo 35)

Consumed by todo 38 (roofline charts + arithmetic-intensity). All rates are
GIGA units: GFLOPS = flops / s / 1e9, GB/s = bytes / s / 1e9 (decimal
gigabytes, 1 GB = 1e9 bytes).

## Timing methodology

- Warmup: 5 untimed launches (clock ramp + device-side state settle).
- Measurement: 20 individually timed launches via GPU events
  (EventTimer::record_start -> launch -> record_end -> synchronize ->
  elapsed_milliseconds), reporting the MEDIAN.
- Clock pinning is best-effort (Metis G28): the harness tries
  `/sys/class/drm/card*/device/power_dpm_force_performance_level =
  profile_standard`; on EPERM it warns and continues (the median absorbs the
  residual variance).

## CSV schema (Metis G32)

`kernel,shape,backend,ms_median,gflops_or_gbps,timestamp,device`

- `shape` is `k=v,k=v,...` text (same convention as the tune CSV).
- `backend` carries the variant label directly: `kernelh`, `rocblas`,
  `hipref-naive`, `hipref-vec`, `cublas`, `cutlass`, `roofline`.
- `gflops_or_gbps` is a single column: GFLOPS for compute-bound kernels
  (gemm, attention, roofline_fma), effective GB/s for bandwidth-bound
  kernels (softmax, layernorm, scan, fused_gelu, roofline_triad).

## Per-kernel FLOPs / bytes-moved (fp32 everywhere)

| kernel | flops | bytes moved |
|---|---|---|
| gemm (m,n,k) | 2\*m\*n\*k | (m\*k + k\*n + m\*n) \* 4 |
| softmax (rows,cols) | - | 2 \* rows \* cols \* 4 (read x, write y) |
| layernorm (rows,cols) | - | 2 \* rows \* cols \* 4 + 2 \* cols \* 4 (read x,gamma,beta; write y) |
| scan (n) | - | 2 \* n \* 4 (read x, write y; the two-pass intermediate traffic is an implementation detail, NOT counted) |
| fused_gelu (n) | - | 3 \* n \* 4 (read x, read+write tmp intermediate, write y) |
| attention (b,h,s,d=64) | 4 \* b \* h \* s^2 \* d (QK^T + PV, 2 MACs each) | - |
| roofline_triad (n elems) | - | 3 \* n \* 4 (read a, read b, write c) |
| roofline_fma (fmas) | 2 \* fmas | - |

Effective rates: `gflops_or_gbps = flops_or_bytes / (ms * 1e-3) / 1e9`.

## Arithmetic intensity for todo 38 (Williams et al., doi:10.1145/1498765.1498785)

AI = FLOPs / bytes moved, derived from the table above:

- GEMM: `2mnk / ((mk + kn + mn) * 4)`.
- softmax / layernorm / fused_gelu / scan: `~0` (no flops counted - they are
  bandwidth-bound rows; the charts plot effective GB/s, not AI points, and
  the roofline ceiling they sit under is roofline_triad).
- attention: `4bh s^2 d / (4 bh s d * 4) = s / 4` (the four Q/K/V/O tensors
  are each b\*h\*s\*d elements; the chart uses the documented per-tile
  derivation).
- roofline ceilings: roofline_triad (GB/s ceiling) and roofline_fma (GFLOPS
  ceiling) are read from the results.csv rows `kernel = roofline_triad` /
  `kernel = roofline_fma` - measured, never assumed.

## Suite matrix

- GEMM shapes: the square diagonal {512, 1024, 2048, 4096, 8192} (M=K=N).
  The plan's "{512,1024,2048,4096,8192}^3" is read as this diagonal: the
  full 125-combo cartesian product would put ~44s x 25-launch GEMM cells in
  the matrix (pre-fix kernelh) and exceed any practical session runtime;
  the diagonal still exercises the full size range on all three GEMM
  backends (kernelh / rocblas / hipref-naive).
- softmax / layernorm: {(4096,1024), (16384,4096)}.
- scan: {1e5, 1e6}. fused_gelu: {2^20, 2^22} (not pinned by the plan; chosen
  to match the fused_gelu e2e's power-of-two sizes).
- attention: seq {512,1024}, batch 2, heads 8, head_dim 64 (fixed by the
  template).
- roofline: triad over 2^28 f32 elements per buffer (1 GiB x 3 = 3 GiB
  moved), fma over 2^22 elements x 8192 FMAs/thread = 2^35 FMAs.

## Backends

- `kernelh` - our compiled kernels (DSL -> default pipeline -> HSACO, loaded
  + launched in-process via the runtime).
- `rocblas` - rocblas_gemm_ex + rocblas_gemm_algo_standard (fp32, alpha=1,
  beta=0), the LOCAL vendor baseline. GEMM only.
- `hipref-naive` / `hipref-vec` - hand-written HIP reference kernels in
  bench/refs (scalar + vectorized-4 softmax/layernorm/scan/fused_gelu; naive
  tiled GEMM has only the naive variant), compiled `hipcc --genco
  --offload-arch=gfx1101`.
- `cublas` / `cutlass` - the nvcc sm_80 cross harness (cublasGemmEx +
  CUTLASS device::Gemm). It builds locally and runs on any NVIDIA machine;
  without an NVIDIA driver it exits 77 "skip: no CUDA driver". If the
  CUTLASS portion fails to build, a `cutlass,build-failed` row is recorded
  and the matrix continues cuBLAS-only.

No performance parity is claimed anywhere - these are measurements recorded
for todo 38's charts.
