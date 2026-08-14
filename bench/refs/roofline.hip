// Roofline microbenchmark kernels (plan todo 35, Oracle N8): the measured
// machine ceilings that todo 38 reads from results.csv.
//
//   roofline_triad - streaming read a + read b + write c over 1 GiB f32
//                    buffers per operand (3 GiB moved) -> measured bandwidth
//                    ceiling. grid-stride, 256 blocks x 256 threads.
//   roofline_fma   - dense FMA saturation -> measured compute ceiling. Each
//                    thread runs 8 INDEPENDENT accumulator chains so the FMA
//                    pipeline stays full (a single dependent chain would be
//                    FMA-latency-bound and under-measure the ceiling); the
//                    buffer is 16 MiB so the kernel is purely compute.
//
// Compiled with: hipcc --genco --offload-arch=gfx1101 roofline.hip -o
// roofline.o (a COF code object hipModuleLoadData accepts directly). The
// kernel ABI is raw device pointers + scalars (plain HIP kernels, NOT the
// memref-descriptor ABI), so the harness launches them with
// Backend::launch_kernel. `extern "C"` keeps the symbols unmangled for
// GpuModule::get_function.

#include <hip/hip_runtime.h>

extern "C" __global__ void roofline_triad(const float *__restrict__ a,
                                          const float *__restrict__ b,
                                          float *__restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

extern "C" __global__ void roofline_fma(float *__restrict__ c, int n,
                                        int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int rounds = iters / 8; // 8 independent chains -> full FMA throughput
  const float k = 1.0000001f;
  for (; i < n; i += stride) {
    float a0 = c[i], a1 = c[i], a2 = c[i], a3 = c[i];
    float a4 = c[i], a5 = c[i], a6 = c[i], a7 = c[i];
    for (int r = 0; r < rounds; ++r) {
      a0 = fmaf(a0, k, 1.0f);
      a1 = fmaf(a1, k, 1.0f);
      a2 = fmaf(a2, k, 1.0f);
      a3 = fmaf(a3, k, 1.0f);
      a4 = fmaf(a4, k, 1.0f);
      a5 = fmaf(a5, k, 1.0f);
      a6 = fmaf(a6, k, 1.0f);
      a7 = fmaf(a7, k, 1.0f);
    }
    c[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  }
}
