#include <hip/hip_runtime.h>
// Reference fused_gelu kernels (plan todo 35, backend (c)): hand-written HIP
// versions of the todo-30 kernelh fused_gelu template (the two-op chain
// t = a*x + b, out = gelu(t) that the fuse pass collapses to one loop nest).
//
// ABI (raw pointers):
//   fused_gelu_naive(const float* x, float* y, int n)
//   fused_gelu_vec4 (const float4* x, float4* y, int n)
//
// Chain constants pinned in todo 30: a = 1.5, b = 0.5,
// gelu(t) = 0.5*t*(1 + tanh(c*(t + d*t^3))), c = sqrt(2/pi) ~ 0.7978845608,
// d = 0.044715. Grid-stride over the block (grid = ceil(n/256), block 256).
// vec4 requires n % 4 == 0 (the harness uses 2^20 / 2^22).

#define KERNELH_GELU_A 1.5f
#define KERNELH_GELU_B 0.5f
#define KERNELH_GELU_C 0.7978845608f
#define KERNELH_GELU_D 0.044715f

__device__ __forceinline__ float kernelh_gelu(float t) {
  const float t2 = t * t;
  const float t3 = t2 * t;
  const float z = KERNELH_GELU_C * (t + KERNELH_GELU_D * t3);
  return 0.5f * t * (1.0f + tanhf(z));
}

extern "C" __global__ void fused_gelu_naive(const float *__restrict__ x,
                                            float *__restrict__ y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    y[i] = kernelh_gelu(KERNELH_GELU_A * x[i] + KERNELH_GELU_B);
  }
}

extern "C" __global__ void fused_gelu_vec4(const float4 *__restrict__ x4,
                                           float4 *__restrict__ y4, int n) {
  const int c4 = n >> 2; // n % 4 == 0
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c4;
       i += gridDim.x * blockDim.x) {
    const float4 v = x4[i];
    float4 o;
    o.x = kernelh_gelu(KERNELH_GELU_A * v.x + KERNELH_GELU_B);
    o.y = kernelh_gelu(KERNELH_GELU_A * v.y + KERNELH_GELU_B);
    o.z = kernelh_gelu(KERNELH_GELU_A * v.z + KERNELH_GELU_B);
    o.w = kernelh_gelu(KERNELH_GELU_A * v.w + KERNELH_GELU_B);
    y4[i] = o;
  }
}
