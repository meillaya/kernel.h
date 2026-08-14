#include <hip/hip_runtime.h>
// Reference layernorm kernels (plan todo 35, backend (c)): hand-written HIP
// versions of the todo-28 kernelh layernorm template.
//
// ABI (raw pointers):
//   layernorm_naive(const float* x, const float* gamma, const float* beta,
//                   float* y, int rows, int cols)
//   layernorm_vec4 (const float4* x, const float4* gamma, const float4* beta,
//                   float4* y, int rows, int cols)
//
// y[r,c] = gamma[c] * (x[r,c] - mean_r) * rsqrt(var_r + eps) + beta[c] with
// eps = 1e-5, mean/var over the row's columns. One block per row, columns
// grid-strided over the block (blockDim.x <= 256, power of two). vec4 uses
// 16-byte vectorized access over float4 chunks (cols % 4 == 0).

#define KERNELH_LN_EPS 1e-5f

extern "C" __global__ void layernorm_naive(
    const float *__restrict__ x, const float *__restrict__ gamma,
    const float *__restrict__ beta, float *__restrict__ y, int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const float *xr = x + (long)row * cols;
  float *yr = y + (long)row * cols;

  __shared__ float s_red[256];

  // Mean.
  float local_sum = 0.0f;
  for (int c = tid; c < cols; c += bdim) {
    local_sum += xr[c];
  }
  s_red[tid] = local_sum;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] += s_red[tid + off];
    }
    __syncthreads();
  }
  const float mean = s_red[0] / (float)cols;

  // Variance (second pass over the row).
  float local_var = 0.0f;
  for (int c = tid; c < cols; c += bdim) {
    const float d = xr[c] - mean;
    local_var += d * d;
  }
  s_red[tid] = local_var;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] += s_red[tid + off];
    }
    __syncthreads();
  }
  const float inv_std = rsqrtf(s_red[0] / (float)cols + KERNELH_LN_EPS);

  for (int c = tid; c < cols; c += bdim) {
    yr[c] = (xr[c] - mean) * inv_std * gamma[c] + beta[c];
  }
}

extern "C" __global__ void layernorm_vec4(
    const float4 *__restrict__ x4, const float4 *__restrict__ gamma4,
    const float4 *__restrict__ beta4, float4 *__restrict__ y4, int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const int c4 = cols >> 2; // float4 chunks per row (cols % 4 == 0)
  const float4 *xr = x4 + (long)row * c4;
  float4 *yr = y4 + (long)row * c4;

  __shared__ float s_red[256];

  float local_sum = 0.0f;
  for (int i = tid; i < c4; i += bdim) {
    const float4 v = xr[i];
    local_sum += v.x + v.y + v.z + v.w;
  }
  s_red[tid] = local_sum;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] += s_red[tid + off];
    }
    __syncthreads();
  }
  const float mean = s_red[0] / (float)cols;

  float local_var = 0.0f;
  for (int i = tid; i < c4; i += bdim) {
    const float4 v = xr[i];
    const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean,
                d3 = v.w - mean;
    local_var += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
  }
  s_red[tid] = local_var;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] += s_red[tid + off];
    }
    __syncthreads();
  }
  const float inv_std = rsqrtf(s_red[0] / (float)cols + KERNELH_LN_EPS);

  for (int i = tid; i < c4; i += bdim) {
    const float4 v = xr[i];
    const float4 g = gamma4[i];
    const float4 b = beta4[i];
    float4 o;
    o.x = (v.x - mean) * inv_std * g.x + b.x;
    o.y = (v.y - mean) * inv_std * g.y + b.y;
    o.z = (v.z - mean) * inv_std * g.z + b.z;
    o.w = (v.w - mean) * inv_std * g.w + b.w;
    yr[i] = o;
  }
}
