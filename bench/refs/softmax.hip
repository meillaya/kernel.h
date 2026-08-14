#include <hip/hip_runtime.h>
// Reference softmax kernels (plan todo 35, backend (c)): hand-written HIP
// versions of the todo-27 kernelh softmax template.
//
// ABI (raw pointers - these refs bypass the memref-descriptor ABI and are
// launched by the harness with plain hipModuleLaunchKernel-style params):
//   softmax_naive(const float* x, float* y, int rows, int cols)
//   softmax_vec4 (const float4* x, float4* y, int rows, int cols)
//
// y[r,c] = exp(x[r,c] - max_r) / sum_c exp(x[r,c] - max_r) with the SAFE
// max-subtracting form (todo-27 stability requirement: finite outputs for
// inputs ~+/-1e3). One block per row; the row's columns are grid-strided
// over the block (blockDim.x <= 256 must be a power of two for the tree
// reduction).
//
// vec4 variant: the row is processed as float4 chunks (cols % 4 == 0
// required - the harness only uses 1024/4096). Correctness is equivalent to
// the scalar form; only the memory access is 16-byte vectorized.

extern "C" __global__ void softmax_naive(const float *__restrict__ x,
                                         float *__restrict__ y, int rows,
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

  // Pass 1: row maximum (local grid-stride max, then shared tree reduce).
  float local_max = -INFINITY;
  for (int c = tid; c < cols; c += bdim) {
    local_max = fmaxf(local_max, xr[c]);
  }
  s_red[tid] = local_max;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
    }
    __syncthreads();
  }
  const float row_max = s_red[0];

  // Pass 2: row sum of exp(x - max), tree reduce.
  float local_sum = 0.0f;
  for (int c = tid; c < cols; c += bdim) {
    local_sum += __expf(xr[c] - row_max);
  }
  s_red[tid] = local_sum;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] += s_red[tid + off];
    }
    __syncthreads();
  }
  const float inv = 1.0f / s_red[0];

  // Pass 3: normalize.
  for (int c = tid; c < cols; c += bdim) {
    yr[c] = __expf(xr[c] - row_max) * inv;
  }
}

extern "C" __global__ void softmax_vec4(const float4 *__restrict__ x4,
                                        float4 *__restrict__ y4, int rows,
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

  float local_max = -INFINITY;
  for (int i = tid; i < c4; i += bdim) {
    const float4 v = xr[i];
    local_max = fmaxf(local_max, fmaxf(v.x, fmaxf(v.y, fmaxf(v.z, v.w))));
  }
  s_red[tid] = local_max;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
    }
    __syncthreads();
  }
  const float row_max = s_red[0];

  float local_sum = 0.0f;
  for (int i = tid; i < c4; i += bdim) {
    const float4 v = xr[i];
    local_sum += __expf(v.x - row_max) + __expf(v.y - row_max) +
                 __expf(v.z - row_max) + __expf(v.w - row_max);
  }
  s_red[tid] = local_sum;
  __syncthreads();
  for (int off = bdim / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s_red[tid] += s_red[tid + off];
    }
    __syncthreads();
  }
  const float inv = 1.0f / s_red[0];

  for (int i = tid; i < c4; i += bdim) {
    const float4 v = xr[i];
    float4 o;
    o.x = __expf(v.x - row_max) * inv;
    o.y = __expf(v.y - row_max) * inv;
    o.z = __expf(v.z - row_max) * inv;
    o.w = __expf(v.w - row_max) * inv;
    yr[i] = o;
  }
}
