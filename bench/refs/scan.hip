#include <hip/hip_runtime.h>
// Reference scan kernels (plan todo 35, backend (c)): hand-written HIP
// versions of the todo-29 kernelh two-pass Blelloch scan.
//
// ABI (raw pointers):
//   scan_pass1_naive(const float* x, float* y, float* totals, int n, int t)
//   scan_pass1_vec4 (const float* x, float* y, float* totals, int n, int t)
//   scan_totals_naive(const float* totals, float* y, int t, int n)
//   scan_totals_vec4 (const float* totals, float* y, int t, int n)
//
// Pass 1: grid = ceil(n/256), 256-thread workgroups. Each block computes the
// INCLUSIVE prefix scan of its 256-element chunk (padded with zeros) into y
// and writes the chunk total to totals[block]. Hillis-Steele over a
// ping-pong shared pair (no read/write race: the read source buffer is never
// the write target in the same iteration).
//
// Pass 2: single 256-thread workgroup. Exclusive prefix over the t block
// totals (thread-partials + Hillis-Steele), then each block's elements in y
// are incremented by their block prefix offset.
//
// vec4 variant: pass 1 processes 1024-element chunks (4 float4s per thread),
// per-thread partial sums scanned across the 256 threads, then the thread
// offset is added to all four lanes before the float4 store. Pass 2 is
// identical to the naive form (totals traffic dominates at t < 4096).

#define KERNELH_SCAN_CHUNK 256

extern "C" __global__ void scan_pass1_naive(const float *__restrict__ x,
                                            float *__restrict__ y,
                                            float *__restrict__ totals,
                                            int n, int t) {
  __shared__ float buf[2][KERNELH_SCAN_CHUNK];
  const int tid = threadIdx.x;
  const int base = blockIdx.x * KERNELH_SCAN_CHUNK;
  const float v = (base + tid < n) ? x[base + tid] : 0.0f;
  buf[0][tid] = v;
  __syncthreads();

  int src = 0, dst = 1;
  for (int stride = 1; stride < KERNELH_SCAN_CHUNK; stride <<= 1) {
    const float carry = (tid >= stride) ? buf[src][tid - stride] : 0.0f;
    buf[dst][tid] = buf[src][tid] + carry;
    src = 1 - src;
    dst = 1 - dst;
    __syncthreads();
  }

  if (base + tid < n) {
    y[base + tid] = buf[src][tid];
  }
  if (tid == KERNELH_SCAN_CHUNK - 1) {
    totals[blockIdx.x] = buf[src][KERNELH_SCAN_CHUNK - 1];
  }
}

extern "C" __global__ void scan_pass1_vec4(const float *__restrict__ x,
                                           float *__restrict__ y,
                                           float *__restrict__ totals,
                                           int n, int t) {
  // 1024-element chunks: each thread loads 4 consecutive floats as one
  // float4, keeps 4 lanes in registers, then the 256 per-thread partial sums
  // are scanned across the block (Hillis-Steele over the partials) and the
  // exclusive thread offset is added to all 4 lanes before the float4 store.
  __shared__ float partials[2][256];
  __shared__ float offsets[256];
  const int tid = threadIdx.x;
  const int base = blockIdx.x * 1024;
  const int elem = base + tid * 4;
  const bool inRange = (elem + 3) < n || elem < n;

  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  if (elem + 3 < n) {
    const float4 v = *reinterpret_cast<const float4 *>(x + elem);
    v0 = v.x;
    v1 = v.y;
    v2 = v.z;
    v3 = v.w;
  } else if (inRange) { // ragged tail chunk: scalar fallback for the last
    // partial elements (keeps the vectorized fast path aligned).
    if (elem < n) v0 = x[elem];
    if (elem + 1 < n) v1 = x[elem + 1];
    if (elem + 2 < n) v2 = x[elem + 2];
    if (elem + 3 < n) v3 = x[elem + 3];
  }
  partials[0][tid] = (v0 + v1) + (v2 + v3);
  __syncthreads();

  int src = 0, dst = 1;
  for (int stride = 1; stride < 256; stride <<= 1) {
    const float carry = (tid >= stride) ? partials[src][tid - stride] : 0.0f;
    partials[dst][tid] = partials[src][tid] + carry;
    src = 1 - src;
    dst = 1 - dst;
    __syncthreads();
  }
  offsets[tid] = (tid == 0) ? 0.0f : partials[src][tid - 1];
  __syncthreads();

  const float run = offsets[tid];
  if (inRange) {
    if (elem + 3 < n) {
      float4 o;
      o.x = v0 + run;
      o.y = v1 + run;
      o.z = v2 + run;
      o.w = v3 + run;
      *reinterpret_cast<float4 *>(y + elem) = o;
    } else {
      if (elem < n) y[elem] = v0 + run;
      if (elem + 1 < n) y[elem + 1] = v1 + run;
      if (elem + 2 < n) y[elem + 2] = v2 + run;
      if (elem + 3 < n) y[elem + 3] = v3 + run;
    }
  }
  // The block total is the LAST thread's inclusive prefix (it covers every
  // thread's partial sum exactly once).
  if (tid == 255) {
    totals[blockIdx.x] = partials[src][255];
  }
}

// Shared two-pass totals-add body (device helper; a __global__ may not call
// a __global__ in HIP/CUDA, so both entry points delegate here).
__device__ void scan_totals_impl(const float *__restrict__ totals,
                                 float *__restrict__ y, int t, int n) {
  __shared__ float partials[2][256];
  __shared__ float offsets[256];
  const int tid = threadIdx.x;
  const int bdim = blockDim.x;

  // Per-thread partial sum over its strided slice of the t block totals.
  float part = 0.0f;
  for (int i = tid; i < t; i += bdim) {
    part += totals[i];
  }
  partials[0][tid] = part;
  __syncthreads();

  // Exclusive scan over the thread partials.
  int src = 0, dst = 1;
  for (int stride = 1; stride < bdim; stride <<= 1) {
    const float carry = (tid >= stride) ? partials[src][tid - stride] : 0.0f;
    partials[dst][tid] = partials[src][tid] + carry;
    src = 1 - src;
    dst = 1 - dst;
    __syncthreads();
  }
  offsets[tid] = (tid == 0) ? 0.0f : partials[src][tid - 1];
  __syncthreads();

  // Add each block's exclusive prefix offset into its 256-element chunk.
  float run = offsets[tid];
  for (int i = tid; i < t; i += bdim) {
    const int base = i * KERNELH_SCAN_CHUNK;
    for (int j = 0; j < KERNELH_SCAN_CHUNK; ++j) {
      if (base + j < n) {
        y[base + j] += run;
      }
    }
    run += totals[i];
  }
}

extern "C" __global__ void scan_totals_naive(const float *__restrict__ totals,
                                             float *__restrict__ y, int t,
                                             int n) {
  scan_totals_impl(totals, y, t, n);
}

extern "C" __global__ void scan_totals_vec4(const float *__restrict__ totals,
                                            float *__restrict__ y, int t,
                                            int n) {
  scan_totals_impl(totals, y, t, n);
}
