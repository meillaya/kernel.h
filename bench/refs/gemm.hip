#include <hip/hip_runtime.h>
// Reference GEMM kernel (plan todo 35, backend (c)): a naive tiled GEMM
// (32x32 tiles, static shared staging) as the hand-written HIP baseline.
//
// ABI (raw pointers):
//   gemm_tiled(const float* a, const float* b, float* c, int m, int n, int k)
//
// C = A*B, A is MxK row-major, B is KxN row-major, C is MxN row-major.
// Launch: grid = (n/32, m/32), block = (32, 8) = 256 threads; the 32x32 A/B
// tiles are staged into shared memory with a 256-thread strided load loop
// (1024 elements per tile), then each thread accumulates 32 MACs from shared.
// Boundary guards keep out-of-range tile elements zeroed so ragged edges
// (M/N/K not divisible by 32) stay correct. Static shared only (plan G9).

extern "C" __global__ void gemm_tiled(const float *__restrict__ a,
                                      const float *__restrict__ b,
                                      float *__restrict__ c, int m, int n,
                                      int k) {
  __shared__ float sa[32][32];
  __shared__ float sb[32][32];

  const int tx = threadIdx.x; // [0, 32)
  const int ty = threadIdx.y; // [0, 8)
  const int row = blockIdx.y * 32 + ty;
  const int col = blockIdx.x * 32 + tx;
  const int tid = ty * 32 + tx; // flat thread id in [0, 256)

  float acc = 0.0f;
  for (int kk = 0; kk < k; kk += 32) {
    // Stage A tile [32][32] (1024 elements, 256 threads -> 4 rounds).
    for (int idx = tid; idx < 1024; idx += 256) {
      const int i = idx >> 5; // idx / 32
      const int j = idx & 31; // idx % 32
      const int gi = row;
      const int gj = kk + j;
      sa[i][j] = (gi < m && gj < k) ? a[(long)gi * k + gj] : 0.0f;
    }
    // Stage B tile [32][32].
    for (int idx = tid; idx < 1024; idx += 256) {
      const int i = idx >> 5;
      const int j = idx & 31;
      const int gi = kk + i;
      const int gj = col;
      sb[i][j] = (gi < k && gj < n) ? b[(long)gi * n + gj] : 0.0f;
    }
    __syncthreads();
    for (int i = 0; i < 32; ++i) {
      acc += sa[ty][i] * sb[i][tx];
    }
    __syncthreads();
  }
  if (row < m && col < n) {
    c[(long)row * n + col] = acc;
  }
}
