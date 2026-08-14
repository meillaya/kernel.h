// test/fixtures/vectoradd.hip
// AMD runtime smoke gate fixture (atomgpu plan todo 6, stage 2).
//
// HIP vector add over 1<<20 floats. Constraints pinned by the plan:
//   - plain hipMalloc ONLY (D11: never hipMallocAsync/hipMallocFromPoolAsync)
//   - no WMMA/MFMA intrinsics
//   - 256-thread blocks, sharedMemBytes=0
//
// Prints VECTORADD-PASS only if EVERY element is verified correct host-side
// (exact float comparison: operands are small integers, sums are exact).
// Any HIP error or mismatch -> clear error on stderr + non-zero exit.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    hipError_t hip_err_ = (expr);                                             \
    if (hip_err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "HIP error %s:%d: %s: %s (%d)\n", __FILE__,        \
                   __LINE__, #expr, hipGetErrorString(hip_err_),              \
                   static_cast<int>(hip_err_));                               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__global__ void vectoradd(const float* a, float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int N = 1 << 20;
  const size_t bytes = sizeof(float) * static_cast<size_t>(N);

  float* h_a = static_cast<float*>(std::malloc(bytes));
  float* h_b = static_cast<float*>(std::malloc(bytes));
  float* h_c = static_cast<float*>(std::malloc(bytes));
  if (!h_a || !h_b || !h_c) {
    std::fprintf(stderr, "ERROR: host malloc failed\n");
    return 1;
  }
  for (int i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(i % 1000);
    h_b[i] = static_cast<float>((i * 3) % 1000);
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));
  HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

  const int block = 256;
  const int grid = N / block;  // 1<<20 divisible by 256 -> exact cover
  hipLaunchKernelGGL(vectoradd, dim3(grid), dim3(block), 0, 0, d_a, d_b, d_c,
                     N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

  long bad = 0;
  for (int i = 0; i < N; ++i) {
    const float want = h_a[i] + h_b[i];
    if (h_c[i] != want) {
      if (bad < 8) {
        std::fprintf(stderr, "MISMATCH i=%d got=%f want=%f\n", i,
                     static_cast<double>(h_c[i]), static_cast<double>(want));
      }
      ++bad;
    }
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  std::free(h_a);
  std::free(h_b);
  std::free(h_c);

  if (bad != 0) {
    std::fprintf(stderr, "VECTORADD-FAIL: %ld/%d elements incorrect\n", bad,
                 N);
    return 1;
  }
  std::printf("VECTORADD-PASS\n");
  return 0;
}
