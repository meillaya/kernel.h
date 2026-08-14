// Cross harness: cuBLAS cublasGemmEx + CUTLASS GEMM baselines (plan todo 35,
// backend (d)). Compiled with nvcc for sm_80 (cross - never runs on this
// box, which has no NVIDIA driver).
//
// Runtime contract (Metis G24): when no NVIDIA driver is present the harness
// prints "skip: no CUDA driver" and exits 77 BEFORE touching any CUDA API
// beyond the availability probe. On an NVIDIA machine it runs the GEMM
// matrix {512,1024,2048,4096,8192} x {cublas, cutlass} with the pinned
// warmup-5 + median-of-20 event methodology and appends G32-schema rows to
// the --out CSV.
//
// CUTLASS build-failed fallback (Oracle N6): if the CUTLASS portion failed
// to compile (KERNELH_CROSS_NO_CUTLASS defined by the build), a
// `cutlass,build-failed` row is recorded per GEMM shape and the matrix
// continues cuBLAS-only - never a silent missing baseline.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef KERNELH_CROSS_HAS_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#endif

namespace {

constexpr int kExitOk = 0;
constexpr int kExitNoDriver = 77;

const int kGemmSizes[] = {512, 1024, 2048, 4096, 8192};

std::string nowTimestamp() {
  char buf[32];
  const std::time_t t = std::time(nullptr);
  std::tm tm = {};
  localtime_r(&t, &tm);
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
  return buf;
}

std::string csvEscape(const std::string &field) {
  if (field.find(',') == std::string::npos &&
      field.find('"') == std::string::npos &&
      field.find('\n') == std::string::npos) {
    return field;
  }
  std::string out = "\"";
  for (const char c : field) {
    if (c == '"') {
      out.push_back('"');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

// Median of a sample vector (NaN-safe sort like the kernelh harness).
double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  if (n % 2 == 1) {
    return samples[n / 2];
  }
  return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

struct Timer {
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;

  bool create() {
    return cudaEventCreate(&start) == cudaSuccess &&
           cudaEventCreate(&end) == cudaSuccess;
  }
  void begin() { cudaEventRecord(start, 0); }
  // Ends + synchronizes + returns elapsed ms, or -1 on failure.
  double endMs() {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start, end) != cudaSuccess) {
      return -1.0;
    }
    return static_cast<double>(ms);
  }
  ~Timer() {
    if (start) {
      cudaEventDestroy(start);
    }
    if (end) {
      cudaEventDestroy(end);
    }
  }
};

void writeRow(std::FILE *out, const std::string &kernel,
              const std::string &shape, const std::string &backend,
              double ms, const std::string &gflopsOrGbps,
              const std::string &device) {
  char msBuf[32];
  std::snprintf(msBuf, sizeof(msBuf), "%.6f", ms);
  std::fprintf(out, "%s,%s,%s,%s,%s,%s,%s\n", kernel.c_str(),
               shape.c_str(), backend.c_str(), msBuf, gflopsOrGbps.c_str(),
               nowTimestamp().c_str(), device.c_str());
}

std::string shapeText(int n) {
  return "m=" + std::to_string(n) + ",n=" + std::to_string(n) +
         ",k=" + std::to_string(n);
}

// Times `fn` with warmup 5 + median of 20 CUDA-event samples.
template <typename Fn> double timeMedian(const Fn &fn) {
  Timer timer;
  if (!timer.create()) {
    return -1.0;
  }
  for (int i = 0; i < 5; ++i) {
    if (!fn()) {
      return -1.0;
    }
  }
  cudaDeviceSynchronize();
  std::vector<double> samples;
  samples.reserve(20);
  for (int i = 0; i < 20; ++i) {
    timer.begin();
    if (!fn()) {
      return -1.0;
    }
    const double ms = timer.endMs();
    if (ms < 0.0) {
      return -1.0;
    }
    samples.push_back(ms);
  }
  return median(std::move(samples));
}

// ---------------------------------------------------------------------------
// cuBLAS cublasGemmEx fp32 baseline.
// ---------------------------------------------------------------------------
bool runCublas(std::FILE *out, cublasHandle_t handle, int n,
               float *dA, float *dB, float *dC,
               const std::string &device) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // CUDA 12 cublasGemmEx is the 19-arg in-place form (C = alpha*A*B +
  // beta*C): the compute type replaces the old rocblas-style solution_index/
  // flags tail.
  const auto oneGemm = [&]() {
    const cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, CUDA_R_32F, n,
        dB, CUDA_R_32F, n, &beta, dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS;
  };
  const double ms = timeMedian(oneGemm);
  if (ms < 0.0) {
    std::fprintf(stderr, "cross: cublas gemm n=%d failed/timed-out\n", n);
    return false;
  }
  const double gflops =
      2.0 * static_cast<double>(n) * n * n / (ms * 1e-3) / 1e9;
  char rate[32];
  std::snprintf(rate, sizeof(rate), "%.3f", gflops);
  writeRow(out, "gemm", shapeText(n), "cublas", ms, rate, device);
  return true;
}

#ifdef KERNELH_CROSS_HAS_CUTLASS
// ---------------------------------------------------------------------------
// CUTLASS fp32 SIMT GEMM baseline (sm_80). The SIMT specialization of
// device::Gemm (cutlass/gemm/kernel/default_gemm.h) pins the instruction
// shape to GemmShape<1,1,1> (scalar MACs) and requires a scalar epilogue
// (LinearCombination<..., 1, ...>); the threadblock/warp tile shapes follow
// the classic 05_basic_gemm layout (128x128x8 / 64x64x8), 4-stage pipeline,
// RowMajor everywhere.
// ---------------------------------------------------------------------------
using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4>;

bool runCutlass(std::FILE *out, int n, float *dA, float *dB, float *dC,
                float *dD, const std::string &device) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CutlassGemm gemm;
  // CUTLASS 4.x Arguments: problem size + TensorRefs (RowMajor(extent)
  // yields the row-major leading dimension = columns) + epilogue params.
  typename CutlassGemm::Arguments args(
      {n, n, n},
      cutlass::TensorRef<const float, cutlass::layout::RowMajor>(
          dA, cutlass::layout::RowMajor(n)),
      cutlass::TensorRef<const float, cutlass::layout::RowMajor>(
          dB, cutlass::layout::RowMajor(n)),
      cutlass::TensorRef<const float, cutlass::layout::RowMajor>(
          dC, cutlass::layout::RowMajor(n)),
      cutlass::TensorRef<float, cutlass::layout::RowMajor>(
          dD, cutlass::layout::RowMajor(n)),
      {alpha, beta});
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "cross: cutlass can_implement failed for n=%d\n", n);
    return false;
  }
  gemm.initialize(args);
  const auto oneGemm = [&]() {
    gemm();
    return true;
  };
  const double ms = timeMedian(oneGemm);
  if (ms < 0.0) {
    std::fprintf(stderr, "cross: cutlass gemm n=%d failed/timed-out\n", n);
    return false;
  }
  const double gflops =
      2.0 * static_cast<double>(n) * n * n / (ms * 1e-3) / 1e9;
  char rate[32];
  std::snprintf(rate, sizeof(rate), "%.3f", gflops);
  writeRow(out, "gemm", shapeText(n), "cutlass", ms, rate, device);
  return true;
}
#endif // KERNELH_CROSS_HAS_CUTLASS

} // namespace

int main(int argc, char **argv) {
  std::string outPath = "results.csv";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      outPath = argv[++i];
    } else if (std::strncmp(argv[i], "--out=", 6) == 0) {
      outPath = argv[i] + 6;
    } else {
      std::fprintf(stderr, "cross: unknown arg '%s' (usage: cross_harness "
                           "--out <csv>)\n",
                   argv[i]);
      return 2;
    }
  }

  // Availability probe FIRST (Metis G24): no driver -> exit 77 before any
  // CUDA work beyond the count query.
  int devCount = 0;
  if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
    std::fprintf(stderr, "skip: no CUDA driver\n");
    return kExitNoDriver;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "skip: no CUDA driver (set device failed)\n");
    return kExitNoDriver;
  }
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::fprintf(stderr, "skip: no CUDA driver (props failed)\n");
    return kExitNoDriver;
  }
  const std::string device = prop.name;

  std::FILE *out = std::fopen(outPath.c_str(), "w");
  if (!out) {
    std::fprintf(stderr, "cross: cannot open '%s' for writing\n",
                 outPath.c_str());
    return 1;
  }
  std::fprintf(out, "kernel,shape,backend,ms_median,gflops_or_gbps,"
                    "timestamp,device\n");

#ifdef KERNELH_CROSS_NO_CUTLASS
  // Oracle N6: CUTLASS failed to build - record build-failed rows, continue
  // cuBLAS-only.
  for (const int n : kGemmSizes) {
    writeRow(out, "gemm", shapeText(n), "cutlass", 0.0, "build-failed",
             device);
  }
#endif

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cross: cublasCreate failed\n");
    std::fclose(out);
    return 1;
  }

  bool ok = true;
  for (const int n : kGemmSizes) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    if (cudaMalloc(&dA, bytes) != cudaSuccess ||
        cudaMalloc(&dB, bytes) != cudaSuccess ||
        cudaMalloc(&dC, bytes) != cudaSuccess ||
        cudaMalloc(&dD, bytes) != cudaSuccess) {
      std::fprintf(stderr, "cross: cudaMalloc failed at n=%d\n", n);
      ok = false;
      break;
    }
    // Uninitialized input is fine for pure timing (no numeric verification
    // here - the kernelh e2e owns correctness).
    cudaMemset(dA, 0, bytes);
    cudaMemset(dB, 0, bytes);

    ok &= runCublas(out, handle, n, dA, dB, dC, device);
#ifdef KERNELH_CROSS_HAS_CUTLASS
    ok &= runCutlass(out, n, dA, dB, dC, dD, device);
#endif

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
  }

  cublasDestroy(handle);
  std::fclose(out);
  return ok ? kExitOk : 1;
}
