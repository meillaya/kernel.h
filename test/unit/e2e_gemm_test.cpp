// GEMM kernel e2e (plan todo 26): C = A*B through the FULL pipeline, with
// the tiled shared-memory schedule. Covers:
//   - the 5 power-of-two square sizes {128, 512, 1024, 2048, 4096} plus the
//     non-multiple 1000x1000x1000 case (1000 % BM != 0 -> the tile pass's
//     boundary predicates must fire and keep every access in bounds),
//   - the fp64 CPU reference cast to fp32 with the relative Frobenius norm
//     (plan Metis G11 tolerance: ||C_gpu - C_ref||_F / ||C_ref||_F <= 1e-5),
//   - a wrong-reference negative control ([!shouldfail]): comparing against
//     A*B^T must FAIL when the kernel is correct (proves the numeric check
//     has teeth),
//   - a compile-time LDS-overflow probe: a tile config whose staged shared
//     buffers exceed the 64 KiB LDS budget must fire the promote-shared
//     diagnostic BEFORE any launch (non-zero exit, message names LDS).
//
// Full-GPU path (degraded=false): for each size N:
//   kernelh compile --kernel gemm --shape m=N,n=N,k=N --target amdgcn-gfx1101
//     -> gemm.hsaco (todo 24 production driver via std::system; the template
//        applies the default tile(32,32,32).map_blocks() (map_warps OFF)
//        use_shared_memory() schedule - the CLI has no schedule flags; the
//        tiled shared-memory GEMM pre-stages its A/B tiles through static
//        LDS with tid-partitioned staging loops and per-tile barriers)
//   -> GpuModule::load + get_function("gemm")
//   -> KernelLaunch grid=ceil(N/32) block=256, 3 rank-2 memref args
//   -> copy_to_host -> relative Frobenius vs the fp64 CPU reference.
//
// Inputs are exact-in-f32 integers (values in [0,64) with products <= 3969);
// a K-deep partial sum stays < 2^24 = 16777216 (max 4096*3969 = 16257024),
// so every fma is exactly rounded and the fp32 kernel result equals the fp64
// reference bit-for-bit (Frobenius diff ~0). This removes accumulation-order
// noise from the tolerance discussion entirely.
//
// Degraded path (todo 6 degrade branch): no launches - compile for BOTH
// targets and assert the artifacts carry the kernel symbol (binary substring
// probe) plus the s_barrier tile-loop barriers the shared-memory schedule
// lowers to in the AMD ISA. On this box
// (degraded=false) the numeric path is what executes.

#include "kernelh/dsl/program.hpp"
#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include "gpu_gate.hpp"
#include "kernelh_compile_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#endif

namespace {

// Exact-in-f32 integer inputs: values in [0, 64), products <= 63*63 = 3969,
// and any K-deep partial sum (K <= 4096 here) stays < 2^24, so every fp32
// fma is exactly rounded (see file header).
inline float fill_a_value(long i, long k) {
  return static_cast<float>((i * 131 + k * 17 + 7) & 63);
}
inline float fill_b_value(long k, long j) {
  return static_cast<float>((k * 97 + j * 41 + 3) & 63);
}

// The tile extent applied by the gemm template's default schedule (BM=32;
// grid = ceil(M/BM), one 32-row tile per block computed cooperatively by
// the 256 threads over the columns).
constexpr long kTile = 32;
// Launch block = 256 threads: the compiled kernel's max_flat_workgroup_size
// is 256 (larger blocks are rejected at launch); each thread owns the
// columns j = tid + it*256 and the per-thread work stays bounded.

// fp64 CPU reference cast to fp32: C[i,j] = sum_k a[i,k] * b[k,j], computed
// as a rank-1 update (per-row running accumulators) so the inner loop walks
// B row-major (contiguous) and the compiler can SIMD-vectorize it - the
// scalar per-element inner product over K is ~15x slower on this host. The
// inputs are exact integers whose partial sums are exactly representable in
// fp64, so the accumulation order does not matter: the rank-1 form equals the
// per-element form bit-for-bit.
void fill_reference(const std::vector<float> &a, const std::vector<float> &b,
                    std::vector<float> &ref, long m, long n, long k) {
  std::vector<double> acc(static_cast<size_t>(n), 0.0);
  for (long i = 0; i < m; ++i) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (long kk = 0; kk < k; ++kk) {
      const double av = static_cast<double>(a[static_cast<size_t>(i * k + kk)]);
      const float *bRow = &b[static_cast<size_t>(kk * n)];
      for (long j = 0; j < n; ++j) {
        acc[static_cast<size_t>(j)] += av * static_cast<double>(bRow[j]);
      }
    }
    for (long j = 0; j < n; ++j) {
      ref[static_cast<size_t>(i * n + j)] = static_cast<float>(acc[static_cast<size_t>(j)]);
    }
  }
}

// Relative Frobenius norm of c - ref over the M x N outputs.
double relative_frobenius(const std::vector<float> &c,
                          const std::vector<float> &ref, size_t count) {
  double diffSq = 0.0;
  double refSq = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double d = static_cast<double>(c[i]) - static_cast<double>(ref[i]);
    diffSq += d * d;
    const double r = static_cast<double>(ref[i]);
    refSq += r * r;
  }
  return std::sqrt(diffSq / refSq);
}

struct GemmResult {
  long n = 0;
  double rel_frobenius = 0.0;
};

// One full pipeline pass: compile -> load -> launch -> copy -> verify.
GemmResult run_gemm_e2e(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                        long n) {
  const long m = n;
  const long k = n;
  const std::string outDir = "/tmp/kernelh-t26-e2e-" + std::to_string(n);
  const std::string shapeText =
      "m=" + std::to_string(m) + ",n=" + std::to_string(n) + ",k=" +
      std::to_string(k);
  const int rc = kernelh::test::compile_kernel(
      "gemm", shapeText, {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);

  const std::string hsacoPath = outDir + "/gemm.hsaco";
  REQUIRE(kernelh::test::file_exists(hsacoPath));
  const std::vector<char> blob = kernelh::test::read_file_bin(hsacoPath);
  REQUIRE(kernelh::test::is_elf_magic(blob));
  REQUIRE(kernelh::test::artifact_contains_symbol(hsacoPath, "gemm"));

  const size_t aCount = static_cast<size_t>(m) * static_cast<size_t>(k);
  const size_t bCount = static_cast<size_t>(k) * static_cast<size_t>(n);
  const size_t cCount = static_cast<size_t>(m) * static_cast<size_t>(n);
  std::vector<float> h_a(aCount);
  std::vector<float> h_b(bCount);
  std::vector<float> h_c(cCount, 0.0f);
  for (long i = 0; i < m; ++i) {
    for (long kk = 0; kk < k; ++kk) {
      h_a[static_cast<size_t>(i * k + kk)] = fill_a_value(i, kk);
    }
  }
  for (long kk = 0; kk < k; ++kk) {
    for (long j = 0; j < n; ++j) {
      h_b[static_cast<size_t>(kk * n + j)] = fill_b_value(kk, j);
    }
  }

  auto a_buf = kernelh::runtime::Buffer::allocate(backend, aCount * sizeof(float));
  auto b_buf = kernelh::runtime::Buffer::allocate(backend, bCount * sizeof(float));
  auto c_buf = kernelh::runtime::Buffer::allocate(backend, cCount * sizeof(float));
  REQUIRE(a_buf.has_value());
  REQUIRE(b_buf.has_value());
  REQUIRE(c_buf.has_value());
  REQUIRE(a_buf->copy_from_host(h_a.data(), aCount * sizeof(float)).has_value());
  REQUIRE(b_buf->copy_from_host(h_b.data(), bCount * sizeof(float)).has_value());
  REQUIRE(c_buf->copy_from_host(h_c.data(), cCount * sizeof(float)).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("gemm");
  REQUIRE(fn.has_value());

  const long grid = (n + kTile - 1) / kTile;
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(grid)).block(256u);
  launch.memref_arg(a_buf->device_ptr(), std::vector<int64_t>{m, k});
  launch.memref_arg(b_buf->device_ptr(), std::vector<int64_t>{k, n});
  launch.memref_arg(c_buf->device_ptr(), std::vector<int64_t>{m, n});
  // 3 rank-2 buffers x (2 + 2 + 3) descriptor slots.
  REQUIRE(launch.args().param_count() == 21);

  auto launchRes = launch.launch(*fn);
  if (!launchRes) {
    std::printf("E2E-GEMM n=%ld launch-error: %s\n", n,
                launchRes.error().message.c_str());
  }
  REQUIRE(launchRes.has_value());
  REQUIRE(c_buf->copy_to_host(h_c.data(), cCount * sizeof(float)).has_value());

  // Metis G11 reference: fp64 CPU computation cast to fp32, relative
  // Frobenius norm over the whole output.
  std::vector<float> ref(cCount);
  fill_reference(h_a, h_b, ref, m, n, k);
  GemmResult result;
  result.n = n;
  result.rel_frobenius = relative_frobenius(h_c, ref, cCount);
  return result;
}

// Degraded-mode artifact assertion (todo 6 degrade branch): compile for BOTH
// targets and probe the blobs for the kernel symbol + LDS instructions in the
// AMD ISA - no GPU needed.
void assert_degraded_artifacts(long n) {
  const std::string outDir = "/tmp/kernelh-t26-e2e-" + std::to_string(n);
  const std::string shapeText = "m=" + std::to_string(n) + ",n=" +
                                std::to_string(n) + ",k=" + std::to_string(n);
  const int rc = kernelh::test::compile_kernel(
      "gemm", shapeText, {"amdgcn-gfx1101", "nvptx-sm_80"}, outDir);
  REQUIRE(rc == 0);
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/gemm.hsaco",
                                                  "gemm"));
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/gemm.ptx",
                                                  "gemm"));
  const std::vector<char> isa =
      kernelh::test::read_file_bin(outDir + "/gemm.isa");
  const std::string isaText(isa.begin(), isa.end());
  // The template pre-stages its A/B tiles through static LDS (the scan.cpp
  // pattern), so the compiled kernel carries REAL shared-memory traffic: the
  // tile-loop barriers surface as s_barrier AND the staging stores + compute
  // loads surface as ds_* instructions in the AMD ISA.
  REQUIRE(isaText.find("s_barrier") != std::string::npos);
  REQUIRE(isaText.find("ds_") != std::string::npos);
}

// Runs a CLI command capturing stdout+stderr into `logPath`; returns the
// process exit code (WEXITSTATUS semantics).
int run_cli_capture(const std::string &command, const std::string &logPath) {
  const std::string cmd = command + " > '" + logPath + "' 2>&1";
  const int rc = std::system(cmd.c_str());
  if (rc == -1) {
    return -1;
  }
#ifdef _WIN32
  return rc;
#else
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : 1;
#endif
}

std::string read_text_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    return "";
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

} // namespace

TEST_CASE("e2e_gemm", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    // Degraded mode: compile + artifact symbol checks only (no launches).
    for (const long n : {128L, 512L, 1024L}) {
      assert_degraded_artifacts(n);
    }
    std::printf("E2E-GEMM-DEGRADED-ARTIFACTS-OK\n");
    return;
  }

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  std::setvbuf(stdout, nullptr, _IONBF, 0);
  double worst = 0.0;
  for (const long n : {128L, 512L, 1024L, 2048L, 4096L, 1000L}) {
    const GemmResult result = run_gemm_e2e(backend, n);
    // Tolerance pinned in plan todo 26 (Metis G11): relative Frobenius
    // <= 1e-5 vs the fp64 CPU reference cast to fp32.
    REQUIRE(result.rel_frobenius <= 1e-5);
    worst = std::max(worst, result.rel_frobenius);
    std::printf("E2E-GEMM n=%ld rel-frobenius=%.6e (ref: fp64->fp32)\n", n,
                result.rel_frobenius);
  }
  std::printf("E2E-GEMM-PASSED (sizes=6, worst-rel-frobenius=%.6e)\n", worst);
}

TEST_CASE("e2e_gemm_wrong_reference_negative_control", "[gpu][!shouldfail]") {
  // QA scenario (test-the-test): this case deliberately asserts the WRONG
  // reference C == A*B^T against the CORRECT gemm kernel (which computes
  // C = A*B). With a working numeric check this MUST fail loudly, and
  // [!shouldfail] converts the designed failure into a suite-green result.
  // If this case ever fails (i.e. the wrong-reference assertion PASSES), the
  // value comparison is rubber-stamping and the whole e2e harness is broken.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kN = 256;
  const long m = kN, k = kN;

  const std::string outDir = "/tmp/kernelh-t26-negctrl";
  const int rc = kernelh::test::compile_kernel(
      "gemm", "m=256,n=256,k=256", {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);
  const std::vector<char> blob =
      kernelh::test::read_file_bin(outDir + "/gemm.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob));

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  const size_t aCount = static_cast<size_t>(kN) * static_cast<size_t>(kN);
  const size_t cCount = aCount;
  std::vector<float> h_a(aCount);
  std::vector<float> h_b(aCount);
  std::vector<float> h_c(cCount, 0.0f);
  for (long i = 0; i < kN; ++i) {
    for (long kk = 0; kk < kN; ++kk) {
      h_a[static_cast<size_t>(i * kN + kk)] = fill_a_value(i, kk);
      h_b[static_cast<size_t>(i * kN + kk)] = fill_b_value(i, kk);
    }
  }

  auto a_buf = kernelh::runtime::Buffer::allocate(backend, aCount * sizeof(float));
  auto b_buf = kernelh::runtime::Buffer::allocate(backend, aCount * sizeof(float));
  auto c_buf = kernelh::runtime::Buffer::allocate(backend, cCount * sizeof(float));
  REQUIRE(a_buf.has_value());
  REQUIRE(b_buf.has_value());
  REQUIRE(c_buf.has_value());
  REQUIRE(a_buf->copy_from_host(h_a.data(), aCount * sizeof(float)).has_value());
  REQUIRE(b_buf->copy_from_host(h_b.data(), aCount * sizeof(float)).has_value());
  REQUIRE(c_buf->copy_from_host(h_c.data(), cCount * sizeof(float)).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("gemm");
  REQUIRE(fn.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(kN / kTile)).block(256u);
  launch.memref_arg(a_buf->device_ptr(), std::vector<int64_t>{m, k});
  launch.memref_arg(b_buf->device_ptr(), std::vector<int64_t>{k, kN});
  launch.memref_arg(c_buf->device_ptr(), std::vector<int64_t>{m, kN});
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(c_buf->copy_to_host(h_c.data(), cCount * sizeof(float)).has_value());

  // WRONG reference: expect the kernel output to equal A*B^T (i.e.
  // C_wrong[i,j] = sum_k a[i,k] * b[j,k]). The real kernel computes A*B, so
  // with these non-degenerate integer inputs essentially every element
  // differs and this REQUIRE must fail. A PASS here means the comparison is
  // rubber-stamping - [!shouldfail] turns that into a failure.
  long mismatches = 0;
  for (long i = 0; i < kN; ++i) {
    for (long j = 0; j < kN; ++j) {
      double wrong = 0.0;
      for (long kk = 0; kk < k; ++kk) {
        wrong += static_cast<double>(h_a[static_cast<size_t>(i * k + kk)]) *
                 static_cast<double>(h_b[static_cast<size_t>(j * k + kk)]);
      }
      const float wrongF = static_cast<float>(wrong);
      if (h_c[static_cast<size_t>(i * kN + j)] != wrongF) {
        ++mismatches;
      }
    }
  }
  std::printf("NEGATIVE-CONTROL mismatches-vs-AxB^T=%ld of n=%ld "
              "(expected %ld with a correct kernel)\n",
              mismatches, kN * kN, kN * kN);
  REQUIRE(mismatches == 0);
}

TEST_CASE("e2e_gemm_lds_overflow_diagnostic") {
  // Failure probe (plan todo 26 QA): a tile config whose staged shared
  // buffers exceed the 64 KiB LDS budget (tile(512,512,64) -> A/B/C each
  // 512x64 -> 384 KiB) must fire the promote-shared diagnostic during the
  // default pipeline, BEFORE any launch: non-zero exit, message names LDS.
  // The CLI has no schedule flags, so the oversized schedule is emitted
  // through the DSL in-process and compiled via the --input path.
  const std::string outDir = "/tmp/kernelh-t26-ldsprobe";
  const std::string rmCmd =
      "rm -rf '" + outDir + "' && mkdir -p '" + outDir + "'";
  REQUIRE(std::system(rmCmd.c_str()) == 0);

  const kernelh::EmittedKernel emitted =
      kernelh::program()
          .tile(512, 512, 64)
          .map_blocks()
          .map_warps()
          .use_shared_memory()
          .kernel("gemm")
          .shape({{"m", 2048}, {"n", 2048}, {"k", 2048}})
          .emit();
  const std::string atomPath = outDir + "/gemm_ldsprobe.atom";
  {
    std::ofstream out(atomPath, std::ios::binary);
    REQUIRE(out.good());
    out.write(emitted.atomText.data(),
              static_cast<std::streamsize>(emitted.atomText.size()));
  }

  const std::string logPath = outDir + "/compile.log";
  std::string cmd = "'";
  cmd += kernelh::test::kernelh_cli();
  cmd += "' compile --input '" + atomPath + "' --target amdgcn-gfx1101 --out '";
  cmd += outDir + "'";
  const int exitCode = run_cli_capture(cmd, logPath);
  REQUIRE(exitCode != 0);
  const std::string log = read_text_file(logPath);
  REQUIRE(log.find("LDS limit") != std::string::npos);
  std::printf("LDS-OVERFLOW-PROBE exit=%d diagnostic: %s\n", exitCode,
              log.find("shared memory usage") == std::string::npos
                  ? log.c_str()
                  : log.substr(log.find("shared memory usage")).c_str());
}
