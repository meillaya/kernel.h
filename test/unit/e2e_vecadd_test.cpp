// FIRST kernel e2e (plan todo 25): vecadd correctness through the FULL
// pipeline, at three sizes, plus the wrong-reference negative control (QA
// scenario: prove the numeric comparison has teeth, via Catch2 [!shouldfail]).
//
// Full-GPU path (degraded=false): for each n in {1024, 65536, 1<<20}:
//   kernelh compile --kernel vecadd --shape n=<n> --target amdgcn-gfx1101
//     -> vecadd.hsaco         (todo 24 production driver via std::system)
//   -> GpuModule::load + get_function("vecadd")
//   -> KernelLaunch grid=ceil(n/256) block=256, 3 memref args (a, b, c)
//   -> copy_to_host -> host comparison against the FP64 CPU reference cast
//     to fp32 (plan Metis G11): max absolute difference <= 1e-6, printed.
//
// Degraded path (todo 6 degrade branch): no launches - compile only, then
// assert the HSACO and PTX artifacts carry the kernel symbol (binary substring
// probe). On this box (degraded=false) the numeric path is what executes.

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
#include <string>
#include <vector>

namespace {

// Exact-in-f32 inputs (multiples of 0.5 / 0.25; b shifted by +0.5 so it is
// never zero - the negative control needs a+b != a-b on EVERY element), so
// the kernel's f32 add is correctly rounded exactly and the fp64 reference
// matches to 0.0 ulp.
void fill_inputs(std::vector<float> &a, std::vector<float> &b) {
  const size_t n = a.size();
  for (size_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>((i % 257)) * 0.5f;
    b[i] = static_cast<float>((i % 331)) * 0.25f + 0.5f;
  }
}

struct E2eResult {
  long n = 0;
  double max_abs_diff = 0.0;
};

// One full pipeline pass: compile -> load -> launch -> copy -> verify.
// `want_ptx` makes the compile also emit the nvptx artifacts (degraded-mode
// artifact checks use both sides).
E2eResult run_vecadd_e2e(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                         long n, long block) {
  const std::string outDir = "/tmp/kernelh-t25-e2e-" + std::to_string(n);
  const int rc = kernelh::test::compile_kernel(
      "vecadd", "n=" + std::to_string(n), {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);

  const std::string hsacoPath = outDir + "/vecadd.hsaco";
  REQUIRE(kernelh::test::file_exists(hsacoPath));
  const std::vector<char> blob = kernelh::test::read_file_bin(hsacoPath);
  REQUIRE(kernelh::test::is_elf_magic(blob));
  REQUIRE(kernelh::test::artifact_contains_symbol(hsacoPath, "vecadd"));

  const size_t bytes = sizeof(float) * static_cast<size_t>(n);
  std::vector<float> h_a(static_cast<size_t>(n));
  std::vector<float> h_b(static_cast<size_t>(n));
  std::vector<float> h_c(static_cast<size_t>(n), 0.0f);
  fill_inputs(h_a, h_b);

  auto a_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto b_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto c_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(a_buf.has_value());
  REQUIRE(b_buf.has_value());
  REQUIRE(c_buf.has_value());
  REQUIRE(a_buf->copy_from_host(h_a.data(), bytes).has_value());
  REQUIRE(b_buf->copy_from_host(h_b.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("vecadd");
  REQUIRE(fn.has_value());

  const long grid = (n + block - 1) / block;
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(grid)).block(static_cast<unsigned>(block));
  launch.memref_arg(a_buf->device_ptr(), std::vector<int64_t>{n});
  launch.memref_arg(b_buf->device_ptr(), std::vector<int64_t>{n});
  launch.memref_arg(c_buf->device_ptr(), std::vector<int64_t>{n});
  REQUIRE(launch.args().param_count() == 15); // 3 buffers x 5 descriptor fields

  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(c_buf->copy_to_host(h_c.data(), bytes).has_value());

  // Metis G11 reference: fp64 CPU computation cast to fp32, max abs diff.
  E2eResult result;
  result.n = n;
  for (long i = 0; i < n; ++i) {
    const float ref =
        static_cast<float>(static_cast<double>(h_a[static_cast<size_t>(i)]) +
                           static_cast<double>(h_b[static_cast<size_t>(i)]));
    const double diff =
        std::fabs(static_cast<double>(h_c[static_cast<size_t>(i)]) -
                  static_cast<double>(ref));
    if (diff > result.max_abs_diff) {
      result.max_abs_diff = diff;
    }
  }
  return result;
}

// Degraded-mode artifact assertion (todo 6 degrade branch): compile for BOTH
// targets and probe the blobs for the kernel symbol - no GPU needed.
void assert_degraded_artifacts(long n) {
  const std::string outDir = "/tmp/kernelh-t25-e2e-" + std::to_string(n);
  const int rc = kernelh::test::compile_kernel(
      "vecadd", "n=" + std::to_string(n),
      {"amdgcn-gfx1101", "nvptx-sm_80"}, outDir);
  REQUIRE(rc == 0);
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/vecadd.hsaco",
                                                  "vecadd"));
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/vecadd.ptx",
                                                  "vecadd"));
}

} // namespace

TEST_CASE("e2e_vecadd", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    // Degraded mode: compile + artifact symbol checks only (no launches).
    for (const long n : {1024L, 65536L, 1L << 20}) {
      assert_degraded_artifacts(n);
    }
    std::printf("E2E-VECADD-DEGRADED-ARTIFACTS-OK\n");
    return;
  }

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  constexpr long kBlock = 256;
  double worst = 0.0;
  for (const long n : {1024L, 65536L, 1L << 20}) {
    const E2eResult result = run_vecadd_e2e(backend, n, kBlock);
    // Tolerance pinned in plan todo 25 (Metis G11): max abs diff <= 1e-6.
    REQUIRE(result.max_abs_diff <= 1e-6);
    worst = std::max(worst, result.max_abs_diff);
    std::printf("E2E-VECADD n=%ld max-abs-diff=%.9f (ref: fp64->fp32)\n", n,
                result.max_abs_diff);
  }
  std::printf("E2E-VECADD-PASSED (sizes=3, worst-max-abs-diff=%.9f)\n", worst);
}

TEST_CASE("e2e_vecadd_wrong_reference_negative_control", "[gpu][!shouldfail]") {
  // QA scenario (test-the-test): this case deliberately asserts the WRONG
  // reference c[i] == a[i]-b[i] against the CORRECT vecadd kernel (which
  // computes c[i] = a[i]+b[i]). With a working numeric check this MUST fail
  // loudly, and [!shouldfail] converts the designed failure into a
  // suite-green result. If this case ever fails (i.e. the wrong-reference
  // assertion PASSES), the value comparison is rubber-stamping and the whole
  // e2e harness is broken - the acceptance would catch it here.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kBlock = 256;
  constexpr long kN = 1024;

  const std::string outDir = "/tmp/kernelh-t25-negctrl";
  const int rc = kernelh::test::compile_kernel(
      "vecadd", "n=" + std::to_string(kN), {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);
  const std::vector<char> blob =
      kernelh::test::read_file_bin(outDir + "/vecadd.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob));

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  const size_t bytes = sizeof(float) * static_cast<size_t>(kN);
  std::vector<float> h_a(static_cast<size_t>(kN));
  std::vector<float> h_b(static_cast<size_t>(kN));
  std::vector<float> h_c(static_cast<size_t>(kN), 0.0f);
  fill_inputs(h_a, h_b);

  auto a_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto b_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto c_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(a_buf.has_value());
  REQUIRE(b_buf.has_value());
  REQUIRE(c_buf.has_value());
  REQUIRE(a_buf->copy_from_host(h_a.data(), bytes).has_value());
  REQUIRE(b_buf->copy_from_host(h_b.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("vecadd");
  REQUIRE(fn.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(kN / kBlock)).block(static_cast<unsigned>(kBlock));
  launch.memref_arg(a_buf->device_ptr(), std::vector<int64_t>{kN});
  launch.memref_arg(b_buf->device_ptr(), std::vector<int64_t>{kN});
  launch.memref_arg(c_buf->device_ptr(), std::vector<int64_t>{kN});
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(c_buf->copy_to_host(h_c.data(), bytes).has_value());

  // WRONG reference: expect the kernel output to equal a[i] - b[i]. The real
  // kernel computes a[i] + b[i] and b[i] >= 0.25 everywhere, so a+b never
  // equals a-b: this REQUIRE must fail on every element. A PASS here means the
  // comparison is rubber-stamping - [!shouldfail] turns that into a failure.
  long mismatches = 0;
  for (long i = 0; i < kN; ++i) {
    const float wrong =
        static_cast<float>(static_cast<double>(h_a[static_cast<size_t>(i)]) -
                           static_cast<double>(h_b[static_cast<size_t>(i)]));
    if (h_c[static_cast<size_t>(i)] != wrong) {
      ++mismatches;
    }
  }
  std::printf("NEGATIVE-CONTROL mismatches-vs-wrong-ref=%ld of n=%ld "
              "(expected %ld with a correct kernel)\n",
              mismatches, kN, kN);
  REQUIRE(mismatches == 0);
}
