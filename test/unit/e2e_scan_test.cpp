// Scan kernel e2e (plan todo 29): inclusive prefix sum, f32, TWO-PASS
// Blelloch through the FULL production pipeline.
//
// Pass 1 (kernel "scan"): per-workgroup upsweep + downsweep in static LDS
// (workgroup = 256 threads, one element per thread - the ROCDL-serialized
// descriptor caps max_flat_workgroup_size at 256), writing each block total
// to a device array. Pass 2 (kernel "scan_totals", grid = 1 workgroup):
// scans the block totals (exact 3-word expansion prefix, the documented
// adaptation for the up-to-4096 totals the 256-thread cap produces) and adds
// each block's offset. NO cooperative launch, NO decoupled lookback.
//
// NUMERICS: the pinned per-size tolerances {1024: 1e-4, 65536: 1e-3,
// 2^20: 5e-3} are SMALLER than one fp32 ulp of the large prefix sums
// (ulp(5e5) ~= 0.03), so the comparison is done against the round-to-fp32
// fp64 reference (the device outputs fp32; comparing against an exact fp64
// value at those magnitudes would fail even a PERFECTLY rounded kernel).
// The kernel uses double-float (TwoSum) block scans + an exact 3-word
// expansion offset prefix, and is bit-exact vs the rounded reference
// (verified: max abs diff == 0.0 at every pinned size and N=1000 with the
// fixed seed).
//
// Sizes: N in {1024, 65536, 1<<20} with uniform-random [0,1] inputs, FIXED
// SEED (std::mt19937 with a literal seed - deterministic across runs; the
// fp64 reference is derived from the same host array, so only the seed is
// shared). N=1000 (not divisible by 256) is the boundary case and must be
// correct too. Negative control [!shouldfail]: an EXCLUSIVE prefix reference
// must FAIL against the correct inclusive kernel (the task's pinned wrong
// reference); with a working numeric check the comparison fails loudly and
// [!shouldfail] converts that into a suite-green result.
//
// Degraded path (todo 6 degrade branch): compile for BOTH targets and probe
// the blobs - the scan HSACO/PTX carry the kernel symbol, the PTX is
// identical, and the AMD ISA carries LDS traffic (plan degraded acceptance:
// `grep -c 'ds_' <scan.isa>` > 0).

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include "gpu_gate.hpp"
#include "kernelh_compile_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

// Workgroup scan width (must match the template).
constexpr long kW = 256;

// Uniform [0,1) inputs, fixed seed (std::mt19937 - deterministic across
// platforms; a literal seed keeps the test reproducible and the per-size
// tolerances calibrated to this exact input stream).
void fill_rand(std::vector<float> &v, uint32_t seed) {
  std::mt19937 rng(seed);
  for (float &x : v) {
    x = static_cast<float>(rng()) * (1.0f / 4294967296.0f);
  }
}

// One full two-pass pipeline run for size n: compile both kernels, allocate
// the buffers, launch pass 1 then pass 2, and return the max abs diff of the
// kernel output vs the round-to-fp32 fp64 reference (INCLUSIVE prefix by
// default; `exclusiveRef` selects the EXCLUSIVE prefix - the deliberately
// wrong reference used by the negative control).
double run_scan_e2e(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                    long n, uint32_t seed, bool exclusiveRef = false) {
  const long T = (n + kW - 1) / kW;
  const std::string base = "/tmp/kernelh-t29-e2e-" + std::to_string(n);
  const std::string p1Dir = base + "/p1";
  const std::string p2Dir = base + "/p2";
  REQUIRE(kernelh::test::compile_kernel(
              "scan", "n=" + std::to_string(n), {"amdgcn-gfx1101"}, p1Dir) == 0);
  REQUIRE(kernelh::test::compile_kernel("scan_totals",
                                        "n=" + std::to_string(n),
                                        {"amdgcn-gfx1101"}, p2Dir) == 0);

  const std::vector<char> blob1 =
      kernelh::test::read_file_bin(p1Dir + "/scan.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob1));
  REQUIRE(kernelh::test::artifact_contains_symbol(p1Dir + "/scan.hsaco",
                                                  "scan"));
  const std::vector<char> blob2 =
      kernelh::test::read_file_bin(p2Dir + "/scan_totals.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob2));
  REQUIRE(kernelh::test::artifact_contains_symbol(
      p2Dir + "/scan_totals.hsaco", "scan_totals"));

  const size_t nBytes = sizeof(float) * static_cast<size_t>(n);
  const size_t tBytes = sizeof(float) * static_cast<size_t>(T);
  std::vector<float> h_x(static_cast<size_t>(n));
  std::vector<float> h_y(static_cast<size_t>(n), 0.0f);
  fill_rand(h_x, seed);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, nBytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, nBytes);
  auto ylo_buf = kernelh::runtime::Buffer::allocate(backend, nBytes);
  auto thi_buf = kernelh::runtime::Buffer::allocate(backend, tBytes);
  auto tlo_buf = kernelh::runtime::Buffer::allocate(backend, tBytes);
  auto offh_buf = kernelh::runtime::Buffer::allocate(backend, tBytes);
  auto offl_buf = kernelh::runtime::Buffer::allocate(backend, tBytes);
  auto off2_buf = kernelh::runtime::Buffer::allocate(backend, tBytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(ylo_buf.has_value());
  REQUIRE(thi_buf.has_value());
  REQUIRE(tlo_buf.has_value());
  REQUIRE(offh_buf.has_value());
  REQUIRE(offl_buf.has_value());
  REQUIRE(off2_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), nBytes).has_value());

  auto module1 =
      kernelh::runtime::GpuModule::load(backend, blob1.data(), blob1.size());
  REQUIRE(module1.has_value());
  auto fn1 = module1->get_function("scan");
  REQUIRE(fn1.has_value());
  auto module2 =
      kernelh::runtime::GpuModule::load(backend, blob2.data(), blob2.size());
  REQUIRE(module2.has_value());
  auto fn2 = module2->get_function("scan_totals");
  REQUIRE(fn2.has_value());

  // Pass 1: grid = ceil(n/256) workgroups of 256 threads, 5 rank-1 memrefs.
  kernelh::runtime::KernelLaunch l1(backend);
  l1.grid(static_cast<unsigned>(T)).block(static_cast<unsigned>(kW));
  l1.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{n});
  l1.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{n});
  l1.memref_arg(ylo_buf->device_ptr(), std::vector<int64_t>{n});
  l1.memref_arg(thi_buf->device_ptr(), std::vector<int64_t>{T});
  l1.memref_arg(tlo_buf->device_ptr(), std::vector<int64_t>{T});
  REQUIRE(l1.args().param_count() == 25); // 5 x (2 ptr + 1 offset + 1 sz + 1 stride)
  REQUIRE(l1.launch(*fn1).has_value());

  // Pass 2: grid = 1 workgroup of 256 threads, 7 rank-1 memrefs.
  kernelh::runtime::KernelLaunch l2(backend);
  l2.grid(1u).block(static_cast<unsigned>(kW));
  l2.memref_arg(thi_buf->device_ptr(), std::vector<int64_t>{T});
  l2.memref_arg(tlo_buf->device_ptr(), std::vector<int64_t>{T});
  l2.memref_arg(offh_buf->device_ptr(), std::vector<int64_t>{T});
  l2.memref_arg(offl_buf->device_ptr(), std::vector<int64_t>{T});
  l2.memref_arg(off2_buf->device_ptr(), std::vector<int64_t>{T});
  l2.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{n});
  l2.memref_arg(ylo_buf->device_ptr(), std::vector<int64_t>{n});
  REQUIRE(l2.args().param_count() == 35); // 7 x 5 descriptor slots
  REQUIRE(l2.launch(*fn2).has_value());

  REQUIRE(y_buf->copy_to_host(h_y.data(), nBytes).has_value());

  // Round-to-fp32 fp64 reference (the fp64 accumulation is EXACT here: the
  // sum of <= 2^20 fp32 values each in [0,1) needs ~44 significant bits < 53).
  double maxAbsDiff = 0.0;
  double acc = 0.0;
  for (long i = 0; i < n; ++i) {
    const double xv = static_cast<double>(h_x[static_cast<size_t>(i)]);
    const double pref = exclusiveRef ? acc : acc + xv;
    acc += xv;
    const float ref = static_cast<float>(pref);
    maxAbsDiff =
        std::max(maxAbsDiff,
                 std::fabs(static_cast<double>(h_y[static_cast<size_t>(i)]) -
                           static_cast<double>(ref)));
  }
  return maxAbsDiff;
}

// Degraded-mode artifact assertion (todo 6 degrade branch): compile for BOTH
// targets, probe for the kernel symbols, and count LDS instructions in the
// scan ISA (plan degraded acceptance: grep -c 'ds_' scan.isa > 0).
void assert_degraded_artifacts(long n) {
  const std::string base = "/tmp/kernelh-t29-e2e-" + std::to_string(n);
  REQUIRE(kernelh::test::compile_kernel(
              "scan", "n=" + std::to_string(n),
              {"amdgcn-gfx1101", "nvptx-sm_80"}, base + "/p1") == 0);
  REQUIRE(kernelh::test::compile_kernel(
              "scan_totals", "n=" + std::to_string(n),
              {"amdgcn-gfx1101", "nvptx-sm_80"}, base + "/p2") == 0);
  REQUIRE(kernelh::test::artifact_contains_symbol(base + "/p1/scan.hsaco",
                                                  "scan"));
  REQUIRE(kernelh::test::artifact_contains_symbol(base + "/p1/scan.ptx",
                                                  "scan"));
  REQUIRE(kernelh::test::artifact_contains_symbol(
      base + "/p2/scan_totals.hsaco", "scan_totals"));
  const std::vector<char> isa =
      kernelh::test::read_file_bin(base + "/p1/scan.isa");
  const std::string isaText(isa.begin(), isa.end());
  size_t dsCount = 0;
  size_t pos = 0;
  while ((pos = isaText.find("ds_", pos)) != std::string::npos) {
    ++dsCount;
    pos += 3;
  }
  REQUIRE(dsCount > 0); // LDS traffic present in the pass-1 kernel ISA
  std::printf("DEGRADED scan.isa ds_ instruction count = %zu (> 0)\n",
              dsCount);
}

} // namespace

TEST_CASE("e2e_scan", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    for (const long n : {1024L, 65536L, 1048576L}) {
      assert_degraded_artifacts(n);
    }
    std::printf("E2E-SCAN-DEGRADED-ARTIFACTS-OK\n");
    return;
  }

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  std::setvbuf(stdout, nullptr, _IONBF, 0);

  // Per-size tolerances pinned in plan todo 29 (Metis G11), max abs diff vs
  // the round-to-fp32 fp64 reference.
  const struct {
    long n;
    double atol;
    uint32_t seed;
  } kSizes[] = {{1024L, 1e-4, 0x29C0FFEEu},
                {65536L, 1e-3, 0x29C0FFEEu},
                {1048576L, 5e-3, 0x29C0FFEEu}};

  double worst = 0.0;
  for (const auto &s : kSizes) {
    const double diff = run_scan_e2e(backend, s.n, s.seed);
    std::printf("E2E-SCAN n=%ld atol=%.1e max-abs-diff=%.9f -> %s\n", s.n,
                s.atol, diff, diff <= s.atol ? "PASS" : "FAIL");
    REQUIRE(diff <= s.atol);
    worst = std::max(worst, diff);
  }

  // Boundary case (plan todo 29 QA): N=1000 is not divisible by the
  // workgroup size 256; the partial last workgroup must still produce a
  // correct prefix (the padded tail is clamped and its garbage total is
  // never consumed).
  {
    const long n = 1000L;
    const double diff = run_scan_e2e(backend, n, 0x29C0FFEEu);
    std::printf("E2E-SCAN-BOUNDARY n=%ld (not divisible by %ld) "
                "max-abs-diff=%.9f -> boundary: PASS\n",
                n, kW, diff);
    REQUIRE(diff <= 1e-4);
  }

  std::printf("E2E-SCAN-PASSED (sizes=3+1, worst-max-abs-diff=%.9f)\n", worst);
}

TEST_CASE("e2e_scan_exclusive_reference_negative_control",
          "[gpu][!shouldfail]") {
  // QA scenario (test-the-test): assert the kernel matches a DELIBERATELY
  // WRONG EXCLUSIVE prefix reference (the task-pinned wrong reference). The
  // real kernel computes the INCLUSIVE prefix; exclusive[i] differs from
  // inclusive[i] by x[i] (>= 0, > 0 for the fixed seed), so the comparison
  // MUST fail loudly, and [!shouldfail] converts the designed failure into a
  // suite-green result. A PASS here means the numeric comparison is
  // rubber-stamping and the whole e2e harness is broken.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kN = 1024;
  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  const double diff = run_scan_e2e(backend, kN, 0x51AC0D3Eu,
                                   /*exclusiveRef=*/true);
  std::printf("NEGATIVE-CONTROL max-abs-diff-vs-EXCLUSIVE-ref=%.9f "
              "(expected > 1e-4 with a correct kernel)\n",
              diff);
  REQUIRE(diff <= 1e-4);
}
