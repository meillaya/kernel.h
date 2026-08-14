// Softmax kernel e2e (plan todo 27): row-wise SAFE softmax through the FULL
// pipeline against an fp64 CPU reference, plus the mandated stability case
// (inputs ~+1e3/-1e3 must produce FINITE outputs matching the reference - no
// inf/nan) and the wrong-max-offset negative control.
//
// Kernel semantics: y[r,c] = exp(x[r,c]-max_r)/sum_c exp(x[r,c]-max_r), the
// online single-pass form (running max + running sum with rescale through
// loop_for iter_args). Mapping: grid = rows, block = min(cols, 256);
// block_id.x selects the row, columns are grid-strided over the block.
//
// Full-GPU path (degraded=false), per shape (rows,cols) in
// {(4096,1024),(16384,4096),(1,8192)}:
//   kernelh compile --kernel softmax --shape rows=R,cols=C
//     --target amdgcn-gfx1101 -> softmax.hsaco (todo-24 driver via
//     std::system)
//   -> GpuModule::load + get_function("softmax")
//   -> KernelLaunch grid=R block=min(C,256), 2 memref args (x, y)
//   -> copy_to_host -> host comparison against the FP64 CPU reference
//     (plan Metis G11): max abs diff <= 1e-4.
//
// Stability case (MANDATORY): rows filled from {+1000, -1000, 0, 0.5, -0.5}
// with the +1000 first (the online max is then found on column 0, so the
// running sum never has to bridge a huge max gap - exp(x_i - max) is always
// <= 1 and no intermediate partial sum can overflow). A naive softmax
// without max subtraction overflows exp(+1000) to inf and yields inf/inf =
// NaN; the SAFE form must produce finite outputs matching the fp64 reference.
// Prints "stability: PASS".
//
// Negative control [!shouldfail]: a deliberately WRONG max-offset reference.
// The correct softmax is translation-invariant (softmax(x-a) == softmax(x)
// for any per-row constant a), so a wrong CONSTANT offset would silently
// coincide with the kernel output and a rubber-stamping numeric check would
// pass. The wrong reference here uses a per-element wrong max-offset
// offset(r,c) = x[r,c] + 1 ("every element claims to be the max"):
//   ref = exp(x[r,c] - (x[r,c]+1)) / sum_j exp(x[r,j] - (x[r,j]+1)) = 1/cols
// i.e. the UNIFORM distribution. The correct softmax of random data is never
// uniform, so the kernel-vs-ref comparison MUST fail loudly; [!shouldfail]
// converts the designed failure into a suite-green result. A PASS here means
// the numeric comparison is rubber-stamping.
//
// Degraded path (todo 6 degrade branch): no launches - compile only, then
// assert the HSACO/PTX artifacts carry the kernel symbol and the ISA text
// carries the exp lowering (Oracle N4: grep -c 'exp_f32' covers both the
// hardware v_exp_f32 and the __ocml_exp_f32 library call).

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
#include <string>
#include <vector>

namespace {

// Deterministic pseudo-random fill in [-1, 1] (fixed seed - reproducible).
void fill_rand(std::vector<float> &v, uint32_t &seed) {
  for (float &x : v) {
    seed = seed * 1664525u + 1013904223u; // LCG
    x = static_cast<float>(static_cast<int32_t>(seed >> 8) & 0xffff) /
            32768.0f -
        1.0f;
  }
}

// fp64 CPU reference for one row: true max, then exp(x - max) / sum, all in
// double (plan Metis G11). Order-independent (unlike the online kernel).
void softmax_reference_row(const std::vector<float> &xRow, size_t cols,
                           std::vector<double> &outRow) {
  double m = static_cast<double>(xRow[0]);
  for (size_t c = 1; c < cols; ++c) {
    m = std::max(m, static_cast<double>(xRow[c]));
  }
  double s = 0.0;
  for (size_t c = 0; c < cols; ++c) {
    s += std::exp(static_cast<double>(xRow[c]) - m);
  }
  for (size_t c = 0; c < cols; ++c) {
    outRow[c] = std::exp(static_cast<double>(xRow[c]) - m) / s;
  }
}

struct SoftmaxE2e {
  long rows = 0;
  long cols = 0;
  double max_abs_diff = 0.0;
};

// One full pipeline pass: compile -> load -> launch -> copy -> verify against
// the fp64 reference. Returns the max abs diff.
SoftmaxE2e run_softmax_e2e(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                           long rows, long cols) {
  const std::string outDir = "/tmp/kernelh-t27-e2e-" + std::to_string(rows) +
                             "x" + std::to_string(cols);
  const int rc = kernelh::test::compile_kernel(
      "softmax", "rows=" + std::to_string(rows) + ",cols=" + std::to_string(cols),
      {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);

  const std::string hsacoPath = outDir + "/softmax.hsaco";
  REQUIRE(kernelh::test::file_exists(hsacoPath));
  const std::vector<char> blob = kernelh::test::read_file_bin(hsacoPath);
  REQUIRE(kernelh::test::is_elf_magic(blob));
  REQUIRE(kernelh::test::artifact_contains_symbol(hsacoPath, "softmax"));

  const size_t bytes = sizeof(float) * static_cast<size_t>(rows) *
                       static_cast<size_t>(cols);
  std::vector<float> h_x(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  std::vector<float> h_y(static_cast<size_t>(rows) * static_cast<size_t>(cols),
                         0.0f);
  uint32_t seed = 0x27ab13u;
  fill_rand(h_x, seed);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("softmax");
  REQUIRE(fn.has_value());

  // One block per row (grid = rows). The ROCDL-serialized kernel descriptor
  // caps max_flat_workgroup_size at 256 (a serializer default in the pinned
  // core pipeline - the todo-28 layernorm lesson), so the workgroup must be
  // <= 256; the write loop is grid-stride over the row's columns, so ANY
  // workgroup size computes every element (cols=1024 -> 4 cols/thread,
  // cols=4096 -> 16, cols=8192 -> 32).
  const long block = std::min(cols, 256L);
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(rows)).block(static_cast<unsigned>(block));
  launch.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{rows, cols});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{rows, cols});
  // Descriptor packing (todo 23): 3 pointers+offset + N sizes + N strides per
  // memref. x and y are rank-2 (7 fields each): 7+7 = 14.
  REQUIRE(launch.args().param_count() == 14);

  const auto launchResult = launch.launch(*fn);
  if (!launchResult) {
    std::printf("SOFTMAX-LAUNCH-ERROR: %s\n",
                launchResult.error().message.c_str());
  }
  REQUIRE(launchResult.has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), bytes).has_value());

  // Metis G11 reference: fp64 CPU computation, max abs diff.
  SoftmaxE2e result;
  result.rows = rows;
  result.cols = cols;
  std::vector<double> refRow(static_cast<size_t>(cols));
  size_t nonFinite = 0;
  for (long r = 0; r < rows; ++r) {
    const std::vector<float> xRow(h_x.begin() + static_cast<ptrdiff_t>(r) * cols,
                                  h_x.begin() + static_cast<ptrdiff_t>(r + 1) * cols);
    softmax_reference_row(xRow, static_cast<size_t>(cols), refRow);
    for (long c = 0; c < cols; ++c) {
      const double gv = static_cast<double>(
          h_y[static_cast<size_t>(r) * static_cast<size_t>(cols) +
              static_cast<size_t>(c)]);
      if (!std::isfinite(gv) || !std::isfinite(refRow[static_cast<size_t>(c)])) {
        ++nonFinite;
      }
      const double diff = std::fabs(gv - refRow[static_cast<size_t>(c)]);
      result.max_abs_diff = std::max(result.max_abs_diff, diff);
    }
  }
  REQUIRE(nonFinite == 0); // softmax outputs must always be finite
  return result;
}

// Stability case (MANDATORY): rows filled from {+1000, -1000, 0, 0.5, -0.5}
// with the +1000 first. The online max is found on column 0, so every
// exp(x_i - max) is <= 1 and no intermediate partial sum can overflow - the
// SAFE max-subtracting softmax produces finite outputs matching the fp64
// reference, where a naive exp(x_i) softmax would produce inf/inf = NaN.
void run_stability_case(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                        long rows, long cols) {
  const std::string outDir = "/tmp/kernelh-t27-stability";
  const int rc = kernelh::test::compile_kernel(
      "softmax", "rows=" + std::to_string(rows) + ",cols=" + std::to_string(cols),
      {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);
  const std::vector<char> blob =
      kernelh::test::read_file_bin(outDir + "/softmax.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob));

  const size_t bytes = sizeof(float) * static_cast<size_t>(rows) *
                       static_cast<size_t>(cols);
  const float pattern[5] = {1000.0f, -1000.0f, 0.0f, 0.5f, -0.5f};
  std::vector<float> h_x(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  std::vector<float> h_y(static_cast<size_t>(rows) * static_cast<size_t>(cols),
                         0.0f);
  for (long r = 0; r < rows; ++r) {
    for (long c = 0; c < cols; ++c) {
      h_x[static_cast<size_t>(r) * static_cast<size_t>(cols) +
          static_cast<size_t>(c)] =
          pattern[static_cast<size_t>(c) % 5];
    }
  }

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("softmax");
  REQUIRE(fn.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(rows))
      .block(static_cast<unsigned>(std::min(cols, 256L)));
  launch.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{rows, cols});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{rows, cols});
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), bytes).has_value());

  // Assert EVERY output is finite (no inf/nan from exp(+1000) - the whole
  // point of the SAFE max-subtracting form) AND matches the fp64 reference.
  size_t nonFinite = 0;
  double maxDiff = 0.0;
  std::vector<double> refRow(static_cast<size_t>(cols));
  for (long r = 0; r < rows; ++r) {
    const std::vector<float> xRow(h_x.begin() + static_cast<ptrdiff_t>(r) * cols,
                                  h_x.begin() + static_cast<ptrdiff_t>(r + 1) * cols);
    softmax_reference_row(xRow, static_cast<size_t>(cols), refRow);
    for (long c = 0; c < cols; ++c) {
      const double gv = static_cast<double>(
          h_y[static_cast<size_t>(r) * static_cast<size_t>(cols) +
              static_cast<size_t>(c)]);
      if (!std::isfinite(gv)) {
        ++nonFinite;
      }
      maxDiff = std::max(maxDiff, std::fabs(gv - refRow[static_cast<size_t>(c)]));
    }
  }
  REQUIRE(nonFinite == 0);  // no inf/nan with max subtraction
  REQUIRE(maxDiff <= 1e-4); // matches the fp64 reference
  std::printf("STABILITY-CASE (inputs ~+/-1e3) rows=%ld cols=%ld: "
              "non-finite=%zu max-abs-diff=%.6f -> stability: PASS\n",
              rows, cols, nonFinite, maxDiff);
}

// Degraded-mode artifact assertion (todo 6 degrade branch): compile for BOTH
// targets and probe the blobs for the kernel symbol + exp lowering - no GPU
// needed.
void assert_degraded_artifacts(long rows, long cols) {
  const std::string outDir = "/tmp/kernelh-t27-e2e-" + std::to_string(rows) +
                             "x" + std::to_string(cols);
  const int rc = kernelh::test::compile_kernel(
      "softmax", "rows=" + std::to_string(rows) + ",cols=" + std::to_string(cols),
      {"amdgcn-gfx1101", "nvptx-sm_80"}, outDir);
  REQUIRE(rc == 0);
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/softmax.hsaco",
                                                  "softmax"));
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/softmax.ptx",
                                                  "softmax"));
  // Plan degraded acceptance (Oracle N4): exp lowering present in the ISA
  // text. `exp_f32` matches both the hardware v_exp_f32 mnemonic and the
  // __ocml_exp_f32 library call.
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/softmax.isa",
                                                  "exp_f32"));
}

} // namespace

TEST_CASE("e2e_softmax", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    // Degraded mode: compile + artifact symbol/exp checks only.
    for (const auto &[rows, cols] :
         std::vector<std::pair<long, long>>{{4096L, 1024L},
                                            {16384L, 4096L},
                                            {1L, 8192L}}) {
      assert_degraded_artifacts(rows, cols);
    }
    std::printf("E2E-SOFTMAX-DEGRADED-ARTIFACTS-OK\n");
    return;
  }

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  double worst = 0.0;
  for (const auto &[rows, cols] :
       std::vector<std::pair<long, long>>{{4096L, 1024L},
                                          {16384L, 4096L},
                                          {1L, 8192L}}) {
    const SoftmaxE2e result = run_softmax_e2e(backend, rows, cols);
    // Tolerance pinned in plan todo 27 (Metis G11): max abs diff <= 1e-4.
    REQUIRE(result.max_abs_diff <= 1e-4);
    worst = std::max(worst, result.max_abs_diff);
    std::printf("E2E-SOFTMAX rows=%ld cols=%ld max-abs-diff=%.9f "
                "(ref: fp64)\n",
                rows, cols, result.max_abs_diff);
  }

  // Stability case (MANDATORY): inputs ~+1e3/-1e3 must produce finite
  // outputs matching the reference (no inf/nan). Pinned acceptance: output
  // contains "stability: PASS".
  run_stability_case(backend, 1024L, 1024L);

  std::printf("E2E-SOFTMAX-PASSED (sizes=3, worst-max-abs-diff=%.9f)\n", worst);
}

TEST_CASE("e2e_softmax_wrong_max_negative_control", "[gpu][!shouldfail]") {
  // QA scenario (test-the-test): this case asserts the kernel matches a
  // DELIBERATELY WRONG max-offset reference. The correct softmax is
  // translation-invariant - a wrong but per-row-CONSTANT offset would cancel
  // out and silently match the kernel, so the wrong reference here uses a
  // per-element wrong max-offset offset(r,c) = x[r,c] + 1, which collapses to
  // the UNIFORM distribution 1/cols. The correct softmax of random data is
  // never uniform, so the comparison MUST fail loudly, and [!shouldfail]
  // converts the designed failure into a suite-green result. If this case
  // ever fails (the wrong-reference assertion PASSES), the value comparison
  // is rubber-stamping and the whole e2e harness is broken.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kRows = 256;
  constexpr long kCols = 256;

  const std::string outDir = "/tmp/kernelh-t27-negctrl";
  const int rc = kernelh::test::compile_kernel(
      "softmax", "rows=" + std::to_string(kRows) + ",cols=" + std::to_string(kCols),
      {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);
  const std::vector<char> blob =
      kernelh::test::read_file_bin(outDir + "/softmax.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob));

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  const size_t bytes = sizeof(float) * static_cast<size_t>(kRows) *
                       static_cast<size_t>(kCols);
  std::vector<float> h_x(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
  std::vector<float> h_y(static_cast<size_t>(kRows) * static_cast<size_t>(kCols),
                         0.0f);
  uint32_t seed = 0x51ac0d3u;
  fill_rand(h_x, seed);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("softmax");
  REQUIRE(fn.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(kRows)).block(256u);
  launch.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{kRows, kCols});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{kRows, kCols});
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), bytes).has_value());

  // WRONG reference: ref = 1/cols (the uniform distribution), derived from
  // the per-element wrong max-offset offset(r,c) = x[r,c] + 1. The real
  // kernel computes the true softmax, which differs from uniform on (at
  // least) the max element by far more than 1e-4 - this REQUIRE must fail.
  const float wrongRef = 1.0f / static_cast<float>(kCols);
  long mismatches = 0;
  for (size_t i = 0; i < static_cast<size_t>(kRows) * static_cast<size_t>(kCols);
       ++i) {
    if (std::fabs(static_cast<double>(h_y[i]) -
                  static_cast<double>(wrongRef)) > 1e-4) {
      ++mismatches;
    }
  }
  std::printf("NEGATIVE-CONTROL mismatches-vs-wrong-uniform-ref=%ld of %ld "
              "(expected %ld with a correct kernel)\n",
              mismatches, kRows * kCols, kRows * kCols);
  REQUIRE(mismatches == 0);
}
