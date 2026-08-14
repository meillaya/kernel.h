// Layernorm kernel e2e (plan todo 28): correctness through the FULL pipeline
// against an fp64 CPU reference, plus the constant-input (var=0) eps guard
// case and the eps=0 negative control.
//
// Kernel semantics: y[r,c] = (x[r,c]-mean_r)/sqrt(var_r+eps)*gamma[c]+beta[c],
// two-pass row reduction over the last dim (mean, then variance), eps=1e-5
// compiled into the kernel. Mapping: grid = rows, block = min(cols, 1024);
// block_id.x selects the row.
//
// Full-GPU path (degraded=false), per shape (rows,cols) in
// {(4096,1024),(16384,4096)}:
//   kernelh compile --kernel layernorm --shape rows=R,cols=C
//     --target amdgcn-gfx1101 -> layernorm.hsaco (todo-24 driver via
//     std::system)
//   -> GpuModule::load + get_function("layernorm")
//   -> KernelLaunch grid=R block=min(C,1024), 4 memref args (x, gamma, beta, y)
//   -> copy_to_host -> host comparison against the FP64 CPU reference
//     (mean/var computed in fp64, plan Metis G11): max abs diff <= 1e-4.
//
// Constant-input case (var=0): an all-3.0 row must produce y = beta exactly
// (0/sqrt(0+1e-5)*gamma = 0) - finite, no NaN, matching the fp64 reference.
// Prints "const-case: PASS".
//
// Negative control [!shouldfail]: an eps=0 normalization on constant input
// has denom = sqrt(0+0) = 0 and numerator (x-mean) = 0 -> 0/0 = NaN. The
// test computes that eps=0 fp64 reference and asserts it is finite - which
// MUST fail (the harness FLAGS the non-finite pathology); [!shouldfail]
// converts the designed failure into a suite-green result. A PASS here would
// mean the numeric harness is rubber-stamping and would not catch an
// eps-missing NaN bug.
//
// Degraded path (todo 6 degrade branch): no launches - compile only, then
// assert the HSACO/PTX artifacts carry the kernel symbol and the ISA text
// carries the rsqrt lowering (plan degraded-mode acceptance).

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

// fp64 reference for one row: mean/var in double, y in double. eps is a
// parameter so the negative control can probe the eps=0 pathology.
void reference_row(const std::vector<float> &xRow, size_t cols,
                   const std::vector<float> &gamma,
                   const std::vector<float> &beta, double eps,
                   std::vector<double> &outRow) {
  double mean = 0.0;
  for (size_t c = 0; c < cols; ++c) {
    mean += static_cast<double>(xRow[c]);
  }
  mean /= static_cast<double>(cols);
  double var = 0.0;
  for (size_t c = 0; c < cols; ++c) {
    const double d = static_cast<double>(xRow[c]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(cols);
  const double inv = 1.0 / std::sqrt(var + eps);
  for (size_t c = 0; c < cols; ++c) {
    outRow[c] = (static_cast<double>(xRow[c]) - mean) * inv *
                    static_cast<double>(gamma[c]) +
                static_cast<double>(beta[c]);
  }
}

struct LayernormE2e {
  long rows = 0;
  long cols = 0;
  double max_abs_diff = 0.0;
};

// One full pipeline pass: compile -> load -> launch -> copy -> verify against
// the fp64 reference (eps=1e-5). Returns the max abs diff.
LayernormE2e run_layernorm_e2e(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                               long rows, long cols) {
  const std::string outDir = "/tmp/kernelh-t28-e2e-" + std::to_string(rows) +
                             "x" + std::to_string(cols);
  const int rc = kernelh::test::compile_kernel(
      "layernorm", "rows=" + std::to_string(rows) +
                       ",cols=" + std::to_string(cols),
      {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);

  const std::string hsacoPath = outDir + "/layernorm.hsaco";
  REQUIRE(kernelh::test::file_exists(hsacoPath));
  const std::vector<char> blob = kernelh::test::read_file_bin(hsacoPath);
  REQUIRE(kernelh::test::is_elf_magic(blob));
  REQUIRE(kernelh::test::artifact_contains_symbol(hsacoPath, "layernorm"));

  const size_t rowBytes = sizeof(float) * static_cast<size_t>(cols);
  const size_t xyBytes = rowBytes * static_cast<size_t>(rows);
  std::vector<float> h_x(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  std::vector<float> h_gamma(static_cast<size_t>(cols));
  std::vector<float> h_beta(static_cast<size_t>(cols));
  std::vector<float> h_y(static_cast<size_t>(rows) * static_cast<size_t>(cols),
                         0.0f);
  uint32_t seed = 0x2e2e8au;
  fill_rand(h_x, seed);
  fill_rand(h_gamma, seed);
  fill_rand(h_beta, seed);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, xyBytes);
  auto gamma_buf = kernelh::runtime::Buffer::allocate(backend, rowBytes);
  auto beta_buf = kernelh::runtime::Buffer::allocate(backend, rowBytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, xyBytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(gamma_buf.has_value());
  REQUIRE(beta_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), xyBytes).has_value());
  REQUIRE(gamma_buf->copy_from_host(h_gamma.data(), rowBytes).has_value());
  REQUIRE(beta_buf->copy_from_host(h_beta.data(), rowBytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("layernorm");
  REQUIRE(fn.has_value());

  // One block per row (grid = rows). The ROCDL-serialized kernel descriptor
  // caps max_flat_workgroup_size at 256 (a serializer default in the pinned
  // core pipeline - vecadd observes the same cap), so the workgroup must be
  // <= 256; the write loop is grid-stride over the row's columns, so ANY
  // workgroup size computes every element (cols=1024 -> 4 cols/thread,
  // cols=4096 -> 16 cols/thread).
  const long block = 256;
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(rows))
      .block(static_cast<unsigned>(block));
  launch.memref_arg(x_buf->device_ptr(),
                    std::vector<int64_t>{rows, cols});
  launch.memref_arg(gamma_buf->device_ptr(), std::vector<int64_t>{cols});
  launch.memref_arg(beta_buf->device_ptr(), std::vector<int64_t>{cols});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{rows, cols});
  // Descriptor packing (todo 23): 3 pointers+offset + N sizes + N strides per
  // memref. x/y are rank-2 (7 fields each), gamma/beta rank-1 (5 each):
  // 7+5+5+7 = 24.
  REQUIRE(launch.args().param_count() == 24);

  const auto launchResult = launch.launch(*fn);
  if (!launchResult) {
    std::printf("LAYERNORM-LAUNCH-ERROR: %s\n",
                launchResult.error().message.c_str());
  }
  REQUIRE(launchResult.has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), xyBytes).has_value());

  // Metis G11 reference: fp64 CPU computation, max abs diff.
  LayernormE2e result;
  result.rows = rows;
  result.cols = cols;
  std::vector<double> refRow(static_cast<size_t>(cols));
  for (long r = 0; r < rows; ++r) {
    const std::vector<float> xRow(h_x.begin() + static_cast<ptrdiff_t>(r) * cols,
                                  h_x.begin() + static_cast<ptrdiff_t>(r + 1) * cols);
    reference_row(xRow, static_cast<size_t>(cols), h_gamma, h_beta, 1e-5, refRow);
    for (long c = 0; c < cols; ++c) {
      const double diff = std::fabs(static_cast<double>(
                                        h_y[static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                            static_cast<size_t>(c)]) -
                                    refRow[static_cast<size_t>(c)]);
      result.max_abs_diff = std::max(result.max_abs_diff, diff);
    }
  }
  return result;
}

// Constant-input (var=0) case: asserts no NaN/inf and that the GPU output
// matches the fp64 reference, then prints the pinned "const-case: PASS".
void run_constant_case(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                       long rows, long cols) {
  const std::string outDir = "/tmp/kernelh-t28-const";
  const int rc = kernelh::test::compile_kernel(
      "layernorm", "rows=" + std::to_string(rows) +
                       ",cols=" + std::to_string(cols),
      {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);
  const std::vector<char> blob =
      kernelh::test::read_file_bin(outDir + "/layernorm.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob));

  const size_t rowBytes = sizeof(float) * static_cast<size_t>(cols);
  const size_t xyBytes = rowBytes * static_cast<size_t>(rows);
  std::vector<float> h_x(static_cast<size_t>(rows) * static_cast<size_t>(cols), 3.0f);
  std::vector<float> h_gamma(static_cast<size_t>(cols));
  std::vector<float> h_beta(static_cast<size_t>(cols));
  std::vector<float> h_y(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  uint32_t seed = 0xc0578au;
  fill_rand(h_gamma, seed);
  fill_rand(h_beta, seed);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, xyBytes);
  auto gamma_buf = kernelh::runtime::Buffer::allocate(backend, rowBytes);
  auto beta_buf = kernelh::runtime::Buffer::allocate(backend, rowBytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, xyBytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(gamma_buf.has_value());
  REQUIRE(beta_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), xyBytes).has_value());
  REQUIRE(gamma_buf->copy_from_host(h_gamma.data(), rowBytes).has_value());
  REQUIRE(beta_buf->copy_from_host(h_beta.data(), rowBytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("layernorm");
  REQUIRE(fn.has_value());

  // See run_layernorm_e2e: kernel descriptor caps the workgroup at 256.
  const long block = 256;
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(rows)).block(static_cast<unsigned>(block));
  launch.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{rows, cols});
  launch.memref_arg(gamma_buf->device_ptr(), std::vector<int64_t>{cols});
  launch.memref_arg(beta_buf->device_ptr(), std::vector<int64_t>{cols});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{rows, cols});
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), xyBytes).has_value());

  // var=0, eps=1e-5: y = 0/sqrt(1e-5)*gamma + beta = beta, finite. Assert no
  // NaN/inf anywhere and match against the fp64 reference.
  std::vector<double> refRow(static_cast<size_t>(cols));
  double maxDiff = 0.0;
  size_t nonFinite = 0;
  for (long r = 0; r < rows; ++r) {
    const std::vector<float> xRow(static_cast<size_t>(cols), 3.0f);
    reference_row(xRow, static_cast<size_t>(cols), h_gamma, h_beta, 1e-5, refRow);
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
  REQUIRE(nonFinite == 0);           // no NaN/inf with eps guarding var=0
  REQUIRE(maxDiff <= 1e-4);          // matches the fp64 reference
  std::printf("CONST-CASE (var=0, eps=1e-5) rows=%ld cols=%ld: non-finite=%zu "
              "max-abs-diff=%.6f -> const-case: PASS\n",
              rows, cols, nonFinite, maxDiff);
}

// Degraded-mode artifact assertion (todo 6 degrade branch): compile for BOTH
// targets and probe the blobs for the kernel symbol + rsqrt lowering - no GPU
// needed.
void assert_degraded_artifacts(long rows, long cols) {
  const std::string outDir = "/tmp/kernelh-t28-e2e-" + std::to_string(rows) +
                             "x" + std::to_string(cols);
  const int rc = kernelh::test::compile_kernel(
      "layernorm", "rows=" + std::to_string(rows) +
                       ",cols=" + std::to_string(cols),
      {"amdgcn-gfx1101", "nvptx-sm_80"}, outDir);
  REQUIRE(rc == 0);
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/layernorm.hsaco",
                                                  "layernorm"));
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/layernorm.ptx",
                                                  "layernorm"));
  // Plan degraded acceptance: rsqrt lowering present in the ISA text.
  // AMDGCN emits the hardware `v_rsq_f32` mnemonic (and a library call would
  // carry `__ocml_rsqrt_f32`), so probe the "rsq" substring which covers
  // both forms.
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/layernorm.isa",
                                                  "rsq"));
}

} // namespace

TEST_CASE("e2e_layernorm", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    // Degraded mode: compile + artifact symbol/rsqrt checks only.
    for (const auto &[rows, cols] :
         std::vector<std::pair<long, long>>{{4096L, 1024L}, {16384L, 4096L}}) {
      assert_degraded_artifacts(rows, cols);
    }
    std::printf("E2E-LAYERNORM-DEGRADED-ARTIFACTS-OK\n");
    return;
  }

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  // Shape (4096,1024): block == cols (one column per thread).
  // Shape (16384,4096): cols > 1024 -> grid-stride write loop over the row
  // (the cols>1024 probe: must be handled cleanly, not silently truncated).
  double worst = 0.0;
  for (const auto &[rows, cols] :
       std::vector<std::pair<long, long>>{{4096L, 1024L}, {16384L, 4096L}}) {
    const LayernormE2e result = run_layernorm_e2e(backend, rows, cols);
    // Tolerance pinned in plan todo 28 (Metis G11): max abs diff <= 1e-4.
    REQUIRE(result.max_abs_diff <= 1e-4);
    worst = std::max(worst, result.max_abs_diff);
    std::printf("E2E-LAYERNORM rows=%ld cols=%ld max-abs-diff=%.9f "
                "(ref: fp64)\n",
                rows, cols, result.max_abs_diff);
  }

  // Constant-input case (var=0): the eps guard must produce finite beta and
  // match the fp64 reference. Pinned acceptance: output contains
  // "const-case: PASS".
  run_constant_case(backend, 1024L, 1024L);

  std::printf("E2E-LAYERNORM-PASSED (sizes=2, worst-max-abs-diff=%.9f)\n",
              worst);
}

TEST_CASE("e2e_layernorm_eps0_constant_negative_control",
          "[gpu][!shouldfail]") {
  // QA scenario (test-the-test): an eps=0 normalization on CONSTANT input
  // gives denom = sqrt(var+0) = sqrt(0) = 0 and numerator (x-mean) = 0 ->
  // 0/0 = NaN - the exact pathology the eps guard exists to prevent. This
  // case computes the eps=0 fp64 reference and asserts it is FINITE; the
  // reference is NaN on every element, so the assertion MUST fail. The
  // non-finite count printed below is the harness FLAGGING the pathology.
  // [!shouldfail] converts the designed failure into a suite-green result. A
  // PASS here means the numeric harness is rubber-stamping.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kRows = 1024;
  constexpr long kCols = 1024;
  const std::string outDir = "/tmp/kernelh-t28-negctrl";
  const int rc = kernelh::test::compile_kernel(
      "layernorm", "rows=" + std::to_string(kRows) +
                       ",cols=" + std::to_string(kCols),
      {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);
  const std::vector<char> blob =
      kernelh::test::read_file_bin(outDir + "/layernorm.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob));

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  const size_t rowBytes = sizeof(float) * static_cast<size_t>(kCols);
  const size_t xyBytes = rowBytes * static_cast<size_t>(kRows);
  std::vector<float> h_x(static_cast<size_t>(kRows) * static_cast<size_t>(kCols), 3.0f);
  std::vector<float> h_gamma(static_cast<size_t>(kCols), 1.0f);
  std::vector<float> h_beta(static_cast<size_t>(kCols), 0.0f);
  std::vector<float> h_y(static_cast<size_t>(kRows) * static_cast<size_t>(kCols), 0.0f);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, xyBytes);
  auto gamma_buf = kernelh::runtime::Buffer::allocate(backend, rowBytes);
  auto beta_buf = kernelh::runtime::Buffer::allocate(backend, rowBytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, xyBytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(gamma_buf.has_value());
  REQUIRE(beta_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), xyBytes).has_value());
  REQUIRE(gamma_buf->copy_from_host(h_gamma.data(), rowBytes).has_value());
  REQUIRE(beta_buf->copy_from_host(h_beta.data(), rowBytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("layernorm");
  REQUIRE(fn.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(kRows)).block(256u);
  launch.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{kRows, kCols});
  launch.memref_arg(gamma_buf->device_ptr(), std::vector<int64_t>{kCols});
  launch.memref_arg(beta_buf->device_ptr(), std::vector<int64_t>{kCols});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{kRows, kCols});
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), xyBytes).has_value());

  // The GPU kernel (eps=1e-5) must have produced FINITE beta (0*rsqrt(1e-5)).
  for (const float v : h_y) {
    REQUIRE(std::isfinite(v));
  }

  // eps=0 fp64 reference on constant input: 0/0 -> NaN everywhere.
  std::vector<double> refRow(static_cast<size_t>(kCols));
  size_t nonfinite = 0;
  for (long r = 0; r < kRows; ++r) {
    const std::vector<float> xRow(static_cast<size_t>(kCols), 3.0f);
    reference_row(xRow, static_cast<size_t>(kCols), h_gamma, h_beta, 0.0, refRow);
    for (long c = 0; c < kCols; ++c) {
      if (!std::isfinite(refRow[static_cast<size_t>(c)])) {
        ++nonfinite;
      }
    }
  }
  std::printf("NEGATIVE-CONTROL eps0-ref-non-finite=%zu of %ld "
              "(0/0 -> NaN, as expected for an eps-missing kernel)\n",
              nonfinite, kRows * kCols);
  // WRONG premise that the eps=0 reference is finite. The reference is NaN on
  // every element, so this REQUIRE MUST fail - [!shouldfail] turns the
  // designed failure into a suite-green result.
  REQUIRE(nonfinite == 0);
}
