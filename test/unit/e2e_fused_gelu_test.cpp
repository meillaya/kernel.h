// fused_gelu kernel e2e (plan todo 30): the elementwise TWO-OP CHAIN
//   t = a*x + b ;  out = gelu(t)   with  gelu(z) = 0.5*z*(1+tanh(c*(z+d*z^3)))
// through the FULL pipeline against an fp64 CPU reference, PLUS the MANDATORY
// fusion-proof differential (Metis G4).
//
// The template emits the chain as a producer/consumer loop_for PAIR staged
// through a rank-1 global intermediate, so the fuse pass (todo 14) must
// collapse it into ONE loop nest. The proof runs `kernelh compile` twice -
// with and without the new `--passes-no-fuse` pipeline override - and counts
// `atom.loop_for` in the two .atom.final files: fused == 1 nest vs unfused
// == 2 nests. "fusion-fired: PASS" is printed ONLY when that differential
// holds (a loop-nest-count DIFFERENTIAL, not an exit-code check).
//
// Correctness [gpu] per shape (rows,cols) in {(4096,1024),(1<<20,)}:
//   kernelh compile --kernel fused_gelu --shape rows=R,cols=C | --shape n=N
//     --target amdgcn-gfx1101 -> fused_gelu.hsaco (todo-24 driver via
//     std::system)
//   -> GpuModule::load + get_function("fused_gelu")
//   -> KernelLaunch grid=ceil(N/256) block=256, 3 memref args (x, tmp, out)
//   -> copy_to_host -> host comparison against the FP64 CPU reference
//     (plan Metis G11): max abs diff <= 1e-4.
//
// Negative control [!shouldfail]: a deliberately WRONG reference - gelu
// WITHOUT the tanh term: wrong_ref(t) = 0.5*t*(1 + c*(t + d*t^3)) (tanh(x)
// replaced by x). For t in [-1, 2] the linear form deviates from the true
// gelu by ~0.1-0.3, so the comparison MUST fail loudly; [!shouldfail] flips
// the designed failure into a suite-green result. A PASS here means the
// numeric comparison is rubber-stamping.
//
// Degraded path (todo 6 degrade branch): no launches - compile only, assert
// the HSACO/PTX artifacts carry the kernel symbol, and still run the fusion
// differential (it only reads the compiled .atom.final files - no GPU
// needed), printing "fusion-fired: PASS".

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
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Chain constants (must match the template exactly: lib/dsl/kernels/
// fused_gelu.cpp). Computed in fp64 for the reference.
constexpr double kAlpha = 1.5;             // a
constexpr double kBeta = 0.5;              // b
constexpr double kC = 0.7978845608;        // sqrt(2/pi)
constexpr double kD = 0.044715;            // 0.044715
constexpr double kTol = 1e-4;              // Metis G11 max abs diff
constexpr long kBlock = 256;               // launch workgroup size

// fp64 reference for the fused chain: out[i] = gelu(a*x[i] + b).
double gelu_reference(double x) {
  const double t = kAlpha * x + kBeta;
  return 0.5 * t * (1.0 + std::tanh(kC * (t + kD * t * t * t)));
}

// Deliberately WRONG reference: gelu WITHOUT the tanh term (tanh(x) -> x).
double gelu_no_tanh_reference(double x) {
  const double t = kAlpha * x + kBeta;
  return 0.5 * t * (1.0 + kC * (t + kD * t * t * t));
}

// Deterministic pseudo-random fill in [-1, 1] (fixed seed - reproducible).
void fill_rand(std::vector<float> &v, uint32_t &seed) {
  for (float &x : v) {
    seed = seed * 1664525u + 1013904223u; // LCG
    x = static_cast<float>(static_cast<int32_t>(seed >> 8) & 0xffff) /
            32768.0f -
        1.0f;
  }
}

// compile_kernel with EXTRA CLI flags appended before --out (e.g. the
// --passes-no-fuse fusion-proof override). Local mirror of the shared helper.
int compile_kernel_extra(const std::string &kernel, const std::string &shapeText,
                         const std::vector<std::string> &targets,
                         const std::string &outDir,
                         const std::vector<std::string> &extraFlags) {
  std::string cmd = "rm -rf '" + outDir + "' && mkdir -p '" + outDir + "' && '";
  cmd += kernelh::test::kernelh_cli();
  cmd += "' compile --kernel " + kernel + " --shape " + shapeText;
  for (const auto &target : targets) {
    cmd += " --target " + target;
  }
  for (const auto &flag : extraFlags) {
    cmd += " " + flag;
  }
  cmd += " --out '" + outDir + "'";
  const int rc = std::system(cmd.c_str());
  if (rc == -1) {
    return -1;
  }
  if (WIFEXITED(rc)) {
    return WEXITSTATUS(rc);
  }
  return 1;
}

std::string read_text_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return {};
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

size_t count_substr(const std::string &haystack, const std::string &needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

std::string sanitize(const std::string &text) {
  std::string out;
  out.reserve(text.size());
  for (char ch : text) {
    out += std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_';
  }
  return out;
}

// THE FUSION PROOF (Metis G4 / mandatory assertion): compile the SAME kernel
// with and without the fuse pass and count atom.loop_for in the final IR.
// fused must be ONE nest, unfused TWO. Prints "fusion-fired: PASS" only when
// the differential holds; the REQUIREs below additionally fail the case if a
// rubber-stamping pipeline ever produced identical counts.
void run_fusion_proof(const std::string &shapeText) {
  const std::string fusedDir = "/tmp/kernelh-t30-fused-" + sanitize(shapeText);
  const std::string noFuseDir = "/tmp/kernelh-t30-nofuse-" + sanitize(shapeText);
  REQUIRE(kernelh::test::compile_kernel("fused_gelu", shapeText,
                                        {"amdgcn-gfx1101"}, fusedDir) == 0);
  REQUIRE(compile_kernel_extra("fused_gelu", shapeText, {"amdgcn-gfx1101"},
                               noFuseDir, {"--passes-no-fuse"}) == 0);

  const std::string fusedFinal =
      read_text_file(fusedDir + "/fused_gelu.atom.final");
  const std::string noFuseFinal =
      read_text_file(noFuseDir + "/fused_gelu.atom.final");
  const size_t fusedNests = count_substr(fusedFinal, "atom.loop_for");
  const size_t noFuseNests = count_substr(noFuseFinal, "atom.loop_for");

  if (fusedNests == 1 && noFuseNests == 2) {
    std::printf(
        "FUSION-PROOF shape=%s fused-loop-nests=%zu unfused-loop-nests=%zu "
        "-> fusion-fired: PASS\n",
        shapeText.c_str(), fusedNests, noFuseNests);
  } else {
    std::printf(
        "FUSION-PROOF shape=%s fused-loop-nests=%zu unfused-loop-nests=%zu "
        "-> fusion-fired: FAIL (expected 1 vs 2)\n",
        shapeText.c_str(), fusedNests, noFuseNests);
  }
  REQUIRE(fusedNests == 1);
  REQUIRE(noFuseNests == 2);
}

struct GeluE2e {
  long n = 0;
  double max_abs_diff = 0.0;
};

// One full pipeline pass: compile -> load -> launch -> copy -> verify against
// the fp64 reference. Returns the max abs diff.
GeluE2e run_gelu_e2e(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                     long n) {
  const std::string outDir = "/tmp/kernelh-t30-e2e-n" + std::to_string(n);
  const int rc = kernelh::test::compile_kernel(
      "fused_gelu", "n=" + std::to_string(n), {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);

  const std::string hsacoPath = outDir + "/fused_gelu.hsaco";
  REQUIRE(kernelh::test::file_exists(hsacoPath));
  const std::vector<char> blob = kernelh::test::read_file_bin(hsacoPath);
  REQUIRE(kernelh::test::is_elf_magic(blob));
  REQUIRE(kernelh::test::artifact_contains_symbol(hsacoPath, "fused_gelu"));

  const size_t bytes =
      sizeof(float) * static_cast<size_t>(n);
  std::vector<float> h_x(static_cast<size_t>(n));
  std::vector<float> h_tmp(static_cast<size_t>(n), 0.0f);
  std::vector<float> h_y(static_cast<size_t>(n), 0.0f);
  uint32_t seed = 0x3a5f91u;
  fill_rand(h_x, seed);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto tmp_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(tmp_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("fused_gelu");
  REQUIRE(fn.has_value());

  // Grid-stride kernel over the flat extent: grid = ceil(n/256), block = 256.
  // 3 rank-1 memref args (x, tmp, out): 3 * (2*1+3) = 15 descriptor slots.
  const unsigned grid =
      static_cast<unsigned>((n + kBlock - 1) / kBlock);
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(grid).block(kBlock);
  launch.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{n});
  launch.memref_arg(tmp_buf->device_ptr(), std::vector<int64_t>{n});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{n});
  REQUIRE(launch.args().param_count() == 15);

  const auto launchResult = launch.launch(*fn);
  if (!launchResult) {
    std::printf("FUSED-GELU-LAUNCH-ERROR: %s\n",
                launchResult.error().message.c_str());
  }
  REQUIRE(launchResult.has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), bytes).has_value());

  // Metis G11 reference: fp64 CPU computation, max abs diff.
  GeluE2e result;
  result.n = n;
  size_t nonFinite = 0;
  for (long i = 0; i < n; ++i) {
    const double ref = gelu_reference(static_cast<double>(h_x[static_cast<size_t>(i)]));
    const double gv = static_cast<double>(h_y[static_cast<size_t>(i)]);
    if (!std::isfinite(gv) || !std::isfinite(ref)) {
      ++nonFinite;
    }
    result.max_abs_diff =
        std::max(result.max_abs_diff, std::fabs(gv - ref));
  }
  REQUIRE(nonFinite == 0);
  return result;
}

// Degraded-mode artifact assertion (todo 6 degrade branch): compile for BOTH
// targets and probe the blobs for the kernel symbol - no GPU needed.
void assert_degraded_artifacts(long n) {
  const std::string outDir = "/tmp/kernelh-t30-e2e-n" + std::to_string(n);
  const int rc = kernelh::test::compile_kernel(
      "fused_gelu", "n=" + std::to_string(n),
      {"amdgcn-gfx1101", "nvptx-sm_80"}, outDir);
  REQUIRE(rc == 0);
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/fused_gelu.hsaco",
                                                  "fused_gelu"));
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/fused_gelu.ptx",
                                                  "fused_gelu"));
}

} // namespace

TEST_CASE("e2e_fused_gelu", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    // Degraded mode: compile + artifact symbol checks, and the fusion
    // differential (CPU-only - it just counts loops in .atom.final).
    for (const long n : {4194304L, 1048576L}) {
      assert_degraded_artifacts(n);
    }
    run_fusion_proof("n=1048576");
    std::printf("E2E-FUSED-GELU-DEGRADED-ARTIFACTS-OK\n");
    return;
  }

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  // Plan todo 30 shapes: (4096,1024) via rows/cols (flat n = 4 Mi), and
  // 1<<20 via n.
  double worst = 0.0;
  for (const long n : {4194304L, 1048576L}) {
    const GeluE2e result = run_gelu_e2e(backend, n);
    REQUIRE(result.max_abs_diff <= kTol);
    worst = std::max(worst, result.max_abs_diff);
    std::printf("E2E-FUSED-GELU n=%ld max-abs-diff=%.9f (ref: fp64)\n",
                result.n, result.max_abs_diff);
  }

  // MANDATORY fusion-proof assertion (Metis G4 / plan acceptance): fused
  // variant has ONE loop nest, unfused TWO -> the fuse pass FIRED.
  run_fusion_proof("n=1048576");

  std::printf("E2E-FUSED-GELU-PASSED (sizes=2, worst-max-abs-diff=%.9f)\n",
              worst);
}

TEST_CASE("e2e_fused_gelu_wrong_reference_negative_control", "[gpu][!shouldfail]") {
  // QA scenario (test-the-test): this case asserts the kernel matches a
  // DELIBERATELY WRONG reference - gelu WITHOUT the tanh term (tanh(x)
  // replaced by x). For the input range (t = a*x+b in [-1, 2]) the linear
  // form deviates from the true gelu by ~0.1-0.3, so the comparison MUST
  // fail loudly, and [!shouldfail] converts the designed failure into a
  // suite-green result. If this case ever fails (the wrong-reference
  // assertion PASSES), the value comparison is rubber-stamping and the whole
  // e2e harness is broken.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kN = 65536;

  const std::string outDir = "/tmp/kernelh-t30-negctrl";
  const int rc = kernelh::test::compile_kernel(
      "fused_gelu", "n=" + std::to_string(kN), {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);
  const std::vector<char> blob =
      kernelh::test::read_file_bin(outDir + "/fused_gelu.hsaco");
  REQUIRE(kernelh::test::is_elf_magic(blob));

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  const size_t bytes = sizeof(float) * static_cast<size_t>(kN);
  std::vector<float> h_x(static_cast<size_t>(kN));
  std::vector<float> h_tmp(static_cast<size_t>(kN), 0.0f);
  std::vector<float> h_y(static_cast<size_t>(kN), 0.0f);
  uint32_t seed = 0x9c21a7u;
  fill_rand(h_x, seed);

  auto x_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto tmp_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto y_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(x_buf.has_value());
  REQUIRE(tmp_buf.has_value());
  REQUIRE(y_buf.has_value());
  REQUIRE(x_buf->copy_from_host(h_x.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("fused_gelu");
  REQUIRE(fn.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>((kN + kBlock - 1) / kBlock))
      .block(kBlock);
  launch.memref_arg(x_buf->device_ptr(), std::vector<int64_t>{kN});
  launch.memref_arg(tmp_buf->device_ptr(), std::vector<int64_t>{kN});
  launch.memref_arg(y_buf->device_ptr(), std::vector<int64_t>{kN});
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(y_buf->copy_to_host(h_y.data(), bytes).has_value());

  // WRONG reference: gelu without the tanh term. The real kernel computes
  // the true tanh-based gelu, which differs from the linear form by far more
  // than 1e-4 over this input range - this REQUIRE must fail.
  long mismatches = 0;
  for (long i = 0; i < kN; ++i) {
    const double wrongRef =
        gelu_no_tanh_reference(static_cast<double>(h_x[static_cast<size_t>(i)]));
    if (std::fabs(static_cast<double>(h_y[static_cast<size_t>(i)]) - wrongRef) >
        kTol) {
      ++mismatches;
    }
  }
  std::printf(
      "NEGATIVE-CONTROL mismatches-vs-wrong-no-tanh-ref=%ld of %ld "
      "(expected %ld with a correct kernel)\n",
      mismatches, kN, kN);
  REQUIRE(mismatches == 0);
}
