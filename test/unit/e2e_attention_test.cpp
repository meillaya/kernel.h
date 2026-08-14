// Attention kernel e2e (plan todo 31): NON-CAUSAL FlashAttention-lite, f32,
// O = softmax(Q*K^T/sqrt(d)) * V, through the FULL production pipeline.
//
// Kernel semantics: O[b,h,i,j] = sum_t P[t] * V[b,h,t,j] with P[t] =
// exp(S[t] - m) / l, S[t] = (sum_d Q[b,h,i,d]*K[b,h,t,d]) / sqrt(head_dim),
// m = max_t S[t], l = sum_t exp(S[t] - m). Computed in the kernel with the
// ONLINE tile-by-tile form: iter_args carry the running max m, the
// max-anchored denominator l, and the per-output-element accumulator acc
// across K/V tiles; within each tile every element does the branch-free
// rescale (m'=max(m,S); l'=l*exp(m-m')+exp(S-m'); acc'=acc*exp(m-m')+
// exp(S-m')*V[t,j]). K/V tiles are staged through static LDS (one flat
// buffer, the emitter's type-dedup lesson) and the query rows are mapped one
// block per row by the tile pass (grid = (seq, heads, batch), block = 64 =
// head_dim).
//
// Full-GPU path (degraded=false), per seq in {512, 1024}:
//   kernelh compile --kernel attention
//     --shape batch=2,heads=8,seq=S,head_dim=64 --target amdgcn-gfx1101
//     -> attention.hsaco (todo-24 driver via std::system)
//   -> GpuModule::load + get_function("attention")
//   -> KernelLaunch grid=(seq, heads, batch) block=64, 4 rank-4 memref args
//   -> copy_to_host -> comparison against the FP64 CPU reference (plan Metis
//     G11): per-element |gpu - ref| <= atol + rtol*|ref| with rtol = atol =
//     2e-3, both printed.
//
// Inputs are deterministic pseudo-random (fixed LCG seed) in [-1, 1] for
// Q/K/V. The fp64 reference computes the full softmax in double precision
// (order-independent, unlike the online kernel).
//
// Negative control [!shouldfail]: a deliberately WRONG reference computed
// WITHOUT the sqrt(d) scaling (S[t] = sum_d Q*K, not /sqrt(d)). The kernel
// scales by 1/sqrt(64) = 0.125, so its softmax operates on 8x-smaller scores
// and produces a materially different output - the comparison MUST fail
// loudly, and [!shouldfail] converts the designed failure into a suite-green
// result. A PASS here means the numeric comparison is rubber-stamping.
//
// Degraded path (todo 6 degrade branch): no launches - compile only, then
// assert the HSACO/PTX artifacts carry the kernel symbol, the AMD ISA carries
// LDS traffic (plan degraded acceptance: grep -c 'ds_' attention.isa > 0),
// and the exp lowering is present (Oracle N4: 'exp_f32' matches both the
// hardware v_exp_f32 and the __ocml_exp_f32 library call).

#include "kernelh/dsl/program.hpp"
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

constexpr long kBatch = 2;
constexpr long kHeads = 8;
constexpr long kHeadDim = 64;
constexpr long kBlock = 64; // one thread per output element (head_dim)

// Deterministic pseudo-random fill in [-1, 1] (fixed seed - reproducible; the
// fp64 reference is derived from the same host arrays, so only the seed is
// shared).
void fill_rand(std::vector<float> &v, uint32_t seed) {
  for (float &x : v) {
    seed = seed * 1664525u + 1013904223u; // LCG
    x = static_cast<float>(static_cast<int32_t>(seed >> 8) & 0xffff) /
            32768.0f -
        1.0f;
  }
}

// FP64 CPU reference for one (b, h, i) query row:
//   S[t] = sum_d Q[d]*K[t,d]  (optionally / sqrt(head_dim))
//   m = max S; l = sum exp(S - m); P[t] = exp(S[t]-m)/l
//   O[j] = sum_t P[t] * V[t,j]
// Computed entirely in double; order-independent (the kernel's online form is
// mathematically identical, so only fp32 rounding separates them).
void attention_reference_row(const std::vector<float> &qRow,
                             const std::vector<float> &kMat,
                             const std::vector<float> &vMat,
                             std::vector<double> &outRow, long head_dim,
                             bool scaled) {
  const double scale = scaled ? 1.0 / std::sqrt(static_cast<double>(head_dim))
                              : 1.0;
  const long seq = static_cast<long>(kMat.size()) / head_dim;
  std::vector<double> s(static_cast<size_t>(seq));
  double m = -std::numeric_limits<double>::infinity();
  for (long t = 0; t < seq; ++t) {
    double dot = 0.0;
    for (long d = 0; d < head_dim; ++d) {
      dot += static_cast<double>(qRow[static_cast<size_t>(d)]) *
             static_cast<double>(
                 kMat[static_cast<size_t>(t * head_dim + d)]);
    }
    s[static_cast<size_t>(t)] = dot * scale;
    m = std::max(m, s[static_cast<size_t>(t)]);
  }
  double l = 0.0;
  std::vector<double> p(static_cast<size_t>(seq));
  for (long t = 0; t < seq; ++t) {
    p[static_cast<size_t>(t)] = std::exp(s[static_cast<size_t>(t)] - m);
    l += p[static_cast<size_t>(t)];
  }
  for (long j = 0; j < head_dim; ++j) {
    double acc = 0.0;
    for (long t = 0; t < seq; ++t) {
      acc += p[static_cast<size_t>(t)] /
             l * static_cast<double>(
                     vMat[static_cast<size_t>(t * head_dim + j)]);
    }
    outRow[static_cast<size_t>(j)] = acc;
  }
}

struct AttentionE2e {
  long seq = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;       // max |gpu-ref| / max(|ref|, tiny)
  double max_violation = 0.0; // max (|gpu-ref| - rtol*|ref|) over elements
};

// One full pipeline pass: compile -> load -> launch -> copy -> verify.
// `scaled` selects the correct reference (Q*K/sqrt(d)) vs the deliberately
// wrong unscaled one (negative control). Returns the worst per-element
// |gpu - ref| and the worst relative deviation.
AttentionE2e run_attention_e2e(
    const std::shared_ptr<kernelh::runtime::Backend> &backend, long seq,
    bool scaled = true) {
  const std::string outDir =
      "/tmp/kernelh-t31-e2e-" + std::to_string(seq) + (scaled ? "" : "-nosqrt");
  const int rc = kernelh::test::compile_kernel(
      "attention",
      "batch=" + std::to_string(kBatch) + ",heads=" + std::to_string(kHeads) +
          ",seq=" + std::to_string(seq) +
          ",head_dim=" + std::to_string(kHeadDim),
      {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);

  const std::string hsacoPath = outDir + "/attention.hsaco";
  REQUIRE(kernelh::test::file_exists(hsacoPath));
  const std::vector<char> blob = kernelh::test::read_file_bin(hsacoPath);
  REQUIRE(kernelh::test::is_elf_magic(blob));
  REQUIRE(kernelh::test::artifact_contains_symbol(hsacoPath, "attention"));

  const size_t count = static_cast<size_t>(kBatch) *
                       static_cast<size_t>(kHeads) * static_cast<size_t>(seq) *
                       static_cast<size_t>(kHeadDim);
  const size_t bytes = count * sizeof(float);
  std::vector<float> h_q(count);
  std::vector<float> h_k(count);
  std::vector<float> h_v(count);
  std::vector<float> h_o(count, 0.0f);
  fill_rand(h_q, 0x313A171Bu);
  fill_rand(h_k, 0x51AC0D3Eu);
  fill_rand(h_v, 0x29C0FFEEu);

  auto q_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto k_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto v_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto o_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(q_buf.has_value());
  REQUIRE(k_buf.has_value());
  REQUIRE(v_buf.has_value());
  REQUIRE(o_buf.has_value());
  REQUIRE(q_buf->copy_from_host(h_q.data(), bytes).has_value());
  REQUIRE(k_buf->copy_from_host(h_k.data(), bytes).has_value());
  REQUIRE(v_buf->copy_from_host(h_v.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("attention");
  REQUIRE(fn.has_value());

  // Grid = (seq, heads, batch): block_id.x = query row (tile pass maps the
  // domain loop to it), block_id.y = head, block_id.z = batch. Block = 64
  // threads = head_dim (one thread per output element).
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(seq), static_cast<unsigned>(kHeads),
              static_cast<unsigned>(kBatch))
      .block(static_cast<unsigned>(kBlock));
  const std::vector<int64_t> shape{kBatch, kHeads, seq, kHeadDim};
  launch.memref_arg(q_buf->device_ptr(), shape);
  launch.memref_arg(k_buf->device_ptr(), shape);
  launch.memref_arg(v_buf->device_ptr(), shape);
  launch.memref_arg(o_buf->device_ptr(), shape);
  // 4 rank-4 memrefs x (2 ptr + 1 offset + 4 sizes + 4 strides) = 4 x 11.
  REQUIRE(launch.args().param_count() == 44);

  const auto launchResult = launch.launch(*fn);
  if (!launchResult) {
    std::printf("ATTENTION-LAUNCH-ERROR: %s\n",
                launchResult.error().message.c_str());
  }
  REQUIRE(launchResult.has_value());
  REQUIRE(o_buf->copy_to_host(h_o.data(), bytes).has_value());

  // FP64 CPU reference + tolerance check (plan Metis G11): per-element
  // |gpu - ref| <= atol + rtol*|ref|, rtol = atol = 2e-3. The caller's
  // acceptance is max(|gpu-ref| - rtol*|ref|) <= atol (tracked per element).
  const double kRtol = 2e-3;
  AttentionE2e result;
  result.seq = seq;
  std::vector<double> refRow(static_cast<size_t>(kHeadDim));
  size_t nonFinite = 0;
  for (long b = 0; b < kBatch; ++b) {
    for (long h = 0; h < kHeads; ++h) {
      for (long i = 0; i < seq; ++i) {
        const size_t base = static_cast<size_t>(((b * kHeads + h) * seq + i) *
                                                kHeadDim);
        std::vector<float> qRow(h_q.begin() + static_cast<ptrdiff_t>(base),
                                h_q.begin() +
                                    static_cast<ptrdiff_t>(base + kHeadDim));
        std::vector<float> kMat(h_k.begin() + static_cast<ptrdiff_t>((b * kHeads + h) * seq * kHeadDim),
                                h_k.begin() + static_cast<ptrdiff_t>((b * kHeads + h + 1) * seq * kHeadDim));
        std::vector<float> vMat(h_v.begin() + static_cast<ptrdiff_t>((b * kHeads + h) * seq * kHeadDim),
                                h_v.begin() + static_cast<ptrdiff_t>((b * kHeads + h + 1) * seq * kHeadDim));
        attention_reference_row(qRow, kMat, vMat, refRow, kHeadDim, scaled);
        for (long j = 0; j < kHeadDim; ++j) {
          const double gv =
              static_cast<double>(h_o[base + static_cast<size_t>(j)]);
          const double rv = refRow[static_cast<size_t>(j)];
          if (!std::isfinite(gv) || !std::isfinite(rv)) {
            ++nonFinite;
          }
          const double absDiff = std::fabs(gv - rv);
          result.max_abs = std::max(result.max_abs, absDiff);
          const double denom = std::max(std::fabs(rv), 1e-12);
          result.max_rel = std::max(result.max_rel, absDiff / denom);
          result.max_violation =
              std::max(result.max_violation, absDiff - kRtol * std::fabs(rv));
        }
      }
    }
  }
  REQUIRE(nonFinite == 0);
  return result;
}

// Degraded-mode artifact assertion (todo 6 degrade branch): compile for BOTH
// targets and probe the blobs for the kernel symbol + LDS traffic + exp
// lowering in the AMD ISA - no GPU needed.
void assert_degraded_artifacts(long seq) {
  const std::string outDir = "/tmp/kernelh-t31-e2e-" + std::to_string(seq);
  const int rc = kernelh::test::compile_kernel(
      "attention",
      "batch=" + std::to_string(kBatch) + ",heads=" + std::to_string(kHeads) +
          ",seq=" + std::to_string(seq) +
          ",head_dim=" + std::to_string(kHeadDim),
      {"amdgcn-gfx1101", "nvptx-sm_80"}, outDir);
  REQUIRE(rc == 0);
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/attention.hsaco",
                                                  "attention"));
  REQUIRE(kernelh::test::artifact_contains_symbol(outDir + "/attention.ptx",
                                                  "attention"));
  const std::vector<char> isa =
      kernelh::test::read_file_bin(outDir + "/attention.isa");
  const std::string isaText(isa.begin(), isa.end());
  // Plan degraded acceptance: LDS traffic present (the K/V/Q staging through
  // static LDS lowers to ds_ instructions)...
  REQUIRE(isaText.find("ds_") != std::string::npos);
  // ... and the exp lowering is present (Oracle N4: 'exp_f32' matches both
  // v_exp_f32 and __ocml_exp_f32).
  REQUIRE(isaText.find("exp_f32") != std::string::npos);
  std::printf("DEGRADED attention.isa ds_=%s exp_f32=%s\n",
              isaText.find("ds_") != std::string::npos ? "present" : "absent",
              isaText.find("exp_f32") != std::string::npos ? "present"
                                                           : "absent");
}

} // namespace

TEST_CASE("e2e_attention", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    for (const long seq : {512L, 1024L}) {
      assert_degraded_artifacts(seq);
    }
    std::printf("E2E-ATTENTION-DEGRADED-ARTIFACTS-OK\n");
    return;
  }

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  std::setvbuf(stdout, nullptr, _IONBF, 0);

  // Tolerances pinned in plan todo 31 (Metis G11/G32): rtol = atol = 2e-3.
  const double kRtol = 2e-3;
  const double kAtol = 2e-3;
  double worst = 0.0;
  for (const long seq : {512L, 1024L}) {
    const AttentionE2e result = run_attention_e2e(backend, seq);
    // Acceptance: EVERY element satisfies |gpu-ref| <= atol + rtol*|ref|,
    // i.e. max(|gpu-ref| - rtol*|ref|) <= atol (the exact per-element rule).
    const bool pass = result.max_violation <= kAtol;
    std::printf("E2E-ATTENTION seq=%ld rtol=%.1e atol=%.1e "
                "max-abs=%.6e max-rel=%.6e -> %s\n",
                seq, kRtol, kAtol, result.max_abs, result.max_rel,
                pass ? "PASS" : "FAIL");
    REQUIRE(pass);
    worst = std::max(worst, result.max_violation);
  }

  std::printf("E2E-ATTENTION-PASSED (seqs=2, worst-max-violation=%.6e)\n",
              worst);
}

TEST_CASE("e2e_attention_no_sqrt_reference_negative_control",
          "[gpu][!shouldfail]") {
  // QA scenario (test-the-test): assert the kernel matches a DELIBERATELY
  // WRONG reference computed WITHOUT the sqrt(d) score scaling (S = Q*K, not
  // Q*K/sqrt(d)). The kernel scales scores by 1/sqrt(64) = 0.125, so its
  // softmax is far flatter than the unscaled reference's - the comparison
  // MUST fail loudly, and [!shouldfail] converts the designed failure into a
  // suite-green result. A PASS here means the numeric comparison is
  // rubber-stamping and the whole e2e harness is broken.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kSeq = 256;
  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  const AttentionE2e result =
      run_attention_e2e(backend, kSeq, /*scaled=*/false);
  std::printf("NEGATIVE-CONTROL (no sqrt(d) scaling) seq=%ld "
              "max-abs-vs-wrong-ref=%.6e max-rel=%.6e "
              "(expected > 2e-3 with a correct kernel)\n",
              kSeq, result.max_abs, result.max_rel);
  // With a working numeric check this REQUIRE fails loudly (the unscaled
  // reference disagrees with the scaled kernel by far more than the
  // tolerance); [!shouldfail] makes the designed failure suite-green.
  REQUIRE(result.max_abs <= 2e-3);
}
