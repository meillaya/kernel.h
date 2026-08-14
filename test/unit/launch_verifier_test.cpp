// Launch-config verifier unit tests (plan todo 32).
//
// The verifier checks a concrete LAUNCH CONFIG (workgroup size, grid shape,
// static shared bytes) against the device-limits table (todo 10) BEFORE a
// kernel is launched - DISTINCT from the todo-9 structural IR verifier. One
// rejection test per rule, one accept case, one all-rules-at-once case (every
// violation must be reported, not just the first), plus the wired runtime
// launch-refusal and the [gpu] occupancy-API case (both SKIP when degraded).

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/launch_verifier.hpp"

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include "gpu_gate.hpp"
#include "kernelh_compile_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using kernelh::compiler::LaunchConfig;
using kernelh::compiler::LaunchVerifier;
using kernelh::compiler::query_occupancy_max_blocks_runtime;
using kernelh::gfx1101_limits;

namespace {

// Case-folded copy, for the "contains 'lds'" assertions (the diagnostics use
// the uppercase vendor spelling "LDS").
std::string lower(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// A DeviceLimits with an artificially small per-CU wavefront budget (4 waves x
// wave32 = 128 concurrent threads) so the analytic occupancy rule can fire
// WITHOUT tripping the max-threads rule (max stays 1024).
kernelh::DeviceLimits low_wave_budget_limits() {
  kernelh::DeviceLimits l = gfx1101_limits;
  l.waves_per_cu = 4;
  return l;
}

// ---------------------------------------------------------------------------
// [gpu] fixture helpers (same contract as runtime_integration_test.cpp): the
// todo-7 HSACO is not committed - candidate locations are checked in order,
// falling back to an in-test re-emit from the committed fixture + the pinned
// ROCDL pipeline.
// ---------------------------------------------------------------------------
std::vector<std::string> hsaco_candidates() {
  return {
      "/tmp/kernelh-w2/xchg.hsaco",
      std::string(KERNELH_SOURCE_DIR) + "/.omo/evidence/task-7-kernelh/xchg.hsaco",
      std::string(KERNELH_SOURCE_DIR) + "/build/test/fixtures/xchg.hsaco",
      std::string(KERNELH_SOURCE_DIR) + "/test/fixtures/xchg.hsaco",
  };
}

std::string find_hsaco() {
  for (const auto &candidate : hsaco_candidates()) {
    if (kernelh::test::file_exists(candidate)) {
      return candidate;
    }
  }
  return "";
}

// Inverts MLIR AsmPrinter string escaping for the gpu.binary `bin = "..."`
// payload (mirrors .omo/evidence/task-8-kernelh/extract_gpu_object.py).
std::vector<char> decode_mlir_bin(const std::vector<char> &mlir) {
  const std::string text(mlir.begin(), mlir.end());
  const std::string needle = "bin = \"";
  const auto pos = text.find(needle);
  if (pos == std::string::npos) {
    return {};
  }
  const auto hex = [](char h) -> int {
    if (h >= '0' && h <= '9') return h - '0';
    if (h >= 'a' && h <= 'f') return h - 'a' + 10;
    if (h >= 'A' && h <= 'F') return h - 'A' + 10;
    return -1;
  };
  std::vector<char> out;
  size_t i = pos + needle.size();
  while (i < text.size()) {
    const char c = text[i];
    if (c == '\\') {
      const char n = text[i + 1];
      if (n == '"') {
        out.push_back('"');
        i += 2;
      } else if (n == '\\') {
        out.push_back('\\');
        i += 2;
      } else {
        const int hi = hex(text[i + 1]);
        const int lo = hex(text[i + 2]);
        if (hi < 0 || lo < 0) {
          return {};
        }
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 3;
      }
    } else if (c == '"') {
      break;
    } else {
      out.push_back(c);
      ++i;
    }
  }
  return out;
}

// Fallback: re-emit the todo-7 HSACO from the committed fixture using the
// pinned ROCDL pipeline (mlir-opt must be on PATH inside the devenv shell).
std::vector<char> emit_hsaco_from_fixture() {
  const char *toolkit = std::getenv("ROCM_TOOLKIT_ROOT");
  if (toolkit == nullptr || *toolkit == '\0') {
    return {};
  }
  const std::string fixture =
      std::string(KERNELH_SOURCE_DIR) + "/test/fixtures/w2_vecadd_gpu.mlir";
  const std::string out_mlir = "/tmp/kernelh-lv-emit.mlir";
  const std::string pipeline =
      "builtin.module(gpu.module(convert-vector-to-llvm,convert-scf-to-cf,"
      "convert-gpu-to-rocdl{chipset=gfx1101 runtime=HIP}),rocdl-attach-target{"
      "chip=gfx1101},gpu-to-llvm,reconcile-unrealized-casts,"
      "gpu-module-to-binary{format=bin toolkit=" +
      std::string(toolkit) + "})";
  const std::string cmd = "mlir-opt '" + fixture + "' -pass-pipeline='" +
                          pipeline + "' -o '" + out_mlir + "' 2>/dev/null";
  if (std::system(cmd.c_str()) != 0) {
    return {};
  }
  std::vector<char> mlir = kernelh::test::read_file_bin(out_mlir);
  std::remove(out_mlir.c_str());
  return decode_mlir_bin(mlir);
}

std::vector<char> obtain_xchg_hsaco() {
  const std::string cached = find_hsaco();
  if (!cached.empty()) {
    return kernelh::test::read_file_bin(cached);
  }
  return emit_hsaco_from_fixture();
}

} // namespace

// ---------------------------------------------------------------------------
// One rejection test per rule (unit, no GPU needed).
// ---------------------------------------------------------------------------

TEST_CASE("launch_verifier_shared_over_lds_rejected", "[unit][launch_verifier]") {
  LaunchVerifier verifier;
  LaunchConfig config{/*threads_per_workgroup=*/256, /*grid=*/1, 1, 1,
                      /*static_shared_bytes=*/70000};
  auto r = verifier.verify(config, gfx1101_limits);
  REQUIRE_FALSE(r.has_value());
  const std::string msg = r.error();
  CHECK(msg.find("shared memory 70000 bytes exceeds the 65536-byte LDS limit")
        != std::string::npos);
  CHECK(msg.find("AMD Navi 32 (gfx1101)") != std::string::npos);
}

TEST_CASE("launch_verifier_threads_over_max_rejected", "[unit][launch_verifier]") {
  LaunchVerifier verifier;
  LaunchConfig config{/*threads_per_workgroup=*/2048, /*grid=*/1, 1, 1, 0};
  auto r = verifier.verify(config, gfx1101_limits);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().find("threads per workgroup 2048 exceeds the max 1024")
        != std::string::npos);
}

TEST_CASE("launch_verifier_threads_not_wave_multiple_rejected",
          "[unit][launch_verifier]") {
  LaunchVerifier verifier;
  LaunchConfig config{/*threads_per_workgroup=*/100, /*grid=*/1, 1, 1, 0};
  auto r = verifier.verify(config, gfx1101_limits);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().find(
            "threads per workgroup 100 is not a multiple of the wavefront size 32")
        != std::string::npos);
}

TEST_CASE("launch_verifier_grid_over_limit_rejected", "[unit][launch_verifier]") {
  LaunchVerifier verifier;
  LaunchConfig config{/*threads_per_workgroup=*/256, /*grid=*/1, 1, 70000, 0};
  auto r = verifier.verify(config, gfx1101_limits);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().find("grid dim (1,1,70000) exceeds the target grid limits")
        != std::string::npos);
}

TEST_CASE("launch_verifier_occupancy_zero_analytic_rejected",
          "[unit][launch_verifier]") {
  LaunchVerifier verifier;
  // 4 waves x wave32 = a 128-thread per-CU budget: 256 threads needs 8
  // wavefronts and cannot be resident at all -> occupancy 0. All the OTHER
  // rules stay silent (256 <= 1024, 256 % 32 == 0, shared 0, grid fine), so
  // only the occupancy diagnostic appears.
  const kernelh::DeviceLimits low = low_wave_budget_limits();
  LaunchConfig bad{/*threads_per_workgroup=*/256, /*grid=*/1, 1, 1, 0};
  auto r = verifier.verify(bad, low);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().find("occupancy would be 0 on target 'AMD Navi 32 (gfx1101)'")
        != std::string::npos);
  CHECK(r.error().find("needs 8 wavefronts") != std::string::npos);

  // Paired positive: 64 threads fits in the 4-wave budget (2 wavefronts) and
  // verifies clean.
  LaunchConfig ok{/*threads_per_workgroup=*/64, /*grid=*/1, 1, 1, 0};
  CHECK(verifier.verify(ok, low).has_value());
}

// ---------------------------------------------------------------------------
// Accept case + malformed-input probe (ALL violations, not just the first).
// ---------------------------------------------------------------------------

TEST_CASE("launch_verifier_valid_config_accepted", "[unit][launch_verifier]") {
  LaunchVerifier verifier;
  LaunchConfig config{/*threads_per_workgroup=*/256, /*grid=*/40, 1, 1,
                      /*static_shared_bytes=*/0};
  CHECK(verifier.verify(config, gfx1101_limits).has_value());
}

TEST_CASE("launch_verifier_all_rules_at_once_reports_all_five",
          "[unit][launch_verifier]") {
  LaunchVerifier verifier;
  // 4100 threads: over max (4100 > 1024) AND not a wavefront multiple
  // (4100 % 32 = 4); 70000 shared bytes over LDS; grid z 70000 over 65535;
  // 129 wavefronts > the 32-wave per-CU budget -> occupancy 0. EVERY rule
  // must fire and appear in the single diagnostic - never just the first.
  LaunchConfig config{/*threads_per_workgroup=*/4100, /*grid=*/1, 1, 70000,
                      /*static_shared_bytes=*/70000};
  auto r = verifier.verify(config, gfx1101_limits);
  REQUIRE_FALSE(r.has_value());
  const std::string msg = r.error();
  CHECK(msg.find("shared memory 70000 bytes exceeds the 65536-byte LDS limit")
        != std::string::npos);
  CHECK(msg.find("threads per workgroup 4100 exceeds the max 1024")
        != std::string::npos);
  CHECK(msg.find(
            "threads per workgroup 4100 is not a multiple of the wavefront size 32")
        != std::string::npos);
  CHECK(msg.find("grid dim (1,1,70000) exceeds the target grid limits")
        != std::string::npos);
  CHECK(msg.find("occupancy would be 0") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Wired into the todo-23 launch path: failed verification = launch refused.
// ---------------------------------------------------------------------------

TEST_CASE("launch_verifier_wired_refuses_over_lds_launch",
          "[gpu][launch_verifier]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  std::vector<char> blob = obtain_xchg_hsaco();
  REQUIRE_FALSE(blob.empty());

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("xchg_kernel");
  REQUIRE(fn.has_value());

  constexpr unsigned kBlock = 256;
  constexpr long kN = 1L << 16;
  const size_t bytes = sizeof(float) * static_cast<size_t>(kN);
  auto in_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(in_buf.has_value());
  auto out_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(out_buf.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(kN / kBlock)).block(kBlock);
  launch.memref_arg(in_buf->device_ptr(), std::vector<int64_t>{kN});
  launch.memref_arg(out_buf->device_ptr(), std::vector<int64_t>{kN});

  // Sanity: the SAME launch with 0 shared bytes passes verification and runs.
  CHECK(launch.launch(*fn).has_value());

  // Plan todo 32: shared=70000 (> the live 65536 LDS) must be REFUSED before
  // submission, with the verifier's diagnostic (asserted case-insensitively
  // for "lds", plus the refused marker and the offending byte count).
  launch.shared_mem_bytes(70000);
  auto refused = launch.launch(*fn);
  REQUIRE_FALSE(refused.has_value());
  const std::string msg = lower(refused.error().message);
  CHECK(msg.find("lds") != std::string::npos);
  CHECK(msg.find("70000") != std::string::npos);
  CHECK(msg.find("refused") != std::string::npos);
}

TEST_CASE("launch_verifier_occupancy_api_with_function",
          "[gpu][launch_verifier]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  std::vector<char> blob = obtain_xchg_hsaco();
  REQUIRE_FALSE(blob.empty());

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());
  auto fn = module->get_function("xchg_kernel");
  REQUIRE(fn.has_value());

  // The STRONG runtime override is linked into kernelh_unit (WHOLE_ARCHIVE).
  // Contract: a real function handle yields either the vendor occupancy API's
  // resident-block count (>= 1 for a 256-thread config) or -1 when the API is
  // unavailable. On this box ROCm 7.2.3 returns hipErrorInvalidDeviceFunction
  // (98) from hipOccupancyMaxActiveBlocksPerMultiprocessor for EVERY kernel
  // (MLIR-emitted AND hipcc-emitted), so -1 and the analytic fallback are the
  // expected path here - the verifier must degrade gracefully, never
  // crash, and never fabricate a violation.
  const int blocks = query_occupancy_max_blocks_runtime(fn->handle, 256, 0);
  INFO("query_occupancy_max_blocks_runtime(fn, 256, 0) = " << blocks
       << " (>= 1: vendor API ok; -1: API unavailable, analytic fallback)");
  REQUIRE((blocks > 0 || blocks == -1));

  // verify() with the function handle must ACCEPT a valid config - whichever
  // occupancy path ran (API or analytic), a valid launch is never refused.
  LaunchVerifier verifier;
  LaunchConfig config{/*threads_per_workgroup=*/256, /*grid=*/1, 1, 1, 0};
  CHECK(verifier.verify(config, gfx1101_limits, fn->handle).has_value());

  // And it must still REJECT an over-LDS config with the LDS diagnostic.
  LaunchConfig bad{/*threads_per_workgroup=*/256, /*grid=*/1, 1, 1,
                   /*static_shared_bytes=*/70000};
  auto refused = verifier.verify(bad, gfx1101_limits, fn->handle);
  REQUIRE_FALSE(refused.has_value());
  CHECK(refused.error().find("LDS") != std::string::npos);
}
