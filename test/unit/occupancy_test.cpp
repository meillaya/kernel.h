// Occupancy report unit tests (plan todo 36).
//
// Covers:
//   - the analytic computation against HAND-COMPUTED values for 3 synthetic
//     gfx1101 configs (the pinned Oracle B5 formula; the expected numbers are
//     derived by hand in each test's comments),
//   - the sm_80 cross-target analytic (Metis G30) from the same formula,
//   - the todo-36 QA scenario: a config with LDS == 64 KiB exactly must report
//     occupancy limited by LDS with blocks_per_cu = 1,
//   - the not-limiting semantics for unknown bounds (vgpr/lds/threads missing),
//   - measured occupancy: a nullptr function handle -> measured_pct 0 with a
//     note (the graceful degradation the todo-32 hook contract requires; on
//     this box the strong hook ALSO returns -1 because
//     hipOccupancyMaxActiveBlocksPerMultiprocessor returns
//     hipErrorInvalidDeviceFunction for every kernel - learnings todo 32),
//   - the occupancy.{json,md} writers,
//   - the [gpu] VALIDATION test (Oracle B5): the analytic estimate must agree
//     with the hipOccupancy API result within +/-1 block on 3 representative
//     configs. It SKIPs when the API is unavailable/degraded - the KNOWN state
//     on this box (the API returns hipErrorInvalidDeviceFunction for every
//     kernel, so the vendor-API fallback to analytic occupancy is the
//     documented path),
//   - the `kernelh bench --occupancy --kernel ... --target ...` CLI wiring
//     (the todo-36 OWN bench hook; occupancy-only mode) and the unknown-kernel
//     clean-error probe (malformed_input).

#include "kernelh/compiler/occupancy.hpp"

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/launch_verifier.hpp"
#include "kernelh/compiler/resource_reporter.hpp"

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"

#include "gpu_gate.hpp"
#include "kernelh_compile_util.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using Catch::Approx;
using kernelh::compiler::OccupancyInput;
using kernelh::compiler::OccupancyLimiter;
using kernelh::compiler::OccupancyReport;
using kernelh::compiler::occupancy_report;
using kernelh::compiler::write_occupancy_json;
using kernelh::compiler::write_occupancy_json_all;
using kernelh::compiler::write_occupancy_markdown;
using kernelh::gfx1101_limits;
using kernelh::sm_80_limits;

namespace {

std::string read_text(const std::string &path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// jq-style integer field extractor (same helper as resource_reporter_test.cpp;
// -1 when missing).
long json_int_field(const std::string &json, const std::string &field) {
  const std::string key = "\"" + field + "\":";
  const size_t pos = json.find(key);
  if (pos == std::string::npos) {
    return -1;
  }
  size_t start = pos + key.size();
  while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) {
    ++start;
  }
  const size_t end = json.find_first_of(",}\n", start);
  if (end == std::string::npos) {
    return -1;
  }
  char *value_end = nullptr;
  const long value = std::strtol(json.c_str() + start, &value_end, 10);
  return value_end == json.c_str() + end ? value : -1;
}

// jq-style double field extractor (theoretical_pct is a % like 12.5).
double json_double_field(const std::string &json, const std::string &field) {
  const std::string key = "\"" + field + "\":";
  const size_t pos = json.find(key);
  if (pos == std::string::npos) {
    return -1.0;
  }
  size_t start = pos + key.size();
  while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) {
    ++start;
  }
  char *value_end = nullptr;
  const double value = std::strtod(json.c_str() + start, &value_end);
  return value_end != json.c_str() + start ? value : -1.0;
}

std::string json_string_field(const std::string &json,
                              const std::string &field) {
  const std::string key = "\"" + field + "\":";
  const size_t pos = json.find(key);
  if (pos == std::string::npos) {
    return std::string();
  }
  const size_t start = json.find('"', pos + key.size());
  if (start == std::string::npos) {
    return std::string();
  }
  const size_t end = json.find('"', start + 1);
  if (end == std::string::npos) {
    return std::string();
  }
  return json.substr(start + 1, end - start - 1);
}

OccupancyInput synthetic_input(const std::string &kernel, int vgpr, long lds,
                               int threads) {
  OccupancyInput input;
  input.kernel = kernel;
  input.target = "gfx1101";
  input.config = "synthetic";
  input.vgpr_count = vgpr;
  input.lds_bytes = lds;
  input.threads_per_workgroup = threads;
  return input;
}

// [gpu] fixture helpers (same pattern as launch_verifier_test.cpp): the todo-7
// HSACO is not committed; candidate locations are checked in order, falling
// back to an in-test re-emit from the committed fixture.
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

} // namespace

// ---------------------------------------------------------------------------
// Analytic computation vs HAND-COMPUTED values (Oracle B5, exactly as pinned):
//
//   VGPR-limited waves    = floor(reg_file_per_cu / (vgpr x wavefront_size))
//                           capped by waves_per_cu
//   LDS-limited blocks    = floor(lds_bytes_per_workgroup / lds_bytes)
//   thread-limited blocks = floor(max_threads_per_workgroup / threads)
//   achieved_waves        = min(...) ; occupancy_pct = achieved/waves_per_cu
//   limiter               = the bound achieved_waves came from
//
// gfx1101 (todo 10): reg_file_per_cu 6144, wavefront 32, lds 65536, max
// threads 1024, waves_per_cu 32.
// ---------------------------------------------------------------------------

TEST_CASE("occupancy_analytic_synthetic_config1", "[occupancy]") {
  // gfx1101, vgpr 32, lds 16384, threads 256. By hand:
  //   VGPR-limited  = floor(6144 / (32 x 32))            = floor(6144/1024) = 6
  //   LDS-limited   = floor(65536 / 16384)               = 4
  //   thread-limited= floor(1024 / 256)                  = 4
  //   achieved      = min(6, 4, 4)                       = 4  -> limiter lds
  //   pct           = 4 / 32 x 100                       = 12.5
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 32, 16384, 256), gfx1101_limits);
  CHECK(r.vgpr_limited_waves == 6);
  CHECK(r.lds_limited_blocks == 4);
  CHECK(r.thread_limited_blocks == 4);
  CHECK(r.achieved_waves == 4);
  CHECK(r.limiter == OccupancyLimiter::kLds);
  CHECK(r.theoretical_pct == Approx(12.5));
}

TEST_CASE("occupancy_analytic_synthetic_config2", "[occupancy]") {
  // gfx1101, vgpr 64, lds 32768, threads 128. By hand:
  //   VGPR-limited  = floor(6144 / (64 x 32))            = floor(6144/2048) = 3
  //   LDS-limited   = floor(65536 / 32768)               = 2
  //   thread-limited= floor(1024 / 128)                  = 8
  //   achieved      = min(3, 2, 8)                       = 2  -> limiter lds
  //   pct           = 2 / 32 x 100                       = 6.25
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 64, 32768, 128), gfx1101_limits);
  CHECK(r.vgpr_limited_waves == 3);
  CHECK(r.lds_limited_blocks == 2);
  CHECK(r.thread_limited_blocks == 8);
  CHECK(r.achieved_waves == 2);
  CHECK(r.limiter == OccupancyLimiter::kLds);
  CHECK(r.theoretical_pct == Approx(6.25));
}

TEST_CASE("occupancy_analytic_synthetic_config3", "[occupancy]") {
  // gfx1101, vgpr 48, lds 8192, threads 512. By hand:
  //   VGPR-limited  = floor(6144 / (48 x 32))            = floor(6144/1536) = 4
  //   LDS-limited   = floor(65536 / 8192)                = 8
  //   thread-limited= floor(1024 / 512)                  = 2
  //   achieved      = min(4, 8, 2)                       = 2  -> limiter threads
  //   pct           = 2 / 32 x 100                       = 6.25
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 48, 8192, 512), gfx1101_limits);
  CHECK(r.vgpr_limited_waves == 4);
  CHECK(r.lds_limited_blocks == 8);
  CHECK(r.thread_limited_blocks == 2);
  CHECK(r.achieved_waves == 2);
  CHECK(r.limiter == OccupancyLimiter::kThreads);
  CHECK(r.theoretical_pct == Approx(6.25));
}

TEST_CASE("occupancy_analytic_vgpr_limiter", "[occupancy]") {
  // gfx1101, vgpr 128, lds 8192, threads 128. By hand:
  //   VGPR-limited  = floor(6144 / (128 x 32))           = floor(6144/4096) = 1
  //   LDS-limited   = floor(65536 / 8192)                = 8
  //   thread-limited= floor(1024 / 128)                  = 8
  //   achieved      = min(1, 8, 8)                       = 1  -> limiter vgpr
  //   pct           = 1 / 32 x 100                       = 3.125
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 128, 8192, 128), gfx1101_limits);
  CHECK(r.vgpr_limited_waves == 1);
  CHECK(r.lds_limited_blocks == 8);
  CHECK(r.thread_limited_blocks == 8);
  CHECK(r.achieved_waves == 1);
  CHECK(r.limiter == OccupancyLimiter::kVgpr);
  CHECK(r.theoretical_pct == Approx(3.125));
}

TEST_CASE("occupancy_analytic_sm80_cross_target", "[occupancy]") {
  // Cross-target (nvptx) analytic from ptxas -v registers + static smem
  // (Metis G30) against the sm_80 table. sm_80 (todo 10): reg_file_per_cu
  // 65536, wavefront 32, lds 49152, max threads 1024, waves_per_cu 64.
  // vgpr 20 (regs), lds 3072 (static smem), threads 256. By hand:
  //   VGPR-limited  = floor(65536 / (20 x 32)) = floor(65536/640) = 102
  //                   capped by waves_per_cu 64                   = 64
  //   LDS-limited   = floor(49152 / 3072)                          = 16
  //   thread-limited= floor(1024 / 256)                            = 4
  //   achieved      = min(64, 16, 4)                               = 4
  //   limiter       = threads ; pct = 4 / 64 x 100                 = 6.25
  OccupancyInput input;
  input.kernel = "scan";
  input.target = "sm_80";
  input.config = "n=1024";
  input.vgpr_count = 20;
  input.lds_bytes = 3072;
  input.threads_per_workgroup = 256;
  const OccupancyReport r = occupancy_report(input, sm_80_limits);
  CHECK(r.vgpr_limited_waves == 64); // 102 -> capped at 64
  CHECK(r.lds_limited_blocks == 16);
  CHECK(r.thread_limited_blocks == 4);
  CHECK(r.achieved_waves == 4);
  CHECK(r.limiter == OccupancyLimiter::kThreads);
  CHECK(r.theoretical_pct == Approx(6.25));
}

TEST_CASE("occupancy_analytic_lds_64k_exact_lds_limited", "[occupancy]") {
  // The todo-36 QA scenario: a config with LDS == 64 KiB EXACTLY must report
  // occupancy limited by LDS with blocks_per_cu = 1.
  // gfx1101, vgpr 64, lds 65536, threads 256. By hand:
  //   VGPR-limited  = floor(6144 / (64 x 32))  = 3
  //   LDS-limited   = floor(65536 / 65536)     = 1   <-- exactly one block
  //   thread-limited= floor(1024 / 256)        = 4
  //   achieved      = min(3, 1, 4)             = 1   (blocks_per_cu = 1)
  //   limiter       = lds ; pct = 1 / 32 x 100 = 3.125
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 64, 65536, 256), gfx1101_limits);
  CHECK(r.lds_limited_blocks == 1); // the LDS=64KiB-exact bound
  CHECK(r.achieved_waves == 1);     // blocks_per_cu = 1
  CHECK(r.limiter == OccupancyLimiter::kLds);
  CHECK(r.theoretical_pct == Approx(3.125));
}

TEST_CASE("occupancy_analytic_unknown_bounds_not_limiting", "[occupancy]") {
  // A missing resource field (vgpr 0, e.g. the todo-33 ptxas-missing case) is
  // NOT a zero bound - it is skipped. gfx1101, lds 16384, threads 256:
  //   LDS-limited = floor(65536/16384) = 4 ; thread-limited = floor(1024/256)=4
  //   achieved = min(4, 4) = 4 -> limiter lds (earlier bound wins the tie).
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 0, 16384, 256), gfx1101_limits);
  CHECK(r.vgpr_limited_waves == 0); // not computed
  CHECK(r.lds_limited_blocks == 4);
  CHECK(r.thread_limited_blocks == 4);
  CHECK(r.achieved_waves == 4);
  CHECK(r.limiter == OccupancyLimiter::kLds);
  CHECK(r.theoretical_pct == Approx(12.5));

  // All bounds unknown -> achieved 0, pct 0, limiter none, no division by zero.
  const OccupancyReport none =
      occupancy_report(synthetic_input("k", 0, 0, 0), gfx1101_limits);
  CHECK(none.achieved_waves == 0);
  CHECK(none.theoretical_pct == 0.0);
  CHECK(none.limiter == OccupancyLimiter::kNone);
}

TEST_CASE("occupancy_measured_null_function_unavailable", "[occupancy]") {
  // No loaded kernel function -> measured_pct 0 with a note (graceful). The
  // analytic part is unaffected. (With a REAL handle on this box the strong
  // hook also returns -1 - hipOccupancyMaxActiveBlocksPerMultiprocessor is
  // broken here - so measured stays analytic-fallback either way.)
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 32, 16384, 256), gfx1101_limits,
                       /*function=*/nullptr);
  CHECK(r.measured_blocks == -1);
  CHECK(r.measured_pct == 0.0);
  CHECK_FALSE(r.measured_note.empty());
  CHECK(r.measured_note.find("no loaded kernel function") !=
        std::string::npos);
  CHECK(r.achieved_waves == 4); // analytic unaffected by the missing handle
}

TEST_CASE("occupancy_json_writer_single_report", "[occupancy]") {
  const std::string path = "/tmp/kernelh-t36-occ.json";
  const OccupancyReport r =
      occupancy_report(synthetic_input("k", 32, 16384, 256), gfx1101_limits);
  REQUIRE(write_occupancy_json(path, r));
  const std::string json = read_text(path);
  REQUIRE_FALSE(json.empty());
  // The todo-36 acceptance fields are all present with the hand-computed
  // values (jq-equivalent assertions on parsed values, not file existence).
  CHECK(json_double_field(json, "theoretical_pct") == Approx(12.5));
  CHECK(json_string_field(json, "limiter") == "lds");
  CHECK(json_int_field(json, "theoretical_waves") == 4);
  CHECK(json_int_field(json, "vgpr_count") == 32);
  CHECK(json_int_field(json, "lds_bytes") == 16384);
  CHECK(json_double_field(json, "measured_pct") == 0.0);
  CHECK(json_string_field(json, "measured_note").find("analytic only") !=
        std::string::npos);
}

TEST_CASE("occupancy_json_writer_multi_report", "[occupancy]") {
  const std::string path = "/tmp/kernelh-t36-occ-all.json";
  std::vector<OccupancyReport> reports;
  reports.push_back(
      occupancy_report(synthetic_input("a", 32, 16384, 256), gfx1101_limits));
  reports.push_back(
      occupancy_report(synthetic_input("b", 48, 8192, 512), gfx1101_limits));
  REQUIRE(write_occupancy_json_all(path, reports));
  const std::string json = read_text(path);
  REQUIRE_FALSE(json.empty());
  CHECK(json.find("\"reports\"") != std::string::npos);
  CHECK(json.find("\"kernel\": \"a\"") != std::string::npos);
  CHECK(json.find("\"kernel\": \"b\"") != std::string::npos);
}

TEST_CASE("occupancy_markdown_writer_table", "[occupancy]") {
  const std::string path = "/tmp/kernelh-t36-occ.md";
  std::vector<OccupancyReport> reports;
  reports.push_back(
      occupancy_report(synthetic_input("k", 32, 16384, 256), gfx1101_limits));
  REQUIRE(write_occupancy_markdown(path, reports));
  const std::string md = read_text(path);
  REQUIRE_FALSE(md.empty());
  // The pinned markdown table: (kernel, config, vgpr, lds, theoretical %,
  // measured %, limiter). Measured shows n/a (no function handle).
  CHECK(md.find("| kernel | config | vgpr | lds (B) | theoretical % | "
                "measured % | limiter |") != std::string::npos);
  CHECK(md.find("| k | synthetic | 32 | 16384 | 12.5 | n/a | lds |") !=
        std::string::npos);
}

// ---------------------------------------------------------------------------
// The todo-36 CLI wiring (`kernelh bench --occupancy` - the OWN bench hook of
// this todo; the explore-subcommand integration lands with todo 37).
// ---------------------------------------------------------------------------

TEST_CASE("occupancy_cli_bench_occupancy_only_writes_files", "[occupancy]") {
  // The todo-36 acceptance: `kernelh bench --occupancy --kernel gemm --target
  // amdgcn-gfx1101 --out <dir>` writes occupancy.json with
  // theoretical_pct in [0,100] and occupancy.md containing a markdown table.
  const std::string outDir = "/tmp/kernelh-t36-cli";
  const std::string cmd = "rm -rf '" + outDir + "' && '" +
                          kernelh::test::kernelh_cli() +
                          "' bench --occupancy --kernel gemm --target "
                          "amdgcn-gfx1101 --out '" +
                          outDir + "'";
  REQUIRE(std::system(cmd.c_str()) == 0);

  const std::string json = read_text(outDir + "/occupancy.json");
  REQUIRE_FALSE(json.empty());
  // Hand-computed for the default-config GEMM (todo-26 resources: vgpr 41,
  // lds 8192; threads 256):
  //   vgpr-limited  = floor(6144/(41*32)) = 4
  //   lds-limited   = floor(65536/8192)   = 8
  //   thread-limited= floor(1024/256)     = 4
  //   achieved      = min(4,8,4)          = 4 -> limiter vgpr (earlier tie)
  //   pct           = 4/32*100            = 12.5
  const double pct = json_double_field(json, "theoretical_pct");
  CHECK(pct >= 0.0);
  CHECK(pct <= 100.0);
  CHECK(pct == Approx(12.5));
  CHECK(json_string_field(json, "limiter") == "vgpr");
  CHECK(json_string_field(json, "kernel") == "gemm");

  const std::string md = read_text(outDir + "/occupancy.md");
  REQUIRE_FALSE(md.empty());
  CHECK(md.find("| kernel | config | vgpr | lds (B) | theoretical % | "
                "measured % | limiter |") != std::string::npos);
  CHECK(md.find("| gemm |") != std::string::npos);
}

TEST_CASE("occupancy_cli_unknown_kernel_clean_error", "[occupancy]") {
  // malformed_input probe: an unknown kernel must fail with a CLEAN error
  // naming the kernel (never a crash, never a partial write).
  const std::string outDir = "/tmp/kernelh-t36-bad";
  const std::string log = "/tmp/kernelh-t36-bad.log";
  const std::string cmd = "rm -rf '" + outDir + "' && '" +
                          kernelh::test::kernelh_cli() +
                          "' bench --occupancy --kernel no_such_kernel "
                          "--target amdgcn-gfx1101 --out '" +
                          outDir + "' > '" + log + "' 2>&1";
  REQUIRE(std::system(cmd.c_str()) != 0);
  const std::string output = read_text(log);
  CHECK(output.find("no_such_kernel") != std::string::npos);
}

// ---------------------------------------------------------------------------
// [gpu] VALIDATION (Oracle B5): the analytic estimate must agree with the
// hipOccupancy API result within +/-1 block on 3 representative configs. If a
// todo-10 constant is wrong, this test catches it before the report ships.
// On THIS box the API is unavailable (hipOccupancyMaxActiveBlocksPerMultiprocessor
// returns hipErrorInvalidDeviceFunction for every kernel - learnings todo 32),
// so the test SKIPs with that documented as the known state.
// ---------------------------------------------------------------------------

TEST_CASE("occupancy_analytic_vs_hip_api_within_one_block", "[gpu][occupancy]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  const std::string cached = find_hsaco();
  REQUIRE_FALSE(cached.empty());
  const std::vector<char> blob = kernelh::test::read_file_bin(cached);
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

  // The loaded kernel's resources (todo-33 reporter over a scratch HSACO):
  // the analytic bounds below need its real vgpr_count + lds_bytes. The todo-7
  // xchg kernel has group_segment_fixed_size = 1024 (cross-thread LDS exchange)
  // and ~20 VGPRs.
  const std::string scratch = "/tmp/kernelh-t36-val";
  std::filesystem::remove_all(scratch);
  std::filesystem::create_directories(scratch);
  {
    std::ofstream out(scratch + "/xchg_kernel.hsaco", std::ios::binary);
    out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  }
  auto resources =
      kernelh::compiler::resource_report("xchg_kernel", "amdgcn-gfx1101",
                                         scratch);
  REQUIRE(resources.has_value());
  REQUIRE(resources->vgpr_count > 0);

  // 3 representative configs (threads, shared bytes): a small block, the
  // bench's typical 256-thread block, and a shared-carrying variant.
  const int configs[][2] = {{64, 0}, {256, 0}, {256, 1024}};
  int api_blocks[3] = {0, 0, 0};
  bool api_available = false;
  for (int i = 0; i < 3; ++i) {
    api_blocks[i] = kernelh::compiler::query_occupancy_max_blocks_runtime(
        fn->handle, configs[i][0],
        static_cast<size_t>(configs[i][1]));
    api_available = api_available || api_blocks[i] >= 0;
  }
  if (!api_available) {
    // KNOWN STATE on this box (documented): the vendor occupancy API returns
    // hipErrorInvalidDeviceFunction (98) for every kernel on ROCm 7.2.3 +
    // this kernel/driver combo (learnings todo 32). The todo-32 hook returns
    // -1 and every consumer falls back to the analytic path - the validation
    // simply cannot run here and must not fail the suite.
    SKIP("hipOccupancyMaxActiveBlocksPerMultiprocessor is unavailable on this "
         "box (returns hipErrorInvalidDeviceFunction for every kernel); the "
         "analytic-vs-API agreement validation cannot run - the vendor-API "
         "fallback to analytic occupancy is the documented path");
  }

  for (int i = 0; i < 3; ++i) {
    if (api_blocks[i] < 0) {
      continue; // a partial API failure degrades per-config
    }
    // Analytic estimate in BLOCKS per CU (the unit the API reports). The
    // pinned Oracle B5 bounds are compared directly by occupancy_report, so
    // the validation converts the VGPR bound (waves) to full blocks:
    //   waves_per_block = ceil(threads / wavefront_size)
    //   vgpr_blocks     = vgpr_limited_waves / waves_per_block
    //   analytic_blocks = min(vgpr_blocks, lds_limited_blocks,
    //                         thread_limited_blocks)
    OccupancyInput input;
    input.kernel = "xchg_kernel";
    input.target = "gfx1101";
    input.config = std::to_string(configs[i][0]) + " threads";
    input.vgpr_count = resources->vgpr_count;
    input.lds_bytes = configs[i][1] > 0 ? resources->lds_bytes
                                        : 0; // shared 0 -> LDS bound skipped
    input.threads_per_workgroup = configs[i][0];
    const OccupancyReport r = occupancy_report(input, gfx1101_limits);
    const int waves_per_block =
        (configs[i][0] + gfx1101_limits.wavefront_size - 1) /
        gfx1101_limits.wavefront_size;
    const int vgpr_blocks =
        r.vgpr_limited_waves > 0 ? r.vgpr_limited_waves / waves_per_block
                                 : 0;
    int analytic_blocks = vgpr_blocks;
    if (r.lds_limited_blocks > 0 && r.lds_limited_blocks < analytic_blocks) {
      analytic_blocks = r.lds_limited_blocks;
    }
    if (r.thread_limited_blocks > 0 &&
        r.thread_limited_blocks < analytic_blocks) {
      analytic_blocks = r.thread_limited_blocks;
    }
    INFO("config threads=" << configs[i][0]
         << " shared=" << configs[i][1] << " analytic_blocks="
         << analytic_blocks << " api_blocks=" << api_blocks[i]);
    REQUIRE(std::abs(analytic_blocks - api_blocks[i]) <= 1);
  }
}
