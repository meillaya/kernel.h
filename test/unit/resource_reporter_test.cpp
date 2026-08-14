// Resource reporter unit tests (plan todo 33).
//
// Covers:
//   - parsing a REAL captured `llvm-readobj --notes` dump fixture
//     (test/fixtures/notes_gemm_hsaco.txt) and a REAL captured ptxas -v
//     output (test/fixtures/ptxas_gemm_v.txt with registers only,
//     test/fixtures/ptxas_smem_v.txt with "M bytes smem"),
//   - the PTX static-shared scan (ptxas does NOT report module-scope
//     `.shared` arrays - our emitter's LDS form - so lds_bytes for NVIDIA
//     must come from the PTX text),
//   - the kernel_trace.csv (rocprofv3) parser against a representative ROCm
//     7.x sample,
//   - truncated/corrupt HSACO -> clean "unparseable code object" error with
//     a non-zero exit and no crash (malformed_input),
//   - the `kernelh compile --resources` CLI wiring end to end, with jq-style
//     assertions on the PARSED JSON VALUES (not file existence),
//   - the analytic occupancy_estimate against hand-computed waves,
//   - NVIDIA toolkit-missing -> non-fatal zeros + warning (still reporting
//     static shared bytes),
//   - the [gpu] rocprofv3 priority path, which SKIPs when rocprofv3 is
//     unavailable/degraded (expected in the devenv).
//
// Each Catch2 case runs in its own process (catch_discover_tests), so the
// tests that temporarily override KERNELH_CUDA_TOOLKIT_ROOT cannot race
// sibling cases.

#include "kernelh/compiler/resource_reporter.hpp"

#include "gpu_gate.hpp"
#include "kernelh_compile_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef KERNELH_SOURCE_DIR
#error "resource_reporter_test.cpp requires the KERNELH_SOURCE_DIR definition"
#endif

namespace {

// Reads a text fixture from test/fixtures (KERNELH_SOURCE_DIR baked by CMake).
std::string read_fixture(const std::string &name) {
  std::ifstream in(std::string(KERNELH_SOURCE_DIR) + "/test/fixtures/" + name);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string read_text(const std::string &path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// jq-style single-field integer extractor for the resources.json schema.
// Returns -1 when the field is missing (so tests cannot pass on "file exists"
// - they must find and compare the actual parsed value).
long json_int_field(const std::string &json, const std::string &field) {
  const std::string key = "\"" + field + "\":";
  const size_t pos = json.find(key);
  if (pos == std::string::npos) {
    return -1;
  }
  size_t start = pos + key.size();
  while (start < json.size() &&
         (json[start] == ' ' || json[start] == '\t')) {
    ++start;
  }
  size_t end = start;
  while (end < json.size() && json[end] >= '0' && json[end] <= '9') {
    ++end;
  }
  if (end == start) {
    return -1;
  }
  char *value_end = nullptr;
  const long value = std::strtol(json.c_str() + start, &value_end, 10);
  return value_end == json.c_str() + end ? value : -1;
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

} // namespace

using kernelh::compiler::KernelResources;
using kernelh::compiler::parse_amdhsa_notes;
using kernelh::compiler::parse_ptxas_report;
using kernelh::compiler::resource_report;
using kernelh::compiler::write_resources_json;

TEST_CASE("resource_reporter_parse_amdhsa_notes_fixture",
          "[resource_reporter]") {
  // REAL captured output: `llvm-readobj --notes <gemm.hsaco>` on this box.
  const std::string notes = read_fixture("notes_gemm_hsaco.txt");
  REQUIRE_FALSE(notes.empty());

  auto parsed = parse_amdhsa_notes(notes, "gemm");
  REQUIRE(parsed.has_value());
  CHECK(parsed->kernel == "gemm");
  CHECK(parsed->vgpr_count == 19);
  CHECK(parsed->sgpr_count == 34);
  CHECK(parsed->lds_bytes == 0); // todo-26 phantom LDS: staging did not fire
  CHECK(parsed->wavefront_size == 32);
}

TEST_CASE("resource_reporter_parse_amdhsa_notes_first_entry",
          "[resource_reporter]") {
  // An unmatched hint falls back to the first amdhsa.kernels entry.
  const std::string notes = read_fixture("notes_gemm_hsaco.txt");
  auto parsed = parse_amdhsa_notes(notes, "no_such_kernel");
  REQUIRE(parsed.has_value());
  CHECK(parsed->vgpr_count == 19);
  CHECK(parsed->kernel == "gemm");
}

TEST_CASE("resource_reporter_parse_amdhsa_notes_rejects_garbage",
          "[resource_reporter]") {
  const auto garbage = parse_amdhsa_notes("this is not a notes dump", "");
  REQUIRE_FALSE(garbage.has_value());
  CHECK(garbage.error().find("unparseable code object") != std::string::npos);

  // A notes dump with the marker but no kernel descriptor is also corrupt.
  const auto empty_kernels = parse_amdhsa_notes(
      "AMDGPU Metadata: ---\namdhsa.kernels:\n  nothing here\n", "");
  REQUIRE_FALSE(empty_kernels.has_value());
  CHECK(empty_kernels.error().find("unparseable code object") !=
        std::string::npos);
}

TEST_CASE("resource_reporter_parse_ptxas_gemm_fixture", "[resource_reporter]") {
  // REAL captured output: `ptxas -v -arch=sm_80 -c gemm.ptx` on this box.
  // Note: the MLIR-emitted PTX has module-scope `.shared` arrays that ptxas
  // does NOT report as smem, so this fixture has registers but no smem line.
  const std::string ptxas_out = read_fixture("ptxas_gemm_v.txt");
  REQUIRE_FALSE(ptxas_out.empty());

  auto parsed = parse_ptxas_report(ptxas_out, "");
  REQUIRE(parsed.has_value());
  CHECK(parsed->vgpr_count == 58);
  CHECK(parsed->sgpr_count == 0);
  CHECK(parsed->lds_bytes == 0);
}

TEST_CASE("resource_reporter_parse_ptxas_smem_fixture", "[resource_reporter]") {
  // REAL captured output: `nvcc -arch=sm_80 -Xptxas -v` on a kernel with a
  // 16 KiB static __shared__ tile - the classic "Used N registers, M bytes
  // smem" form.
  const std::string ptxas_out = read_fixture("ptxas_smem_v.txt");
  REQUIRE_FALSE(ptxas_out.empty());

  auto parsed = parse_ptxas_report(ptxas_out, "");
  REQUIRE(parsed.has_value());
  CHECK(parsed->vgpr_count == 8);
  CHECK(parsed->lds_bytes == 16384);
}

TEST_CASE("resource_reporter_parse_ptxas_static_shared", "[resource_reporter]") {
  // Our emitter's PTX form: `.visible .shared .align 4 .b8 name[4096];` -
  // ptxas -v does NOT report this, so lds_bytes must come from the PTX text.
  const std::string ptx_sample =
      ".version 8.0\n"
      ".target sm_80\n"
      ".visible .shared .align 4 .b8 gemm_shared_0[4096];\n"
      ".visible .entry gemm(\n"
      "    .param .u64 gemm_param_0, .param .u64 gemm_param_1\n"
      ") {\n"
      "    .reg .b64 \t%rd<20>;\n"
      "    mov.b64 \t%rd1, gemm_shared_0;\n"
      "    ret;\n"
      "}\n";
  const std::string ptxas_out =
      "ptxas info    : Compiling entry function 'gemm' for 'sm_80'\n"
      "ptxas info    : Used 58 registers, used 1 barriers, 520 bytes cmem[0]\n";

  auto parsed = parse_ptxas_report(ptxas_out, ptx_sample);
  REQUIRE(parsed.has_value());
  CHECK(parsed->vgpr_count == 58);
  CHECK(parsed->lds_bytes == 4096); // static shared, not ptxas smem
}

TEST_CASE("resource_reporter_parse_ptxas_rejects_garbage",
          "[resource_reporter]") {
  const auto garbage = parse_ptxas_report("nothing here", "");
  REQUIRE_FALSE(garbage.has_value());
}

TEST_CASE("resource_reporter_kernel_trace_csv_sample", "[resource_reporter]") {
  // Representative ROCm 7.x kernel_trace.csv layout. Columns are located by
  // name (Kernel_Name/VGPR_Count/SGPR_Count/LDS_Block_Size), so this sample
  // deliberately orders them differently than the amdhsa-notes fixture and
  // the row-selection must pick the row matching the requested kernel.
  const std::string csv =
      "ID,Dispatch_ID,GridX,GridY,GridZ,BlockX,BlockY,BlockZ,Kernel_Name,"
      "Global_Size_X,Global_Size_Y,Global_Size_Z,GroupSize_X,GroupSize_Y,"
      "GroupSize_Z,SIMD_Count,SIMD_Width,VGPR_Count,SGPR_Count,"
      "LDS_Block_Size,LDS_Total_Size,SCRATCH_Total_Size,Timestamps\n"
      "2,1,8,1,1,256,1,1,gemm,2048,1,1,256,1,1,8,64,19,34,0,0,0,\n"
      "4,2,4,1,1,256,1,1,scan,1024,1,1,256,1,1,8,64,21,16,3072,3072,0,\n";

  auto gemm = kernelh::compiler::parse_kernel_trace_csv(csv, "gemm");
  REQUIRE(gemm.has_value());
  CHECK(gemm->kernel == "gemm");
  CHECK(gemm->vgpr_count == 19); // parity with the amdhsa notes fixture
  CHECK(gemm->sgpr_count == 34);
  CHECK(gemm->lds_bytes == 0);

  auto scan = kernelh::compiler::parse_kernel_trace_csv(csv, "scan");
  REQUIRE(scan.has_value());
  CHECK(scan->vgpr_count == 21);
  CHECK(scan->sgpr_count == 16);
  CHECK(scan->lds_bytes == 3072);

  const auto missing =
      kernelh::compiler::parse_kernel_trace_csv(csv, "no_such_kernel");
  REQUIRE_FALSE(missing.has_value());
}

namespace {
// Shared helper: compile <kernel> with the given --shape for the given target
// and return the temp out dir. Uses the PRODUCTION CLI (todo 24 driver).
std::string compile_for_reporter(const std::string &kernel,
                                 const std::string &shape,
                                 const std::string &target,
                                 const std::string &tag) {
  const std::string outDir = "/tmp/kernelh-t33-" + tag;
  const int rc = kernelh::test::compile_kernel(kernel, shape, {target}, outDir);
  REQUIRE(rc == 0);
  return outDir;
}
} // namespace

TEST_CASE("resource_reporter_truncated_hsaco_clean_error",
          "[resource_reporter]") {
  // Compile a real GEMM HSACO, then truncate it to a corrupt prefix.
  const std::string outDir = compile_for_reporter(
      "gemm", "m=256,k=256,n=256", "amdgcn-gfx1101", "trunc");
  const std::string hsaco = outDir + "/gemm.hsaco";
  {
    std::ifstream in(hsaco, std::ios::binary);
    REQUIRE(in.good());
    std::vector<char> prefix(64);
    in.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    std::ofstream out(hsaco, std::ios::binary | std::ios::trunc);
    out.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
  }

  // Must fail cleanly: "unparseable code object", non-zero, no crash.
  auto res = resource_report("gemm", "amdgcn-gfx1101", outDir);
  REQUIRE_FALSE(res.has_value());
  CHECK(res.error().find("unparseable code object") != std::string::npos);
}

TEST_CASE("resource_reporter_cli_gemm_writes_json", "[resource_reporter]") {
  // The acceptance path: `kernelh compile --kernel gemm --shape ... --target
  // amdgcn-gfx1101 --out <dir> --resources`. Assertions are on the PARSED
  // JSON VALUES (jq-equivalent), never on file existence alone.
  const std::string outDir = "/tmp/kernelh-t33-cligemm";
  const std::string cmd = "rm -rf '" + outDir + "' && mkdir -p '" + outDir +
                          "' && '" + kernelh::test::kernelh_cli() +
                          "' compile --kernel gemm --shape m=256,k=256,n=256 "
                          "--target amdgcn-gfx1101 --out '" +
                          outDir + "' --resources";
  REQUIRE(std::system(cmd.c_str()) == 0);

  const std::string json = read_text(outDir + "/resources.json");
  REQUIRE_FALSE(json.empty());
  CHECK(json_int_field(json, "vgpr_count") == 19);
  CHECK(json_int_field(json, "sgpr_count") == 34);
  CHECK(json_int_field(json, "wavefront_size") == 32);
  // Hand-computed occupancy: floor(6144/(19*32)) = 10.
  CHECK(json_int_field(json, "occupancy_estimate") == 10);
  CHECK(json_string_field(json, "kernel") == "gemm");
  CHECK(json_string_field(json, "target") == "amdgcn-gfx1101");
}

TEST_CASE("resource_reporter_scan_lds_bytes_positive", "[resource_reporter]") {
  // GEMM's group_segment_fixed_size is 0 on this pipeline (todo-26 phantom
  // LDS: staging did not fire for the default config), so the lds_bytes > 0
  // acceptance uses the SCAN kernel, which stages 3 x 256 f32 (3072 B).
  //   AMD:   group_segment_fixed_size == 3072 from the amdhsa notes.
  //   NVIDIA: ptxas registers > 0 AND static shared scan == 3072 (ptxas -v
  //           does not report module-scope .shared, the PTX text does).
  const std::string amdDir = compile_for_reporter(
      "scan", "n=1024", "amdgcn-gfx1101", "scan-amd");
  auto amd = resource_report("scan", "amdgcn-gfx1101", amdDir);
  REQUIRE(amd.has_value());
  CHECK(amd->vgpr_count > 0);
  CHECK(amd->lds_bytes == 3072);
  // Hand-computed occupancy (gfx1101: 6144 regs/CU, wave 32): the scan
  // HSACO descriptor reports vgpr 21 -> floor(6144/(21*32)) = floor(6144/672)
  // = 9.
  CHECK(amd->occupancy_estimate == 9);

  const std::string nvDir =
      compile_for_reporter("scan", "n=1024", "nvptx-sm_80", "scan-nv");
  auto nv = resource_report("scan", "nvptx-sm_80", nvDir);
  REQUIRE(nv.has_value());
  CHECK(nv->vgpr_count > 0);
  CHECK(nv->lds_bytes == 3072);
  // sm_80: 65536 regs/SM, warp 32, waves_per_cu 64. ptxas reports ~20 regs ->
  // floor(65536/(20*32)) = 102, capped by waves_per_cu = 64.
  CHECK(nv->occupancy_estimate == 64);
}

TEST_CASE("resource_reporter_nvidia_toolkit_missing_nonfatal",
          "[resource_reporter]") {
  // ptxas unavailable -> NON-FATAL: zeros with a warning, but the static
  // shared bytes are still read from the PTX text. This process overrides
  // KERNELH_CUDA_TOOLKIT_ROOT to a missing dir; catch_discover_tests runs
  // each case in its own process, so no sibling can see the override.
  const std::string outDir = compile_for_reporter(
      "scan", "n=1024", "nvptx-sm_80", "notoolkit");
  const std::string saved =
      std::getenv("KERNELH_CUDA_TOOLKIT_ROOT") == nullptr
          ? std::string()
          : std::string(std::getenv("KERNELH_CUDA_TOOLKIT_ROOT"));
  std::string restore = saved;
  setenv("KERNELH_CUDA_TOOLKIT_ROOT", "/nonexistent-toolkit-root", 1);

  auto res = resource_report("scan", "nvptx-sm_80", outDir);
  unsetenv("KERNELH_CUDA_TOOLKIT_ROOT");
  if (!restore.empty()) {
    setenv("KERNELH_CUDA_TOOLKIT_ROOT", restore.c_str(), 1);
  }
  REQUIRE(res.has_value()); // non-fatal: no error despite missing ptxas
  CHECK(res->vgpr_count == 0); // filled zeros with a warning
  CHECK(res->lds_bytes == 3072); // static shared still reported
}

TEST_CASE("resource_reporter_rocprofv3_gate", "[gpu][resource_reporter]") {
  // Source-priority-1 path. SKIPs when rocprofv3 is unavailable (the expected
  // devenv state) or the box is degraded - the reporter then falls back to
  // the static amdhsa-notes parsing, which the other cases cover.
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }
  if (!kernelh::compiler::rocprofv3_available()) {
    SKIP("rocprofv3 unavailable; source priority falls back to static parsing");
  }
  const char *launch = std::getenv("KERNELH_MICRO_LAUNCH");
  if (launch == nullptr || *launch == '\0') {
    SKIP("rocprofv3 present but KERNELH_MICRO_LAUNCH unset");
  }
  const std::string outDir = compile_for_reporter(
      "gemm", "m=256,k=256,n=256", "amdgcn-gfx1101", "rocprof");
  auto res = resource_report("gemm", "amdgcn-gfx1101", outDir);
  REQUIRE(res.has_value());
  CHECK(res->vgpr_count > 0);
}
