// [gpu]-tagged integration tests (plan todo 23): load the todo-7
// cross-thread-exchange HSACO (xchg.hsaco) via GpuModule and launch it via
// KernelLaunch with the PINNED descriptor packing, verifying the harness
// contract out[i] = (i+1)%256 HOST-side. A corrupted-blob probe proves the
// error path surfaces as the typed Error (no crash). All cases SKIP when the
// degraded flag says the GPU is unavailable (gpu_gate.hpp).

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/event_timer.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include "gpu_gate.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// The todo-7 HSACO is not committed (regenerable from the committed fixture +
// pipeline); candidate locations are checked in order.
std::vector<std::string> hsaco_candidates() {
  return {
      "/tmp/kernelh-w2/xchg.hsaco",
      std::string(KERNELH_SOURCE_DIR) + "/.omo/evidence/task-7-kernelh/xchg.hsaco",
      std::string(KERNELH_SOURCE_DIR) + "/build/test/fixtures/xchg.hsaco",
      std::string(KERNELH_SOURCE_DIR) + "/test/fixtures/xchg.hsaco",
  };
}

bool file_exists(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return in.good();
}

std::string find_hsaco() {
  for (const auto &candidate : hsaco_candidates()) {
    if (file_exists(candidate)) {
      return candidate;
    }
  }
  return "";
}

std::vector<char> read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto size = in.tellg();
  std::vector<char> data(size > 0 ? static_cast<size_t>(size) : 0);
  in.seekg(0);
  if (size > 0) {
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
  }
  return data;
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
  const std::string out_mlir = "/tmp/kernelh-runtime-emit.mlir";
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
  std::vector<char> mlir = read_file(out_mlir);
  std::remove(out_mlir.c_str());
  return decode_mlir_bin(mlir);
}

// Loads the xchg.hsaco bytes: reuse a cached copy if present, else re-emit
// from the committed fixture + pipeline (deterministic per todo-7 learnings).
std::vector<char> obtain_xchg_hsaco() {
  const std::string cached = find_hsaco();
  if (!cached.empty()) {
    return read_file(cached);
  }
  return emit_hsaco_from_fixture();
}

} // namespace

TEST_CASE("runtime_gpu_load_launch_xchg_descriptor_abi", "[gpu][runtime]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  std::vector<char> blob = obtain_xchg_hsaco();
  INFO("xchg.hsaco blob bytes: " << blob.size());
  REQUIRE_FALSE(blob.empty());
  // ELF magic proves it is a real code object, not an empty/garbage file.
  REQUIRE(blob.size() >= 4);
  REQUIRE(static_cast<unsigned char>(blob[0]) == 0x7f);
  REQUIRE(blob[1] == 'E');
  REQUIRE(blob[2] == 'L');
  REQUIRE(blob[3] == 'F');

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);
  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  // Harness contract (todo 6/7): N = 1<<20 floats, 256-thread blocks,
  // sharedMemBytes=0, in[i]=(i%256), expected out[i]=((i+1)%256).
  constexpr long kBlock = 256;
  constexpr long kN = 1L << 20;
  const size_t bytes = sizeof(float) * static_cast<size_t>(kN);

  std::vector<float> h_in(static_cast<size_t>(kN));
  std::vector<float> h_out(static_cast<size_t>(kN), 0.0f);
  for (long i = 0; i < kN; ++i) {
    h_in[static_cast<size_t>(i)] = static_cast<float>(i % kBlock);
  }

  auto in_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(in_buf.has_value());
  auto out_buf = kernelh::runtime::Buffer::allocate(backend, bytes);
  REQUIRE(out_buf.has_value());
  REQUIRE(in_buf->copy_from_host(h_in.data(), bytes).has_value());

  auto module =
      kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  REQUIRE(module.has_value());

  // The PRODUCTION descriptor-ABI kernel (10 user args = 2 memref descriptors
  // x 5 fields); launched with the pinned descriptor packing.
  auto fn = module->get_function("xchg_kernel");
  REQUIRE(fn.has_value());

  auto timer_or = kernelh::runtime::EventTimer::create(backend);
  REQUIRE(timer_or.has_value());

  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(kN / kBlock)).block(static_cast<unsigned>(kBlock));
  launch.memref_arg(in_buf->device_ptr(), std::vector<int64_t>{kN});
  launch.memref_arg(out_buf->device_ptr(), std::vector<int64_t>{kN});
  REQUIRE(launch.args().param_count() == 10); // 2 buffers x 5 descriptor fields

  REQUIRE(timer_or->record_start().has_value());
  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(timer_or->record_end().has_value());
  REQUIRE(timer_or->synchronize().has_value());

  auto elapsed = timer_or->elapsed_milliseconds();
  REQUIRE(elapsed.has_value());
  REQUIRE(*elapsed >= 0.0f);

  REQUIRE(out_buf->copy_to_host(h_out.data(), bytes).has_value());

  // HOST-side verification of the cross-thread exchange (misleading_success
  // guard: values are checked, not just the launch exit code).
  long bad = 0;
  for (long i = 0; i < kN; ++i) {
    const float want = static_cast<float>((i + 1) % kBlock);
    if (h_out[static_cast<size_t>(i)] != want) {
      if (bad < 8) {
        std::fprintf(stderr, "MISMATCH i=%ld got=%f want=%f\n", i,
                     static_cast<double>(h_out[static_cast<size_t>(i)]),
                     static_cast<double>(want));
      }
      ++bad;
    }
  }
  REQUIRE(bad == 0);
  std::printf("RUNTIME-XCHG-PASS (N=%ld, block=%ld, elapsed=%.3f ms)\n", kN,
              kBlock, static_cast<double>(*elapsed));
}

TEST_CASE("runtime_gpu_corrupted_blob_typed_error", "[gpu][runtime]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  std::vector<char> blob = obtain_xchg_hsaco();
  REQUIRE_FALSE(blob.empty());

  // Corrupt the first 64 bytes (zero them) - hipModuleLoadData must reject
  // the image and the failure must surface as a typed Error: non-zero code,
  // non-empty message, no crash (malformed_input probe).
  std::vector<char> corrupt = blob;
  std::memset(corrupt.data(), 0, std::min<size_t>(64, corrupt.size()));

  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);

  auto module =
      kernelh::runtime::GpuModule::load(backend, corrupt.data(), corrupt.size());
  REQUIRE_FALSE(module.has_value());
  CHECK(module.error().code != 0);
  CHECK_FALSE(module.error().message.empty());
}
