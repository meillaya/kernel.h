// CANARY TRIPWIRE (plan todo 25, Oracle R2-B1): the FIRST kernel end-to-end
// execution through the FULL production pipeline, kept deliberately minimal
// so it doubles as the degraded-flag wiring tripwire.
//
//   kernelh compile --kernel vecadd --shape n=1024 --target amdgcn-gfx1101
//     -> vecadd.hsaco      (todo 24 production driver, shelled out via
//                            std::system - same precedent as the mlir-opt
//                            re-emission fallback in runtime_integration_test.cpp)
//   -> GpuModule::load     (kernelh_runtime, todo 23)
//   -> get_function("vecadd") + KernelLaunch, 3 memref args
//   -> numeric sentinel    (c[i] == a[i]+b[i] for every element)
//
// The numeric check IS the sentinel: a launched-but-broken kernel (or a
// kernel that never ran) leaves the output buffer at its initial value and
// fails the comparison. If this test ever SKIPs while the degraded flag says
// degraded=false, the [gpu] gate wiring is broken - STOP and report it.

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include "gpu_gate.hpp"
#include "kernelh_compile_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace {

// Fill `h_a[i] = (i%257)*0.5f`, `h_b[i] = (i%331)*0.25f + 0.5f`: all inputs
// are multiples of 0.25, exactly representable in f32, and their sums are
// exact too, so the sentinel comparison is an exact equality (no tolerance).
// b[i] >= 0.5 keeps b[i] never zero: the wrong-reference negative control
// needs a+b != a-b on EVERY element (a zero b would make them coincide).
void fill_inputs(std::vector<float> &a, std::vector<float> &b) {
  const size_t n = a.size();
  for (size_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>((i % 257)) * 0.5f;
    b[i] = static_cast<float>((i % 331)) * 0.25f + 0.5f;
  }
}

} // namespace

TEST_CASE("canary_gpu", "[gpu]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }

  constexpr long kBlock = 256;
  constexpr long kN = 1024;

  // 1. Compile vecadd through the production driver (todo 24 CLI).
  const std::string outDir = "/tmp/kernelh-t25-canary";
  const int rc = kernelh::test::compile_kernel(
      "vecadd", "n=" + std::to_string(kN), {"amdgcn-gfx1101"}, outDir);
  REQUIRE(rc == 0);

  const std::string hsacoPath = outDir + "/vecadd.hsaco";
  REQUIRE(kernelh::test::file_exists(hsacoPath));
  const std::vector<char> blob = kernelh::test::read_file_bin(hsacoPath);
  REQUIRE(kernelh::test::is_elf_magic(blob));
  // The produced blob actually carries the kernel symbol (content probe, not
  // just file existence).
  REQUIRE(kernelh::test::artifact_contains_symbol(hsacoPath, "vecadd"));

  // 2. Load + launch once.
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
  REQUIRE(launch.args().param_count() == 15); // 3 buffers x 5 descriptor fields

  REQUIRE(launch.launch(*fn).has_value());
  REQUIRE(c_buf->copy_to_host(h_c.data(), bytes).has_value());

  // 3. Sentinel: the kernel ran and is correct iff c[i] == a[i]+b[i].
  long bad = 0;
  for (long i = 0; i < kN; ++i) {
    const float want =
        static_cast<float>(static_cast<double>(h_a[static_cast<size_t>(i)]) +
                           static_cast<double>(h_b[static_cast<size_t>(i)]));
    if (h_c[static_cast<size_t>(i)] != want) {
      if (bad < 8) {
        std::fprintf(stderr, "MISMATCH i=%ld got=%f want=%f\n", i,
                     static_cast<double>(h_c[static_cast<size_t>(i)]),
                     static_cast<double>(want));
      }
      ++bad;
    }
  }
  REQUIRE(bad == 0);
  std::printf("CANARY-GPU-PASSED (n=%ld, block=%ld)\n", kN, kBlock);
}
