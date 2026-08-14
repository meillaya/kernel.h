// Device-context + strong weak-hook override tests (plan todo 23).
//
// These are [gpu]-tagged: they need a live device, so they SKIP when the
// degraded flag says the GPU is unavailable (gpu_gate.hpp).

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/device_context.hpp"

#include "gpu_gate.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("runtime_query_cu_count_strong_override", "[gpu][runtime]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }
  // The STRONG override (lib/runtime/query_cu_count.cpp) must be active, not
  // the weak default of 0: this binary links kernelh_runtime via
  // WHOLE_ARCHIVE, so query_cu_count_runtime() returns the live CU count.
  const kernelh::DeviceLimits &limits = kernelh::device_limits_for("gfx1101");
  const long runtime_cu = kernelh::query_cu_count_runtime();
  const long cu = kernelh::query_cu_count(limits);

  REQUIRE(runtime_cu > 0); // strong override (the weak default is 0)
  REQUIRE(cu == runtime_cu);
  REQUIRE(limits.cu_count == 0); // table: "runtime query required"
}

TEST_CASE("runtime_device_context_props", "[gpu][runtime]") {
  if (kernelh::test::kernelh_gpu_degraded()) {
    SKIP("GPU runtime unavailable (degraded mode)");
  }
  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);

  auto ctx = kernelh::runtime::DeviceContext::create(backend);
  REQUIRE(ctx.has_value());

  CHECK_FALSE(ctx->name().empty());
  CHECK(ctx->backend_kind() == kernelh::runtime::BackendKind::kHip);
  CHECK(ctx->lds_bytes_per_workgroup() == 65536); // RDNA3 gfx1101
  CHECK(ctx->wavefront_size() == 32);             // wave32
  CHECK(ctx->max_threads_per_workgroup() == 1024);
  CHECK(ctx->cu_count() > 0);
  CHECK(ctx->vram_bytes() > 0);

  // Runtime props must agree with the device-limits table (plan todos 10/23:
  // the runtime overrides feed the table).
  const kernelh::DeviceLimits &limits = kernelh::device_limits_for("gfx1101");
  CHECK(ctx->lds_bytes_per_workgroup() == limits.lds_bytes_per_workgroup);
  CHECK(ctx->wavefront_size() == limits.wavefront_size);
  CHECK(ctx->cu_count() == kernelh::query_cu_count(limits));
}
