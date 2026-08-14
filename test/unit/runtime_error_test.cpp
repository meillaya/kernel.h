// Runtime error-mapping tests (plan todo 23, F2): every failure must surface
// as a typed Error with a non-zero code and a non-empty message - never an
// exception and never a crash.

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/runtime_error.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("runtime_hip_error_string_nonempty", "[unit][runtime]") {
  // The HIP error-string helper works without any device; an out-of-range
  // code must still produce a non-empty description.
  CHECK_FALSE(kernelh::runtime::hip_error_string(9999).empty());
  CHECK_FALSE(kernelh::runtime::hip_error_string(0).empty());
}

TEST_CASE("runtime_bad_device_ordinal_typed_error", "[unit][runtime]") {
  // hipSetDevice with a bogus ordinal must be mapped to the runtime Error
  // type (non-zero code + non-empty message), not thrown and not crashing.
  auto backend = kernelh::runtime::make_hip_backend();
  REQUIRE(backend != nullptr);

  auto ctx = kernelh::runtime::DeviceContext::create(backend, 1234567);
  REQUIRE_FALSE(ctx.has_value());
  CHECK(ctx.error().code != 0);
  CHECK_FALSE(ctx.error().message.empty());
}

TEST_CASE("runtime_null_backend_typed_error", "[unit][runtime]") {
  auto ctx = kernelh::runtime::DeviceContext::create(nullptr, 0);
  REQUIRE_FALSE(ctx.has_value());
  CHECK(ctx.error().code != 0);
  CHECK_FALSE(ctx.error().message.empty());

  auto buf = kernelh::runtime::Buffer::allocate(nullptr, 1024);
  REQUIRE_FALSE(buf.has_value());
  CHECK_FALSE(buf.error().message.empty());
}

TEST_CASE("runtime_cuda_backend_absent_skips_cleanly", "[unit][runtime]") {
  // Cross-ready (plan G10): make_cuda_backend either succeeds (a machine with
  // an NVIDIA driver) or returns a clean Error with the skip code - never
  // crashes and never links libcuda (verified separately via ldd).
  auto cuda = kernelh::runtime::make_cuda_backend();
  if (cuda.has_value()) {
    CHECK((*cuda)->kind() == kernelh::runtime::BackendKind::kCuda);
  } else {
    CHECK(cuda.error().code == kernelh::runtime::kCudaDriverUnavailable);
    CHECK_FALSE(cuda.error().message.empty());
    CHECK(cuda.error().message.find("CUDA") != std::string::npos);
  }
}
