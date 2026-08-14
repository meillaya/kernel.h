// Device-limits table unit tests (plan todo 10).
//
// The table is the SINGLE SOURCE OF TRUTH for every device constant used by
// passes/verifier/occupancy; these tests pin the exact value of EVERY field
// (zero fields included) so a silent value change anywhere is caught, plus
// the lookup contract and the weak runtime-hook default.

#include "kernelh/compiler/device_limits.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using kernelh::DeviceLimits;
using kernelh::device_limits_for;
using kernelh::gfx1101_limits;
using kernelh::sm_80_limits;

namespace {

// Exact-match helper: checks that the returned entry matches `expected` on
// EVERY field - including zeros - so a misleading "all nonzero fields match"
// assertion can never pass over a wrong constant.
void check_entry(const DeviceLimits &got, const DeviceLimits &expected) {
  CHECK(got.lds_bytes_per_workgroup == expected.lds_bytes_per_workgroup);
  CHECK(got.wavefront_size == expected.wavefront_size);
  CHECK(got.max_threads_per_workgroup == expected.max_threads_per_workgroup);
  CHECK(got.max_grid_dim[0] == expected.max_grid_dim[0]);
  CHECK(got.max_grid_dim[1] == expected.max_grid_dim[1]);
  CHECK(got.max_grid_dim[2] == expected.max_grid_dim[2]);
  CHECK(got.vram_bytes == expected.vram_bytes);
  CHECK(got.waves_per_cu == expected.waves_per_cu);
  CHECK(got.reg_file_per_cu == expected.reg_file_per_cu);
  CHECK(got.cu_count == expected.cu_count);
  CHECK(std::string(got.name) == std::string(expected.name));
}

} // namespace

TEST_CASE("device_limits_gfx1101_exact_values", "[unit][device_limits]") {
  const DeviceLimits expected = {
      /*lds_bytes_per_workgroup=*/65536,
      /*wavefront_size=*/32,
      /*max_threads_per_workgroup=*/1024,
      /*max_grid_dim=*/{2147483647, 65535, 65535},
      /*vram_bytes=*/17163091968L,
      /*waves_per_cu=*/32,
      /*reg_file_per_cu=*/6144,
      /*cu_count=*/0,
      /*name=*/"AMD Navi 32 (gfx1101)",
  };
  check_entry(gfx1101_limits, expected);
}

TEST_CASE("device_limits_sm80_exact_values", "[unit][device_limits]") {
  const DeviceLimits expected = {
      /*lds_bytes_per_workgroup=*/49152,
      /*wavefront_size=*/32,
      /*max_threads_per_workgroup=*/1024,
      /*max_grid_dim=*/{2147483647, 65535, 65535},
      /*vram_bytes=*/0L,
      /*waves_per_cu=*/64,
      /*reg_file_per_cu=*/65536,
      /*cu_count=*/0,
      /*name=*/"NVIDIA A100-class (sm_80, cross)",
  };
  check_entry(sm_80_limits, expected);
}

TEST_CASE("device_limits_lookup_hits", "[unit][device_limits]") {
  // Exact-match lookup must return the very table entries (same address =
  // static storage, the reference never dangles).
  CHECK(&device_limits_for("gfx1101") == &gfx1101_limits);
  CHECK(&device_limits_for("sm_80") == &sm_80_limits);
  CHECK(&device_limits_for("gfx1101") == &device_limits_for("gfx1101"));
  CHECK(&device_limits_for("sm_80") == &device_limits_for("sm_80"));
}

TEST_CASE("device_limits_unknown_target_rejected", "[unit][device_limits]") {
  // Unknown target must throw std::invalid_argument whose message names the
  // offending target AND lists every known target token.
  bool threw = false;
  try {
    (void)device_limits_for("gfx9999");
  } catch (const std::invalid_argument &e) {
    threw = true;
    const std::string msg = e.what();
    CHECK(msg.find("gfx9999") != std::string::npos);
    CHECK(msg.find("gfx1101") != std::string::npos);
    CHECK(msg.find("sm_80") != std::string::npos);
  }
  CHECK(threw);
}

TEST_CASE("device_limits_null_and_empty_target_rejected", "[unit][device_limits]") {
  // Null/empty are unknown targets too; the message must not dereference null.
  CHECK_THROWS_AS(device_limits_for(nullptr), std::invalid_argument);
  CHECK_THROWS_AS(device_limits_for(""), std::invalid_argument);
}
// NOTE (todo 23): device_limits_weak_hook_defaults_to_zero moved to
// weak_default_test.cpp - this binary now links the strong runtime override.
