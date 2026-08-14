// Weak-hook default verification (plan todo 10 contract + todo 23).
//
// This binary links lib/compiler WITHOUT the todo-23 runtime library, so
// kernelh::query_cu_count_runtime() must keep its WEAK default of 0 ("no GPU
// runtime linked / no device visible"). The strong override lives in
// lib/runtime/query_cu_count.cpp and is only linked into kernelh_unit - it is
// verified there (runtime_query_cu_count_strong_override). Keeping the two
// binaries separate is required: plain archive extraction keeps the weak
// definition, and the strong one would override it in the same link.

#include "kernelh/compiler/device_limits.hpp"

#include <catch2/catch_test_macros.hpp>

using kernelh::gfx1101_limits;
using kernelh::query_cu_count;
using kernelh::query_cu_count_runtime;
using kernelh::sm_80_limits;

TEST_CASE("device_limits_weak_hook_defaults_to_zero", "[unit][device_limits]") {
  // No runtime library is linked into THIS binary, so the weak default must
  // report "no device": 0 CUs. cu_count = 0 means "runtime query required",
  // and without a strong override query_cu_count() stays 0.
  CHECK(query_cu_count_runtime() == 0);
  CHECK(query_cu_count(gfx1101_limits) == 0);
  CHECK(query_cu_count(sm_80_limits) == 0);
}
