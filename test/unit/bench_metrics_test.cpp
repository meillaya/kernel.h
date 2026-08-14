// Metric formula unit tests (plan todo 35).
//
// Every value below is hand-computed from the formula definitions in
// bench/harness/metrics.hpp (also documented in scripts/README.md for the
// todo-38 roofline arithmetic-intensity charts). These guard the
// gflops_or_gbps column against formula drift - the "real numbers, not
// misleading zeros" adversarial class.

#include "metrics.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("bench_metrics gflops_from_ms", "[bench_metrics]") {
  // 2^30 flops in 1 ms = 2^30 * 1000 / 1e9 = 1073.741824 GFLOPS.
  REQUIRE(kernelh::bench::gflops_from_ms(1073741824.0, 1.0) ==
          Catch::Approx(1073.741824));
  // 1e9 flops in 1 ms = 1000 GFLOPS.
  REQUIRE(kernelh::bench::gflops_from_ms(1e9, 1.0) == Catch::Approx(1000.0));
  // 2^36 flops in 2 ms = 2^36 * 1000 / 2 / 1e9 = 34359.738368 GFLOPS.
  REQUIRE(kernelh::bench::gflops_from_ms(68719476736.0, 2.0) ==
          Catch::Approx(34359.738368));
}

TEST_CASE("bench_metrics gbps_from_ms", "[bench_metrics]") {
  // 3 GiB in 6 ms = 3 * 2^30 * 1000 / 6 / 1e9 = 536.870912 GB/s.
  REQUIRE(kernelh::bench::gbps_from_ms(3.0 * (1LL << 30), 6.0) ==
          Catch::Approx(536.870912));
}

TEST_CASE("bench_metrics gemm", "[bench_metrics]") {
  // 2 * 512^3 = 268435456 flops.
  REQUIRE(kernelh::bench::gemm_flops(512, 512, 512) ==
          Catch::Approx(268435456.0));
  // 2 * 1024^3 = 2147483648 flops.
  REQUIRE(kernelh::bench::gemm_flops(1024, 1024, 1024) ==
          Catch::Approx(2147483648.0));
  // (512*512 + 512*512 + 512*512) * 4 = 3145728 bytes.
  REQUIRE(kernelh::bench::gemm_bytes(512, 512, 512) == Catch::Approx(3145728.0));
}

TEST_CASE("bench_metrics elementwise", "[bench_metrics]") {
  // (4096,1024): 2 * 4096 * 1024 * 4 = 33554432 bytes.
  REQUIRE(kernelh::bench::elementwise_bytes(4096, 1024) ==
          Catch::Approx(33554432.0));
}

TEST_CASE("bench_metrics layernorm", "[bench_metrics]") {
  // 2 * 4096*1024*4 (x+y) + 2 * 1024 * 4 (gamma+beta) = 33562624 bytes.
  REQUIRE(kernelh::bench::layernorm_bytes(4096, 1024) ==
          Catch::Approx(33562624.0));
}

TEST_CASE("bench_metrics scan", "[bench_metrics]") {
  // 2 * 100000 * 4 = 800000 bytes.
  REQUIRE(kernelh::bench::scan_bytes(100000) == Catch::Approx(800000.0));
}

TEST_CASE("bench_metrics fused_gelu", "[bench_metrics]") {
  // 3 * 1048576 * 4 = 12582912 bytes.
  REQUIRE(kernelh::bench::fused_gelu_bytes(1048576) ==
          Catch::Approx(12582912.0));
}

TEST_CASE("bench_metrics attention", "[bench_metrics]") {
  // 4 * 2 * 8 * 512^2 * 64 = 1073741824 flops.
  REQUIRE(kernelh::bench::attention_flops(2, 8, 512, 64) ==
          Catch::Approx(1073741824.0));
}

TEST_CASE("bench_metrics roofline", "[bench_metrics]") {
  // triad: 3 * 2^28 * 4 = 3221225472 bytes.
  REQUIRE(kernelh::bench::roofline_triad_bytes(1L << 28) ==
          Catch::Approx(3221225472.0));
  // fma: 2^35 FMAs -> 2^36 flops = 68719476736.
  REQUIRE(kernelh::bench::roofline_fma_flops(1LL << 35) ==
          Catch::Approx(68719476736.0));
}

TEST_CASE("bench_metrics end_to_end_gemm_rate", "[bench_metrics]") {
  // 512^3 GEMM in 0.25 ms = 2*512^3 / 0.25e-3 / 1e9 = 1073.741824 GFLOPS.
  const double flops = kernelh::bench::gemm_flops(512, 512, 512);
  REQUIRE(kernelh::bench::gflops_from_ms(flops, 0.25) ==
          Catch::Approx(1073.741824));
}
