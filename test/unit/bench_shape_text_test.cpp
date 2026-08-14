// Shape-text round-trip + malformed-input tests (plan todo 35).
//
// The G32 shape column uses "k=v,k=v,..." text; a malformed segment must be
// rejected (returned as an empty map) so the runner never launches a row
// with a silently-corrupted shape (the malformed_input adversarial class).

#include "shape_text.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("bench_shape_text roundtrip", "[bench_shape_text]") {
  const kernelh::bench::ShapeMap shape{{"m", 512}, {"n", 512}, {"k", 512}};
  const std::string text = kernelh::bench::shape_to_text(shape);
  REQUIRE(text == "k=512,m=512,n=512"); // std::map -> sorted keys
  const kernelh::bench::ShapeMap parsed =
      kernelh::bench::parse_shape_text(text);
  REQUIRE(parsed == shape);
}

TEST_CASE("bench_shape_text single pair", "[bench_shape_text]") {
  REQUIRE(kernelh::bench::shape_to_text({{"n", 268435456}}) == "n=268435456");
  REQUIRE(kernelh::bench::parse_shape_text("n=100000").at("n") == 100000);
}

TEST_CASE("bench_shape_text malformed rejected", "[bench_shape_text]") {
  REQUIRE(kernelh::bench::parse_shape_text("").empty());
  REQUIRE(kernelh::bench::parse_shape_text("m=512,n").empty());
  REQUIRE(kernelh::bench::parse_shape_text("=512").empty());
  REQUIRE(kernelh::bench::parse_shape_text("m=").empty());
  REQUIRE(kernelh::bench::parse_shape_text("m=abc").empty());
  REQUIRE(kernelh::bench::parse_shape_text("m=5.5").empty());
  REQUIRE(kernelh::bench::parse_shape_text("m=512,,n=1").empty());
}
