// Catch2 v3 unit-test entry for kernel.h.
// Real tests arrive with later todos; this locks the ctest wiring with one
// trivial case so `ctest -R unit` discovers >= 1 test.

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("unit_trivial_arithmetic", "[unit]") {
  REQUIRE(1 + 1 == 2);
}

int main(int argc, char *argv[]) {
  return Catch::Session().run(argc, argv);
}
