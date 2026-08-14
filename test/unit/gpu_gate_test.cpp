// gpu_gate unit tests (plan todo 5).
//
// kernelh_gpu_degraded() reads the flag file baked in via the
// KERNELH_DEGRADED_FLAG compile definition:
//   - content "degraded=false" -> false (GPU available)
//   - content "degraded=true"  -> true  (skip [gpu] tests)
//   - file missing             -> true  (FAIL-SAFE)
//
// The flag file is shared state (todo 6 writes the real value), so every case
// saves the current state and restores it exactly on exit - no test residue.

#include "gpu_gate.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace {

namespace fs = std::filesystem;

const char *const kFlagPath = KERNELH_DEGRADED_FLAG;

// Saves the flag file's content (or absence) and restores it on destruction.
class FlagStateGuard {
public:
  FlagStateGuard() {
    std::ifstream in(kFlagPath, std::ios::binary);
    if (in) {
      std::ostringstream ss;
      ss << in.rdbuf();
      original_ = ss.str();
    }
  }

  ~FlagStateGuard() {
    if (original_.has_value()) {
      std::ofstream out(kFlagPath, std::ios::binary | std::ios::trunc);
      out << *original_;
    } else {
      std::error_code ec;
      fs::remove(kFlagPath, ec);
    }
  }

private:
  std::optional<std::string> original_;
};

void write_flag(const std::string &content) {
  fs::create_directories(fs::path(kFlagPath).parent_path());
  std::ofstream out(kFlagPath, std::ios::binary | std::ios::trunc);
  REQUIRE(out);
  out << content;
}

} // namespace

TEST_CASE("gpu_gate_degraded_false_flag", "[unit][gpu_gate]") {
  FlagStateGuard guard;
  write_flag("degraded=false\n");
  REQUIRE(kernelh::test::kernelh_gpu_degraded() == false);
}

TEST_CASE("gpu_gate_degraded_true_flag", "[unit][gpu_gate]") {
  FlagStateGuard guard;
  write_flag("degraded=true\n");
  REQUIRE(kernelh::test::kernelh_gpu_degraded() == true);
}

TEST_CASE("gpu_gate_missing_flag_fails_safe_degraded", "[unit][gpu_gate]") {
  FlagStateGuard guard;
  std::error_code ec;
  fs::remove(kFlagPath, ec);
  REQUIRE(kernelh::test::kernelh_gpu_degraded() == true);
}
