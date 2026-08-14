// Degraded-mode gate for [gpu]-tagged tests (plan todos 5 + 6).
//
// The AMD runtime smoke gate (todo 6) writes `degraded=true` or
// `degraded=false` to the flag file whose path is baked into test binaries
// via the KERNELH_DEGRADED_FLAG compile definition (CMake expands
// $ENV{KERNELH_DEGRADED_FLAG} at configure time - see test/CMakeLists.txt).
//
// Semantics (fail-safe):
//   - file contains "degraded=false" -> kernelh_gpu_degraded() == false
//   - file contains "degraded=true"  -> kernelh_gpu_degraded() == true
//   - file missing or unreadable     -> kernelh_gpu_degraded() == true
//
// Usage in a [gpu]-tagged Catch2 case:
//   if (kernelh::test::kernelh_gpu_degraded()) {
//     SKIP("GPU runtime unavailable (degraded mode)");
//   }

#ifndef KERNELH_TEST_UNIT_GPU_GATE_HPP
#define KERNELH_TEST_UNIT_GPU_GATE_HPP

#include <fstream>
#include <string>

#ifndef KERNELH_DEGRADED_FLAG
#error "gpu_gate.hpp requires the KERNELH_DEGRADED_FLAG compile definition"
#endif

namespace kernelh::test {

// Returns true when GPU tests must be skipped. Fail-safe: any uncertainty
// (missing file, unreadable file, unexpected content) reports degraded.
inline bool kernelh_gpu_degraded() {
  std::ifstream in(KERNELH_DEGRADED_FLAG);
  if (!in) {
    return true;
  }
  const std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
  return content.find("degraded=false") == std::string::npos;
}

} // namespace kernelh::test

#endif // KERNELH_TEST_UNIT_GPU_GATE_HPP
