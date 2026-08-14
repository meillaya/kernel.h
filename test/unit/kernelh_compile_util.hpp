// Shared helpers for the plan-todo-25 [gpu] e2e/canary tests.
//
// The e2e/canary tests shell out to the PRODUCTION `kernelh compile` driver
// (todo 24) via std::system - the same precedent as runtime_integration_test.cpp
// re-emitting the todo-7 HSACO via `mlir-opt ... | std::system`. The CLI is the
// proven end-to-end path (registry emit -> default pipeline -> emitter ->
// vendor backends), so these tests stay focused on CORRECTNESS:
// compile -> load -> launch -> verify.
//
// Requires the KERNELH_BIN compile definition (test/unit/CMakeLists.txt
// expands ${CMAKE_BINARY_DIR}/tools/kernelh/kernelh) and, for the compile
// itself, the ROCM_TOOLKIT_ROOT/CUDA_TOOLKIT_ROOT env vars that the devenv
// shell exports.

#ifndef KERNELH_TEST_UNIT_KERNELH_COMPILE_UTIL_HPP
#define KERNELH_TEST_UNIT_KERNELH_COMPILE_UTIL_HPP

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef KERNELH_BIN
#error "kernelh_compile_util.hpp requires the KERNELH_BIN compile definition"
#endif

namespace kernelh::test {

// Absolute path to the built production driver.
inline const char *kernelh_cli() { return KERNELH_BIN; }

// Runs `kernelh compile --kernel <kernel> --shape <shapeText>
// [--target <t> ...] --out <outDir>`. The out dir is wiped + recreated first
// (self-cleaning, avoids the todo-19 stale-dump class). Returns the CLI exit
// code (0 ok, 1 operational, 2 usage).
inline int compile_kernel(const std::string &kernel, const std::string &shapeText,
                          const std::vector<std::string> &targets,
                          const std::string &outDir) {
  std::string cmd = "rm -rf '" + outDir + "' && mkdir -p '" + outDir + "' && '";
  cmd += KERNELH_BIN;
  cmd += "' compile --kernel " + kernel + " --shape " + shapeText;
  for (const auto &target : targets) {
    cmd += " --target " + target;
  }
  cmd += " --out '" + outDir + "'";
  const int rc = std::system(cmd.c_str());
  if (rc == -1) {
    return -1; // the shell itself failed to start
  }
#ifdef _WIN32
  return rc;
#else
  if (WIFEXITED(rc)) {
    return WEXITSTATUS(rc);
  }
  return 1; // signal-terminated -> operational failure
#endif
}

inline bool file_exists(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return in.good();
}

inline std::vector<char> read_file_bin(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto size = in.tellg();
  std::vector<char> data(size > 0 ? static_cast<size_t>(size) : 0);
  in.seekg(0);
  if (size > 0) {
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
  }
  return data;
}

inline bool is_elf_magic(const std::vector<char> &bytes) {
  return bytes.size() >= 4 &&
         static_cast<unsigned char>(bytes[0]) == 0x7f && bytes[1] == 'E' &&
         bytes[2] == 'L' && bytes[3] == 'F';
}

// Binary-safe substring probe: does a compiled artifact (HSACO/PTX/cubin/ISA)
// contain the kernel symbol name? This is the degraded-mode artifact
// assertion (todo 6 degrade branch): it proves the produced blob actually
// carries the kernel, without loading or launching it.
inline bool artifact_contains_symbol(const std::string &path,
                                     const std::string &symbol) {
  const std::vector<char> data = read_file_bin(path);
  if (data.empty()) {
    return false;
  }
  const std::string haystack(data.begin(), data.end());
  return haystack.find(symbol) != std::string::npos;
}

} // namespace kernelh::test

#endif // KERNELH_TEST_UNIT_KERNELH_COMPILE_UTIL_HPP
