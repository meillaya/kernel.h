// Runtime error type + Result alias (plan todo 23, F2).
//
// The whole kernelh runtime API returns std::expected - NO exceptions across
// the public API (plan F2). Every failing operation produces a typed Error
// with a non-zero vendor code and a non-empty human-readable message.

#ifndef KERNELH_RUNTIME_RUNTIME_ERROR_HPP
#define KERNELH_RUNTIME_RUNTIME_ERROR_HPP

#include <expected>
#include <string>

namespace kernelh::runtime {

// Exit-code convention for "skip" (plan G10 / Metis G24): the CUDA backend is
// bound at RUNTIME via dlopen("libcuda.so.1"); when the driver is absent
// every call returns a clean Error with this code. Callers that must skip
// NVIDIA-only work (e.g. the bench CLI on this machine) treat it like the
// automake/ctest SKIP convention (77).
inline constexpr int kCudaDriverUnavailable = 77;

// A runtime failure: a non-zero vendor error code (hipError_t / CUresult / a
// small internal code) plus a description that is never empty.
struct Error {
  int code = 0;        // vendor/API error code; 0 == success
  std::string message; // human-readable description (non-empty on failure)

  bool is_success() const { return code == 0; }
};

template <typename T> using Result = std::expected<T, Error>;

// HIP-specific error-string helper (the local backend links libamdhip64).
// Returns the human-readable string for a hipError_t code; safe to call
// without any device - this is the core of the hipError_t -> Error mapping.
std::string hip_error_string(int hip_error_code);

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_RUNTIME_ERROR_HPP
