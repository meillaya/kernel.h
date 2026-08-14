// Backend factory functions (plan todo 23).

#include "kernelh/runtime/backend.hpp"

#include <utility>

namespace kernelh::runtime {

// Concrete backend factories live in the backend_*.cpp TUs (kept out of the
// public header so no HIP/CUDA types leak into the API).
namespace detail {
std::shared_ptr<Backend> make_hip_backend_impl();
Result<std::shared_ptr<Backend>> make_cuda_backend_impl();
} // namespace detail

std::shared_ptr<Backend> make_hip_backend() {
  return detail::make_hip_backend_impl();
}

Result<std::shared_ptr<Backend>> make_cuda_backend() {
  return detail::make_cuda_backend_impl();
}

std::shared_ptr<Backend> default_backend() { return make_hip_backend(); }

} // namespace kernelh::runtime
