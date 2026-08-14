// KernelLaunch implementation (plan todo 23).

#include "kernelh/runtime/kernel_launch.hpp"

#include <utility>

namespace kernelh::runtime {

KernelLaunch::KernelLaunch(std::shared_ptr<Backend> backend)
    : backend_(std::move(backend)) {}

KernelLaunch &KernelLaunch::grid(unsigned x, unsigned y, unsigned z) {
  grid_ = {x, y, z};
  return *this;
}

KernelLaunch &KernelLaunch::block(unsigned x, unsigned y, unsigned z) {
  block_ = {x, y, z};
  return *this;
}

KernelLaunch &KernelLaunch::shared_mem_bytes(size_t bytes) {
  shared_ = bytes;
  return *this;
}

KernelLaunch &KernelLaunch::stream(StreamHandle stream) {
  stream_ = stream;
  return *this;
}

KernelLaunch &KernelLaunch::memref_arg(const void *dev_ptr,
                                       const std::vector<int64_t> &shape) {
  args_.add_memref(dev_ptr, shape);
  return *this;
}

Result<void> KernelLaunch::launch(KernelFunction function) const {
  if (!backend_) {
    return std::unexpected(Error{1, "KernelLaunch::launch: null backend"});
  }
  if (function.handle == nullptr) {
    return std::unexpected(
        Error{1, "KernelLaunch::launch: null kernel function handle"});
  }
  if (args_.empty()) {
    return std::unexpected(
        Error{1, "KernelLaunch::launch: no kernel arguments packed"});
  }
  void **params = args_.params();
  return backend_->launch_kernel(function.handle, grid_, block_, shared_,
                                 stream_, params);
}

} // namespace kernelh::runtime
