// Kernel launch configuration (plan todo 23).
//
// hipModuleLaunchKernel (or cuLaunchKernel) with:
//   - grid/block dims
//   - sharedMemBytes pinned to 0 by default (plan G9: static LDS only - no
//     dynamic shared memory; static shared lives in the kernel descriptor)
//   - a stream parameter (nullptr = default stream)
//   - the PINNED kernel-argument packing (plan B3 / Oracle B3): each buffer
//     argument is an expanded memref descriptor, scalars by value after all
//     descriptors (see descriptor_packing.hpp for the byte layout).

#ifndef KERNELH_RUNTIME_KERNEL_LAUNCH_HPP
#define KERNELH_RUNTIME_KERNEL_LAUNCH_HPP

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/descriptor_packing.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace kernelh::runtime {

class KernelLaunch {
public:
  explicit KernelLaunch(std::shared_ptr<Backend> backend);

  KernelLaunch &grid(unsigned x, unsigned y = 1, unsigned z = 1);
  KernelLaunch &block(unsigned x, unsigned y = 1, unsigned z = 1);
  // Pinned to 0 unless the caller explicitly overrides (plan G9 default).
  KernelLaunch &shared_mem_bytes(size_t bytes);
  KernelLaunch &stream(StreamHandle stream);

  // Buffer argument packed as an expanded memref descriptor.
  KernelLaunch &memref_arg(const void *dev_ptr,
                           const std::vector<int64_t> &shape);

  // Scalar argument passed by value after all descriptors.
  template <typename T> KernelLaunch &scalar(T value) {
    args_.add_scalar(value);
    return *this;
  }

  Result<void> launch(KernelFunction function) const;

  // Test seams: the packed layout + launch config sent to the kernel.
  const LaunchArgs &args() const { return args_; }
  const LaunchDims &grid_dims() const { return grid_; }
  const LaunchDims &block_dims() const { return block_; }
  size_t shared_bytes() const { return shared_; }
  StreamHandle stream_handle() const { return stream_; }

private:
  std::shared_ptr<Backend> backend_;
  LaunchDims grid_{1, 1, 1};
  LaunchDims block_{1, 1, 1};
  size_t shared_ = 0;
  StreamHandle stream_ = nullptr;
  mutable LaunchArgs args_;
};

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_KERNEL_LAUNCH_HPP
