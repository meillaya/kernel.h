// Compiled kernel module (plan todo 23).
//
// Loads the device-only module blobs produced by the compiler (todo 22) from
// memory via hipModuleLoadData / cuModuleLoadData, resolves kernel entry
// points, and unloads the module. RAII: unloads on destruction; move-only.

#ifndef KERNELH_RUNTIME_GPU_MODULE_HPP
#define KERNELH_RUNTIME_GPU_MODULE_HPP

#include "kernelh/runtime/backend.hpp"

#include <cstddef>
#include <memory>

namespace kernelh::runtime {

class GpuModule {
public:
  // Loads `image_size` bytes of a compiled module (HSACO for HIP / cubin or
  // PTX for CUDA). A corrupted/truncated image surfaces as a typed Error.
  static Result<GpuModule> load(std::shared_ptr<Backend> backend,
                                const void *image, size_t image_size);

  // Resolves a kernel entry point (e.g. the todo-7 `xchg_kernel` symbol).
  Result<KernelFunction> get_function(const char *name) const;

  // Explicitly unloads the module; also runs in the destructor.
  Result<void> unload();

  ~GpuModule();
  GpuModule(const GpuModule &) = delete;
  GpuModule &operator=(const GpuModule &) = delete;
  GpuModule(GpuModule &&other) noexcept;
  GpuModule &operator=(GpuModule &&other) noexcept;

private:
  GpuModule(std::shared_ptr<Backend> backend, void *module)
      : backend_(std::move(backend)), module_(module) {}

  std::shared_ptr<Backend> backend_;
  void *module_ = nullptr; // hipModule_t / CUmodule
};

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_GPU_MODULE_HPP
