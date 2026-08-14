// GpuModule implementation (plan todo 23).

#include "kernelh/runtime/gpu_module.hpp"

#include <utility>

namespace kernelh::runtime {

Result<GpuModule> GpuModule::load(std::shared_ptr<Backend> backend,
                                  const void *image, size_t image_size) {
  if (!backend) {
    return std::unexpected(Error{1, "GpuModule::load: null backend"});
  }
  if (image == nullptr || image_size == 0) {
    return std::unexpected(
        Error{1, "GpuModule::load: empty module image"});
  }
  auto module = backend->module_load_data(image);
  if (!module) {
    return std::unexpected(std::move(module.error()));
  }
  return GpuModule(std::move(backend), *module);
}

Result<KernelFunction> GpuModule::get_function(const char *name) const {
  if (module_ == nullptr) {
    return std::unexpected(
        Error{1, "GpuModule::get_function: module already unloaded"});
  }
  auto fn = backend_->module_get_function(module_, name);
  if (!fn) {
    return std::unexpected(std::move(fn.error()));
  }
  return KernelFunction{*fn};
}

Result<void> GpuModule::unload() {
  if (module_ == nullptr) {
    return std::unexpected(
        Error{1, "GpuModule::unload: module already unloaded"});
  }
  auto r = backend_->module_unload(module_);
  if (r) {
    module_ = nullptr;
  }
  return r;
}

GpuModule::~GpuModule() {
  if (module_ != nullptr && backend_) {
    backend_->module_unload(module_); // best-effort
  }
  module_ = nullptr;
}

GpuModule::GpuModule(GpuModule &&other) noexcept
    : backend_(std::move(other.backend_)), module_(other.module_) {
  other.module_ = nullptr;
}

GpuModule &GpuModule::operator=(GpuModule &&other) noexcept {
  if (this != &other) {
    if (module_ != nullptr && backend_) {
      backend_->module_unload(module_);
    }
    backend_ = std::move(other.backend_);
    module_ = other.module_;
    other.module_ = nullptr;
  }
  return *this;
}

} // namespace kernelh::runtime
