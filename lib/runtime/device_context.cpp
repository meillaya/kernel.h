// DeviceContext implementation (plan todo 23).

#include "kernelh/runtime/device_context.hpp"

#include <utility>

namespace kernelh::runtime {

Result<DeviceContext> DeviceContext::create(std::shared_ptr<Backend> backend,
                                            int device_ordinal) {
  if (!backend) {
    return std::unexpected(
        Error{1, "DeviceContext::create: null backend"});
  }
  if (auto r = backend->init_device(device_ordinal); !r) {
    return std::unexpected(std::move(r.error()));
  }
  auto info = backend->device_info();
  if (!info) {
    return std::unexpected(std::move(info.error()));
  }
  return DeviceContext(std::move(backend), std::move(*info));
}

} // namespace kernelh::runtime
