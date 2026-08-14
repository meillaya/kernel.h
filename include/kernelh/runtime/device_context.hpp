// DeviceContext (plan todo 23).
//
// Owns a device ordinal (default 0): selects the device and queries its live
// properties. The queried CU count also feeds the device-limits table through
// the strong kernelh::query_cu_count_runtime() override (see
// include/kernelh/compiler/device_limits.hpp), defined by this library's
// lib/runtime/query_cu_count.cpp.

#ifndef KERNELH_RUNTIME_DEVICE_CONTEXT_HPP
#define KERNELH_RUNTIME_DEVICE_CONTEXT_HPP

#include "kernelh/runtime/backend.hpp"

#include <memory>

namespace kernelh::runtime {

class DeviceContext {
public:
  // Selects `device_ordinal` (default 0) on `backend` and queries its live
  // properties. Returns a typed Error when no device is available.
  static Result<DeviceContext> create(std::shared_ptr<Backend> backend,
                                      int device_ordinal = 0);

  const DeviceInfo &info() const { return info_; }
  const std::string &name() const { return info_.name; }
  long lds_bytes_per_workgroup() const { return info_.lds_bytes_per_workgroup; }
  int wavefront_size() const { return info_.wavefront_size; }
  int max_threads_per_workgroup() const {
    return info_.max_threads_per_workgroup;
  }
  long cu_count() const { return info_.cu_count; }
  long vram_bytes() const { return info_.vram_bytes; }
  BackendKind backend_kind() const { return backend_->kind(); }
  const std::shared_ptr<Backend> &backend() const { return backend_; }

private:
  DeviceContext(std::shared_ptr<Backend> backend, DeviceInfo info)
      : backend_(std::move(backend)), info_(std::move(info)) {}

  std::shared_ptr<Backend> backend_;
  DeviceInfo info_;
};

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_DEVICE_CONTEXT_HPP
