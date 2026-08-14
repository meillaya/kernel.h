// Device buffer (plan todo 23).
//
// Plain hipMalloc/hipFree + hipMemcpy H2D/D2H ONLY (plan D11: hipMallocAsync
// and hipMallocFromPoolAsync are BANNED - in-tree ROCm driver regression,
// ROCm/ROCm#6178). RAII: the device pointer is freed on destruction; the
// class is move-only.

#ifndef KERNELH_RUNTIME_BUFFER_HPP
#define KERNELH_RUNTIME_BUFFER_HPP

#include "kernelh/runtime/backend.hpp"

#include <cstddef>
#include <memory>

namespace kernelh::runtime {

class Buffer {
public:
  static Result<Buffer> allocate(std::shared_ptr<Backend> backend,
                                 size_t bytes);

  // H2D copy of `bytes` (<= buffer size) into the device buffer.
  Result<void> copy_from_host(const void *src, size_t bytes);
  // D2H copy of `bytes` (<= buffer size) from the device buffer.
  Result<void> copy_to_host(void *dst, size_t bytes) const;

  void *device_ptr() { return ptr_; }
  const void *device_ptr() const { return ptr_; }
  size_t size() const { return bytes_; }
  bool valid() const { return ptr_ != nullptr; }

  ~Buffer();
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

private:
  Buffer(std::shared_ptr<Backend> backend, void *ptr, size_t bytes)
      : backend_(std::move(backend)), ptr_(ptr), bytes_(bytes) {}

  std::shared_ptr<Backend> backend_;
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
};

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_BUFFER_HPP
