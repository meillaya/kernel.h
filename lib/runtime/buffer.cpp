// Buffer implementation (plan todo 23). Plain hipMalloc/hipFree + H2D/D2H
// copies ONLY (plan D11); RAII frees on destruction.

#include "kernelh/runtime/buffer.hpp"

#include <utility>

namespace kernelh::runtime {

Result<Buffer> Buffer::allocate(std::shared_ptr<Backend> backend,
                                size_t bytes) {
  if (!backend) {
    return std::unexpected(Error{1, "Buffer::allocate: null backend"});
  }
  if (bytes == 0) {
    return std::unexpected(Error{1, "Buffer::allocate: zero-size buffer"});
  }
  auto ptr = backend->malloc(bytes);
  if (!ptr) {
    return std::unexpected(std::move(ptr.error()));
  }
  return Buffer(std::move(backend), *ptr, bytes);
}

Result<void> Buffer::copy_from_host(const void *src, size_t bytes) {
  if (bytes > bytes_) {
    return std::unexpected(Error{
        1, "Buffer::copy_from_host: copy size exceeds buffer size"});
  }
  return backend_->memcpy_h2d(ptr_, src, bytes);
}

Result<void> Buffer::copy_to_host(void *dst, size_t bytes) const {
  if (bytes > bytes_) {
    return std::unexpected(
        Error{1, "Buffer::copy_to_host: copy size exceeds buffer size"});
  }
  return backend_->memcpy_d2h(dst, ptr_, bytes);
}

Buffer::~Buffer() {
  if (ptr_ != nullptr && backend_) {
    backend_->free(ptr_); // best-effort; no error path in a destructor
  }
  ptr_ = nullptr;
  bytes_ = 0;
}

Buffer::Buffer(Buffer &&other) noexcept
    : backend_(std::move(other.backend_)), ptr_(other.ptr_),
      bytes_(other.bytes_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    if (ptr_ != nullptr && backend_) {
      backend_->free(ptr_);
    }
    backend_ = std::move(other.backend_);
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

} // namespace kernelh::runtime
