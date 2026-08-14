// LaunchArgs implementation (plan B3 / Oracle B3 descriptor packing).
//
// Byte layout for one rank-N memref descriptor, hand-verified in
// test/unit/runtime_descriptor_test.cpp:
//   slot[0]            allocated pointer
//   slot[1]            aligned pointer
//   slot[2]            offset (i64, always 0 for contiguous buffers)
//   slot[3..3+N-1]     sizes[0..N-1] (one i64 per dim)
//   slot[3+N..3+2N-1]  strides[0..N-1] (contiguous row-major)
// => 2*N+3 slots per buffer (5 for rank 1, 7 for rank 2).

#include "kernelh/runtime/descriptor_packing.hpp"

#include <algorithm>
#include <cstring>

namespace kernelh::runtime {

void LaunchArgs::add_memref(const void *dev_ptr,
                            const std::vector<int64_t> &shape) {
  const uintptr_t ptr = reinterpret_cast<uintptr_t>(dev_ptr);
  slots_.push_back(static_cast<uint64_t>(ptr)); // allocated pointer
  slots_.push_back(static_cast<uint64_t>(ptr)); // aligned pointer
  slots_.push_back(0);                          // byte offset (always 0)

  for (int64_t dim : shape) {
    slots_.push_back(static_cast<uint64_t>(dim)); // sizes
  }
  // Row-major contiguous strides, in the SAME order as sizes:
  // stride[i] = product(shape[i+1..]).
  std::vector<uint64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = static_cast<uint64_t>(stride);
    stride *= shape[static_cast<size_t>(i)];
  }
  slots_.insert(slots_.end(), strides.begin(), strides.end());
  params_stale_ = true;
}

void LaunchArgs::add_scalar_bytes(const void *value, size_t byte_size) {
  uint64_t word = 0;
  if (value != nullptr && byte_size > 0) {
    std::memcpy(&word, value, std::min<size_t>(byte_size, sizeof(word)));
  }
  slots_.push_back(word);
  params_stale_ = true;
}

void **LaunchArgs::params() {
  if (params_stale_) {
    params_.clear();
    params_.reserve(slots_.size());
    for (uint64_t &slot : slots_) {
      params_.push_back(reinterpret_cast<void *>(&slot));
    }
    params_stale_ = false;
  }
  return params_.data();
}

} // namespace kernelh::runtime
