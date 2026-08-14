// Kernel argument packing to the launch ABI (plan B3 / Oracle B3 - the
// load-bearing todo-7 D-ABI finding).
//
// Every kernel BUFFER argument is passed as the standard EXPANDED memref
// descriptor - the exact struct the compiler emits:
//
//   !llvm.struct<(ptr, ptr, i64, array<N x i64>, array<N x i64>)>
//     = (allocated_ptr, aligned_ptr, offset, sizes[N], strides[N])
//
// which the LLVM type converter expands into 2*N+3 scalar kernel parameters.
// A memref<?xf32> pair (2 buffers, rank 1) therefore yields (2+1+1+1)*2 = 10
// kernel params; a rank-2 pair yields (2+2+2)*2 = 14.
//
// Each packed argument occupies one 8-byte slot (every descriptor field is a
// pointer or i64; scalars are 8 bytes or less). The launch kernelParams array
// is a void* per argument pointing at that slot; the runtime copies the
// argument's declared size from each slot.

#ifndef KERNELH_RUNTIME_DESCRIPTOR_PACKING_HPP
#define KERNELH_RUNTIME_DESCRIPTOR_PACKING_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kernelh::runtime {

// Packs memref descriptors and scalars into the launch kernelParams layout.
class LaunchArgs {
public:
  // Buffer argument: expanded memref descriptor for a contiguous row-major
  // buffer of `shape`; strides are computed as the shape[i+1..] products.
  void add_memref(const void *dev_ptr, const std::vector<int64_t> &shape);

  // Scalar argument passed by value: the first `byte_size` bytes (<= 8) are
  // stored in the slot's low bytes (little-endian host layout).
  void add_scalar_bytes(const void *value, size_t byte_size);

  template <typename T> void add_scalar(T value) {
    add_scalar_bytes(&value, sizeof(T));
  }

  // Number of kernel parameters (= number of 8-byte slots).
  size_t param_count() const { return slots_.size(); }

  // The kernelParams array (a void* per argument, each pointing at its slot).
  // Stable until the next add_* call.
  void **params();

  // Raw 8-byte slots - the exact byte layout sent to the kernel; exposed for
  // the hand-computed descriptor-packing unit tests.
  const std::vector<uint64_t> &raw_slots() const { return slots_; }

  bool empty() const { return slots_.empty(); }

private:
  std::vector<uint64_t> slots_; // one 8-byte slot per kernel parameter
  std::vector<void *> params_;  // params_[i] = &slots_[i]
  bool params_stale_ = true;
};

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_DESCRIPTOR_PACKING_HPP
