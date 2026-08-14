// Descriptor-packing byte-layout tests (plan todo 23, B3 / Oracle B3).
//
// Every buffer kernel argument is packed as the standard EXPANDED memref
// descriptor (allocated, aligned, offset, sizes[rank], strides[rank]) - the
// exact struct the compiler emits (!llvm.struct<(ptr, ptr, i64, array<N x
// i64>, array<N x i64])>). All expected layouts below are hand-computed.

#include "kernelh/runtime/descriptor_packing.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace {

const void *as_ptr(uint64_t address) {
  return reinterpret_cast<const void *>(address);
}

} // namespace

TEST_CASE("runtime_descriptor_1d_byte_layout", "[unit][runtime]") {
  // memref<?xf32> (rank 1, shape 1024): 5 slots.
  //   slot[0] allocated ptr = 0x12340000
  //   slot[1] aligned ptr   = 0x12340000
  //   slot[2] offset        = 0
  //   slot[3] size[0]       = 1024
  //   slot[4] stride[0]     = 1   (contiguous)
  kernelh::runtime::LaunchArgs args;
  args.add_memref(as_ptr(0x12340000ULL), {1024});

  REQUIRE(args.param_count() == 5);
  const auto &slots = args.raw_slots();
  REQUIRE(slots.size() == 5);
  CHECK(slots[0] == 0x12340000ULL);
  CHECK(slots[1] == 0x12340000ULL);
  CHECK(slots[2] == 0ULL);
  CHECK(slots[3] == 1024ULL);
  CHECK(slots[4] == 1ULL);
}

TEST_CASE("runtime_descriptor_2d_byte_layout", "[unit][runtime]") {
  // memref<?x?xf32> (rank 2, shape 16x8, row-major): 7 slots.
  //   slot[0] allocated ptr = 0x00AABBCC
  //   slot[1] aligned ptr   = 0x00AABBCC
  //   slot[2] offset        = 0
  //   slot[3] size[0]       = 16
  //   slot[4] size[1]       = 8
  //   slot[5] stride[0]     = 8   (contiguous row-major: inner extent)
  //   slot[6] stride[1]     = 1
  kernelh::runtime::LaunchArgs args;
  args.add_memref(as_ptr(0x00AABBCCULL), {16, 8});

  REQUIRE(args.param_count() == 7);
  const auto &slots = args.raw_slots();
  REQUIRE(slots.size() == 7);
  CHECK(slots[0] == 0x00AABBCCULL);
  CHECK(slots[1] == 0x00AABBCCULL);
  CHECK(slots[2] == 0ULL);
  CHECK(slots[3] == 16ULL);
  CHECK(slots[4] == 8ULL);
  CHECK(slots[5] == 8ULL);
  CHECK(slots[6] == 1ULL);
}

TEST_CASE("runtime_descriptor_scalars_after_descriptors", "[unit][runtime]") {
  // Descriptor then scalars by value: 5 + 1 + 1 = 7 slots.
  kernelh::runtime::LaunchArgs args;
  args.add_memref(as_ptr(0x1000ULL), {256});
  args.add_scalar(static_cast<int>(42));       // i32 by value
  args.add_scalar(static_cast<double>(3.5));   // f64 by value

  REQUIRE(args.param_count() == 7);
  const auto &slots = args.raw_slots();
  CHECK(slots[0] == 0x1000ULL);
  CHECK(slots[4] == 1ULL); // stride of the descriptor
  CHECK(slots[5] == 42ULL);
  // 3.5 in IEEE-754 double = 0x400C000000000000.
  CHECK(slots[6] == 0x400C000000000000ULL);
}

TEST_CASE("runtime_descriptor_xchg_pair_10_params", "[unit][runtime]") {
  // The todo-7 xchg kernel ABI: memref<?xf32> pair -> 2 * 5 = 10 params.
  kernelh::runtime::LaunchArgs args;
  args.add_memref(as_ptr(0x1000ULL), {4096});
  args.add_memref(as_ptr(0x2000ULL), {4096});

  REQUIRE(args.param_count() == 10);
  const auto &slots = args.raw_slots();
  REQUIRE(slots.size() == 10);
  CHECK(slots[0] == 0x1000ULL);
  CHECK(slots[1] == 0x1000ULL);
  CHECK(slots[2] == 0ULL);
  CHECK(slots[3] == 4096ULL);
  CHECK(slots[4] == 1ULL);
  CHECK(slots[5] == 0x2000ULL);
  CHECK(slots[6] == 0x2000ULL);
  CHECK(slots[7] == 0ULL);
  CHECK(slots[8] == 4096ULL);
  CHECK(slots[9] == 1ULL);
}

TEST_CASE("runtime_descriptor_params_array_points_to_slots", "[unit][runtime]") {
  // The kernelParams array is one void* per argument pointing AT its slot;
  // the runtime copies the argument's declared size from each slot.
  kernelh::runtime::LaunchArgs args;
  args.add_memref(as_ptr(0x1000ULL), {64});
  args.add_scalar(static_cast<int>(7));

  void **params = args.params();
  REQUIRE(params != nullptr);
  REQUIRE(args.param_count() == 6);
  const auto &slots = args.raw_slots();
  for (size_t i = 0; i < slots.size(); ++i) {
    CHECK(params[i] == reinterpret_cast<const void *>(&slots[i]));
  }
}
