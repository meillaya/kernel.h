// Strong override of the launch-verifier occupancy weak hook (plan todo 32
// contract; see include/kernelh/compiler/launch_verifier.hpp): calls
// hipOccupancyMaxActiveBlocksPerMultiprocessor on the loaded function handle.
//
// The weak default in lib/compiler/launch_verifier.cpp returns -1 ("occupancy
// API unavailable / no GPU runtime linked"). This strong definition wins at
// FINAL LINK time whenever this library is linked together with lib/compiler -
// so binaries linking both get the real per-multiprocessor occupancy, and
// binaries without a runtime keep the weak default. lib/compiler itself never
// links HIP.
//
// GOTCHAS (mirror lib/runtime/query_cu_count.cpp, both verified empirically):
//  1. Do NOT include launch_verifier.hpp here: its prior declaration carries
//     __attribute__((weak)), and C++ propagates the attribute from a prior
//     declaration to the definition - the "strong" override would compile
//     weak too and lose the link. The function is therefore declared locally,
//     WITHOUT the attribute.
//  2. Plain archive extraction keeps the WEAK definition (a weak def satisfies
//     the reference, so the archive member is never pulled in). Consumers must
//     force this TU into the final link (WHOLE_ARCHIVE on kernelh_runtime,
//     exactly like the todo-23 query_cu_count override). kernelh_unit does.
//
// Never throws and never does more than necessary: returns -1 when no usable
// function handle or device is present, matching the weak default exactly.

#include <hip/hip_runtime_api.h>

#include <cstddef>

namespace kernelh::compiler {

// Forward declaration WITHOUT __attribute__((weak)) - see gotcha 1 above.
int query_occupancy_max_blocks_runtime(void *function, int block_size,
                                       size_t shared_bytes);

int query_occupancy_max_blocks_runtime(void *function, int block_size,
                                       size_t shared_bytes) {
  if (function == nullptr || block_size <= 0) {
    return -1;
  }
  int num_blocks = 0;
  hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, reinterpret_cast<const void *>(function), block_size,
      shared_bytes);
  if (err != hipSuccess || num_blocks < 0) {
    return -1;
  }
  return num_blocks;
}

} // namespace kernelh::compiler
