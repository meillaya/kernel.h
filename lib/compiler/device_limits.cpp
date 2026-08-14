// Device limits table definitions (plan todo 10) - the single source of
// truth for device constants (see include/kernelh/compiler/device_limits.hpp
// for the API conventions and the weak-hook contract).
//
// PLAN F2 PROVENANCE: every constant below cites its source. Do not add
// device constants anywhere else in the tree (F1/F2 grep-audited).

#include "kernelh/compiler/device_limits.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace kernelh {

// ---------------------------------------------------------------------------
// AMD Navi 32 (gfx1101, RDNA3) - local execution device (plan todos 6/7).
//
//  lds_bytes_per_workgroup = 65536
//      RDNA3 gives each CU 64 KiB of LDS, and a single workgroup may use the
//      full 64 KiB (RDNA3 ISA reference; matches HIP
//      hipDeviceAttributeMaxSharedMemoryPerBlock = 65536 on gfx1101).
//  wavefront_size = 32
//      RDNA3 native wavefront is wave32 (RDNA3 ISA reference).
//  max_threads_per_workgroup = 1024
//      Maximum workgroup size on RDNA3 (RDNA3 ISA reference; matches HIP
//      hipDeviceAttributeMaxThreadsPerBlock).
//  max_grid_dim = {2147483647, 65535, 65535}
//      HIP grid-dimension limits (hipDeviceAttributeMaxGridDimX/Y/Z).
//  vram_bytes = 17163091968
//      Exact byte count reported for THIS machine's gfx1101 card by the
//      ROCm runtime pool query (.omo/drafts/atomgpu.md HARDWARE findings;
//      ~16 GiB usable of a 16 GiB board).
//  waves_per_cu = 32
//      RDNA3: a CU runs at most 1024 threads concurrently; 1024 max
//      threads/CU / wave32 = 32 waves per CU (RDNA3 ISA reference).
//  reg_file_per_cu = 6144
//      RDNA3: 4 SIMDs per CU x 1536 32-bit VGPRs per SIMD = 6144 32-bit
//      VGPRs per CU (RDNA3 ISA reference).
//  cu_count = 0
//      "Runtime query required": the exact SKU (Radeon RX 7700 XT vs
//      7800 XT - different CU counts) is not pinned by lspci on this box
//      (Oracle B5), so the count must come from the runtime via the
//      query_cu_count_runtime() weak hook (todo 23).
constexpr DeviceLimits gfx1101_limits = {
    /*lds_bytes_per_workgroup=*/65536,
    /*wavefront_size=*/32,
    /*max_threads_per_workgroup=*/1024,
    /*max_grid_dim=*/{2147483647, 65535, 65535},
    /*vram_bytes=*/17163091968L,
    /*waves_per_cu=*/32,
    /*reg_file_per_cu=*/6144,
    /*cu_count=*/0,
    /*name=*/"AMD Navi 32 (gfx1101)"};

// ---------------------------------------------------------------------------
// NVIDIA A100-class (sm_80) - cross-compile-only target (plan todo 8);
// there is no NVIDIA card in this machine, so analytics are static.
//
//  lds_bytes_per_workgroup = 49152
//      A100 default shared memory per block is 48 KiB = 49152 bytes
//      (CUDA C++ Programming Guide, compute capability 8.0 feature table;
//      larger carve-outs require an explicit opt-in and are out of scope).
//  wavefront_size = 32
//      NVIDIA warp size is 32 (CUDA C++ Programming Guide).
//  max_threads_per_workgroup = 1024
//      Max threads per block on sm_80 (CUDA C++ Programming Guide).
//  max_grid_dim = {2147483647, 65535, 65535}
//      CUDA grid-dimension limits for compute capability >= 3.0.
//  vram_bytes = 0
//      No local device to query; cross-target only.
//  waves_per_cu = 64
//      A100: 64 warps resident per SM concurrently (CUDA C++ Programming
//      Guide, compute capability 8.0).
//  reg_file_per_cu = 65536
//      A100: 65536 32-bit registers per SM (CUDA C++ Programming Guide,
//      compute capability 8.0).
//  cu_count = 0
//      Cross-only target; where an A100-class card exists, the count comes
//      from the runtime via the query_cu_count_runtime() override.
constexpr DeviceLimits sm_80_limits = {
    /*lds_bytes_per_workgroup=*/49152,
    /*wavefront_size=*/32,
    /*max_threads_per_workgroup=*/1024,
    /*max_grid_dim=*/{2147483647, 65535, 65535},
    /*vram_bytes=*/0L,
    /*waves_per_cu=*/64,
    /*reg_file_per_cu=*/65536,
    /*cu_count=*/0,
    /*name=*/"NVIDIA A100-class (sm_80, cross)"};

// ---------------------------------------------------------------------------

namespace {

// Target token -> table entry. The token is the exact string accepted by
// device_limits_for() ("gfx1101" / "sm_80"); it is intentionally NOT a
// DeviceLimits field so the public struct stays exactly as specified.
struct TargetEntry {
  const char *target;
  const DeviceLimits *limits;
};

constexpr TargetEntry kTargetTable[] = {
    {"gfx1101", &gfx1101_limits},
    {"sm_80", &sm_80_limits},
};

std::string known_targets_list() {
  std::string list;
  for (const TargetEntry &entry : kTargetTable) {
    if (!list.empty()) {
      list += ", ";
    }
    list += entry.target;
  }
  return list;
}

} // namespace

const DeviceLimits &device_limits_for(const char *target_name) {
  if (target_name != nullptr) {
    for (const TargetEntry &entry : kTargetTable) {
      if (std::strcmp(target_name, entry.target) == 0) {
        return *entry.limits;
      }
    }
  }
  throw std::invalid_argument(
      "unknown device target \"" +
      std::string(target_name != nullptr ? target_name : "(null)") +
      "\"; known targets: " + known_targets_list());
}

// Weak default: no GPU runtime is linked into lib/compiler, so there is no
// device to query. Plan todo 23's runtime library provides the strong
// override (contract documented in device_limits.hpp).
__attribute__((weak)) long query_cu_count_runtime() { return 0; }

long query_cu_count(const DeviceLimits &limits) {
  const long runtime_cu_count = query_cu_count_runtime();
  return runtime_cu_count > 0 ? runtime_cu_count : limits.cu_count;
}

} // namespace kernelh
