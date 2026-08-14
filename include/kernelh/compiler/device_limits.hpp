// Device limits table - the SINGLE SOURCE OF TRUTH for every device constant
// used by the passes, the launch-config verifier, and occupancy calculations
// (plan todo 10). All later consumers MUST read constants from this table via
// kernelh::device_limits_for() instead of hardcoding them; the F1/F2 final
// audits grep lib/passes, lib/compiler and lib/ir for stray constants.
//
// The table itself (constexpr entries gfx1101_limits / sm_80_limits) is
// defined in lib/compiler/device_limits.cpp together with the lookup and the
// runtime CU-count hook; this header declares the public API. Per-entry
// provenance for every constant lives with the definitions in that .cpp
// (plan F2: device constants require documented provenance).
//
// LOOKUP ERROR CONVENTION
// -----------------------
// device_limits_for(target_name) matches the target token EXACTLY
// ("gfx1101", "sm_80") and returns a reference to the matching table entry
// (entries have static storage duration, so the reference never dangles).
// An unknown, empty, or null target throws std::invalid_argument whose
// message names the offending target AND lists every known target token.
// No runtime query happens at static-initialization time: the table is pure
// constexpr data.
//
// RUNTIME CU-COUNT HOOK CONTRACT (weak symbol)
// --------------------------------------------
// query_cu_count_runtime() is declared __attribute__((weak)) and its default
// definition (in device_limits.cpp) returns 0, meaning "no GPU runtime
// linked / no device visible". A later runtime-aware translation unit -
// plan todo 23, the HIP runtime library - provides a STRONG definition with
// the same signature that returns the actual CU count (via
// hipDeviceGetAttribute(hipDeviceAttributeMultiprocessorCount, ...)).
// Because the override is resolved at LINK time, lib/compiler itself never
// links HIP/ROCm. Any binary that links both lib/compiler and the todo-23
// runtime automatically picks up the strong override; binaries without a
// runtime keep the weak default of 0.
//
// query_cu_count(limits) returns the runtime value when it is > 0, otherwise
// it falls back to limits.cu_count. Table entries carry cu_count = 0 to mean
// "runtime query required", so without an override query_cu_count() is 0.

#ifndef KERNELH_COMPILER_DEVICE_LIMITS_HPP
#define KERNELH_COMPILER_DEVICE_LIMITS_HPP

namespace kernelh {

// Static hardware limits for one device target. All values are compile-time
// constants from the vendor ISA/programming references (see the definitions
// in lib/compiler/device_limits.cpp for per-field provenance); fields that
// depend on the exact physical card (e.g. cu_count) are 0 and must be
// resolved at runtime instead.
struct DeviceLimits {
  long lds_bytes_per_workgroup;  // max shared/LDS bytes per workgroup
  int wavefront_size;            // threads per wavefront (AMD) / warp (NV)
  int max_threads_per_workgroup; // max workgroup/block size
  long max_grid_dim[3];          // max grid extent {x, y, z}
  long vram_bytes;               // device VRAM in bytes (0 = unknown/absent)
  int waves_per_cu;              // max concurrent waves per CU (AMD) / SM (NV)
  int reg_file_per_cu;           // 32-bit registers per CU (AMD) / SM (NV)
  long cu_count;                 // CU (AMD) / SM (NV) count; 0 = runtime query
  const char *name;              // human-readable device name
};

// AMD Navi 32 (gfx1101, RDNA3) - the local execution device.
extern const DeviceLimits gfx1101_limits;

// NVIDIA A100-class (sm_80) - cross-compile-only target (no local card).
extern const DeviceLimits sm_80_limits;

// Returns the table entry whose target token matches `target_name` exactly.
// Throws std::invalid_argument for unknown/empty/null targets; the message
// names the offending target and lists all known target tokens.
const DeviceLimits &device_limits_for(const char *target_name);

// Weak hook - default returns 0 ("no GPU runtime linked"). Plan todo 23
// supplies the strong override that queries the live device; see the
// contract at the top of this header. Do not call at static-init time.
long query_cu_count_runtime() __attribute__((weak));

// CU count to use for occupancy math: the runtime value when
// query_cu_count_runtime() > 0, otherwise limits.cu_count (0 = unknown).
long query_cu_count(const DeviceLimits &limits);

} // namespace kernelh

#endif // KERNELH_COMPILER_DEVICE_LIMITS_HPP
