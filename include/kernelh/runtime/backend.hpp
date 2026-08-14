// GPU backend abstraction (plan todo 23).
//
// One Backend interface implemented twice:
//   - HIP locally (gfx1101, degraded=false confirmed): lib/runtime/backend_hip.cpp
//   - CUDA Driver API bound at RUNTIME via dlopen("libcuda.so.1")
//     (cross-ready, plan G10): lib/runtime/backend_cuda.cpp
//
// libcuda is NEVER linked at build time. If the driver is absent, the CUDA
// backend factory returns a clean Error with code kCudaDriverUnavailable and
// no call ever crashes.

#ifndef KERNELH_RUNTIME_BACKEND_HPP
#define KERNELH_RUNTIME_BACKEND_HPP

#include "kernelh/runtime/runtime_error.hpp"

#include <memory>
#include <string>

namespace kernelh::runtime {

enum class BackendKind { kHip, kCuda };

inline const char *to_string(BackendKind kind) {
  return kind == BackendKind::kHip ? "hip" : "cuda";
}

// Live-device properties queried at DeviceContext creation. These feed the
// device-limits table (include/kernelh/compiler/device_limits.hpp): the
// runtime CU count overrides the table through the strong
// kernelh::query_cu_count_runtime() hook (plan todo 10 contract).
struct DeviceInfo {
  std::string name; // vendor device name, e.g. "Radeon RX 7700 XT"
  long lds_bytes_per_workgroup = 0; // max shared/LDS bytes per workgroup
  int wavefront_size = 0;           // threads per wavefront (AMD) / warp (NV)
  int max_threads_per_workgroup = 0;
  long cu_count = 0; // CU (AMD) / SM (NV) count on the live device
  long vram_bytes = 0;
};

struct LaunchDims {
  unsigned x = 1, y = 1, z = 1;
};

// Opaque handles shared across backends.
using EventHandle = void *;
using StreamHandle = void *;

// A loaded kernel entry (hipFunction_t / CUfunction).
struct KernelFunction {
  void *handle = nullptr;
};

// Abstract GPU backend. Every method returns Result so failures are always
// surfaced as typed errors, never exceptions (plan F2).
class Backend {
public:
  virtual ~Backend() = default;

  virtual BackendKind kind() const = 0;

  // Device selection + live-property query (hipSetDevice / cuInit+cuDeviceGet).
  virtual Result<void> init_device(int device_ordinal) = 0;
  virtual Result<DeviceInfo> device_info() = 0;
  virtual Result<void> device_synchronize() = 0;

  // Memory (plan D11: plain malloc/free + H2D/D2H memcpy ONLY - never
  // stream-ordered allocation like hipMallocAsync).
  virtual Result<void *> malloc(size_t bytes) = 0;
  virtual Result<void> free(void *ptr) = 0;
  virtual Result<void> memcpy_h2d(void *dst, const void *src, size_t bytes) = 0;
  virtual Result<void> memcpy_d2h(void *dst, const void *src, size_t bytes) = 0;

  // Module lifecycle (hipModuleLoadData / hipModuleGetFunction / unload).
  virtual Result<void *> module_load_data(const void *image) = 0;
  virtual Result<void *> module_get_function(void *module, const char *name) = 0;
  virtual Result<void> module_unload(void *module) = 0;

  // Kernel launch. `params` is the kernelParams array: one void* per kernel
  // argument, each pointing at the packed argument bytes (expanded memref
  // descriptors + scalars; see descriptor_packing.hpp). shared_bytes is 0 by
  // construction of KernelLaunch (plan G9: static LDS only).
  virtual Result<void> launch_kernel(void *function, const LaunchDims &grid,
                                     const LaunchDims &block,
                                     size_t shared_bytes,
                                     StreamHandle stream,
                                     void *const *params) = 0;

  // Event timing (hipEventCreate/Record/ElapsedTime/Destroy).
  virtual Result<EventHandle> event_create() = 0;
  virtual Result<void> event_record(EventHandle event, StreamHandle stream) = 0;
  virtual Result<void> event_synchronize(EventHandle event) = 0;
  virtual Result<float> event_elapsed_time(EventHandle start,
                                           EventHandle end) = 0;
  virtual Result<void> event_destroy(EventHandle event) = 0;
};

// Backend factories. Construction does not touch a device (device selection
// happens in DeviceContext::create).
std::shared_ptr<Backend> make_hip_backend();

// dlopen("libcuda.so.1") + dlsym every needed symbol; on failure returns a
// clean Error with code kCudaDriverUnavailable (plan G10: NVIDIA targets
// skip, never crash). libcuda is NEVER linked at build time.
Result<std::shared_ptr<Backend>> make_cuda_backend();

// The default local backend: HIP (this machine's gfx1101, plan todo 6 gate).
std::shared_ptr<Backend> default_backend();

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_BACKEND_HPP
