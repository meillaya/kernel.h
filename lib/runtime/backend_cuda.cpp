// CUDA Driver API backend, bound at RUNTIME via dlopen("libcuda.so.1")
// (plan todo 23, G10 - cross-ready): the SAME Backend interface over the
// CUDA Driver API. libcuda is NEVER linked at build time. On machines without
// an NVIDIA driver (this one) dlopen fails and the factory returns a clean
// Error with code kCudaDriverUnavailable (exit-77 skip semantics); no call
// ever crashes.
//
// Driver-API types/constants are reproduced locally (from cuda.h) because no
// CUDA header is available at build time.

#include "kernelh/runtime/backend.hpp"

#include <dlfcn.h>

#include <string>
#include <utility>

namespace kernelh::runtime {
namespace detail {
namespace {

// Local stand-ins for driver-API types (stable ABI integers).
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUresult = int;

// Attribute ids from cuda.h (CUdevice_attribute).
// (CUDA 12.9, cuda.h: WARP_SIZE = 10 line 721, MULTIPROCESSOR_COUNT = 16 line 728)
enum : int {
  kCuSuccess = 0,
  kCuDeviceAttributeMaxThreadsPerBlock = 1,
  kCuDeviceAttributeMaxSharedMemoryPerBlock = 8,
  kCuDeviceAttributeWarpSize = 10,
  kCuDeviceAttributeMultiprocessorCount = 16,
};

// Function-pointer typedefs for every driver symbol we resolve via dlsym.
#define KERNELH_CUDA_FN_TYPES(FN)                                             \
  FN(cuInit, CUresult, (unsigned int))                                        \
  FN(cuDeviceGet, CUresult, (CUdevice *, int))                                \
  FN(cuDeviceGetName, CUresult, (char *, int, CUdevice))                      \
  FN(cuDeviceGetAttribute, CUresult, (int *, int, CUdevice))                  \
  FN(cuDeviceTotalMem, CUresult, (size_t *, CUdevice))                        \
  FN(cuDevicePrimaryCtxRetain, CUresult, (void **, CUdevice))                 \
  FN(cuCtxSetCurrent, CUresult, (void *))                                     \
  FN(cuMemAlloc_v2, CUresult, (CUdeviceptr *, size_t))                          \
  FN(cuMemFree_v2, CUresult, (CUdeviceptr))                                     \
  FN(cuMemcpyHtoD_v2, CUresult, (CUdeviceptr, const void *, size_t))            \
  FN(cuMemcpyDtoH_v2, CUresult, (void *, CUdeviceptr, size_t))                  \
  FN(cuModuleLoadData, CUresult, (void **, const void *))                     \
  FN(cuModuleGetFunction, CUresult, (void **, void *, const char *))          \
  FN(cuModuleUnload, CUresult, (void *))                                      \
  FN(cuLaunchKernel, CUresult,                                                \
     (void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,    \
      unsigned, void *, void **, void **))                                    \
  FN(cuStreamSynchronize, CUresult, (void *))                                 \
  FN(cuCtxSynchronize, CUresult, ())                                          \
  FN(cuEventCreate, CUresult, (void **, unsigned int))                        \
  FN(cuEventRecord, CUresult, (void *, void *))                               \
  FN(cuEventSynchronize, CUresult, (void *))                                  \
  FN(cuEventElapsedTime, CUresult, (float *, void *, void *))                 \
  FN(cuEventDestroy, CUresult, (void *))                                      \
  FN(cuGetErrorString, CUresult, (CUresult, const char **))                   \
  FN(cuGetErrorName, CUresult, (CUresult, const char **))

#define KERNELH_DECLARE_FN_PTR(name, ret, args) using name##Fn = ret (*) args;

KERNELH_CUDA_FN_TYPES(KERNELH_DECLARE_FN_PTR)

#undef KERNELH_DECLARE_FN_PTR

// The resolved driver API surface (one function pointer per symbol).
struct CudaApi {
  void *handle = nullptr;
#define KERNELH_CUDA_FN_FIELD(name, ret, args) name##Fn name = nullptr;
  KERNELH_CUDA_FN_TYPES(KERNELH_CUDA_FN_FIELD)
#undef KERNELH_CUDA_FN_FIELD

  std::string error_string(CUresult code) const {
    const char *s = nullptr;
    if (cuGetErrorString != nullptr && cuGetErrorString(code, &s) == kCuSuccess &&
        s != nullptr) {
      return s;
    }
    return "CUDA error " + std::to_string(code);
  }
};

// Maps a CUresult to the runtime Error type (plan F2).
Error map_cu_error(const CudaApi &api, CUresult code, const char *what) {
  return Error{static_cast<int>(code),
               std::string(what) + " failed: " + api.error_string(code) + " (" +
                   std::to_string(static_cast<int>(code)) + ")"};
}

} // namespace

class CudaBackend final : public Backend {
public:
  explicit CudaBackend(CudaApi api) : api_(std::move(api)) {}

  BackendKind kind() const override { return BackendKind::kCuda; }

  Result<void> init_device(int device_ordinal) override {
    if (CUresult r = api_.cuInit(0); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuInit"));
    }
    CUdevice dev = 0;
    if (CUresult r = api_.cuDeviceGet(&dev, device_ordinal); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuDeviceGet"));
    }
    void *ctx = nullptr;
    if (CUresult r = api_.cuDevicePrimaryCtxRetain(&ctx, dev);
        r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuDevicePrimaryCtxRetain"));
    }
    if (CUresult r = api_.cuCtxSetCurrent(ctx); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuCtxSetCurrent"));
    }
    return {};
  }

  Result<DeviceInfo> device_info() override {
    CUdevice dev = 0;
    if (CUresult r = api_.cuDeviceGet(&dev, 0); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuDeviceGet"));
    }
    DeviceInfo info;
    char name[256] = {0};
    if (CUresult r = api_.cuDeviceGetName(name, static_cast<int>(sizeof(name)),
                                          dev);
        r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuDeviceGetName"));
    }
    info.name = name;
    size_t total = 0;
    if (CUresult r = api_.cuDeviceTotalMem(&total, dev); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuDeviceTotalMem"));
    }
    info.vram_bytes = static_cast<long>(total);

    auto attr = [&](int attribute, const char *what, int *out) -> Result<void> {
      if (CUresult r = api_.cuDeviceGetAttribute(out, attribute, dev);
          r != kCuSuccess) {
        return std::unexpected(map_cu_error(api_, r, what));
      }
      return {};
    };

    int v = 0;
    if (auto r = attr(kCuDeviceAttributeMaxThreadsPerBlock,
                      "cuDeviceGetAttribute(MaxThreadsPerBlock)", &v);
        !r) {
      return std::unexpected(std::move(r.error()));
    }
    info.max_threads_per_workgroup = v;
    if (auto r = attr(kCuDeviceAttributeMaxSharedMemoryPerBlock,
                      "cuDeviceGetAttribute(MaxSharedMemoryPerBlock)", &v);
        !r) {
      return std::unexpected(std::move(r.error()));
    }
    info.lds_bytes_per_workgroup = v;
    if (auto r = attr(kCuDeviceAttributeWarpSize,
                      "cuDeviceGetAttribute(WarpSize)", &v);
        !r) {
      return std::unexpected(std::move(r.error()));
    }
    info.wavefront_size = v;
    if (auto r = attr(kCuDeviceAttributeMultiprocessorCount,
                      "cuDeviceGetAttribute(MultiprocessorCount)", &v);
        !r) {
      return std::unexpected(std::move(r.error()));
    }
    info.cu_count = v;
    return info;
  }

  Result<void> device_synchronize() override {
    if (CUresult r = api_.cuCtxSynchronize(); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuCtxSynchronize"));
    }
    return {};
  }

  Result<void *> malloc(size_t bytes) override {
    CUdeviceptr ptr = 0;
    if (CUresult r = api_.cuMemAlloc_v2(&ptr, bytes); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuMemAlloc_v2"));
    }
    return reinterpret_cast<void *>(ptr);
  }

  Result<void> free(void *ptr) override {
    if (CUresult r = api_.cuMemFree_v2(reinterpret_cast<CUdeviceptr>(ptr));
        r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuMemFree_v2"));
    }
    return {};
  }

  Result<void> memcpy_h2d(void *dst, const void *src, size_t bytes) override {
    if (CUresult r = api_.cuMemcpyHtoD_v2(reinterpret_cast<CUdeviceptr>(dst),
                                          src, bytes);
        r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuMemcpyHtoD_v2"));
    }
    return {};
  }

  Result<void> memcpy_d2h(void *dst, const void *src, size_t bytes) override {
    if (CUresult r = api_.cuMemcpyDtoH_v2(dst, reinterpret_cast<CUdeviceptr>(src),
                                          bytes);
        r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuMemcpyDtoH_v2"));
    }
    return {};
  }

  Result<void *> module_load_data(const void *image) override {
    void *module = nullptr;
    if (CUresult r = api_.cuModuleLoadData(&module, image); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuModuleLoadData"));
    }
    return module;
  }

  Result<void *> module_get_function(void *module, const char *name) override {
    void *fn = nullptr;
    if (CUresult r = api_.cuModuleGetFunction(&fn, module, name);
        r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuModuleGetFunction"));
    }
    return fn;
  }

  Result<void> module_unload(void *module) override {
    if (CUresult r = api_.cuModuleUnload(module); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuModuleUnload"));
    }
    return {};
  }

  Result<void> launch_kernel(void *function, const LaunchDims &grid,
                             const LaunchDims &block, size_t shared_bytes,
                             StreamHandle stream, void *const *params) override {
    if (CUresult r = api_.cuLaunchKernel(
            function, grid.x, grid.y, grid.z, block.x, block.y, block.z,
            static_cast<unsigned>(shared_bytes), stream,
            const_cast<void **>(params), nullptr);
        r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuLaunchKernel"));
    }
    return {};
  }

  Result<EventHandle> event_create() override {
    void *event = nullptr;
    if (CUresult r = api_.cuEventCreate(&event, 0); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuEventCreate"));
    }
    return event;
  }

  Result<void> event_record(EventHandle event, StreamHandle stream) override {
    if (CUresult r = api_.cuEventRecord(event, stream); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuEventRecord"));
    }
    return {};
  }

  Result<void> event_synchronize(EventHandle event) override {
    if (CUresult r = api_.cuEventSynchronize(event); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuEventSynchronize"));
    }
    return {};
  }

  Result<float> event_elapsed_time(EventHandle start, EventHandle end) override {
    float ms = 0.0f;
    if (CUresult r = api_.cuEventElapsedTime(&ms, start, end); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuEventElapsedTime"));
    }
    return ms;
  }

  Result<void> event_destroy(EventHandle event) override {
    if (CUresult r = api_.cuEventDestroy(event); r != kCuSuccess) {
      return std::unexpected(map_cu_error(api_, r, "cuEventDestroy"));
    }
    return {};
  }

private:
  CudaApi api_;
};

Result<std::shared_ptr<Backend>> make_cuda_backend_impl() {
  void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    const char *why = dlerror();
    return std::unexpected(Error{
        kCudaDriverUnavailable,
        std::string("CUDA driver unavailable: dlopen(libcuda.so.1) failed: ") +
            (why != nullptr ? why : "unknown error") +
            " (NVIDIA targets skip; plan G10 exit-77 semantics)"});
  }

  CudaApi api;
  api.handle = handle;
#define KERNELH_CUDA_RESOLVE(name, ret, args)                                  \
  api.name = reinterpret_cast<name##Fn>(dlsym(handle, #name));                \
  if (api.name == nullptr) {                                                  \
    const char *why = dlerror();                                              \
    dlclose(handle);                                                          \
    return std::unexpected(Error{                                             \
        kCudaDriverUnavailable,                                               \
        std::string("CUDA driver incomplete: dlsym(" #name ") failed: ") +    \
            (why != nullptr ? why : "unknown error") +                        \
            " (NVIDIA targets skip; plan G10 exit-77 semantics)"});           \
  }
  KERNELH_CUDA_FN_TYPES(KERNELH_CUDA_RESOLVE)
#undef KERNELH_CUDA_RESOLVE

  return std::make_shared<CudaBackend>(std::move(api));
}

} // namespace detail
} // namespace kernelh::runtime
