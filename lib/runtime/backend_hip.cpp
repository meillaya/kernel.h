// HIP backend implementation (plan todo 23) - the LOCAL execution backend
// (gfx1101, degraded=false confirmed by todo 6).
//
// Plain hipMalloc/hipFree/hipMemcpy ONLY (plan D11: never hipMallocAsync);
// sharedMemBytes is 0 by construction of KernelLaunch (plan G9).

#include "kernelh/runtime/backend.hpp"

#include <hip/hip_runtime_api.h>

#include <string>
#include <utility>

namespace kernelh::runtime {

std::string hip_error_string(int hip_error_code) {
  const char *s = hipGetErrorString(static_cast<hipError_t>(hip_error_code));
  return s != nullptr ? std::string(s) : "unknown HIP error";
}

namespace detail {
namespace {

// Maps a hipError_t to the runtime Error type (plan F2: no exceptions; every
// failure is a typed Error with a non-zero code and a non-empty message).
Error map_hip_error(hipError_t err, const char *what) {
  return Error{static_cast<int>(err),
               std::string(what) + " failed: " +
                   hip_error_string(static_cast<int>(err)) + " (" +
                   std::to_string(static_cast<int>(err)) + ")"};
}

// Convenience: invoke `expr`, returning unexpected(map_hip_error(...)) on
// failure. Returns true on success.
template <typename Fn>
bool hip_ok(Fn &&fn, const char *what, Error *out) {
  hipError_t err = fn();
  if (err != hipSuccess) {
    *out = map_hip_error(err, what);
    return false;
  }
  return true;
}

} // namespace

class HipBackend final : public Backend {
public:
  BackendKind kind() const override { return BackendKind::kHip; }

  Result<void> init_device(int device_ordinal) override {
    Error err;
    if (!hip_ok([&] { return hipSetDevice(device_ordinal); }, "hipSetDevice",
                &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<DeviceInfo> device_info() override {
    int device = 0;
    Error err;
    if (!hip_ok([&] { return hipGetDevice(&device); }, "hipGetDevice", &err)) {
      return std::unexpected(std::move(err));
    }

    DeviceInfo info;
    hipDeviceProp_t props{};
    if (!hip_ok([&] { return hipGetDeviceProperties(&props, device); },
                "hipGetDeviceProperties", &err)) {
      return std::unexpected(std::move(err));
    }
    info.name = props.name;
    info.vram_bytes = static_cast<long>(props.totalGlobalMem);

    auto attr = [&](hipDeviceAttribute_t attribute, const char *what,
                    int *out) -> bool {
      return hip_ok([&] { return hipDeviceGetAttribute(out, attribute, device); },
                    what, &err);
    };

    int v = 0;
    if (!attr(hipDeviceAttributeMaxSharedMemoryPerBlock,
              "hipDeviceGetAttribute(MaxSharedMemoryPerBlock)", &v)) {
      return std::unexpected(std::move(err));
    }
    info.lds_bytes_per_workgroup = v;
    if (!attr(hipDeviceAttributeWarpSize, "hipDeviceGetAttribute(WarpSize)",
              &v)) {
      return std::unexpected(std::move(err));
    }
    info.wavefront_size = v;
    if (!attr(hipDeviceAttributeMaxThreadsPerBlock,
              "hipDeviceGetAttribute(MaxThreadsPerBlock)", &v)) {
      return std::unexpected(std::move(err));
    }
    info.max_threads_per_workgroup = v;
    if (!attr(hipDeviceAttributeMultiprocessorCount,
              "hipDeviceGetAttribute(MultiprocessorCount)", &v)) {
      return std::unexpected(std::move(err));
    }
    info.cu_count = v;
    return info;
  }

  Result<void> device_synchronize() override {
    Error err;
    if (!hip_ok([&] { return hipDeviceSynchronize(); }, "hipDeviceSynchronize",
                &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<void *> malloc(size_t bytes) override {
    void *ptr = nullptr;
    Error err;
    if (!hip_ok([&] { return hipMalloc(&ptr, bytes); }, "hipMalloc", &err)) {
      return std::unexpected(std::move(err));
    }
    return ptr;
  }

  Result<void> free(void *ptr) override {
    Error err;
    if (!hip_ok([&] { return hipFree(ptr); }, "hipFree", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<void> memcpy_h2d(void *dst, const void *src, size_t bytes) override {
    Error err;
    if (!hip_ok([&] { return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice); },
                "hipMemcpy(H2D)", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<void> memcpy_d2h(void *dst, const void *src, size_t bytes) override {
    Error err;
    if (!hip_ok([&] { return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost); },
                "hipMemcpy(D2H)", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<void *> module_load_data(const void *image) override {
    hipModule_t module = nullptr;
    Error err;
    if (!hip_ok([&] { return hipModuleLoadData(&module, image); },
                "hipModuleLoadData", &err)) {
      return std::unexpected(std::move(err));
    }
    return module;
  }

  Result<void *> module_get_function(void *module, const char *name) override {
    hipFunction_t fn = nullptr;
    Error err;
    if (!hip_ok([&] { return hipModuleGetFunction(&fn, static_cast<hipModule_t>(module), name); },
                "hipModuleGetFunction", &err)) {
      return std::unexpected(std::move(err));
    }
    return fn;
  }

  Result<void> module_unload(void *module) override {
    Error err;
    if (!hip_ok([&] { return hipModuleUnload(static_cast<hipModule_t>(module)); },
                "hipModuleUnload", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<void> launch_kernel(void *function, const LaunchDims &grid,
                             const LaunchDims &block, size_t shared_bytes,
                             StreamHandle stream, void *const *params) override {
    Error err;
    if (!hip_ok(
            [&] {
              return hipModuleLaunchKernel(
                  static_cast<hipFunction_t>(function), grid.x, grid.y, grid.z,
                  block.x, block.y, block.z, shared_bytes,
                  static_cast<hipStream_t>(stream), const_cast<void **>(params),
                  nullptr);
            },
            "hipModuleLaunchKernel", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<EventHandle> event_create() override {
    hipEvent_t event = nullptr;
    Error err;
    if (!hip_ok([&] { return hipEventCreate(&event); }, "hipEventCreate",
                &err)) {
      return std::unexpected(std::move(err));
    }
    return event;
  }

  Result<void> event_record(EventHandle event, StreamHandle stream) override {
    Error err;
    if (!hip_ok([&] { return hipEventRecord(static_cast<hipEvent_t>(event),
                                            static_cast<hipStream_t>(stream)); },
                "hipEventRecord", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<void> event_synchronize(EventHandle event) override {
    Error err;
    if (!hip_ok([&] { return hipEventSynchronize(static_cast<hipEvent_t>(event)); },
                "hipEventSynchronize", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }

  Result<float> event_elapsed_time(EventHandle start, EventHandle end) override {
    float ms = 0.0f;
    Error err;
    if (!hip_ok([&] { return hipEventElapsedTime(&ms, static_cast<hipEvent_t>(start),
                                                 static_cast<hipEvent_t>(end)); },
                "hipEventElapsedTime", &err)) {
      return std::unexpected(std::move(err));
    }
    return ms;
  }

  Result<void> event_destroy(EventHandle event) override {
    Error err;
    if (!hip_ok([&] { return hipEventDestroy(static_cast<hipEvent_t>(event)); },
                "hipEventDestroy", &err)) {
      return std::unexpected(std::move(err));
    }
    return {};
  }
};

std::shared_ptr<Backend> make_hip_backend_impl() {
  return std::make_shared<HipBackend>();
}

} // namespace detail
} // namespace kernelh::runtime
