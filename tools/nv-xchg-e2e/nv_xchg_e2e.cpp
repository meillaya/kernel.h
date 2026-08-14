// nv_xchg_e2e.cpp - standalone NVIDIA e2e for kernel.h (plan todo 8/23 validation
// on remote RTX 6000 Ada via the PrimeIntellect pod).
//
// Uses ONLY the CUDA-only subset of kernelh_runtime (backend_cuda.cpp,
// descriptor_packing.cpp, device_context.cpp, buffer.cpp, gpu_module.cpp,
// kernel_launch.cpp, event_timer.cpp) - zero HIP/MLIR link dependencies, so the
// binary runs on a bare Ubuntu pod with just libcuda.so.1 (dlopen'd at runtime).
//
// Launches the todo-8 xchg cubin (sm_80, descriptor-ABI kernel @xchg_kernel,
// 10 kernel params) and verifies the harness contract host-side:
//   in[i]  = (float)(i % 256)
//   out[i] = (float)((i + 1) % 256)
// with cross-thread exchange through shared memory (proves addrspace-3 shared
// works on real NVIDIA silicon).
//
// Usage: nv_xchg_e2e <xchg.cubin> <n> [block_size]
//
// Compile (host, plain g++):
//   g++ -std=c++23 -I include nv_xchg_e2e.cpp \
//     lib/runtime/backend_cuda.cpp lib/runtime/descriptor_packing.cpp \
//     lib/runtime/device_context.cpp lib/runtime/buffer.cpp \
//     lib/runtime/gpu_module.cpp lib/runtime/kernel_launch.cpp \
//     lib/runtime/event_timer.cpp -ldl -o nv_xchg_e2e

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace rt = kernelh::runtime;

static std::vector<char> readFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  in.read(data.data(), size);
  return data;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <xchg.cubin> <n> [block_size]\n", argv[0]);
    return 2;
  }
  const std::string cubinPath = argv[1];
  const int64_t n = std::strtoll(argv[2], nullptr, 10);
  const int blockSize = argc > 3 ? std::atoi(argv[3]) : 256;

  // 1. CUDA backend via dlopen(libcuda.so.1).
  auto backendRes = rt::make_cuda_backend();
  if (!backendRes) {
    std::fprintf(stderr, "error: CUDA backend unavailable: code=%d msg=%s\n",
                 backendRes.error().code, backendRes.error().message.c_str());
    return 77; // exit-77 skip semantics
  }
  std::shared_ptr<rt::Backend> backend = *backendRes;
  std::printf("NV-BACKEND-OK (dlopen libcuda.so.1)\n");

  // 2. Device context (live device info).
  auto ctxRes = rt::DeviceContext::create(backend, 0);
  if (!ctxRes) {
    std::fprintf(stderr, "error: device init: %s\n",
                 ctxRes.error().message.c_str());
    return 1;
  }
  const rt::DeviceInfo &info = ctxRes->info();
  std::printf("NV-DEVICE name=%s cu=%ld lds=%ld wavefront=%d maxThreads=%d\n",
              info.name.c_str(), info.cu_count, info.lds_bytes_per_workgroup,
              info.wavefront_size, info.max_threads_per_workgroup);

  // 3. Buffers.
  auto inBuf = rt::Buffer::allocate(backend, static_cast<size_t>(n) * 4);
  auto outBuf = rt::Buffer::allocate(backend, static_cast<size_t>(n) * 4);
  if (!inBuf || !outBuf) {
    std::fprintf(stderr, "error: buffer alloc failed\n");
    return 1;
  }
  std::vector<float> hostIn(static_cast<size_t>(n));
  std::vector<float> hostOut(static_cast<size_t>(n), -1.0f);
  for (int64_t i = 0; i < n; ++i) {
    hostIn[static_cast<size_t>(i)] = static_cast<float>(i % 256);
  }
  if (!inBuf->copy_from_host(hostIn.data(), static_cast<size_t>(n) * 4)) {
    std::fprintf(stderr, "error: H2D copy failed\n");
    return 1;
  }

  // 4. Load the cubin.
  std::vector<char> cubin = readFile(cubinPath);
  auto modRes = rt::GpuModule::load(backend, cubin.data(), cubin.size());
  if (!modRes) {
    std::fprintf(stderr, "error: module load: %s\n",
                 modRes.error().message.c_str());
    return 1;
  }
  auto fnRes = modRes->get_function("xchg_kernel");
  if (!fnRes) {
    std::fprintf(stderr, "error: get_function xchg_kernel: %s\n",
                 fnRes.error().message.c_str());
    return 1;
  }
  std::printf("NV-MODULE-LOAD-OK (cubin %zu bytes, symbol xchg_kernel)\n",
              cubin.size());

  // 5. Launch with the descriptor-ABI packing (memref<?xf32> pair = 10 params).
  const int64_t blocks = (n + blockSize - 1) / blockSize;
  rt::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(blocks), 1, 1)
      .block(static_cast<unsigned>(blockSize), 1, 1)
      .memref_arg(inBuf->device_ptr(), {n})
      .memref_arg(outBuf->device_ptr(), {n});
  auto launchRes = launch.launch(*fnRes);
  if (!launchRes) {
    std::fprintf(stderr, "error: launch: %s\n",
                 launchRes.error().message.c_str());
    return 1;
  }
  std::printf("NV-LAUNCH-OK grid=%ld block=%d shared=0\n", blocks, blockSize);

  // 6. Verify host-side (the harness contract).
  if (!outBuf->copy_to_host(hostOut.data(), static_cast<size_t>(n) * 4)) {
    std::fprintf(stderr, "error: D2H copy failed\n");
    return 1;
  }
  int64_t mismatches = 0;
  for (int64_t i = 0; i < n; ++i) {
    const float expected = static_cast<float>((i + 1) % 256);
    if (std::fabs(hostOut[static_cast<size_t>(i)] - expected) > 1e-4f) {
      if (mismatches < 5) {
        std::fprintf(stderr, "mismatch at %lld: got %f want %f\n", i,
                     hostOut[static_cast<size_t>(i)], expected);
      }
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    std::fprintf(stderr, "FAIL: %lld mismatches out of %lld\n", mismatches, n);
    return 1;
  }
  std::printf("NV-XCHG-PASS (N=%lld, out[i]=(i+1)%%256 verified host-side)\n", n);
  return 0;
}
