// nv_vecadd_e2e.cpp - production-path NVIDIA e2e for kernel.h.
//
// Launches the PRODUCTION vecadd cubin produced by the full toolchain
// (`kernelh compile --kernel vecadd --target nvptx-sm_80`, plan todos 19/21/22/24)
// on a remote NVIDIA GPU via the kernelh_runtime CUDA dlopen backend, and
// verifies the vecadd semantics host-side:
//   out[i] = in0[i] + in1[i]  for all i in [0, n)
//
// The vecadd template (lib/dsl/kernels/vecadd.cpp) computes gid = block_id*block_dim
// + thread_id and loops gid .. n with stride block_dim*grid_dim, so the grid
// must cover n (grid = ceil(n/block), gridDim.y/z = 1).
//
// Usage: nv_vecadd_e2e <vecadd.cubin> <n> [block_size]
//
// Compile on the pod (g++-12, C++23):
//   g++-12 -std=c++23 -I include -o nv_vecadd_e2e tools/nv-xchg-e2e/nv_vecadd_e2e.cpp \
//     lib/runtime/backend.cpp lib/runtime/backend_cuda.cpp \
//     lib/runtime/descriptor_packing.cpp lib/runtime/device_context.cpp \
//     lib/runtime/buffer.cpp lib/runtime/gpu_module.cpp \
//     lib/runtime/kernel_launch.cpp lib/runtime/event_timer.cpp \
//     nv_stub.cpp -ldl

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    std::fprintf(stderr, "usage: %s <vecadd.cubin> <n> [block_size]\n", argv[0]);
    return 2;
  }
  const std::string cubinPath = argv[1];
  const int64_t n = std::strtoll(argv[2], nullptr, 10);
  const int blockSize = argc > 3 ? std::atoi(argv[3]) : 256;

  auto backendRes = rt::make_cuda_backend();
  if (!backendRes) {
    std::fprintf(stderr, "error: CUDA backend unavailable: code=%d msg=%s\n",
                 backendRes.error().code, backendRes.error().message.c_str());
    return 77;
  }
  std::shared_ptr<rt::Backend> backend = *backendRes;
  std::printf("NV-BACKEND-OK (dlopen libcuda.so.1)\n");

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

  auto in0 = rt::Buffer::allocate(backend, static_cast<size_t>(n) * 4);
  auto in1 = rt::Buffer::allocate(backend, static_cast<size_t>(n) * 4);
  auto out = rt::Buffer::allocate(backend, static_cast<size_t>(n) * 4);
  if (!in0 || !in1 || !out) {
    std::fprintf(stderr, "error: buffer alloc failed\n");
    return 1;
  }
  std::vector<float> hIn0(static_cast<size_t>(n));
  std::vector<float> hIn1(static_cast<size_t>(n));
  std::vector<float> hOut(static_cast<size_t>(n), -1.0f);
  for (int64_t i = 0; i < n; ++i) {
    hIn0[static_cast<size_t>(i)] = static_cast<float>(i % 1000);
    hIn1[static_cast<size_t>(i)] = static_cast<float>((i * 7) % 1000);
  }
  if (!in0->copy_from_host(hIn0.data(), static_cast<size_t>(n) * 4) ||
      !in1->copy_from_host(hIn1.data(), static_cast<size_t>(n) * 4)) {
    std::fprintf(stderr, "error: H2D copy failed\n");
    return 1;
  }

  std::vector<char> cubin = readFile(cubinPath);
  auto modRes = rt::GpuModule::load(backend, cubin.data(), cubin.size());
  if (!modRes) {
    std::fprintf(stderr, "error: module load: %s\n",
                 modRes.error().message.c_str());
    return 1;
  }
  auto fnRes = modRes->get_function("vecadd");
  if (!fnRes) {
    std::fprintf(stderr, "error: get_function vecadd: %s\n",
                 fnRes.error().message.c_str());
    return 1;
  }
  std::printf("NV-MODULE-LOAD-OK (cubin %zu bytes, symbol vecadd)\n",
              cubin.size());

  const int64_t blocks = (n + blockSize - 1) / blockSize;
  rt::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(blocks), 1, 1)
      .block(static_cast<unsigned>(blockSize), 1, 1)
      .memref_arg(in0->device_ptr(), {n})
      .memref_arg(in1->device_ptr(), {n})
      .memref_arg(out->device_ptr(), {n});
  auto launchRes = launch.launch(*fnRes);
  if (!launchRes) {
    std::fprintf(stderr, "error: launch: %s\n",
                 launchRes.error().message.c_str());
    return 1;
  }
  std::printf("NV-LAUNCH-OK grid=%ld block=%d shared=0\n", blocks, blockSize);

  if (!out->copy_to_host(hOut.data(), static_cast<size_t>(n) * 4)) {
    std::fprintf(stderr, "error: D2H copy failed\n");
    return 1;
  }
  int64_t mismatches = 0;
  for (int64_t i = 0; i < n; ++i) {
    const float expected = hIn0[static_cast<size_t>(i)] +
                           hIn1[static_cast<size_t>(i)];
    if (std::fabs(hOut[static_cast<size_t>(i)] - expected) > 1e-3f) {
      if (mismatches < 5) {
        std::fprintf(stderr, "mismatch at %ld: got %f want %f\n", i,
                     hOut[static_cast<size_t>(i)], expected);
      }
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    std::fprintf(stderr, "FAIL: %ld mismatches out of %ld\n", mismatches, n);
    return 1;
  }
  std::printf("NV-VECADD-PASS (N=%ld, out[i]=in0[i]+in1[i] verified host-side)\n",
              n);
  return 0;
}
