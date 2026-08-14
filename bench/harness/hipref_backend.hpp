// hipref backend: hand-written HIP reference kernels (plan todo 35, backend
// (c)). bench/refs/*.hip are compiled with `hipcc --genco` into COF code
// objects loaded via GpuModule::load and launched with the RAW-POINTER ABI
// (bypassing the memref-descriptor packing - the refs are plain HIP kernels).
//
// Variants: hipref-naive (scalar) + hipref-vec (vectorized-4) for
// softmax/layernorm/scan/fused_gelu, hipref-naive for the naive tiled GEMM.
// Launch configs mirror the kernelh backend's (and the todo-25..31 e2e
// tests); the vec variants use the same buffers, only the entry symbol
// differs (the device pointer is reinterpreted as float4 inside the kernel).

#ifndef KERNELH_BENCH_HIPREF_BACKEND_HPP
#define KERNELH_BENCH_HIPREF_BACKEND_HPP

#include "bench_row.hpp"
#include "kernelh/runtime/backend.hpp"

#include <memory>
#include <optional>
#include <string>

namespace kernelh::bench {

// Runs the hipref backend for `spec` (spec.backend must be "hipref-naive" or
// "hipref-vec"). Returns the measured row, or nullopt with `error` set on any
// failure.
std::optional<BenchRow> run_hipref_row(
    const RowSpec &spec, const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_HIPREF_BACKEND_HPP
