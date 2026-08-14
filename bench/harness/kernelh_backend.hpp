// kernelh backend: our compiled kernels (plan todo 35, backend (a)).
//
// Each row compiles the named DSL template IN-PROCESS with its default
// schedule (registry emit -> structural verify -> default pipeline -> atom
// dialect -> HSACO), loads it, and times it with the pinned warmup-5 +
// median-of-20 methodology. Launch configs replicate the todo-25..31 e2e
// tests (which the launch verifier accepts): gemm grid=ceil(m/32) block=256,
// softmax grid=rows block=min(cols,256), layernorm grid=rows block=256, scan
// two-pass (scan + scan_totals), fused_gelu grid=ceil(n/256) block=256,
// attention grid=(seq,heads,batch) block=64.

#ifndef KERNELH_BENCH_KERNELH_BACKEND_HPP
#define KERNELH_BENCH_KERNELH_BACKEND_HPP

#include "bench_row.hpp"
#include "kernelh/runtime/backend.hpp"

#include <memory>
#include <optional>
#include <string>

namespace kernelh::bench {

// Runs the kernelh backend for `spec` (spec.backend must be "kernelh").
// Returns the measured row, or nullopt with `error` set on any failure.
std::optional<BenchRow> run_kernelh_row(
    const RowSpec &spec, const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_KERNELH_BACKEND_HPP
