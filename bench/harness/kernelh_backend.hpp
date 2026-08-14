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

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kernelh::bench {

// Runs the kernelh backend for `spec` (spec.backend must be "kernelh").
// Returns the measured row, or nullopt with `error` set on any failure.
std::optional<BenchRow> run_kernelh_row(
    const RowSpec &spec, const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error);

// Plan todo 36: the occupancy generator hook consumed by `kernelh bench
// --occupancy` (and exposed here for todo 37's explore occupancy section -
// Oracle R2-B2, the explore wiring lands in todo 37).
//
// Occupancy-only mode (one kernel + one target): compiles the named DSL kernel
// for `target` (a CLI --target token), runs the todo-33 resource reporter for
// vgpr/lds, and computes the theoretical (Oracle B5) + measured (todo-32 hook
// on the loaded kernel, when a device is present) occupancy. `shapeText` may
// be empty (a per-kernel default shape is used); `threads` <= 0 selects the
// per-kernel default launch size. Writes <outDir>/occupancy.{json,md}.
// Returns 0 on success, 1 on any operational failure (error set), 2 on a CLI
// usage error. No device is required (analytic always works; measured degrades
// to a note when the device or the vendor API is unavailable).
int write_occupancy_for_kernel(const std::string &kernel,
                               const std::string &target,
                               const std::string &shapeText, int threads,
                               const std::filesystem::path &outDir,
                               std::string &error);

// Suite mode: generates occupancy.{json,md} for every kernelh-backend row in
// `rows` (analytic + measured via the supplied live backend). Used by
// `kernelh bench --suite <name> --occupancy` after the rows are measured.
// Returns 0 on success, 1 on failure (error set).
int write_occupancy_for_rows(
    const std::vector<RowSpec> &rows,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::filesystem::path &outDir, std::string &error);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_KERNELH_BACKEND_HPP
