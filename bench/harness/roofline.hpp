// Roofline microbenchmark backends (plan todo 35, Oracle N8): the measured
// machine ceilings that todo 38 reads from results.csv.
//
//   roofline_triad  - streaming read a + read b + write c over 1 GiB f32
//                     buffers per operand (3 GiB moved) -> measured bandwidth
//                     ceiling in GB/s.
//   roofline_fma    - dense FMA saturation (8 independent accumulator chains
//                     per thread so the FMA pipeline stays full; the memory
//                     footprint is tiny so the kernel is purely compute) ->
//                     measured compute ceiling in GFLOPS.
//
// Both kernels live in bench/refs/roofline.hip (hipcc --genco). Launch ABI is
// raw device pointers + scalars (NOT memref descriptors - these are plain
// HIP kernels), so the harness calls Backend::launch_kernel directly.

#ifndef KERNELH_BENCH_ROOFLINE_HPP
#define KERNELH_BENCH_ROOFLINE_HPP

#include "bench_row.hpp"
#include "kernelh/runtime/backend.hpp"

#include <memory>
#include <optional>
#include <string>

namespace kernelh::bench {

// Runs the roofline_triad microbenchmark for the row spec (n = 2^28 f32
// elements per buffer = 1 GiB). Returns the measured row, or nullopt with the
// diagnostic in `error` on any failure.
std::optional<BenchRow> run_roofline_triad(
    const RowSpec &spec, const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error);

// Runs the roofline_fma microbenchmark (n = 2^22 elements, 2^35 total FMAs).
std::optional<BenchRow> run_roofline_fma(
    const RowSpec &spec, const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_ROOFLINE_HPP
