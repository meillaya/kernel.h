// `kernelh bench` orchestration (plan todo 35).
//
// Runs a suite's {kernel x shape x backend} matrix with the pinned timing
// methodology (warmup 5 + median of 20 via EventTimer), writes results.csv
// (G32 schema), and returns a process exit code:
//   0   - every row measured + CSV written
//   1   - operational failure (device unavailable, a row failed, bad suite)
//   2   - CLI usage error (unknown suite, handled here for the QA probe)
//   77  - skip: no CUDA driver (cuda-cross suite on a machine without one),
//         or no HIP device (degraded mode - the [gpu] backends are skipped)

#ifndef KERNELH_BENCH_RUN_BENCH_HPP
#define KERNELH_BENCH_RUN_BENCH_HPP

#include <string>

namespace kernelh::bench {

struct BenchFlags {
  std::string suite = "all";
  std::string out; // a .csv file path OR a directory (results.csv inside)
  // --occupancy (plan todo 36): also generate occupancy.{json,md} after the
  // suite rows are measured.
  bool occupancy = false;
  // Occupancy-only mode (plan todo 36 acceptance): with --occupancy AND
  // --kernel, the bench produces occupancy.{json,md} for the one named kernel
  // + target WITHOUT running any timing rows. --shape/--threads are optional
  // (per-kernel defaults apply); --target is required.
  std::string kernel;
  std::string target;
  std::string shape;
  int threads = 0; // launch workgroup size; <= 0 -> per-kernel default
};

// Resolves `out` to the CSV path: paths already ending in ".csv" are used
// verbatim; anything else is treated as a directory (created if missing) and
// gets "results.csv" appended.
std::string resolve_csv_path(const std::string &out);

int run_bench(const BenchFlags &flags);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_RUN_BENCH_HPP
