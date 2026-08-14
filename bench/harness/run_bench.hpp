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
  // --occupancy hook (consumed by plan todo 36): accepted by the CLI now so
  // the flag surface is stable; the occupancy report generation lands with
  // todo 36 and reads this flag.
  bool occupancy = false;
};

// Resolves `out` to the CSV path: paths already ending in ".csv" are used
// verbatim; anything else is treated as a directory (created if missing) and
// gets "results.csv" appended.
std::string resolve_csv_path(const std::string &out);

int run_bench(const BenchFlags &flags);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_RUN_BENCH_HPP
