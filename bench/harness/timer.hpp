// Warmup + median-of-N EventTimer timing (plan todo 35, Metis G32).
//
// The pinned methodology: `warmup` untimed launches (device-side state / clock
// ramping settle), then `iterations` individually-timed launches (record_start
// -> launch -> record_end -> synchronize -> elapsed_milliseconds), and the
// MEDIAN of the samples is reported. Median absorbs the variance that
// best-effort clock pinning (Metis G28) cannot fully remove.

#ifndef KERNELH_BENCH_TIMER_HPP
#define KERNELH_BENCH_TIMER_HPP

#include "kernelh/runtime/backend.hpp"

#include <functional>
#include <vector>

namespace kernelh::bench {

// Median of a non-empty sample vector (sorted copy). NaN is treated as larger
// than any value so it sorts last; a NaN-heavy sample still returns a number.
double median_of(std::vector<double> samples);

// Times `one_launch` (which must perform EXACTLY one kernel launch when
// invoked) with the pinned methodology. Returns the median elapsed
// milliseconds, or -1.0 on any runtime failure (device sync / event failure).
// A per-launch failure returns -1.0 immediately.
double time_launch_median(
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::function<kernelh::runtime::Result<void>()> &one_launch,
    int warmup = 5, int iterations = 20);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_TIMER_HPP
