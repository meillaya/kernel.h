// Best-effort clock pinning (plan todo 35, Metis G28).
//
// Attempts to write "profile_standard" to every
// /sys/class/drm/card*/device/power_dpm_force_performance_level so the GPU
// runs at a stable clock during measurement. Best-effort: any failure
// (EPERM without root, no sysfs access in the container, absent cards) logs
// a warning to stderr and CONTINUES - the median-of-20 methodology absorbs
// the residual clock variance.

#ifndef KERNELH_BENCH_CLOCK_PIN_HPP
#define KERNELH_BENCH_CLOCK_PIN_HPP

namespace kernelh::bench {

// Returns the number of cards successfully pinned (0..N). Never throws.
int pin_clocks_best_effort();

} // namespace kernelh::bench

#endif // KERNELH_BENCH_CLOCK_PIN_HPP
