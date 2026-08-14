#include "timer.hpp"

#include "kernelh/runtime/event_timer.hpp"

#include <algorithm>
#include <cmath>

namespace kernelh::bench {

double median_of(std::vector<double> samples) {
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(), samples.end(),
            [](double a, double b) {
              // NaN sorts last (treat as +inf) so a NaN-heavy sample's
              // median is NaN -> the caller can detect the failure.
              if (std::isnan(a)) return false;
              if (std::isnan(b)) return true;
              return a < b;
            });
  const size_t n = samples.size();
  if (n % 2 == 1) {
    return samples[n / 2];
  }
  return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

double time_launch_median(
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::function<kernelh::runtime::Result<void>()> &one_launch,
    int warmup, int iterations) {
  auto timer = kernelh::runtime::EventTimer::create(backend);
  if (!timer) {
    return -1.0;
  }
  for (int i = 0; i < warmup; ++i) {
    if (!one_launch()) {
      return -1.0;
    }
  }
  if (!backend->device_synchronize()) {
    return -1.0;
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    if (!timer->record_start()) {
      return -1.0;
    }
    if (!one_launch()) {
      return -1.0;
    }
    if (!timer->record_end()) {
      return -1.0;
    }
    if (!timer->synchronize()) {
      return -1.0;
    }
    auto ms = timer->elapsed_milliseconds();
    if (!ms) {
      return -1.0;
    }
    samples.push_back(static_cast<double>(*ms));
  }
  return median_of(std::move(samples));
}

} // namespace kernelh::bench
