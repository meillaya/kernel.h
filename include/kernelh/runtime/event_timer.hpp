// Event-based timer (plan todo 23).
//
// hipEventCreate / hipEventRecord / hipEventElapsedTime / hipEventDestroy
// (CUDA driver equivalents for the cross backend): records start and end
// events on a stream and reports the elapsed wall time in milliseconds.
// RAII: both events are destroyed on destruction; move-only.

#ifndef KERNELH_RUNTIME_EVENT_TIMER_HPP
#define KERNELH_RUNTIME_EVENT_TIMER_HPP

#include "kernelh/runtime/backend.hpp"

#include <memory>

namespace kernelh::runtime {

class EventTimer {
public:
  static Result<EventTimer> create(std::shared_ptr<Backend> backend);

  Result<void> record_start(StreamHandle stream = nullptr);
  Result<void> record_end(StreamHandle stream = nullptr);
  // Blocks until the recorded end event completes.
  Result<void> synchronize() const;
  // Milliseconds between the two recorded events (both must be complete).
  Result<float> elapsed_milliseconds() const;

  ~EventTimer();
  EventTimer(const EventTimer &) = delete;
  EventTimer &operator=(const EventTimer &) = delete;
  EventTimer(EventTimer &&other) noexcept;
  EventTimer &operator=(EventTimer &&other) noexcept;

private:
  EventTimer(std::shared_ptr<Backend> backend, EventHandle start,
             EventHandle end)
      : backend_(std::move(backend)), start_(start), end_(end) {}

  std::shared_ptr<Backend> backend_;
  EventHandle start_ = nullptr;
  EventHandle end_ = nullptr;
};

} // namespace kernelh::runtime

#endif // KERNELH_RUNTIME_EVENT_TIMER_HPP
