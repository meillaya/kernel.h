// EventTimer implementation (plan todo 23).

#include "kernelh/runtime/event_timer.hpp"

#include <utility>

namespace kernelh::runtime {

Result<EventTimer> EventTimer::create(std::shared_ptr<Backend> backend) {
  if (!backend) {
    return std::unexpected(Error{1, "EventTimer::create: null backend"});
  }
  auto start = backend->event_create();
  if (!start) {
    return std::unexpected(std::move(start.error()));
  }
  auto end = backend->event_create();
  if (!end) {
    backend->event_destroy(*start); // best-effort cleanup
    return std::unexpected(std::move(end.error()));
  }
  return EventTimer(std::move(backend), *start, *end);
}

Result<void> EventTimer::record_start(StreamHandle stream) {
  return backend_->event_record(start_, stream);
}

Result<void> EventTimer::record_end(StreamHandle stream) {
  return backend_->event_record(end_, stream);
}

Result<void> EventTimer::synchronize() const {
  return backend_->event_synchronize(end_);
}

Result<float> EventTimer::elapsed_milliseconds() const {
  return backend_->event_elapsed_time(start_, end_);
}

EventTimer::~EventTimer() {
  if (backend_) {
    if (start_ != nullptr) {
      backend_->event_destroy(start_);
    }
    if (end_ != nullptr) {
      backend_->event_destroy(end_);
    }
  }
  start_ = nullptr;
  end_ = nullptr;
}

EventTimer::EventTimer(EventTimer &&other) noexcept
    : backend_(std::move(other.backend_)), start_(other.start_),
      end_(other.end_) {
  other.start_ = nullptr;
  other.end_ = nullptr;
}

EventTimer &EventTimer::operator=(EventTimer &&other) noexcept {
  if (this != &other) {
    if (backend_) {
      if (start_ != nullptr) {
        backend_->event_destroy(start_);
      }
      if (end_ != nullptr) {
        backend_->event_destroy(end_);
      }
    }
    backend_ = std::move(other.backend_);
    start_ = other.start_;
    end_ = other.end_;
    other.start_ = nullptr;
    other.end_ = nullptr;
  }
  return *this;
}

} // namespace kernelh::runtime
