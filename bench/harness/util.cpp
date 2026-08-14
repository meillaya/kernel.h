#include "util.hpp"

#include <chrono>
#include <ctime>

namespace kernelh::bench {

std::string now_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  return buf;
}

} // namespace kernelh::bench
