#include "shape_text.hpp"

#include <cerrno>
#include <cstdlib>

namespace kernelh::bench {

std::string shape_to_text(const ShapeMap &shape) {
  std::string out;
  for (const auto &[key, value] : shape) {
    if (!out.empty()) {
      out += ",";
    }
    out += key + "=" + std::to_string(value);
  }
  return out;
}

ShapeMap parse_shape_text(const std::string &text) {
  ShapeMap out;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    const size_t end = comma == std::string::npos ? text.size() : comma;
    const std::string item = text.substr(start, end - start);
    const size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= item.size()) {
      return {};
    }
    const std::string value = item.substr(eq + 1);
    errno = 0;
    char *valueEnd = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &valueEnd, 10);
    if (errno != 0 || valueEnd == value.c_str() || *valueEnd != '\0') {
      return {};
    }
    out[item.substr(0, eq)] = parsed;
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

} // namespace kernelh::bench
