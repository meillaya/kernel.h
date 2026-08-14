#include "clock_pin.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>

namespace kernelh::bench {

int pin_clocks_best_effort() {
  int pinned = 0;
  DIR *dir = ::opendir("/sys/class/drm");
  if (dir == nullptr) {
    std::cerr << "bench: warning: clock pinning skipped (no /sys/class/drm): "
              << std::strerror(errno) << "\n";
    return 0;
  }
  while (struct dirent *entry = ::readdir(dir)) {
    const std::string name = entry->d_name;
    if (name.rfind("card", 0) != 0) {
      continue;
    }
    const std::string sysfs =
        std::string("/sys/class/drm/") + name + "/device/" +
        "power_dpm_force_performance_level";
    const int fd = ::open(sysfs.c_str(), O_WRONLY);
    if (fd < 0) {
      std::cerr << "bench: warning: clock pinning " << sysfs << ": "
                << std::strerror(errno) << " (EPERM -> run as root; "
                << "median-of-20 absorbs variance)\n";
      continue;
    }
    const char level[] = "profile_standard";
    const ssize_t written = ::write(fd, level, sizeof(level) - 1);
    ::close(fd);
    if (written < 0) {
      std::cerr << "bench: warning: clock pinning " << sysfs << ": "
                << std::strerror(errno) << "\n";
      continue;
    }
    ++pinned;
  }
  ::closedir(dir);
  if (pinned > 0) {
    std::cout << "bench: pinned " << pinned
              << " card(s) to profile_standard\n";
  }
  return pinned;
}

} // namespace kernelh::bench
