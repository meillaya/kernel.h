// Occupancy report implementation (plan todo 36) - see
// include/kernelh/compiler/occupancy.hpp for the API contract and the pinned
// Oracle B5 formulas.
//
// kernelh_compiler stays MLIR-free and HIP-free (the todo-23 weak-default link
// depends on it): the measured path goes through the todo-32 weak hook
// query_occupancy_max_blocks_runtime() defined in launch_verifier.cpp (same
// library); the vendor API call itself lives in the todo-23 runtime.

#include "kernelh/compiler/occupancy.hpp"

#include "kernelh/compiler/launch_verifier.hpp"

#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

namespace kernelh::compiler {

namespace {

// Wavefronts a workgroup of `threads` threads occupies.
int waves_per_block(const DeviceLimits &limits, int threads) {
  if (limits.wavefront_size <= 0 || threads <= 0) {
    return 1;
  }
  return (threads + limits.wavefront_size - 1) / limits.wavefront_size;
}

// "vgpr" | "lds" | "threads" | "none" (the JSON/markdown limiter spelling).
const char *limiter_text(OccupancyLimiter limiter) {
  switch (limiter) {
  case OccupancyLimiter::kVgpr:
    return "vgpr";
  case OccupancyLimiter::kLds:
    return "lds";
  case OccupancyLimiter::kThreads:
    return "threads";
  case OccupancyLimiter::kNone:
    return "none";
  }
  return "none";
}

// Shortest %-string ("12.5", "6.25", "0"): %g keeps the values exact for the
// hand-computed test constants.
std::string pct_text(double pct) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%g", pct);
  return std::string(buf);
}

// Minimal JSON string escaping (kernel/config/note strings; same contract as
// resource_reporter.cpp's json_escape).
std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (const unsigned char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out += static_cast<char>(c);
      }
    }
  }
  return out;
}

// The flat per-report JSON object (the occupancy.{json,md} payload schema).
std::string report_json(const OccupancyReport &r) {
  std::ostringstream os;
  os << "{\n"
     << "  \"kernel\": \"" << json_escape(r.kernel) << "\",\n"
     << "  \"target\": \"" << json_escape(r.target) << "\",\n"
     << "  \"config\": \"" << json_escape(r.config) << "\",\n"
     << "  \"vgpr_count\": " << r.vgpr_count << ",\n"
     << "  \"lds_bytes\": " << r.lds_bytes << ",\n"
     << "  \"threads_per_workgroup\": " << r.threads_per_workgroup << ",\n"
     << "  \"vgpr_limited_waves\": " << r.vgpr_limited_waves << ",\n"
     << "  \"lds_limited_blocks\": " << r.lds_limited_blocks << ",\n"
     << "  \"thread_limited_blocks\": " << r.thread_limited_blocks << ",\n"
     << "  \"theoretical_waves\": " << r.achieved_waves << ",\n"
     << "  \"theoretical_pct\": " << pct_text(r.theoretical_pct) << ",\n"
     << "  \"limiter\": \"" << limiter_text(r.limiter) << "\",\n"
     << "  \"measured_blocks\": " << r.measured_blocks << ",\n"
     << "  \"measured_pct\": " << pct_text(r.measured_pct) << ",\n"
     << "  \"measured_note\": \"" << json_escape(r.measured_note) << "\"\n"
     << "}";
  return os.str();
}

bool write_text_file(const std::filesystem::path &path,
                     const std::string &text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  return out.good();
}

} // namespace

OccupancyReport occupancy_report(const OccupancyInput &input,
                                 const DeviceLimits &limits,
                                 void *function) {
  OccupancyReport r;
  r.kernel = input.kernel;
  r.target = input.target;
  r.config = input.config;
  r.vgpr_count = input.vgpr_count;
  r.lds_bytes = input.lds_bytes;
  r.threads_per_workgroup = input.threads_per_workgroup;

  // 1. VGPR-limited waves (Oracle B5): floor(reg_file_per_cu /
  //    (vgpr_count x wavefront_size)), capped to waves_per_cu. Unknown inputs
  //    (vgpr_count <= 0 or a missing table field) -> bound not limiting.
  const bool have_vgpr = input.vgpr_count > 0 && limits.reg_file_per_cu > 0 &&
                         limits.wavefront_size > 0;
  if (have_vgpr) {
    long long waves = limits.reg_file_per_cu /
                      (1LL * input.vgpr_count * limits.wavefront_size);
    if (waves < 1) {
      waves = 1;
    }
    if (waves > limits.waves_per_cu) {
      waves = limits.waves_per_cu;
    }
    r.vgpr_limited_waves = static_cast<int>(waves);
  }

  // 2. LDS-limited blocks: floor(lds_bytes_per_workgroup / lds_bytes).
  const bool have_lds = input.lds_bytes > 0 &&
                        limits.lds_bytes_per_workgroup > 0;
  if (have_lds) {
    r.lds_limited_blocks = static_cast<int>(limits.lds_bytes_per_workgroup /
                                            input.lds_bytes);
  }

  // 3. Thread-limited blocks: floor(max_threads_per_workgroup / threads).
  const bool have_threads = input.threads_per_workgroup > 0 &&
                            limits.max_threads_per_workgroup > 0;
  if (have_threads) {
    r.thread_limited_blocks = limits.max_threads_per_workgroup /
                              input.threads_per_workgroup;
  }

  // 4. achieved_waves = min over the three available bounds, compared directly
  //    exactly as Oracle B5 pins them. Ties resolve to the EARLIER bound
  //    (vgpr < lds < threads). occupancy_pct = achieved_waves/waves_per_cu.
  int best = std::numeric_limits<int>::max();
  r.limiter = OccupancyLimiter::kNone;
  if (have_vgpr) {
    best = r.vgpr_limited_waves;
    r.limiter = OccupancyLimiter::kVgpr;
  }
  if (have_lds && r.lds_limited_blocks < best) {
    best = r.lds_limited_blocks;
    r.limiter = OccupancyLimiter::kLds;
  }
  if (have_threads && r.thread_limited_blocks < best) {
    best = r.thread_limited_blocks;
    r.limiter = OccupancyLimiter::kThreads;
  }
  if (best != std::numeric_limits<int>::max()) {
    r.achieved_waves = best;
    if (limits.waves_per_cu > 0) {
      r.theoretical_pct = 100.0 * r.achieved_waves / limits.waves_per_cu;
    }
  }

  // 5. Measured occupancy via the todo-32 weak hook when a loaded function
  //    handle is available. The hook returns -1 when the runtime is not linked
  //    OR the vendor API failed (hipOccupancyMaxActiveBlocksPerMultiprocessor
  //    returns hipErrorInvalidDeviceFunction for every kernel on the local
  //    ROCm box - learnings todo 32): measured_pct stays 0 with a note.
  if (function != nullptr && input.threads_per_workgroup > 0) {
    const int blocks = query_occupancy_max_blocks_runtime(
        function, input.threads_per_workgroup,
        static_cast<size_t>(input.lds_bytes));
    if (blocks >= 0) {
      r.measured_blocks = blocks;
      const int wpb = waves_per_block(limits, input.threads_per_workgroup);
      const long long measured_waves = 1LL * blocks * wpb;
      if (limits.waves_per_cu > 0) {
        double pct = 100.0 * measured_waves / limits.waves_per_cu;
        r.measured_pct = pct > 100.0 ? 100.0 : pct;
      }
    } else {
      r.measured_blocks = -1;
      r.measured_note =
          "vendor occupancy API unavailable "
          "(hipOccupancyMaxActiveBlocksPerMultiprocessor failed - "
          "hipErrorInvalidDeviceFunction on the local ROCm box); "
          "measured_pct = 0 (analytic only)";
    }
  } else {
    r.measured_blocks = -1;
    r.measured_note = "no loaded kernel function; measured_pct = 0 "
                      "(analytic only)";
  }
  return r;
}

bool write_occupancy_json(const std::filesystem::path &file,
                          const OccupancyReport &report) {
  return write_text_file(file, report_json(report) + "\n");
}

bool write_occupancy_json_all(const std::filesystem::path &file,
                              const std::vector<OccupancyReport> &reports) {
  std::ostringstream os;
  os << "{\n  \"reports\": [\n";
  for (size_t i = 0; i < reports.size(); ++i) {
    os << "    " << report_json(reports[i]);
    if (i + 1 < reports.size()) {
      os << ",\n";
    } else {
      os << "\n";
    }
  }
  os << "  ]\n}\n";
  return write_text_file(file, os.str());
}

bool write_occupancy_markdown(const std::filesystem::path &file,
                              const std::vector<OccupancyReport> &reports) {
  std::ostringstream os;
  os << "# Occupancy report\n\n"
     << "Generated by `kernelh bench --occupancy` (plan todo 36). Theoretical "
        "occupancy comes from the todo-33 resources and the todo-10 "
        "device-limits table (Oracle B5); measured occupancy uses "
        "hipOccupancyMaxActiveBlocksPerMultiprocessor on the loaded kernel "
        "when the vendor API is available.\n\n"
     << "| kernel | config | vgpr | lds (B) | theoretical % | measured % | "
        "limiter |\n"
     << "| --- | --- | ---: | ---: | ---: | ---: | --- |\n";
  for (const OccupancyReport &r : reports) {
    std::string config = r.config;
    for (size_t pos = 0; (pos = config.find('|', pos)) != std::string::npos;
         pos += 2) {
      config.replace(pos, 1, "\\|");
    }
    os << "| " << r.kernel << " | " << config << " | " << r.vgpr_count << " | "
       << r.lds_bytes << " | " << pct_text(r.theoretical_pct) << " | "
       << (r.measured_blocks >= 0 ? pct_text(r.measured_pct) : "n/a") << " | "
       << limiter_text(r.limiter) << " |\n";
  }
  os << "\nNotes:\n"
     << "- achieved_waves = min(VGPR-limited waves, LDS-limited blocks, "
        "thread-limited blocks), the Oracle B5 formula; occupancy_pct = "
        "achieved_waves / waves_per_cu x 100.\n"
     << "- limiter names the bound achieved_waves came from "
        "(vgpr | lds | threads); \"none\" when no bound was computable.\n"
     << "- measured % shows \"n/a\" when the vendor occupancy API is "
        "unavailable (no device, or the todo-32 hook returned -1 - the local "
        "ROCm box returns hipErrorInvalidDeviceFunction for every kernel, so "
        "measured stays analytic-fallback here).\n";
  return write_text_file(file, os.str());
}

} // namespace kernelh::compiler
