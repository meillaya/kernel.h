// Resource reporter implementation (plan todo 33).
//
// Produces resources.json per compiled kernel (schema in the header) through
// the pinned source priority: rocprofv3 kernel-trace when available, else
// static parsing of the vendor artifacts (amdhsa notes via llvm-readobj for
// AMD; ptxas -v + PTX static-shared scan for NVIDIA). See the header for the
// full contract.
//
// kernelh_compiler stays MLIR-free and runtime-free (the todo-23 weak-default
// link depends on it), so all subprocess work uses plain POSIX fork/exec/poll
// with a timeout - no MLIR, no HIP/CUDA runtime.

#include "kernelh/compiler/resource_reporter.hpp"

#include "kernelh/compiler/device_limits.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <poll.h>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace kernelh::compiler {
namespace {

// ---------------------------------------------------------------------------
// Subprocess runner: fork + exec + poll with a timeout. stdout and stderr are
// merged into one stream (ptxas -v prints its info to stderr). On timeout the
// child is SIGKILLed so a hung ptxas/llvm-readobj/rocprofv3 can never hang
// the reporter (todo-33 adversarial class hung_commands).
// ---------------------------------------------------------------------------
struct CommandResult {
  bool ran = false; // fork+exec succeeded (child was spawned)
  bool timed_out = false;
  int exit_code = -1; // WEXITSTATUS; -1 when signaled
  std::string output; // merged stdout + stderr
};

CommandResult run_capture(const std::vector<std::string> &argv, int timeout_ms,
                          const std::filesystem::path &cwd) {
  CommandResult result;
  int fds[2];
  if (pipe(fds) != 0) {
    return result;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    close(fds[0]);
    close(fds[1]);
    return result;
  }
  if (pid == 0) {
    // Child: merge stdout+stderr into the write end; stdin from /dev/null.
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[0]);
    close(fds[1]);
    const int devnull = open("/dev/null", O_RDONLY);
    if (devnull >= 0) {
      dup2(devnull, STDIN_FILENO);
      close(devnull);
    }
    if (!cwd.empty()) {
      (void)chdir(cwd.c_str()); // best-effort; exec failure below still exits
    }
    std::vector<char *> cargv;
    cargv.reserve(argv.size() + 1);
    for (const std::string &a : argv) {
      cargv.push_back(const_cast<char *>(a.c_str()));
    }
    cargv.push_back(nullptr);
    execvp(cargv[0], cargv.data());
    _exit(127); // exec failure (e.g. tool not on PATH)
  }
  // Parent: read until EOF or timeout.
  close(fds[1]);
  std::string output;
  char buf[8192];
  bool timed_out = false;
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    const int remaining =
        timeout_ms > elapsed ? static_cast<int>(timeout_ms - elapsed) : 0;
    struct pollfd pfd = {fds[0], static_cast<short>(POLLIN), 0};
    const int pr = poll(&pfd, 1, remaining);
    if (pr == 0) {
      timed_out = true;
      kill(pid, SIGKILL);
      break;
    }
    if (pr < 0 && errno != EINTR) {
      break;
    }
    if (pr > 0 && (pfd.revents & (POLLIN | POLLHUP))) {
      const ssize_t n = read(fds[0], buf, sizeof(buf));
      if (n > 0) {
        output.append(buf, static_cast<size_t>(n));
      } else if (n == 0) {
        break; // EOF
      } else if (errno != EINTR) {
        break;
      }
    }
  }
  close(fds[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  result.ran = true;
  result.timed_out = timed_out;
  result.output = std::move(output);
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  }
  return result;
}

CommandResult run_capture(const std::vector<std::string> &argv,
                          int timeout_ms) {
  return run_capture(argv, timeout_ms, std::filesystem::path());
}

// ---------------------------------------------------------------------------
// Small text helpers.
// ---------------------------------------------------------------------------
std::string read_file_text(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::string();
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Parses a strictly-decimal integer; returns nullopt on any malformed input
// (empty / non-digit / overflow), so callers can distinguish "missing or
// corrupt field" from a legitimate 0.
std::optional<long> parse_decimal(std::string_view text) {
  if (text.empty()) {
    return std::nullopt;
  }
  long value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
    if (value > (9223372036854775807L - (c - '0')) / 10) {
      return std::nullopt; // overflow
    }
    value = value * 10 + (c - '0');
  }
  return value;
}

// Trims a view in place (no allocation; the returned view aliases `s`, which
// must outlive the caller's use - the CSV/notes buffers do).
std::string_view trim(std::string_view s) {
  size_t b = 0;
  size_t e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) {
    ++b;
  }
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) {
    --e;
  }
  return s.substr(b, e - b);
}

// ---------------------------------------------------------------------------
// Tool resolution (mirrors backends.cpp's resolveToolkit env contract).
// ---------------------------------------------------------------------------
std::string rocm_toolkit_root() {
  const char *override = std::getenv("KERNELH_ROCM_TOOLKIT_ROOT");
  if (override != nullptr && *override != '\0') {
    return override;
  }
  const char *root = std::getenv("ROCM_TOOLKIT_ROOT");
  return root != nullptr ? root : "";
}

std::string cuda_toolkit_root() {
  const char *override = std::getenv("KERNELH_CUDA_TOOLKIT_ROOT");
  if (override != nullptr && *override != '\0') {
    return override;
  }
  const char *root = std::getenv("CUDA_TOOLKIT_ROOT");
  return root != nullptr ? root : "";
}

// "amdgcn-gfx1101" / "nvptx-sm_80" -> "gfx1101" / "sm_80".
std::string chip_for_target(const std::string &target) {
  const size_t dash = target.find('-');
  if (dash == std::string::npos || dash + 1 >= target.size()) {
    return target;
  }
  return target.substr(dash + 1);
}

// Maps a CLI target token to a device-limits token ("gfx1101"/"sm_80"); bare
// tokens pass through (the parsers/tests use them directly).
std::string device_token_for(const std::string &target) {
  if (target == "amdgcn-gfx1101") {
    return "gfx1101";
  }
  if (target == "nvptx-sm_80") {
    return "sm_80";
  }
  return target;
}

// ---------------------------------------------------------------------------
// Occupancy estimate: analytic VGPR/register-limited waves per CU from the
// todo-10 table. floor(reg_file_per_cu / (vgpr_count x wavefront_size)),
// clamped to [1, waves_per_cu]. The full theoretical/measured occupancy
// arrives with todo 36; this is the todo-33 seam the autotuner consumes.
// ---------------------------------------------------------------------------
int occupancy_estimate_for(const std::string &target, int vgpr_count) {
  if (vgpr_count <= 0) {
    return 0;
  }
  const DeviceLimits *limits = nullptr;
  try {
    limits = &device_limits_for(device_token_for(target).c_str());
  } catch (const std::invalid_argument &) {
    return 0; // unknown target -> no estimate
  }
  if (limits->reg_file_per_cu <= 0 || limits->wavefront_size <= 0 ||
      limits->waves_per_cu <= 0) {
    return 0;
  }
  long long waves = limits->reg_file_per_cu /
                    (1LL * vgpr_count * limits->wavefront_size);
  if (waves < 1) {
    waves = 1;
  }
  if (waves > limits->waves_per_cu) {
    waves = limits->waves_per_cu;
  }
  return static_cast<int>(waves);
}

// Static shared-memory scan of PTX text. Our emitter emits module-scope
// `.visible .shared .align N .b8 <name>[<bytes>];` declarations that ptxas -v
// does NOT report as "bytes smem" (verified on LLVM-emitted PTX), so the
// NVIDIA path complements the ptxas report with this scan. nvcc-style
// entry-scope `.shared` arrays are matched too (their size IS reported by
// ptxas, and max() in parse_ptxas_report avoids double counting).
long static_shared_bytes(std::string_view ptx_text) {
  long total = 0;
  size_t pos = 0;
  while (pos < ptx_text.size()) {
    const size_t nl = ptx_text.find('\n', pos);
    const size_t end = nl == std::string_view::npos ? ptx_text.size() : nl;
    const std::string_view line = ptx_text.substr(pos, end - pos);
    if (line.find(".shared") != std::string_view::npos &&
        line.find(".extern") == std::string_view::npos) {
      const size_t lb = line.rfind('[');
      const size_t rb = line.rfind(']');
      if (lb != std::string_view::npos && rb != std::string_view::npos &&
          rb > lb + 1) {
        if (const auto n = parse_decimal(line.substr(lb + 1, rb - lb - 1))) {
          total += *n;
        }
      }
    }
    pos = end + 1;
  }
  return total;
}

// Priority-1 probe: runs rocprofv3 kernel-trace on a micro-launch binary
// (KERNELH_MICRO_LAUNCH, invoked as `<launch> <hsaco>`) and parses
// kernel_trace.csv. Returns true only on a fully successful trace; any
// failure (rocprofv3 absent, no micro-launch, trace error) fills `warn` and
// returns false so resource_report falls back to static parsing.
bool try_rocprofv3(const std::string &kernel, const std::string &target,
                   const std::filesystem::path &out_dir, KernelResources &res,
                   std::string &warn) {
  if (!rocprofv3_available()) {
    warn = "rocprofv3 not on PATH; falling back to static artifact parsing";
    return false;
  }
  const char *launch = std::getenv("KERNELH_MICRO_LAUNCH");
  if (launch == nullptr || *launch == '\0') {
    warn =
        "rocprofv3 present but KERNELH_MICRO_LAUNCH unset; falling back to "
        "static artifact parsing";
    return false;
  }
  const std::filesystem::path hsaco = out_dir / (kernel + ".hsaco");
  if (!std::filesystem::exists(hsaco)) {
    warn = "no " + hsaco.string() +
           " for the rocprofv3 micro-launch; falling back to static parsing";
    return false;
  }
  // rocprof writes kernel_trace.csv into the scratch dir (-d).
  const std::filesystem::path scratch = out_dir / ".resources-rocprof";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);
  const CommandResult r = run_capture(
      {"rocprofv3", "-d", scratch.string(), "--kernel-trace",
       "--output-format", "csv", "--", launch, hsaco.string()},
      120000);
  if (!r.ran || r.timed_out) {
    warn = "rocprofv3 failed to run (timed out or not spawned); falling back "
           "to static artifact parsing";
    return false;
  }
  const std::string csv = read_file_text(scratch / "kernel_trace.csv");
  auto parsed = parse_kernel_trace_csv(csv, kernel);
  if (!parsed) {
    warn = parsed.error() + "; falling back to static artifact parsing";
    return false;
  }
  res = *parsed;
  return true;
}

// AMD path: `llvm-readobj --notes <hsaco>`.
CommandResult run_llvm_readobj_notes(const std::string &hsaco) {
  CommandResult r = run_capture({"llvm-readobj", "--notes", hsaco}, 30000);
  // exit 127 == execvp failure (tool not on PATH): retry via the ROCm root.
  if (r.ran && r.exit_code == 127) {
    const std::string root = rocm_toolkit_root();
    if (!root.empty()) {
      r = run_capture({root + "/llvm/bin/llvm-readobj", "--notes", hsaco},
                      30000);
    }
  }
  return r;
}

// NVIDIA path: `$CUDA_TOOLKIT_ROOT/bin/ptxas -v <ptx>` as a SEPARATE explicit
// step (Oracle N5: in-process serialization never exposes ptxas stderr).
// Returns false when the toolkit is missing or ptxas fails - the caller then
// fills zeros with a warning (non-fatal).
bool run_ptxas_v(const std::string &ptx, const std::string &chip,
                 std::string &out, std::string &warn) {
  const std::string root = cuda_toolkit_root();
  const std::string ptxas = root.empty() ? "ptxas" : root + "/bin/ptxas";
  if (!root.empty() && !std::filesystem::exists(ptxas)) {
    warn = "CUDA toolkit ptxas not found at '" + ptxas +
           "' (CUDA_TOOLKIT_ROOT=" + root + "); filling resource zeros";
    return false;
  }
  const CommandResult r =
      run_capture({ptxas, "-v", "-arch=" + chip, "-c", "-o", "/dev/null", ptx},
                  30000);
  if (!r.ran) {
    warn = "failed to spawn ptxas; filling resource zeros";
    return false;
  }
  if (r.timed_out) {
    warn = "ptxas timed out on '" + ptx + "'; filling resource zeros";
    return false;
  }
  if (r.exit_code == 127) {
    warn = "ptxas not found on PATH; filling resource zeros";
    return false;
  }
  if (r.exit_code != 0) {
    warn = "ptxas -v exited " + std::to_string(r.exit_code) + " on '" + ptx +
           "'; keeping static shared bytes only";
    return false;
  }
  out = r.output;
  return true;
}

// Hand-rolled JSON string escaping (minimal: the JSON spec's required
// escapes; kernel names are identifiers so this rarely fires).
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

} // namespace

// Parses a rocprofv3 kernel_trace.csv (ROCm 7.x `--output-format csv`). See
// the header for the contract; this implementation locates columns by name so
// their ordering in the CSV does not matter.
std::expected<KernelResources, std::string>
parse_kernel_trace_csv(std::string_view csv, std::string_view kernel) {
  const size_t header_end = csv.find('\n');
  if (header_end == std::string_view::npos) {
    return std::unexpected("kernel_trace.csv has no header row");
  }
  std::vector<std::string_view> columns;
  const std::string_view header = csv.substr(0, header_end);
  size_t col_pos = 0;
  while (col_pos <= header.size()) {
    const size_t comma = header.find(',', col_pos);
    const size_t end = comma == std::string_view::npos ? header.size() : comma;
    columns.push_back(trim(header.substr(col_pos, end - col_pos)));
    if (comma == std::string_view::npos) {
      break;
    }
    col_pos = comma + 1;
  }
  auto column_index = [&](const char *name) -> int {
    for (size_t i = 0; i < columns.size(); ++i) {
      if (columns[i] == name) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };
  const int name_col = column_index("Kernel_Name");
  const int vgpr_col = column_index("VGPR_Count");
  const int sgpr_col = column_index("SGPR_Count");
  const int lds_col = column_index("LDS_Block_Size");
  if (name_col < 0 || vgpr_col < 0) {
    return std::unexpected(
        "kernel_trace.csv missing Kernel_Name/VGPR_Count columns");
  }

  size_t row_pos = header_end + 1;
  while (row_pos < csv.size()) {
    const size_t nl = csv.find('\n', row_pos);
    const size_t end = nl == std::string_view::npos ? csv.size() : nl;
    const std::string_view row = csv.substr(row_pos, end - row_pos);
    std::vector<std::string_view> fields;
    size_t fp = 0;
    while (fp <= row.size()) {
      const size_t comma = row.find(',', fp);
      const size_t fe = comma == std::string_view::npos ? row.size() : comma;
      fields.push_back(trim(row.substr(fp, fe - fp)));
      if (comma == std::string_view::npos) {
        break;
      }
      fp = comma + 1;
    }
    if (fields.size() > static_cast<size_t>(name_col) &&
        fields[static_cast<size_t>(name_col)] == kernel) {
      KernelResources res;
      res.kernel = std::string(kernel);
      if (vgpr_col >= 0 && fields.size() > static_cast<size_t>(vgpr_col)) {
        if (const auto n =
                parse_decimal(fields[static_cast<size_t>(vgpr_col)])) {
          res.vgpr_count = static_cast<int>(*n);
        }
      }
      if (sgpr_col >= 0 && fields.size() > static_cast<size_t>(sgpr_col)) {
        if (const auto n =
                parse_decimal(fields[static_cast<size_t>(sgpr_col)])) {
          res.sgpr_count = static_cast<int>(*n);
        }
      }
      if (lds_col >= 0 && fields.size() > static_cast<size_t>(lds_col)) {
        if (const auto n =
                parse_decimal(fields[static_cast<size_t>(lds_col)])) {
          res.lds_bytes = *n;
        }
      }
      return res;
    }
    row_pos = end + 1;
  }
  return std::unexpected("kernel_trace.csv has no row for kernel '" +
                         std::string(kernel) + "'");
}

// ---------------------------------------------------------------------------
// amdhsa notes parsing (pure text; see header).
// ---------------------------------------------------------------------------
std::expected<KernelResources, std::string>
parse_amdhsa_notes(std::string_view notes, std::string_view kernel_hint) {
  if (notes.find("amdhsa.kernels:") == std::string_view::npos) {
    return std::unexpected(
        "unparseable code object: no amdhsa.kernels section in notes");
  }

  // Split the metadata into per-kernel sections at `  - .args:` markers.
  struct Section {
    std::string name;
    int vgpr = -1;
    int sgpr = -1;
    long lds = -1;
    int wave = -1;
  };
  std::vector<Section> sections;
  size_t search_pos = 0;
  while (search_pos < notes.size()) {
    const size_t marker = notes.find("  - .args:", search_pos);
    if (marker == std::string_view::npos) {
      break;
    }
    const size_t next = notes.find("  - .args:", marker + 9);
    const size_t sec_end = next == std::string_view::npos ? notes.size() : next;
    const std::string_view sec = notes.substr(marker, sec_end - marker);

    Section s;
    size_t line_pos = 0;
    while (line_pos < sec.size()) {
      const size_t nl = sec.find('\n', line_pos);
      const size_t line_end = nl == std::string_view::npos ? sec.size() : nl;
      const std::string line = std::string(trim(sec.substr(line_pos, line_end - line_pos)));
      if (!line.empty() && line[0] == '.') {
        const size_t colon = line.find(':');
        if (colon != std::string::npos) {
          const std::string key = line.substr(1, colon - 1);
          const std::string value = std::string(trim(line.substr(colon + 1)));
          if (key == "name") {
            s.name = value;
          } else if (key == "vgpr_count") {
            if (const auto n = parse_decimal(value)) {
              s.vgpr = static_cast<int>(*n);
            }
          } else if (key == "sgpr_count") {
            if (const auto n = parse_decimal(value)) {
              s.sgpr = static_cast<int>(*n);
            }
          } else if (key == "group_segment_fixed_size") {
            if (const auto n = parse_decimal(value)) {
              s.lds = *n;
            }
          } else if (key == "wavefront_size") {
            if (const auto n = parse_decimal(value)) {
              s.wave = static_cast<int>(*n);
            }
          }
        }
      }
      line_pos = line_end + 1;
    }
    sections.push_back(std::move(s));
    search_pos = sec_end;
  }

  if (sections.empty()) {
    return std::unexpected(
        "unparseable code object: no amdhsa kernel descriptor found");
  }

  // Prefer the entry whose .name matches the hint; otherwise the first entry
  // carrying a .vgpr_count (single-kernel HSACOs land here).
  const Section *chosen = nullptr;
  for (const Section &s : sections) {
    if (!kernel_hint.empty() && s.name == kernel_hint) {
      chosen = &s;
      break;
    }
  }
  if (chosen == nullptr) {
    for (const Section &s : sections) {
      if (s.vgpr >= 0) {
        chosen = &s;
        break;
      }
    }
  }
  if (chosen == nullptr || chosen->vgpr < 0) {
    return std::unexpected(
        "unparseable code object: kernel descriptor has no .vgpr_count");
  }

  KernelResources res;
  res.kernel = chosen->name;
  res.vgpr_count = chosen->vgpr;
  res.sgpr_count = chosen->sgpr > 0 ? chosen->sgpr : 0;
  res.lds_bytes = chosen->lds > 0 ? chosen->lds : 0;
  res.wavefront_size = chosen->wave > 0 ? chosen->wave : 0;
  return res;
}

// ---------------------------------------------------------------------------
// ptxas -v + PTX static-shared parsing (pure text; see header).
// ---------------------------------------------------------------------------
std::expected<KernelResources, std::string>
parse_ptxas_report(std::string_view ptxas_output, std::string_view ptx_text) {
  // "Used N registers" (capital U, from `ptxas info : Used N registers, ...`).
  int registers = 0;
  long ptxas_smem = 0;
  size_t pos = 0;
  while ((pos = ptxas_output.find("Used ", pos)) != std::string_view::npos) {
    pos += 5;
    size_t end = pos;
    while (end < ptxas_output.size() && ptxas_output[end] >= '0' &&
           ptxas_output[end] <= '9') {
      ++end;
    }
    if (end > pos && ptxas_output.substr(end).rfind(" registers", 0) == 0) {
      if (const auto n = parse_decimal(ptxas_output.substr(pos, end - pos))) {
        registers = static_cast<int>(*n);
      }
      break;
    }
  }
  // "M bytes smem" (entry-scope/dynamic smem; absent for MLIR-emitted PTX
  // whose `.shared` arrays are module-scope and unreported).
  const size_t smem_mark = ptxas_output.find(" bytes smem");
  if (smem_mark != std::string_view::npos) {
    size_t digits_end = smem_mark;
    while (digits_end > 0 && ptxas_output[digits_end - 1] >= '0' &&
           ptxas_output[digits_end - 1] <= '9') {
      --digits_end;
    }
    if (const auto n = parse_decimal(
            ptxas_output.substr(digits_end, smem_mark - digits_end))) {
      ptxas_smem = *n;
    }
  }

  const long static_smem = static_shared_bytes(ptx_text);
  if (registers == 0 && ptxas_smem == 0 && static_smem == 0) {
    return std::unexpected(
        "unparseable ptxas output: no 'Used N registers' / smem lines");
  }

  KernelResources res;
  res.vgpr_count = registers;
  res.sgpr_count = 0; // ptxas does not expose an SGPR-equivalent
  res.lds_bytes = ptxas_smem > static_smem ? ptxas_smem : static_smem;
  res.wavefront_size = 0; // filled from the device table by resource_report
  return res;
}

// ---------------------------------------------------------------------------
// rocprofv3 availability probe (source-priority-1 gate).
// ---------------------------------------------------------------------------
bool rocprofv3_available() {
  const CommandResult r =
      run_capture({"bash", "-c", "command -v rocprofv3"}, 10000);
  return r.ran && !r.timed_out && r.exit_code == 0 && !r.output.empty();
}

// ---------------------------------------------------------------------------
// JSON writer (hand-rolled; the project has no JSON dependency).
// ---------------------------------------------------------------------------
bool write_resources_json(const std::filesystem::path &artifact_dir,
                          const KernelResources &res) {
  std::ostringstream ss;
  ss << "{\n"
     << "  \"kernel\": \"" << json_escape(res.kernel) << "\",\n"
     << "  \"target\": \"" << json_escape(res.target) << "\",\n"
     << "  \"vgpr_count\": " << res.vgpr_count << ",\n"
     << "  \"sgpr_count\": " << res.sgpr_count << ",\n"
     << "  \"lds_bytes\": " << res.lds_bytes << ",\n"
     << "  \"wavefront_size\": " << res.wavefront_size << ",\n"
     << "  \"occupancy_estimate\": " << res.occupancy_estimate << "\n"
     << "}\n";
  std::ofstream out(artifact_dir / "resources.json", std::ios::binary);
  if (!out) {
    return false;
  }
  out << ss.str();
  return out.good();
}

// ---------------------------------------------------------------------------
// resource_report: the todo-33 pipeline with the pinned source priority.
// ---------------------------------------------------------------------------
std::expected<KernelResources, std::string>
resource_report(const std::string &kernel, const std::string &target,
                const std::filesystem::path &artifact_dir) {
  KernelResources res;
  res.kernel = kernel;
  res.target = target;

  // Priority 1: rocprofv3 kernel-trace. Expected to be absent; any failure
  // falls back with a warning.
  std::string warn;
  if (try_rocprofv3(kernel, target, artifact_dir, res, warn)) {
    res.occupancy_estimate = occupancy_estimate_for(target, res.vgpr_count);
    if (!write_resources_json(artifact_dir, res)) {
      return std::unexpected("cannot write resources.json in '" +
                             artifact_dir.string() + "'");
    }
    return res;
  }

  if (target.rfind("amdgcn", 0) == 0) {
    // Priority 2 (AMD): llvm-readobj --notes <kernel>.hsaco.
    const std::filesystem::path hsaco = artifact_dir / (kernel + ".hsaco");
    if (!std::filesystem::exists(hsaco)) {
      return std::unexpected("cannot open hsaco artifact '" + hsaco.string() +
                             "'");
    }
    const CommandResult notes = run_llvm_readobj_notes(hsaco.string());
    if (!notes.ran || notes.timed_out || notes.exit_code != 0) {
      // Truncated/corrupt code object: llvm-readobj fails -> clean error.
      return std::unexpected(
          "unparseable code object: llvm-readobj failed on '" + hsaco.string() +
          "'");
    }
    auto parsed = parse_amdhsa_notes(notes.output, kernel);
    if (!parsed) {
      return parsed; // carries "unparseable code object"
    }
    res = *parsed;
    res.kernel = kernel;
    res.target = target;
    res.occupancy_estimate = occupancy_estimate_for(target, res.vgpr_count);
  } else if (target.rfind("nvptx", 0) == 0) {
    // Priority 2 (NVIDIA): ptxas -v as a separate explicit step + a
    // static-shared scan of the PTX. A missing/unusable ptxas is NON-FATAL:
    // fill zeros with a warning (todo-33 contract), still reporting the
    // static shared bytes we can read directly from the PTX.
    const std::filesystem::path ptx = artifact_dir / (kernel + ".ptx");
    if (!std::filesystem::exists(ptx)) {
      return std::unexpected("cannot open ptx artifact '" + ptx.string() + "'");
    }
    const std::string ptx_text = read_file_text(ptx);
    std::string ptxas_out;
    std::string tool_warn;
    if (run_ptxas_v(ptx.string(), chip_for_target(target), ptxas_out,
                    tool_warn)) {
      auto parsed = parse_ptxas_report(ptxas_out, ptx_text);
      if (!parsed) {
        tool_warn = parsed.error();
        res.vgpr_count = 0;
        res.sgpr_count = 0;
        res.lds_bytes = static_shared_bytes(ptx_text);
      } else {
        res = *parsed;
      }
    } else {
      res.vgpr_count = 0;
      res.sgpr_count = 0;
      res.lds_bytes = static_shared_bytes(ptx_text);
    }
    if (!tool_warn.empty()) {
      std::cerr << "warning: " << tool_warn << "\n";
    }
    res.kernel = kernel;
    res.target = target;
    if (res.wavefront_size <= 0) {
      res.wavefront_size = 32; // warp size
    }
    res.occupancy_estimate = occupancy_estimate_for(target, res.vgpr_count);
  } else {
    return std::unexpected("resource_report: unknown target '" + target +
                           "' (expected amdgcn-* or nvptx-*)");
  }

  if (!write_resources_json(artifact_dir, res)) {
    return std::unexpected("cannot write resources.json in '" +
                           artifact_dir.string() + "'");
  }
  return res;
}

} // namespace kernelh::compiler
