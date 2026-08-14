// kernel.h compiler - autotuner search logic (plan todo 34).
//
// The bounded GEMM autotune search: sample the config space with a FIXED seed
// (Metis G22 cap), evaluate candidates through a caller-supplied evaluator,
// refine around the best, and persist/load results in a JSON cache keyed by
// sha256(kernel + shape + device_name) (Metis note 27: the device name is IN
// the key). The CSV of evaluated configs is written by write_tune_csv.
//
// This library is deliberately MLIR-free and GPU-free (kernelh_compiler stays
// dependency-free - the todo-23 weak-default link contract): the SEARCH logic
// (candidate generation, pruning policy, refinement, cache, CSV) lives here,
// while the heavy compile -> launch -> time work happens in the caller's
// evaluator callback (the `kernelh tune` subcommand in tools/kernelh/main.cpp
// runs it in a forked child to enforce the per-candidate timeout).
//
// PRUNING SEAM (Metis G23): the resource-JSON feedback. resource_prune_reason()
// decides, from a todo-33 KernelResources + the todo-10 DeviceLimits, whether
// a compiled config is admissible: configs whose lds_bytes exceed the device
// LDS budget or whose vgpr_count exceeds the register-file budget are pruned
// WITHOUT ever being launched. It is a pure function over the parsed JSON
// payload, so it is unit-testable without any profiler.

#ifndef KERNELH_COMPILER_AUTOTUNER_HPP
#define KERNELH_COMPILER_AUTOTUNER_HPP

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/resource_reporter.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace kernelh::compiler {

// One GEMM tuning candidate (plan todo 34 config space):
//   BM in {64, 128, 256}, BN in {64, 128, 256}, BK in {16, 32},
//   waves in {4, 8}, unroll in {1, 2}, vec in {1, 2, 4}  -> 216 combos.
// `waves` is the wavefront count per workgroup (the launch block size is
// waves * wavefront_size); the tile/map_blocks/map_warps/use_shared_memory/
// vector_width/unroll schedule attrs map 1:1 to the todo-11 ScheduleAttrs.
struct GemmCandidate {
  int bm = 0;     // tile[0]
  int bn = 0;     // tile[1]
  int bk = 0;     // tile[2]
  int waves = 0;  // wavefronts per workgroup
  int unroll = 0; // unroll factor
  int vec = 0;    // vector width

  std::string to_string() const;
  bool operator==(const GemmCandidate &) const = default;
};

// The full GEMM config space (all 216 combos, deterministic order).
std::vector<GemmCandidate> gemm_config_space();

// One evaluated candidate (a row of the tune CSV).
struct EvaluatedCandidate {
  GemmCandidate config;
  double ms_median = 0.0; // EventTimer median-of-10; 0.0 when not timed
  bool accepted = false;  // passed every gate and was timed
  std::string reason;     // "ok" | "pruned: lds" | "pruned: vgpr" |
                          // "pruned: launch" | "reverify: FAIL ..." |
                          // "timeout" | "compile-error: ..."
};

struct TuneOptions {
  int sample_cap = 64;     // max sampled candidates (Metis G22: <= 64)
  int refine_budget = 16;  // max refinement neighbors of the best
  int max_candidates = 80; // hard cap on total evaluated per run
  std::uint64_t seed = 0x2e46d0526de5f342ULL; // fixed literal seed
  int per_candidate_timeout_seconds = 30; // wall clock per evaluated config
};

struct TuneRun {
  std::vector<EvaluatedCandidate> evaluated;
  int best_index = -1;    // index into evaluated (best ACCEPTED, min ms)
  bool cache_hit = false; // results were served from the cache
};

// Evaluates ONE candidate: compile -> launch verify -> resource check ->
// numeric re-verify -> EventTimer median-of-10. Owned by the caller (the CLI
// implements it over the in-process compile pipeline + HIP runtime).
using TuneEvaluator = std::function<EvaluatedCandidate(const GemmCandidate &)>;

// The profiler-feedback prune rule (Metis G23): decides whether a compiled
// kernel's resources make its config admissible. Returns "" when admissible,
// otherwise a prune reason ("pruned: lds" when lds_bytes > the device LDS
// budget, "pruned: vgpr" when vgpr_count exceeds the register-file-per-wavefront
// budget). Pure + profiler-free; the evaluator logs the returned reason.
std::string resource_prune_reason(const KernelResources &res,
                                  const kernelh::DeviceLimits &limits);

// Self-contained SHA-256 over `data`, hex-encoded (kernelh_compiler has no
// dependency set; the CLI builds the key as sha256(kernel+shape+device_name)).
std::string sha256_hex(const std::string &data);

// The bounded search (sample with the fixed seed + local refinement) with a
// whole-run cache. `cache_key` is the sha256(kernel+shape+device_name) hex
// (device name IN the key, Metis note 27). On a cache hit the cached evaluated
// list + best are returned and `evaluate` is NOT called; the caller detects it
// via TuneRun::cache_hit and logs "cache-hit" (the plan's second-run
// assertion). The cache lives at `cache_path` (build/tune-cache.json) and is
// updated after every search. `log` receives the per-candidate progress lines.
//
// Total evaluated never exceeds options.max_candidates (default 80; Metis
// G22 cap) regardless of sample + refine budgets.
TuneRun autotune(const TuneOptions &options, const TuneEvaluator &evaluate,
                 const std::string &cache_key,
                 const std::filesystem::path &cache_path, std::ostream &log);

// Writes the full CSV of evaluated configs (the plan's output artifact).
// Columns: kernel, shape, bm, bn, bk, waves, unroll, vec, ms_median, accepted,
// reason. Returns false only on a write failure.
bool write_tune_csv(const std::filesystem::path &path,
                    const std::string &kernel, const std::string &shape_text,
                    const std::vector<EvaluatedCandidate> &evaluated);

} // namespace kernelh::compiler

#endif // KERNELH_COMPILER_AUTOTUNER_HPP
