// Autotuner search logic unit tests (plan todo 34) - the MLIR-free, GPU-free
// pure part of the autotuner: config-space enumeration, fixed-seed sampling,
// the resource-JSON prune seam (Metis G23 - testable without any profiler),
// local refinement, the <= 80 candidate cap (Metis G22), the whole-run cache
// (device name IN the key, Metis note 27; second run -> cache-hit), and the
// CSV writer.
//
// The on-GPU end-to-end acceptance (compile -> launch -> numeric re-verify ->
// EventTimer median-of-10) lives in the `kernelh tune` CLI and is exercised by
// the manual acceptance in .omo/evidence/task-34-kernelh.txt, NOT here.

#include "kernelh/compiler/autotuner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using kernelh::compiler::EvaluatedCandidate;
using kernelh::compiler::GemmCandidate;
using kernelh::compiler::KernelResources;
using kernelh::compiler::TuneOptions;
using kernelh::compiler::TuneRun;

// A fake evaluator: accepts every candidate with a deterministic synthetic
// time derived from the config (monotone in the tile size), so the search has
// a well-defined best and refinement target. Counts calls.
class FakeEvaluator {
public:
  explicit FakeEvaluator(bool acceptAll = true) : acceptAll_(acceptAll) {}

  kernelh::compiler::EvaluatedCandidate operator()(const GemmCandidate &c) {
    ++calls_;
    EvaluatedCandidate out;
    out.config = c;
    if (!acceptAll_ && (c.bm >= 256 || c.bn >= 256)) {
      out.accepted = false;
      out.reason = "pruned: lds";
      return out;
    }
    out.accepted = true;
    out.ms_median = 10.0 + c.bm + c.bn + c.bk + c.waves + c.unroll * 10 + c.vec;
    out.reason = "ok";
    return out;
  }

  int calls() const { return calls_; }

private:
  bool acceptAll_;
  int calls_ = 0;
};

kernelh::compiler::TuneOptions baseOptions() {
  kernelh::compiler::TuneOptions options;
  options.sample_cap = 64;
  options.refine_budget = 16;
  options.max_candidates = 80;
  return options;
}

std::string cacheKeyFor(const std::string &device) {
  return kernelh::compiler::sha256_hex("gemm|m=512,k=512,n=512|" + device);
}

std::filesystem::path tempCachePath(const std::string &suffix) {
  const std::filesystem::path dir = "/tmp/kernelh-t34-unit-" + suffix;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir / "tune-cache.json";
}

} // namespace

TEST_CASE("autotuner_config_space_216_combos") {
  const std::vector<GemmCandidate> space = kernelh::compiler::gemm_config_space();
  REQUIRE(space.size() == 216);
  for (const GemmCandidate &c : space) {
    REQUIRE((c.bm == 64 || c.bm == 128 || c.bm == 256));
    REQUIRE((c.bn == 64 || c.bn == 128 || c.bn == 256));
    REQUIRE((c.bk == 16 || c.bk == 32));
    REQUIRE((c.waves == 4 || c.waves == 8));
    REQUIRE((c.unroll == 1 || c.unroll == 2));
    REQUIRE((c.vec == 1 || c.vec == 2 || c.vec == 4));
  }
}

TEST_CASE("autotuner_sha256_known_vector") {
  // FIPS 180-4 test vector: SHA-256("abc").
  REQUIRE(kernelh::compiler::sha256_hex("abc") ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // Empty-string digest.
  REQUIRE(kernelh::compiler::sha256_hex("") ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_CASE("autotuner_cache_key_includes_device_name") {
  // Metis note 27: the device name is IN the key - two devices must produce
  // different keys for the same kernel+shape.
  const std::string keyA = cacheKeyFor("AMD Radeon RX 7700 XT");
  const std::string keyB = cacheKeyFor("AMD Radeon RX 7800 XT");
  REQUIRE(keyA.size() == 64);
  REQUIRE(keyA != keyB);
}

TEST_CASE("autotuner_resource_prune_reason") {
  const kernelh::DeviceLimits limits =
      kernelh::device_limits_for("gfx1101"); // lds 65536, reg budget 192/wf

  KernelResources ok;
  ok.kernel = "gemm";
  ok.lds_bytes = 32768;
  ok.vgpr_count = 19;
  ok.occupancy_estimate = 10;
  REQUIRE(kernelh::compiler::resource_prune_reason(ok, limits).empty());

  // LDS overflow -> pruned (Metis G23 seam; no profiler needed).
  KernelResources ldsOver;
  ldsOver.lds_bytes = 65536 + 1;
  ldsOver.vgpr_count = 19;
  ldsOver.occupancy_estimate = 10;
  REQUIRE(kernelh::compiler::resource_prune_reason(ldsOver, limits) ==
          "pruned: lds");

  // VGPR budget: reg_file_per_cu(6144) / wavefront(32) = 192 VGPRs per
  // wavefront; a kernel needing more can never be resident.
  KernelResources vgprOver;
  vgprOver.lds_bytes = 1024;
  vgprOver.vgpr_count = 300;
  vgprOver.occupancy_estimate = 0;
  REQUIRE(kernelh::compiler::resource_prune_reason(vgprOver, limits) ==
          "pruned: vgpr");

  // Unknown (0) fields are "not available" and must NOT prune.
  KernelResources unknown;
  REQUIRE(kernelh::compiler::resource_prune_reason(unknown, limits).empty());
}

TEST_CASE("autotuner_sample_is_deterministic_with_fixed_seed") {
  const auto run = [](const std::string &suffix) {
    FakeEvaluator evaluator;
    std::ostringstream log;
    const kernelh::compiler::TuneRun result =
        kernelh::compiler::autotune(baseOptions(), std::ref(evaluator),
                                    cacheKeyFor("dev-" + suffix),
                                    tempCachePath(suffix), log);
    return result;
  };
  const kernelh::compiler::TuneRun first = run("det1");
  const kernelh::compiler::TuneRun second = run("det2");
  REQUIRE(first.evaluated.size() == second.evaluated.size());
  for (size_t i = 0; i < first.evaluated.size(); ++i) {
    REQUIRE(first.evaluated[i].config == second.evaluated[i].config);
    REQUIRE(first.evaluated[i].ms_median == second.evaluated[i].ms_median);
    REQUIRE(first.evaluated[i].accepted == second.evaluated[i].accepted);
  }
}

TEST_CASE("autotuner_prune_rules_prune_lds_configs_via_evaluator") {
  // A fake evaluator rejects bm/bn >= 256 with "pruned: lds" (the resource
  // seam applied to those tile sizes); the search must record them as pruned
  // and never select one as best.
  FakeEvaluator evaluator(/*acceptAll=*/false);
  std::ostringstream log;
  const kernelh::compiler::TuneRun result =
      kernelh::compiler::autotune(baseOptions(), std::ref(evaluator),
                                  cacheKeyFor("prune"),
                                  tempCachePath("prune"), log);

  REQUIRE(result.evaluated.size() <= 80);
  bool sawPrunedLds = false;
  for (const EvaluatedCandidate &e : result.evaluated) {
    if (e.accepted) {
      REQUIRE(e.config.bm < 256);
      REQUIRE(e.config.bn < 256);
    } else if (e.reason == "pruned: lds") {
      sawPrunedLds = true;
    }
  }
  // With 64 sampled of 216, at least one bm>=256/bn>=256 config must have been
  // sampled and pruned.
  REQUIRE(sawPrunedLds);
  REQUIRE(result.best_index >= 0);
  REQUIRE(result.evaluated[static_cast<size_t>(result.best_index)].accepted);
}

TEST_CASE("autotuner_local_refinement_explores_neighbors_of_best") {
  TuneOptions options = baseOptions();
  options.sample_cap = 10; // small sample so neighbors are NOT pre-sampled
  options.refine_budget = 16;
  options.max_candidates = 80;

  FakeEvaluator evaluator;
  std::ostringstream log;
  const kernelh::compiler::TuneRun result =
      kernelh::compiler::autotune(options, std::ref(evaluator),
                                  cacheKeyFor("refine"),
                                  tempCachePath("refine"), log);

  REQUIRE(evaluator.calls() == static_cast<int>(result.evaluated.size()));
  REQUIRE(result.evaluated.size() <= 80);
  REQUIRE(result.evaluated.size() > 10); // refinement added neighbors
  REQUIRE(result.best_index >= 0);

  // Refinement targets the best WITHIN THE SAMPLE (the first sample_cap
  // entries) - a refined neighbor may later beat it, so the final best_index
  // is not necessarily that config.
  size_t preBest = 0;
  for (size_t i = 1; i < options.sample_cap; ++i) {
    if (result.evaluated[i].accepted &&
        result.evaluated[i].ms_median < result.evaluated[preBest].ms_median) {
      preBest = i;
    }
  }
  const GemmCandidate best = result.evaluated[preBest].config;
  // Every post-sample candidate must be a single-field neighbor of the
  // pre-refinement best (that is what refinement evaluates).
  size_t refined = 0;
  for (size_t i = options.sample_cap; i < result.evaluated.size(); ++i) {
    const GemmCandidate &c = result.evaluated[i].config;
    int diffs = 0;
    if (c.bm != best.bm) ++diffs;
    if (c.bn != best.bn) ++diffs;
    if (c.bk != best.bk) ++diffs;
    if (c.waves != best.waves) ++diffs;
    if (c.unroll != best.unroll) ++diffs;
    if (c.vec != best.vec) ++diffs;
    REQUIRE(diffs == 1);
    ++refined;
  }
  REQUIRE(refined > 0);
}

TEST_CASE("autotuner_never_exceeds_80_candidates") {
  // Metis G22 hard cap: even with an absurd sample budget the total evaluated
  // must stay <= 80 (sample 64 + refinement 16).
  TuneOptions options = baseOptions();
  options.sample_cap = 500; // > the 216-combo space
  options.refine_budget = 16;
  options.max_candidates = 80;

  FakeEvaluator evaluator;
  std::ostringstream log;
  const kernelh::compiler::TuneRun result =
      kernelh::compiler::autotune(options, std::ref(evaluator),
                                  cacheKeyFor("cap"),
                                  tempCachePath("cap"), log);
  REQUIRE(result.evaluated.size() == 80); // sample 64 + all 16 refine slots
  REQUIRE(evaluator.calls() == 80);
}

TEST_CASE("autotuner_no_accepted_candidate_skips_refinement") {
  // An evaluator that rejects EVERYTHING must produce best_index == -1 and
  // never spend the refinement budget.
  kernelh::compiler::TuneOptions options = baseOptions();
  options.sample_cap = 64;
  options.refine_budget = 16;
  options.max_candidates = 80;

  int calls = 0;
  const kernelh::compiler::TuneEvaluator rejectAll =
      [&calls](const GemmCandidate &c) {
        ++calls;
        EvaluatedCandidate out;
        out.config = c;
        out.accepted = false;
        out.reason = "pruned: lds";
        return out;
      };
  std::ostringstream log;
  const kernelh::compiler::TuneRun result =
      kernelh::compiler::autotune(options, rejectAll, cacheKeyFor("reject"),
                                  tempCachePath("reject"), log);
  REQUIRE(result.best_index == -1);
  REQUIRE(result.evaluated.size() == 64); // no refinement happened
  REQUIRE(calls == 64);
  REQUIRE(log.str().find("no accepted candidate") != std::string::npos);
}

TEST_CASE("autotuner_second_run_hits_cache") {
  const std::filesystem::path cachePath = tempCachePath("cache");
  const std::string key = cacheKeyFor("cache-dev");

  int calls = 0;
  const kernelh::compiler::TuneEvaluator evaluator =
      [&calls](const GemmCandidate &c) {
        ++calls;
        EvaluatedCandidate out;
        out.config = c;
        out.accepted = true;
        out.ms_median = 20.0;
        out.reason = "ok";
        return out;
      };

  std::ostringstream log1;
  const kernelh::compiler::TuneRun first =
      kernelh::compiler::autotune(baseOptions(), evaluator, key, cachePath,
                                  log1);
  REQUIRE(first.cache_hit == false);
  const int callsAfterFirst = calls;
  REQUIRE(callsAfterFirst > 0);
  REQUIRE(callsAfterFirst <= 80);

  // Second run, identical key + cache file: results come from the cache and
  // the evaluator is never invoked again.
  std::ostringstream log2;
  const kernelh::compiler::TuneRun second =
      kernelh::compiler::autotune(baseOptions(), evaluator, key, cachePath,
                                  log2);
  REQUIRE(second.cache_hit == true);
  REQUIRE(calls == callsAfterFirst); // evaluator NOT called on cache hit
  REQUIRE(second.evaluated.size() == first.evaluated.size());
  REQUIRE(second.best_index == first.best_index);
  for (size_t i = 0; i < first.evaluated.size(); ++i) {
    REQUIRE(second.evaluated[i].config == first.evaluated[i].config);
    REQUIRE(second.evaluated[i].ms_median == first.evaluated[i].ms_median);
    REQUIRE(second.evaluated[i].accepted == first.evaluated[i].accepted);
    REQUIRE(second.evaluated[i].reason == first.evaluated[i].reason);
  }
  // The plan's second-run assertion is a "cache-hit" log line.
  REQUIRE(log2.str().find("cache-hit") != std::string::npos);
  REQUIRE(log1.str().find("cache-miss") != std::string::npos);
}

TEST_CASE("autotuner_cache_respects_device_in_key") {
  // A different device name (-> different key) must MISS the cache that the
  // first device populated.
  const std::filesystem::path cachePath = tempCachePath("cachedev");
  FakeEvaluator evaluator;
  std::ostringstream log1;
  const kernelh::compiler::TuneRun first =
      kernelh::compiler::autotune(baseOptions(), std::ref(evaluator),
                                  cacheKeyFor("devA"), cachePath, log1);
  REQUIRE(first.cache_hit == false);
  const int callsAfterFirst = evaluator.calls();

  std::ostringstream log2;
  const kernelh::compiler::TuneRun second =
      kernelh::compiler::autotune(baseOptions(), std::ref(evaluator),
                                  cacheKeyFor("devB"), cachePath, log2);
  REQUIRE(second.cache_hit == false); // different device -> different key
  REQUIRE(evaluator.calls() > callsAfterFirst); // search re-ran
}

TEST_CASE("autotuner_write_csv") {
  const std::filesystem::path dir = "/tmp/kernelh-t34-unit-csv";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path csv = dir / "tune-results.csv";

  EvaluatedCandidate accepted;
  accepted.config = GemmCandidate{64, 128, 32, 8, 2, 4};
  accepted.ms_median = 1.25;
  accepted.accepted = true;
  accepted.reason = "ok";
  EvaluatedCandidate pruned;
  pruned.config = GemmCandidate{256, 256, 32, 4, 1, 1};
  pruned.accepted = false;
  pruned.reason = "pruned: lds";
  const std::vector<EvaluatedCandidate> evaluated = {accepted, pruned};

  REQUIRE(kernelh::compiler::write_tune_csv(csv, "gemm", "m=512,k=512,n=512",
                                            evaluated));

  std::ifstream in(csv);
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  REQUIRE(text.find("kernel,shape,bm,bn,bk,waves,unroll,vec,ms_median,"
                    "accepted,reason") != std::string::npos);
  // Shape text contains commas -> quoted.
  REQUIRE(text.find("\"m=512,k=512,n=512\",64,128,32,8,2,4,1.25,true,ok") !=
          std::string::npos);
  REQUIRE(text.find("256,256,32,4,1,1,0,false,pruned: lds") !=
          std::string::npos);
}
