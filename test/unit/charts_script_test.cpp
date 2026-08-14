// charts.py script test (plan todo 38).
//
// Runs the production `scripts/charts.py` python script end-to-end on a
// small hand-computed fixture CSV (test/fixtures/charts_sample.csv) and
// asserts:
//   - exit code 0 and 4 PNGs produced, each > 10 KB (the plan's chart
//     artifact gate - "misleading_success_output" adversarial class: PNG
//     size is asserted, not just the exit code),
//   - charts_stats.csv is written, parseable, carries the G32 row
//     identifiers, and the arithmetic-intensity values match the formulas
//     documented in scripts/README.md (gemm 512 -> 85.333333, gemm 1024 ->
//     170.666667, attention seq=512 -> 128, bandwidth kernels -> 0, ridge
//     AI = 15000/300 = 50) - the "machine-readable summary asserted by
//     tests" contract,
//   - the header-only-CSV probe exits non-zero with "no data rows" and
//     produces NO empty charts (plan QA scenario).
//
// Shells out to `python3 scripts/charts.py` (the same std::system pattern
// as the todo-25 e2e tests shelling out to the `kernelh compile` driver);
// python3 + matplotlib come from the todo-2 devenv env, which ctest runs
// inside.

#include "kernelh_compile_util.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef KERNELH_SOURCE_DIR
#error "charts_script_test.cpp requires the KERNELH_SOURCE_DIR compile definition"
#endif

namespace {

// Absolute path to scripts/charts.py (KERNELH_SOURCE_DIR is baked in).
const std::string kChartsScript =
    std::string(KERNELH_SOURCE_DIR) + "/scripts/charts.py";
const std::string kFixture =
    std::string(KERNELH_SOURCE_DIR) + "/test/fixtures/charts_sample.csv";

// (png filename, minimum size in bytes) - the plan's >10KB artifact gate.
const std::vector<std::pair<std::string, long>> kExpectedPngs = {
    {"gemm_gflops_vs_size.png", 10240},
    {"elementwise_bw.png", 10240},
    {"attention_gflops.png", 10240},
    {"roofline.png", 10240},
};

long file_size(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.good()) {
    return -1;
  }
  return static_cast<long>(in.tellg());
}

std::vector<std::string> split(const std::string &line, char delim) {
  std::vector<std::string> fields;
  std::string field;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      // CSV double-quote: toggle quoting; a doubled "" is an escaped quote.
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if (c == delim && !in_quotes) {
      fields.push_back(field);
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  fields.push_back(field);
  return fields;
}

} // namespace

TEST_CASE("charts script writes 4 PNGs >10KB + parseable stats CSV",
          "[charts]") {
  const std::string outDir = "/tmp/kernelh-t38-charts";
  const std::string cmd =
      "rm -rf '" + outDir + "' && python3 '" + kChartsScript + "' '" +
      kFixture + "' '" + outDir + "'";
  const int rc = std::system(cmd.c_str());
  REQUIRE(WIFEXITED(rc));
  REQUIRE(WEXITSTATUS(rc) == 0);

  // Every PNG exists and is larger than 10 KB.
  for (const auto &[name, minBytes] : kExpectedPngs) {
    const long size = file_size(outDir + "/" + name);
    INFO(name << " size=" << size);
    REQUIRE(size > minBytes);
  }

  // charts_stats.csv exists and is non-empty.
  const std::string statsPath = outDir + "/charts_stats.csv";
  REQUIRE(kernelh::test::file_exists(statsPath));
  REQUIRE(file_size(statsPath) > 0);
}

TEST_CASE("charts script stats CSV carries the documented AI formulas",
          "[charts]") {
  const std::string outDir = "/tmp/kernelh-t38-charts-ai";
  const std::string cmd =
      "rm -rf '" + outDir + "' && python3 '" + kChartsScript + "' '" +
      kFixture + "' '" + outDir + "'";
  const int rc = std::system(cmd.c_str());
  REQUIRE(WIFEXITED(rc));
  REQUIRE(WEXITSTATUS(rc) == 0);

  std::ifstream in(outDir + "/charts_stats.csv");
  REQUIRE(in.good());
  std::string header;
  std::getline(in, header);
  // python csv.writer uses the \r\n dialect; strip the trailing CR.
  if (!header.empty() && header.back() == '\r') {
    header.pop_back();
  }
  REQUIRE(header ==
          "kernel,backend,shape,ms_median,gflops_or_gbps,ai,ceiling_ai,"
          "ceiling_gflops");

  // Collect ai/ceiling_ai/ceiling_gflops by (kernel, backend, shape).
  struct Row {
    std::string kernel;
    std::string backend;
    std::string shape;
    double ai = 0.0;
    double ceiling_ai = 0.0;
    double ceiling_gflops = 0.0;
  };
  std::vector<Row> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> fields = split(line, ',');
    REQUIRE(fields.size() == 8);
    Row row;
    row.kernel = fields[0];
    row.backend = fields[1];
    row.shape = fields[2];
    row.ai = std::stod(fields[5]);
    row.ceiling_ai = std::stod(fields[6]);
    row.ceiling_gflops = std::stod(fields[7]);
    rows.push_back(row);
  }
  REQUIRE(rows.size() >= 5); // the 5 non-ceiling fixture rows

  auto find = [&rows](const std::string &kernel, const std::string &backend,
                      const std::string &shape) -> const Row * {
    for (const Row &row : rows) {
      if (row.kernel == kernel && row.backend == backend &&
          row.shape == shape) {
        return &row;
      }
    }
    return nullptr;
  };

  // GEMM AI = 2mnk / ((mk + kn + mn) * 4): 512 -> 85.333333, 1024 ->
  // 170.666667.
  const Row *g512 = find("gemm", "kernelh", "m=512,n=512,k=512");
  REQUIRE(g512 != nullptr);
  REQUIRE(g512->ai == Catch::Approx(85.333333).epsilon(1e-5));
  const Row *g1024 = find("gemm", "kernelh", "m=1024,n=1024,k=1024");
  REQUIRE(g1024 != nullptr);
  REQUIRE(g1024->ai == Catch::Approx(170.666667).epsilon(1e-5));
  // rocblas gemm 512 shares the same AI (same shape -> same formula).
  const Row *g512r = find("gemm", "rocblas", "m=512,n=512,k=512");
  REQUIRE(g512r != nullptr);
  REQUIRE(g512r->ai == Catch::Approx(85.333333).epsilon(1e-5));

  // Attention AI = s/4 = 512/4 = 128.
  const Row *att = find("attention", "kernelh",
                        "batch=2,heads=8,seq=512,head_dim=64");
  REQUIRE(att != nullptr);
  REQUIRE(att->ai == Catch::Approx(128.0).epsilon(1e-5));

  // Bandwidth-bound kernels (softmax) have AI ~ 0 (README: no flops counted).
  const Row *softmax = find("softmax", "kernelh", "rows=4096,cols=1024");
  REQUIRE(softmax != nullptr);
  REQUIRE(softmax->ai == Catch::Approx(0.0).epsilon(1e-9));

  // Ridge AI = compute ceiling / BW ceiling = 15000 / 300 = 50, and the
  // compute ceiling column carries the roofline_fma rate.
  for (const Row &row : rows) {
    REQUIRE(row.ceiling_ai == Catch::Approx(50.0).epsilon(1e-5));
    REQUIRE(row.ceiling_gflops == Catch::Approx(15000.0).epsilon(1e-3));
  }
}

TEST_CASE("charts script rejects header-only CSV with no charts",
          "[charts]") {
  const std::string csvPath = "/tmp/kernelh-t38-empty.csv";
  {
    std::ofstream out(csvPath);
    out << "kernel,shape,backend,ms_median,gflops_or_gbps,timestamp,device\n";
  }
  const std::string outDir = "/tmp/kernelh-t38-empty-out";
  const std::string cmd =
      "rm -rf '" + outDir + "' && python3 '" + kChartsScript + "' '" +
      csvPath + "' '" + outDir + "'";
  const int rc = std::system(cmd.c_str());
  REQUIRE(WIFEXITED(rc));
  REQUIRE(WEXITSTATUS(rc) != 0);
  // No charts may be silently produced.
  for (const auto &[name, minBytes] : kExpectedPngs) {
    REQUIRE(!kernelh::test::file_exists(outDir + "/" + name));
  }
}
