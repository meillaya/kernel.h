// G32 results.csv schema unit tests (plan todo 35).
//
// Guards the exact CSV contract consumed by todo 36 (occupancy) and todo 38
// (charts): the header spelling, quoting/escaping, and the parse-back
// guarantee that every row carries REAL positive numbers in the numeric
// columns (the adversarial "misleading_success_output" class).

#include "bench_row.hpp"
#include "csv.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string &line, char delim) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (start <= line.size()) {
    const size_t comma = line.find(delim, start);
    fields.push_back(line.substr(start, comma == std::string::npos
                                          ? std::string::npos
                                          : comma - start));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return fields;
}

kernelh::bench::BenchRow sampleRow(const std::string &kernel,
                                   const std::string &shape,
                                   const std::string &backend, double ms,
                                   double rate) {
  kernelh::bench::BenchRow row;
  row.kernel = kernel;
  row.shape = shape;
  row.backend = backend;
  row.ms_median = ms;
  row.gflops_or_gbps = rate;
  row.timestamp = "2026-08-14 10:00:00";
  row.device = "Radeon RX 7700 XT";
  return row;
}

} // namespace

TEST_CASE("bench_csv header is the exact G32 schema", "[bench_csv]") {
  REQUIRE(std::string(kernelh::bench::kCsvHeader) ==
          "kernel,shape,backend,ms_median,gflops_or_gbps,timestamp,device");
}

TEST_CASE("bench_csv simple fields stay unquoted", "[bench_csv]") {
  const kernelh::bench::BenchRow row =
      sampleRow("roofline_triad", "n=268435456", "roofline", 6.25, 512.5);
  REQUIRE(kernelh::bench::csv_line(row) ==
          "roofline_triad,n=268435456,roofline,6.25,512.5,2026-08-14 "
          "10:00:00,Radeon RX 7700 XT");
}

TEST_CASE("bench_csv escapes commas quotes and newlines", "[bench_csv]") {
  using kernelh::bench::csv_escape;
  REQUIRE(csv_escape("plain") == "plain");
  REQUIRE(csv_escape("a,b") == "\"a,b\"");
  REQUIRE(csv_escape("say \"hi\"") == "\"say \"\"hi\"\"\"");
  REQUIRE(csv_escape("line1\nline2") == "\"line1\nline2\"");
  REQUIRE(csv_escape("carriage\rreturn") == "\"carriage\rreturn\"");
}

TEST_CASE("bench_csv write_parse_roundtrip_real_numbers", "[bench_csv]") {
  std::vector<kernelh::bench::BenchRow> rows;
  rows.push_back(sampleRow("roofline_triad", "n=268435456", "roofline",
                           6.125, 525.94));
  rows.push_back(sampleRow("roofline_fma", "n=4194304", "roofline", 2.75,
                           24979.908));
  std::ostringstream os;
  REQUIRE(kernelh::bench::write_results_csv(os, rows));

  std::istringstream in(os.str());
  std::string header;
  std::getline(in, header);
  REQUIRE(header == kernelh::bench::kCsvHeader);
  int lineCount = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    ++lineCount;
    const std::vector<std::string> fields = split(line, ',');
    REQUIRE(fields.size() == 7);
    const double ms = std::stod(fields[3]);
    const double rate = std::stod(fields[4]);
    REQUIRE(std::isfinite(ms));
    REQUIRE(ms > 0.0);
    REQUIRE(std::isfinite(rate));
    REQUIRE(rate > 0.0);
    REQUIRE(!fields[0].empty());
    REQUIRE(!fields[1].empty());
    REQUIRE(!fields[2].empty());
  }
  REQUIRE(lineCount == 2);
}
