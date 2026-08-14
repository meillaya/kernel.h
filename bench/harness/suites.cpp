#include "suites.hpp"

#include <algorithm>

namespace kernelh::bench {

namespace {

// GEMM square sizes (M = K = N).
const long kGemmSizes[] = {512, 1024, 2048, 4096, 8192};

std::vector<RowSpec> gemmRows(const std::string &backend) {
  std::vector<RowSpec> rows;
  for (const long n : kGemmSizes) {
    rows.push_back({"gemm", "m=" + std::to_string(n) + ",n=" +
                                  std::to_string(n) + ",k=" + std::to_string(n),
                    backend});
  }
  return rows;
}

std::vector<RowSpec> kernelhRows() {
  std::vector<RowSpec> rows = gemmRows("kernelh");
  for (const auto &[r, c] :
       std::vector<std::pair<long, long>>{{4096, 1024}, {16384, 4096}}) {
    const std::string shape = "rows=" + std::to_string(r) + ",cols=" +
                              std::to_string(c);
    rows.push_back({"softmax", shape, "kernelh"});
    rows.push_back({"layernorm", shape, "kernelh"});
  }
  for (const long n : {100000L, 1000000L}) {
    rows.push_back({"scan", "n=" + std::to_string(n), "kernelh"});
  }
  for (const long n : {1048576L, 4194304L}) {
    rows.push_back({"fused_gelu", "n=" + std::to_string(n), "kernelh"});
  }
  for (const long seq : {512L, 1024L}) {
    rows.push_back({"attention",
                    "batch=2,heads=8,seq=" + std::to_string(seq) +
                        ",head_dim=64",
                    "kernelh"});
  }
  return rows;
}

std::vector<RowSpec> hiprefRows() {
  std::vector<RowSpec> rows;
  for (const auto &[r, c] :
       std::vector<std::pair<long, long>>{{4096, 1024}, {16384, 4096}}) {
    const std::string shape = "rows=" + std::to_string(r) + ",cols=" +
                              std::to_string(c);
    rows.push_back({"softmax", shape, "hipref-naive"});
    rows.push_back({"softmax", shape, "hipref-vec"});
    rows.push_back({"layernorm", shape, "hipref-naive"});
    rows.push_back({"layernorm", shape, "hipref-vec"});
  }
  for (const long n : {100000L, 1000000L}) {
    rows.push_back({"scan", "n=" + std::to_string(n), "hipref-naive"});
    rows.push_back({"scan", "n=" + std::to_string(n), "hipref-vec"});
  }
  for (const long n : {1048576L, 4194304L}) {
    rows.push_back({"fused_gelu", "n=" + std::to_string(n), "hipref-naive"});
    rows.push_back({"fused_gelu", "n=" + std::to_string(n), "hipref-vec"});
  }
  rows.push_back({"gemm", "m=512,n=512,k=512", "hipref-naive"});
  rows.push_back({"gemm", "m=1024,n=1024,k=1024", "hipref-naive"});
  return rows;
}

std::vector<RowSpec> rooflineRows() {
  return {{"roofline_triad", "n=268435456", "roofline"},
          {"roofline_fma", "n=4194304", "roofline"}};
}

} // namespace

std::vector<RowSpec> suite_rows(const std::string &name) {
  if (name == "roofline") {
    return rooflineRows();
  }
  if (name == "kernelh") {
    return kernelhRows();
  }
  if (name == "rocblas") {
    return gemmRows("rocblas");
  }
  if (name == "hipref") {
    return hiprefRows();
  }
  if (name == "all") {
    std::vector<RowSpec> rows = rooflineRows();
    for (const RowSpec &row : kernelhRows()) {
      rows.push_back(row);
    }
    for (const RowSpec &row : gemmRows("rocblas")) {
      rows.push_back(row);
    }
    for (const RowSpec &row : hiprefRows()) {
      rows.push_back(row);
    }
    return rows;
  }
  return {};
}

std::vector<std::string> suite_names() {
  return {"all", "cuda-cross", "hipref", "kernelh", "rocblas", "roofline"};
}

} // namespace kernelh::bench
