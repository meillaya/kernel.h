#include "csv.hpp"

#include <iomanip>
#include <ostream>
#include <sstream>

namespace kernelh::bench {

const char *const kCsvHeader =
    "kernel,shape,backend,ms_median,gflops_or_gbps,timestamp,device";

std::string csv_escape(const std::string &field) {
  const bool needsQuotes =
      field.find(',') != std::string::npos ||
      field.find('"') != std::string::npos ||
      field.find('\n') != std::string::npos ||
      field.find('\r') != std::string::npos;
  if (!needsQuotes) {
    return field;
  }
  std::string out;
  out.reserve(field.size() + 2);
  out.push_back('"');
  for (const char c : field) {
    if (c == '"') {
      out.push_back('"');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

// Compact, deterministic number formatting: up to 10 significant digits,
// no trailing ".000000000" noise. NaN/inf render as "nan"/"inf" (the
// adversarial CSV assertion in the plan rejects zero/negative real rows,
// and the CSV must remain parseable by python csv.DictReader).
std::string formatDouble(double v) {
  std::ostringstream os;
  os << std::setprecision(10) << v;
  return os.str();
}

std::string csv_line(const BenchRow &row) {
  return csv_escape(row.kernel) + "," + csv_escape(row.shape) + "," +
         csv_escape(row.backend) + "," + formatDouble(row.ms_median) + "," +
         formatDouble(row.gflops_or_gbps) + "," + csv_escape(row.timestamp) +
         "," + csv_escape(row.device);
}

bool write_results_csv(std::ostream &out, const std::vector<BenchRow> &rows) {
  out << kCsvHeader << '\n';
  for (const BenchRow &row : rows) {
    out << csv_line(row) << '\n';
  }
  return out.good();
}

} // namespace kernelh::bench
