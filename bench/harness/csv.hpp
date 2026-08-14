// G32 results.csv writer (plan todo 35).
//
// Columns: kernel,shape,backend,ms_median,gflops_or_gbps,timestamp,device.
// RFC 4180-ish quoting: any field containing a comma, quote, or newline is
// wrapped in double quotes with embedded quotes doubled. The header is
// written exactly once; rows append after it.

#ifndef KERNELH_BENCH_CSV_HPP
#define KERNELH_BENCH_CSV_HPP

#include "bench_row.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace kernelh::bench {

// The exact G32 column order (a change here breaks todo 36/38 consumers).
extern const char *const kCsvHeader;

// Escapes one CSV field (double-quoted when needed).
std::string csv_escape(const std::string &field);

// Serializes one row to a single CSV line (no trailing newline).
std::string csv_line(const BenchRow &row);

// Writes the header + all rows to `out`. Returns false on write failure.
bool write_results_csv(std::ostream &out, const std::vector<BenchRow> &rows);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_CSV_HPP
