// Small shared helpers for the harness (plan todo 35).
#ifndef KERNELH_BENCH_UTIL_HPP
#define KERNELH_BENCH_UTIL_HPP

#include <string>

namespace kernelh::bench {

// "YYYY-MM-DD HH:MM:SS" local time for the G32 timestamp column.
std::string now_timestamp();

} // namespace kernelh::bench

#endif // KERNELH_BENCH_UTIL_HPP
