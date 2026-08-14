// Shape-text helpers for the G32 shape column (plan todo 35).
//
// The shape column carries "k=v,k=v,..." text - the same convention as the
// tune CSV (m=512,n=512,k=512). Parsing is strict: a malformed segment
// (empty key/value, non-integer value) yields an empty map so callers can
// reject the row.

#ifndef KERNELH_BENCH_SHAPE_TEXT_HPP
#define KERNELH_BENCH_SHAPE_TEXT_HPP

#include <cstdint>
#include <map>
#include <string>

namespace kernelh::bench {

using ShapeMap = std::map<std::string, int64_t>;

// "m=512,n=512,k=512" (std::map -> sorted keys, deterministic).
std::string shape_to_text(const ShapeMap &shape);

// Empty map on any malformed segment.
ShapeMap parse_shape_text(const std::string &text);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_SHAPE_TEXT_HPP
