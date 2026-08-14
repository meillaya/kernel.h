// kernel.h "atom" IR - parser (plan todo 9).
//
// Hand-written recursive descent with a lexer tracking line/column. The
// parser accepts exactly what the printer emits (round-trip guarantee:
// print(parse(print(M))) == print(M)); malformed input yields a clear error
// naming the line and column.
//
// Public API returns std::expected<Module, std::string> - parse errors are
// reported through the error value, never through exceptions (plan F2
// convention).

#ifndef KERNELH_IR_PARSER_HPP
#define KERNELH_IR_PARSER_HPP

#include "kernelh/ir/module.hpp"

#include <expected>
#include <string>
#include <string_view>

namespace kernelh::ir {

// Parses ".atom" text into a Module. On failure returns an error message that
// includes the line/column of the offending token.
std::expected<Module, std::string> parseModule(std::string_view text);

} // namespace kernelh::ir

#endif // KERNELH_IR_PARSER_HPP
