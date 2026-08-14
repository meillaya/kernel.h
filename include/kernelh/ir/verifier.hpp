// kernel.h "atom" IR - structural verifier (plan todo 9).
//
// The verifier reports EVERY violation found (not just the first) as a
// Diagnostic list. It enforces:
//   - SSA dominance within scopes (a use must be dominated by its definition:
//     same block and defined earlier, or an enclosing scope),
//   - per-op operand/result type rules (see ops.hpp for the documented
//     semantics of each op),
//   - barrier placement: barriers are allowed only in kernel-body blocks and
//     loop_for body blocks (atom.reduce is a leaf op with no region in v1,
//     so there is no reduce scope to forbid them in - the generic
//     allowed-block rule is implemented anyway so it extends when regions
//     grow),
//   - static shared total <= kDefaultLdsLimitBytes,
//   - loop_for iter_args <-> yield / body-arg consistency,
//   - kernel schedule-attr validity (tile positive, vector_width >= 1,
//     unroll >= 1),
//   - scalar const values fit their type (i32 must fit int32_t).

#ifndef KERNELH_IR_VERIFIER_HPP
#define KERNELH_IR_VERIFIER_HPP

#include "kernelh/ir/module.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace kernelh::ir {

// Maximum static shared-memory (LDS) bytes per workgroup. PLACEHOLDER until
// todo 10a switches the verifier to the device-limits table
// (kernelh::device_limits_for); the LDS check lives in ONE function
// (verifyKernelSharedLimit) so that swap is a single edit.
inline constexpr int64_t kDefaultLdsLimitBytes = 65536;

struct Diagnostic {
  enum class Severity : uint8_t { Error, Warning };
  Severity severity = Severity::Error;
  std::string message;

  std::string toString() const {
    return (severity == Severity::Error ? "error: " : "warning: ") + message;
  }
};

using Diagnostics = std::vector<Diagnostic>;

// Verifies the whole module; returns ALL violations (empty on success).
Diagnostics verifyModule(const Module &module);

// True when the module has no error diagnostics.
bool isModuleValid(const Module &module);

} // namespace kernelh::ir

#endif // KERNELH_IR_VERIFIER_HPP
