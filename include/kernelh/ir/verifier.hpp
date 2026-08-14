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
//   - static shared total <= the caller-supplied device LDS budget
//     (DeviceLimits::lds_bytes_per_workgroup; todo 10a switchover from the
//     placeholder LDS constant),
//   - loop_for iter_args <-> yield / body-arg consistency,
//   - kernel schedule-attr validity (tile positive, vector_width >= 1,
//     unroll >= 1),
//   - scalar const values fit their type (i32 must fit int32_t).
//
// The verifier is DEVICE-AGNOSTIC: the shared-memory budget is supplied by the
// CALLER as a kernelh::DeviceLimits (see kernelh/compiler/device_limits.hpp),
// so lib/ir itself stays self-contained and does NOT link kernelh_compiler.
// The one-argument verifyModule(module) convenience overload (which defaults
// to kernelh::device_limits_for("gfx1101")) is defined inline here, so the
// device_limits_for symbol is pulled in only by binaries that call it - and
// those must link kernelh_compiler at final link time.

#ifndef KERNELH_IR_VERIFIER_HPP
#define KERNELH_IR_VERIFIER_HPP

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/ir/module.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace kernelh::ir {

struct Diagnostic {
  enum class Severity : uint8_t { Error, Warning };
  Severity severity = Severity::Error;
  std::string message;

  std::string toString() const {
    return (severity == Severity::Error ? "error: " : "warning: ") + message;
  }
};

using Diagnostics = std::vector<Diagnostic>;

// Verifies the whole module against the given device's limits; returns ALL
// violations (empty on success). The shared-memory budget comes from
// `limits.lds_bytes_per_workgroup` (todo 10a switchover).
Diagnostics verifyModule(const Module &module,
                         const kernelh::DeviceLimits &limits);

// Convenience overload: verifies against the default local device (gfx1101).
// Defined inline so lib/ir itself never references device_limits_for; callers
// that use this overload must link kernelh_compiler.
inline Diagnostics verifyModule(const Module &module) {
  return verifyModule(module, kernelh::device_limits_for("gfx1101"));
}

// True when the module has no error diagnostics.
inline bool isModuleValid(const Module &module) {
  for (const Diagnostic &d : verifyModule(module)) {
    if (d.severity == Diagnostic::Severity::Error) {
      return false;
    }
  }
  return true;
}

} // namespace kernelh::ir

#endif // KERNELH_IR_VERIFIER_HPP
