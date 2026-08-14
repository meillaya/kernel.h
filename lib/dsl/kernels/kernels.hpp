// Internal DSL header shared by lib/dsl translation units only. Not
// installed; kernel template bodies (lib/dsl/kernels/*.cpp) include it.
#ifndef KERNELH_DSL_INTERNAL_HPP
#define KERNELH_DSL_INTERNAL_HPP

#include "kernelh/dsl/program.hpp"

#include <map>
#include <string>

namespace kernelh::detail {

// The registry map itself (one per process). Access via registerKernel/
// registeredKernels on the public side; emit() reads it directly.
std::map<std::string, KernelTemplate> &kernelRegistry();

// Idempotent: registers every built-in kernel template on first call.
// ProgramBuilder::emit() invokes this before lookup so built-ins are always
// present without static-init ordering hazards.
void ensureBuiltinKernels();

// Registers the vecadd template (lib/dsl/kernels/vecadd.cpp). Called by
// ensureBuiltinKernels; further templates (todos 25-31) get their own
// register* functions wired in there.
void registerVecaddKernel();

} // namespace kernelh::detail

#endif // KERNELH_DSL_INTERNAL_HPP
