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

// Registers the gemm template (lib/dsl/kernels/gemm.cpp, plan todo 26).
void registerGemmKernel();
// Registers the layernorm template (lib/dsl/kernels/layernorm.cpp), plan
// todo 28: two-pass row normalization over the last dim with iter_args
// reductions. Called by ensureBuiltinKernels.
void registerLayernormKernel();
// Registers the softmax template (lib/dsl/kernels/softmax.cpp, plan todo
// 27): row-wise safe online single-pass softmax (running max + running sum
// with rescale through loop_for iter_args). Called by ensureBuiltinKernels.
void registerSoftmaxKernel();

} // namespace kernelh::detail

#endif // KERNELH_DSL_INTERNAL_HPP
