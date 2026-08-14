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
// Registers the scan templates (lib/dsl/kernels/scan.cpp, plan todo 29):
// "scan" (pass-1 two-pass Blelloch upsweep+downsweep in static LDS, writing
// block totals) and "scan_totals" (pass-2: scan of the block totals + add
// offsets). Both are called by ensureBuiltinKernels.
void registerScanKernel();
// Registers the fused_gelu template (lib/dsl/kernels/fused_gelu.cpp, plan
// todo 30): an elementwise TWO-OP CHAIN (t = a*x + b; out = gelu(t)) staged
// through a rank-1 global intermediate so the fuse pass (todo 14) collapses
// the producer/consumer pair into ONE loop nest. Called by
// ensureBuiltinKernels.
void registerFusedGeluKernel();
// Registers the attention template (lib/dsl/kernels/attention.cpp, plan todo
// 31): NON-CAUSAL fused tiled attention with ONLINE softmax rescaling across
// K/V tiles (iter_args carry running max + denominator + accumulator),
// head_dim = 64, no matrix ops. Called by ensureBuiltinKernels.
void registerAttentionKernel();

} // namespace kernelh::detail

#endif // KERNELH_DSL_INTERNAL_HPP
