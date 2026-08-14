// Loader for hipcc-compiled reference code objects (plan todo 35).
//
// bench/refs/*.hip are compiled with `hipcc --genco --offload-arch=gfx1101`
// into COF fat-binary code objects (see bench/refs/CMakeLists.txt). The build
// dir is baked into the harness via the KERNELH_REFS_DIR compile definition;
// hipModuleLoadData accepts the COF container directly (verified in the todo-6
// fixture recipe).

#ifndef KERNELH_BENCH_REFS_LOADER_HPP
#define KERNELH_BENCH_REFS_LOADER_HPP

#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/gpu_module.hpp"

#include <memory>
#include <string>

namespace kernelh::bench {

// Absolute path to a compiled reference artifact, or "" when KERNELH_REFS_DIR
// is not defined at build time (degraded env without hipcc).
std::string refs_artifact_path(const std::string &filename);

// Loads the artifact (COF code object) into a new GpuModule. The module is
// returned by value so the caller keeps it alive as long as the resolved
// kernel functions are used (the function handle dies with the module).
kernelh::runtime::Result<kernelh::runtime::GpuModule> load_refs_module(
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &filename);

} // namespace kernelh::bench

#endif // KERNELH_BENCH_REFS_LOADER_HPP
