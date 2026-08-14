// Kernel template registry (plan todo 11): a process-wide name -> template
// map. Public API (registerKernel / registeredKernels) lives in
// include/kernelh/dsl/program.hpp; built-ins are registered lazily by
// ensureBuiltinKernels() on first emit() - no static-init order hazards.

#include "kernelh/dsl/program.hpp"

#include "kernels/kernels.hpp"

#include <string>
#include <utility>
#include <vector>

namespace kernelh {

namespace detail {

std::map<std::string, KernelTemplate> &kernelRegistry() {
  static std::map<std::string, KernelTemplate> registry;
  return registry;
}

void ensureBuiltinKernels() {
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  registerVecaddKernel();
  registerGemmKernel();
  registerLayernormKernel();
  registerSoftmaxKernel();
  registerScanKernel();
  registerFusedGeluKernel();
}

} // namespace detail

void registerKernel(std::string name, KernelTemplate fn) {
  detail::kernelRegistry()[std::move(name)] = std::move(fn);
}

std::vector<std::string> registeredKernels() {
  detail::ensureBuiltinKernels();
  std::vector<std::string> names;
  names.reserve(detail::kernelRegistry().size());
  for (const auto &[name, fn] : detail::kernelRegistry()) {
    (void)fn;
    names.push_back(name);
  }
  return names;
}

} // namespace kernelh
