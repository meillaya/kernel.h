// ProgramBuilder implementation (plan todo 11): emit() selects the named
// kernel template from the registry, runs it over a fresh kernel body with an
// ir::IRBuilder, and returns the built module + its ".atom" text.

#include "kernelh/dsl/program.hpp"

#include "kernelh/ir/printer.hpp"

#include "kernels/kernels.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace kernelh {

EmittedKernel ProgramBuilder::emit() {
  detail::ensureBuiltinKernels();

  if (kernelName_.empty()) {
    throw std::runtime_error(
        "no kernel template selected: call .kernel(name) before .emit()");
  }

  auto &registry = detail::kernelRegistry();
  const auto found = registry.find(kernelName_);
  if (found == registry.end()) {
    std::string names;
    for (const std::string &name : registeredKernels()) {
      if (!names.empty()) {
        names += ", ";
      }
      names += name;
    }
    throw std::runtime_error("unknown kernel template '" + kernelName_ +
                             "'; registered kernels: " + names);
  }

  auto kernel = std::make_unique<ir::Kernel>(kernelName_);
  kernel->schedule = schedule_;
  kernel->shape = shape_;

  ir::Block &body = kernel->body;
  ir::IRBuilder builder(body);
  found->second(builder, kernel->schedule, kernel->shape);

  EmittedKernel out;
  out.module.kernels.push_back(std::move(kernel));
  out.atomText = ir::printModule(out.module);
  return out;
}

} // namespace kernelh
