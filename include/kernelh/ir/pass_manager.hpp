// kernel.h "atom" IR - pass manager skeleton (plan todo 9).
//
// An ordered pass runner plus a registry mapping pass name -> pass function.
// No transformation passes are registered yet (todos 13+ register canonicalize
// etc.); the registry is empty so `--run-passes=anything` fails with a clear
// error listing the registered passes.
//
// Dump hooks: runPasses writes "<dumpDir>/00-input.atom" before the pipeline
// and "<dumpDir>/NN-<passname>.atom" (NN = pass index, zero-padded) after each
// successful pass, so later todos get per-pass IR dumps that re-parse.

#ifndef KERNELH_IR_PASS_MANAGER_HPP
#define KERNELH_IR_PASS_MANAGER_HPP

#include "kernelh/ir/module.hpp"

#include <expected>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace kernelh::ir {

// A pass transforms the Module in place; std::expected carries a pass failure
// message (passes report diagnostics through their own mechanisms when needed).
using PassFn = std::function<std::expected<void, std::string>(Module &)>;

class PassRegistry {
public:
  void registerPass(std::string name, PassFn fn);
  bool hasPass(std::string_view name) const;
  // Registered pass names, sorted (for deterministic listings).
  std::vector<std::string> passNames() const;

  // Runs a single named pass; unknown name -> error listing registered passes.
  std::expected<void, std::string> run(Module &module,
                                       std::string_view name) const;

private:
  std::map<std::string, PassFn, std::less<>> passes_;
};

// Runs a comma-separated pass pipeline ("a,b,c") in order through `registry`.
// An unknown name fails immediately (before/after partial execution) with an
// error listing the registered passes. When `dumpDir` is non-empty the input
// and post-pass states are dumped as described above.
std::expected<void, std::string> runPasses(Module &module,
                                           std::string_view pipeline,
                                           const PassRegistry &registry,
                                           std::string_view dumpDir = "");

} // namespace kernelh::ir

#endif // KERNELH_IR_PASS_MANAGER_HPP
