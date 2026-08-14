// Pass manager skeleton for the atom IR (plan todo 9).

#include "kernelh/ir/pass_manager.hpp"

#include "kernelh/ir/printer.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kernelh::ir {

// ---------------------------------------------------------------------------
// Global registry (plan todo 13). Function-local static (Meyers singleton):
// pass .cpp files self-register from file-scope static initializers, and the
// map is constructed on first use regardless of static-init order.
// ---------------------------------------------------------------------------

namespace {

std::map<std::string, PassFn, std::less<>> &globalPassRegistry() {
  static std::map<std::string, PassFn, std::less<>> registry;
  return registry;
}

} // namespace

void registerPass(std::string name, PassFn fn) {
  globalPassRegistry()[std::move(name)] = std::move(fn);
}

std::vector<std::string> registeredPasses() {
  std::vector<std::string> names;
  names.reserve(globalPassRegistry().size());
  for (const auto &[name, unused] : globalPassRegistry()) {
    (void)unused;
    names.push_back(name);
  }
  return names;
}

std::expected<void, std::string> runPasses(Module &module,
                                           std::string_view pipeline,
                                           std::string_view dumpDir) {
  // Snapshot the global registry into a PassRegistry so the existing
  // pipeline-splitting, unknown-pass errors, and dump hooks are reused as-is.
  PassRegistry registry;
  for (const auto &[name, fn] : globalPassRegistry()) {
    registry.registerPass(name, fn);
  }
  return runPasses(module, pipeline, registry, dumpDir);
}

void PassRegistry::registerPass(std::string name, PassFn fn) {
  passes_[std::move(name)] = std::move(fn);
}

bool PassRegistry::hasPass(std::string_view name) const {
  return passes_.find(name) != passes_.end();
}

std::vector<std::string> PassRegistry::passNames() const {
  std::vector<std::string> names;
  names.reserve(passes_.size());
  for (const auto &[name, unused] : passes_) {
    (void)unused;
    names.push_back(name);
  }
  return names;
}

std::expected<void, std::string> PassRegistry::run(Module &module,
                                                   std::string_view name) const {
  const auto found = passes_.find(name);
  if (found == passes_.end()) {
    std::string message = "unknown pass '" + std::string(name) +
                          "'; registered passes: ";
    const std::vector<std::string> names = passNames();
    if (names.empty()) {
      message += "(none)";
    } else {
      for (size_t i = 0; i < names.size(); ++i) {
        if (i != 0) {
          message += ", ";
        }
        message += names[i];
      }
    }
    return std::unexpected(std::move(message));
  }
  return found->second(module);
}

namespace {

// Splits "a, b,c" into {a, b, c}; empty segments are dropped.
std::vector<std::string> splitPipeline(std::string_view pipeline) {
  std::vector<std::string> names;
  size_t start = 0;
  while (start <= pipeline.size()) {
    const size_t comma = pipeline.find(',', start);
    const size_t end = comma == std::string_view::npos ? pipeline.size() : comma;
    // Trim spaces.
    size_t begin = start;
    while (begin < end && pipeline[begin] == ' ') {
      ++begin;
    }
    size_t stop = end;
    while (stop > begin && pipeline[stop - 1] == ' ') {
      --stop;
    }
    if (begin < stop) {
      names.emplace_back(pipeline.substr(begin, stop - begin));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  return names;
}

void writeDump(std::string_view dumpDir, const std::string &filename,
               const Module &module) {
  std::filesystem::path dir(dumpDir);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::ofstream out((dir / filename).string());
  out << printModule(module);
}

} // namespace

std::expected<void, std::string> runPasses(Module &module,
                                           std::string_view pipeline,
                                           const PassRegistry &registry,
                                           std::string_view dumpDir) {
  const std::vector<std::string> names = splitPipeline(pipeline);

  if (!dumpDir.empty()) {
    writeDump(dumpDir, "00-input.atom", module);
  }

  char counter[16];
  for (size_t i = 0; i < names.size(); ++i) {
    const auto result = registry.run(module, names[i]);
    if (!result) {
      return result; // error already names the pass and lists registered ones
    }
    if (!dumpDir.empty()) {
      std::snprintf(counter, sizeof(counter), "%02zu", i + 1);
      writeDump(dumpDir,
                std::string(counter) + "-" + names[i] + ".atom", module);
    }
  }
  return {};
}

} // namespace kernelh::ir
