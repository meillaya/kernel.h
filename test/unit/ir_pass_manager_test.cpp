// Pass-manager skeleton unit tests for the atom IR (plan todo 9).

#include "ir_corpus.hpp"

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using kernelh::ir::Module;
using kernelh::ir::PassFn;
using kernelh::ir::PassRegistry;
using kernelh::ir::runPasses;
using kernelh::ir::verifyModule;

using kernelh::test::buildMinimal;

namespace {

std::string readFileText(const std::string &path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

} // namespace

TEST_CASE("ir_pass_registry_empty_initially", "[unit][ir]") {
  const PassRegistry registry;
  CHECK(registry.passNames().empty());
  CHECK_FALSE(registry.hasPass("canonicalize"));
}

TEST_CASE("ir_pass_register_and_run", "[unit][ir]") {
  PassRegistry registry;
  bool ran = false;
  registry.registerPass("probe", [&ran](Module &) -> std::expected<void, std::string> {
    ran = true;
    return {};
  });
  CHECK(registry.hasPass("probe"));
  CHECK(registry.passNames() == std::vector<std::string>{"probe"});
  Module module = buildMinimal();
  CHECK(registry.run(module, "probe").has_value());
  CHECK(ran);
}

TEST_CASE("ir_pass_register_pass_mutates_module", "[unit][ir]") {
  PassRegistry registry;
  registry.registerPass(
      "rename", [](Module &m) -> std::expected<void, std::string> {
        if (m.kernels.empty()) {
          return std::unexpected("no kernels");
        }
        m.kernels[0]->name = "renamed";
        return {};
      });
  Module module = buildMinimal();
  CHECK(registry.run(module, "rename").has_value());
  CHECK(module.kernels[0]->name == "renamed");
  CHECK(verifyModule(module).empty());
}

TEST_CASE("ir_pass_unknown_pass_lists_registered", "[unit][ir]") {
  PassRegistry registry;
  registry.registerPass("alpha", [](Module &) -> std::expected<void, std::string> {
    return {};
  });
  registry.registerPass("beta", [](Module &) -> std::expected<void, std::string> {
    return {};
  });
  Module module = buildMinimal();
  const auto result = registry.run(module, "nonexistent");
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().find("unknown pass 'nonexistent'") != std::string::npos);
  CHECK(result.error().find("registered passes") != std::string::npos);
  CHECK(result.error().find("alpha") != std::string::npos);
  CHECK(result.error().find("beta") != std::string::npos);
}

TEST_CASE("ir_pass_unknown_pass_empty_registry", "[unit][ir]") {
  const PassRegistry registry;
  Module module = buildMinimal();
  const auto result = runPasses(module, "canonicalize", registry);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().find("registered passes") != std::string::npos);
  CHECK(result.error().find("(none)") != std::string::npos);
}

TEST_CASE("ir_pass_run_passes_order", "[unit][ir]") {
  PassRegistry registry;
  std::string log;
  registry.registerPass("first", [&log](Module &) -> std::expected<void, std::string> {
    log += "first;";
    return {};
  });
  registry.registerPass("second", [&log](Module &) -> std::expected<void, std::string> {
    log += "second;";
    return {};
  });
  registry.registerPass("third", [&log](Module &) -> std::expected<void, std::string> {
    log += "third;";
    return {};
  });
  Module module = buildMinimal();
  CHECK(runPasses(module, "third, first,second", registry).has_value());
  CHECK(log == "third;first;second;");
}

TEST_CASE("ir_pass_stops_on_failure", "[unit][ir]") {
  PassRegistry registry;
  registry.registerPass("ok", [](Module &) -> std::expected<void, std::string> {
    return {};
  });
  registry.registerPass("boom", [](Module &) -> std::expected<void, std::string> {
    return std::unexpected("pass blew up");
  });
  registry.registerPass("never", [](Module &) -> std::expected<void, std::string> {
    return std::unexpected("must not run");
  });
  Module module = buildMinimal();
  const auto result = runPasses(module, "ok,boom,never", registry);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().find("pass blew up") != std::string::npos);
}

TEST_CASE("ir_pass_empty_pipeline_runs_nothing", "[unit][ir]") {
  const PassRegistry registry;
  Module module = buildMinimal();
  CHECK(runPasses(module, "", registry).has_value());
  CHECK(runPasses(module, " , , ", registry).has_value());
}

TEST_CASE("ir_pass_dump_dir_writes_files", "[unit][ir]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "kernelh_ir_dump_test";
  fs::remove_all(dir);

  PassRegistry registry;
  registry.registerPass("zeroer", [](Module &) -> std::expected<void, std::string> {
    return {};
  });
  Module module = buildMinimal();
  const auto result = runPasses(module, "zeroer", registry, dir.string());
  REQUIRE(result.has_value());

  const std::string input = readFileText((dir / "00-input.atom").string());
  const std::string after = readFileText((dir / "01-zeroer.atom").string());
  CHECK(input == kernelh::ir::printModule(buildMinimal()));
  CHECK(after == input);
  fs::remove_all(dir);
}

TEST_CASE("ir_pass_dump_dir_two_passes_numbering", "[unit][ir]") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "kernelh_ir_dump_test2";
  fs::remove_all(dir);

  PassRegistry registry;
  registry.registerPass("a", [](Module &) -> std::expected<void, std::string> {
    return {};
  });
  registry.registerPass("b", [](Module &) -> std::expected<void, std::string> {
    return {};
  });
  Module module = buildMinimal();
  CHECK(runPasses(module, "a,b", registry, dir.string()).has_value());
  CHECK(fs::exists(dir / "00-input.atom"));
  CHECK(fs::exists(dir / "01-a.atom"));
  CHECK(fs::exists(dir / "02-b.atom"));
  fs::remove_all(dir);
}
