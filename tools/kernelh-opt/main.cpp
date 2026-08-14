// kernelh-opt — mlir-opt-style driver for kernel.h.
//
// Two modes:
//
// 1. Default (no atom flags): delegate to MlirOptMain exactly as before. The
//    atom flags below are scanned MANUALLY out of argv first so they coexist
//    with MlirOptMain's own llvm::cl parsing (which never sees them).
//
// 2. Atom mode (any of the atom flags present): operate on the custom
//    structured-SSA "atom" IR (plan todo 9):
//      --load-atom <file>      parse <file> + structural verify (default)
//      --parse-atom            read ".atom" text from stdin
//      --verify                explicit standalone structural verification
//      --print-atom            print the module as ".atom" text to stdout
//      --emit-mlir             lower the module to a device-only gpu.module
//                              (todo 21 emitter) and print it as MLIR text
//      --run-passes=<a,b,c>    run the pass pipeline in order
//      --run-default-pipeline  run the pinned canonical pipeline order
//      --dump-dir <dir>        per-pass dumps: 00-input.atom, NN-<pass>.atom
//      --atom-target <t>       device limits table target for verification
//                              (gfx1101|sm_80; default gfx1101)
//    Compose freely, e.g.:
//      kernelh-opt --load-atom f.atom --run-passes=... --print-atom

#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/emitter.hpp"
#include "kernelh/Dialect/Atom/IR/AtomDialect.h"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"
#include "kernelh/passes/pipeline.hpp"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct AtomOptions {
  bool loadAtom = false;
  bool parseAtom = false;
  bool verify = false;
  bool printAtom = false;
  bool emitMlir = false;
  bool hasRunPasses = false;
  bool runDefaultPipeline = false;
  bool hasDumpDir = false;
  std::string inputFile;
  std::string passPipeline;
  std::string dumpDir;
  std::string atomTarget = "gfx1101";
};

enum class ScanResult { NoAtom, AtomMode, UsageError };

bool isAtomFlag(const std::string &arg) {
  return arg == "--load-atom" || arg == "--parse-atom" || arg == "--verify" ||
         arg == "--print-atom" || arg == "--emit-mlir" ||
         arg == "--run-default-pipeline" ||
         arg.rfind("--run-passes", 0) == 0 ||
         arg.rfind("--dump-dir", 0) == 0 ||
         arg.rfind("--atom-target", 0) == 0;
}

void printUsage() {
  std::cerr << "kernelh-opt atom flags:\n"
            << "  --load-atom <file>     parse + structural verify (default)\n"
            << "  --parse-atom           read .atom text from stdin\n"
            << "  --verify               explicit structural verification\n"
            << "  --print-atom           print the module as .atom text\n"
            << "  --emit-mlir            lower to a device-only gpu.module and\n"
            << "                         print it as MLIR text (todo 21)\n"
            << "  --run-passes=<a,b,c>   run the pass pipeline in order\n"
            << "  --run-default-pipeline run the pinned canonical pipeline\n"
            << "  --dump-dir <dir>       per-pass dumps (needs --run-passes)\n"
            << "  --atom-target <t>      device limits target for the verify\n"
            << "                         pass (gfx1101|sm_80; default gfx1101)\n"
            << "Any other flag runs the normal mlir-opt mode.\n";
}

// Extracts the value of `--name=<value>` or `--name <value>` into `out`.
// Returns true on success; on a missing value prints an error and returns
// false.
bool takeValue(int argc, char **argv, int &i, const std::string &arg,
               const char *flag, std::string &out) {
  const std::string prefix = std::string(flag) + "=";
  if (arg.rfind(prefix, 0) == 0) {
    out = arg.substr(prefix.size());
    return true;
  }
  if (arg == flag) {
    if (i + 1 >= argc) {
      std::cerr << "error: " << flag << " requires a value\n";
      return false;
    }
    out = argv[++i];
    return true;
  }
  return false;
}

// Scans argv for the atom flags. NoAtom: no atom flag present (mlir mode).
// AtomMode: options filled in and valid. UsageError: malformed atom flags
// (message already printed).
ScanResult scanAtomOptions(int argc, char **argv, AtomOptions &options) {
  bool anyAtomFlag = false;
  for (int i = 1; i < argc; ++i) {
    if (isAtomFlag(argv[i])) {
      anyAtomFlag = true;
      break;
    }
  }
  if (!anyAtomFlag) {
    return ScanResult::NoAtom;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--load-atom") {
      if (!takeValue(argc, argv, i, arg, "--load-atom", options.inputFile)) {
        return ScanResult::UsageError;
      }
      options.loadAtom = true;
    } else if (arg == "--parse-atom") {
      options.parseAtom = true;
    } else if (arg == "--verify") {
      options.verify = true;
    } else if (arg == "--print-atom") {
      options.printAtom = true;
    } else if (arg == "--emit-mlir") {
      options.emitMlir = true;
    } else if (arg == "--run-passes" || arg.rfind("--run-passes=", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--run-passes",
                     options.passPipeline)) {
        return ScanResult::UsageError;
      }
      options.hasRunPasses = true;
    } else if (arg == "--run-default-pipeline") {
      options.runDefaultPipeline = true;
    } else if (arg == "--dump-dir" || arg.rfind("--dump-dir=", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--dump-dir", options.dumpDir)) {
        return ScanResult::UsageError;
      }
      options.hasDumpDir = true;
    } else if (arg == "--atom-target" || arg.rfind("--atom-target=", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--atom-target", options.atomTarget)) {
        return ScanResult::UsageError;
      }
    } else if (arg == "--help") {
      std::cout << "kernelh-opt: MLIR optimizer driver for kernel.h\n";
      printUsage();
      std::exit(0);
    } else {
      std::cerr << "error: unknown flag '" << arg << "' in atom mode\n";
      return ScanResult::UsageError;
    }
  }

  if (options.loadAtom && options.parseAtom) {
    std::cerr << "error: --load-atom and --parse-atom are conflicting input "
                 "sources\n";
    return ScanResult::UsageError;
  }
  if (!options.loadAtom && !options.parseAtom) {
    std::cerr << "error: no input source; use --load-atom <file> or "
                 "--parse-atom (stdin)\n";
    return ScanResult::UsageError;
  }
  if (options.hasRunPasses && options.runDefaultPipeline) {
    std::cerr << "error: --run-passes and --run-default-pipeline are "
                 "conflicting pipeline selectors\n";
    return ScanResult::UsageError;
  }
  if (options.hasDumpDir && !options.hasRunPasses &&
      !options.runDefaultPipeline) {
    std::cerr << "error: --dump-dir requires --run-passes or "
                 "--run-default-pipeline\n";
    return ScanResult::UsageError;
  }
  return ScanResult::AtomMode;
}

// Reads an entire stream into a string.
std::string readAll(std::istream &in) {
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int runAtomMode(const AtomOptions &options) {
  std::string text;
  if (options.loadAtom) {
    std::ifstream in(options.inputFile);
    if (!in) {
      std::cerr << "error: cannot open input file '" << options.inputFile
                << "'\n";
      return 1;
    }
    text = readAll(in);
  } else {
    text = readAll(std::cin);
  }

  auto parsed = kernelh::ir::parseModule(text);
  if (!parsed) {
    std::cerr << "error: " << parsed.error() << "\n";
    return 1;
  }
  kernelh::ir::Module &module = *parsed;

  // --load-atom implies structural verification; --verify forces it too. The
  // shared-memory budget comes from the device-limits table for --atom-target
  // (unknown target -> clean error from device_limits_for, exit 1).
  if (options.loadAtom || options.verify) {
    kernelh::DeviceLimits limits;
    try {
      limits = kernelh::device_limits_for(options.atomTarget.c_str());
    } catch (const std::invalid_argument &e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
    }
    const kernelh::ir::Diagnostics diags =
        kernelh::ir::verifyModule(module, limits);
    bool failed = false;
    for (const kernelh::ir::Diagnostic &d : diags) {
      std::cerr << d.toString() << "\n";
      failed |= (d.severity == kernelh::ir::Diagnostic::Severity::Error);
    }
    if (failed) {
      return 1;
    }
  }

  if (options.hasRunPasses) {
    // Global registry: pass .cpp files self-register via static initializers
    // (lib/passes/CMakeLists.txt contract), so no registry is built by hand.
    const auto result = kernelh::ir::runPasses(
        module, options.passPipeline,
        options.hasDumpDir ? std::string_view(options.dumpDir)
                           : std::string_view());
    if (!result) {
      std::cerr << "error: " << result.error() << "\n";
      return 1;
    }
  }

  if (options.runDefaultPipeline) {
    // The pinned canonical order (Metis G3), run via the same global registry
    // (lib/passes/pipeline.cpp). Same dump behavior as --run-passes.
    const auto result = kernelh::passes::runDefaultPipeline(
        module, options.hasDumpDir ? std::string_view(options.dumpDir)
                                   : std::string_view());
    if (!result) {
      std::cerr << "error: " << result.error() << "\n";
      return 1;
    }
  }

  if (options.printAtom) {
    std::cout << kernelh::ir::printModule(module);
  }

  if (options.emitMlir) {
    mlir::MLIRContext ctx;
    auto gpu = kernelh::compiler::emitGpuModule(module, ctx);
    if (!gpu) {
      std::cerr << "error: " << gpu.error() << "\n";
      return 1;
    }
    (**gpu).print(llvm::outs());
    llvm::outs() << "\n";
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  AtomOptions atomOptions;
  switch (scanAtomOptions(argc, argv, atomOptions)) {
  case ScanResult::UsageError:
    return 1;
  case ScanResult::AtomMode:
    return runAtomMode(atomOptions);
  case ScanResult::NoAtom:
    break;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  // The "atom" MLIR dialect (plan todo 20): the bridge between the custom
  // atom IR and the GPU dialect.
  mlir::kernelh::atom::registerAtomDialect(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "kernelh-opt: MLIR optimizer driver for kernel.h", registry));
}
