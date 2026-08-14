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
//
//    With --compile-target=<t> + --out <dir>, the atom chain is extended with
//    the in-process vendor backend (plan todo 22):
//      kernelh-opt --load-atom vecadd.atom --emit-mlir \
//                  --compile-target=amdgcn-gfx1101 --out dir
//    writes dir/vecadd.isa + dir/vecadd.hsaco (amdgcn) or
//    dir/vecadd.ptx + dir/vecadd.cubin (nvptx) and prints the text artifact.
//
// 3. Mlir-compile mode: --compile-target present WITHOUT atom flags - parse
//    the positional .mlir input (a hand-written gpu.module fixture) and run
//    the same vendor pipeline:
//      kernelh-opt --compile-target=amdgcn-gfx1101 fixture.mlir --out dir

#include "kernelh/compiler/backends.hpp"
#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/emitter.hpp"
#include "kernelh/Dialect/Atom/IR/AtomDialect.h"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"
#include "kernelh/passes/pipeline.hpp"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/FileSystem.h"
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
  std::string compileTarget;
  std::string outDir;
  bool hasOutDir = false;
};

struct CompileOptions {
  bool enabled = false;
  bool hasOutDir = false;
  std::string compileTarget;
  std::string outDir;
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
            << "  --compile-target <t>   vendor backend (todo 22):\n"
            << "                         nvptx-sm_80 -> ptx+cubin,\n"
            << "                         amdgcn-gfx1101 -> isa+hsaco\n"
            << "                         (requires --out <dir>)\n"
            << "  --out <dir>            artifact output dir for --compile-target\n"
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

// Lenient pre-scan for the backend-compile flags (todo 22), which are shared
// by the atom chain and the mlir-input compile mode. Never reports usage
// errors - the mode scanners own those.
void scanCompileFlags(int argc, char **argv, CompileOptions &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--compile-target", 0) == 0) {
      takeValue(argc, argv, i, arg, "--compile-target",
                options.compileTarget);
      options.enabled = true;
    } else if (arg.rfind("--out", 0) == 0) {
      takeValue(argc, argv, i, arg, "--out", options.outDir);
      options.hasOutDir = true;
    }
  }
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
    } else if (arg.rfind("--compile-target", 0) == 0) {
      // Backend-compile flags (todo 22) are valid in atom mode; the values
      // are collected by the lenient scanCompileFlags pre-scan.
      if (!takeValue(argc, argv, i, arg, "--compile-target",
                     options.compileTarget)) {
        return ScanResult::UsageError;
      }
    } else if (arg.rfind("--out", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--out", options.outDir)) {
        return ScanResult::UsageError;
      }
      options.hasOutDir = true;
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

// Writes `bytes` to `path` (binary). Returns false and prints the error on
// failure.
bool writeBlobFile(const std::string &path, const std::vector<char> &bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "error: cannot write '" << path << "'\n";
    return false;
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

bool writeTextFile(const std::string &path, const std::string &text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "error: cannot write '" << path << "'\n";
    return false;
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  return static_cast<bool>(out);
}

// Artifact stem = the first gpu.func's symbol name ("vecadd", "xchg_kernel").
std::string artifactStem(mlir::ModuleOp module) {
  std::string stem;
  module.walk([&](mlir::gpu::GPUFuncOp func) {
    if (stem.empty()) {
      stem = func.getName().str();
    }
  });
  return stem.empty() ? "kernel" : stem;
}

// Runs the todo-22 vendor pipeline on the module (a builtin.module containing
// a gpu.module) and writes the artifacts into --out. The module is mutated by
// each compile (gpu.module -> gpu.binary), so every artifact compiles from a
// fresh clone. Prints the text artifact (ptx/isa) to stdout; binary artifacts
// report their size.
int compileAndWrite(mlir::OwningOpRef<mlir::ModuleOp> &module,
                    const CompileOptions &options) {
  auto spec = kernelh::compiler::parseTargetSpec(options.compileTarget);
  if (!spec) {
    std::cerr << "error: " << spec.error() << "\n";
    return 1;
  }

  std::error_code ec;
  if (ec = llvm::sys::fs::create_directories(options.outDir); ec) {
    std::cerr << "error: cannot create output dir '" << options.outDir
              << "': " << ec.message() << "\n";
    return 1;
  }

  const std::string stem = artifactStem(*module);
  const auto cloneOf = [](mlir::OwningOpRef<mlir::ModuleOp> &src) {
    return mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::cast<mlir::ModuleOp>(src->clone()));
  };

  if (spec->isNVPTX()) {
    {
      auto m = cloneOf(module);
      auto ptx = kernelh::compiler::compileToText(m, *spec, "ptx");
      if (!ptx) {
        std::cerr << "error: " << ptx.error() << "\n";
        return 1;
      }
      if (!writeTextFile(options.outDir + "/" + stem + ".ptx", *ptx)) {
        return 1;
      }
      llvm::outs() << *ptx;
      llvm::outs().flush();
    }
    {
      auto m = cloneOf(module);
      auto cubin = kernelh::compiler::compileToBlob(m, *spec, "cubin");
      if (!cubin) {
        std::cerr << "error: " << cubin.error() << "\n";
        return 1;
      }
      if (!writeBlobFile(options.outDir + "/" + stem + ".cubin", *cubin)) {
        return 1;
      }
      llvm::outs() << "wrote " << stem << ".cubin (" << cubin->size()
                   << " bytes)\n";
    }
  } else {
    {
      auto m = cloneOf(module);
      auto isa = kernelh::compiler::compileToText(m, *spec, "isa");
      if (!isa) {
        std::cerr << "error: " << isa.error() << "\n";
        return 1;
      }
      if (!writeTextFile(options.outDir + "/" + stem + ".isa", *isa)) {
        return 1;
      }
      llvm::outs() << *isa;
      llvm::outs().flush();
    }
    {
      auto m = cloneOf(module);
      auto hsaco = kernelh::compiler::compileToBlob(m, *spec, "hsaco");
      if (!hsaco) {
        std::cerr << "error: " << hsaco.error() << "\n";
        return 1;
      }
      if (!writeBlobFile(options.outDir + "/" + stem + ".hsaco", *hsaco)) {
        return 1;
      }
      llvm::outs() << "wrote " << stem << ".hsaco (" << hsaco->size()
                   << " bytes)\n";
    }
  }
  return 0;
}

// Mlir-compile mode: --compile-target without atom flags. The input is the
// positional .mlir file (or stdin), a hand-written gpu.module fixture.
int runMlirCompileMode(int argc, char **argv, const CompileOptions &options) {
  std::string input;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" || arg == "-o" || arg == "--compile-target") {
      ++i;
      continue;
    }
    if (arg.rfind("--out=", 0) == 0 || arg.rfind("--compile-target=", 0) == 0) {
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      continue;
    }
    input = arg;
    break;
  }

  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (input.empty()) {
    module = mlir::parseSourceString<mlir::ModuleOp>(readAll(std::cin), &ctx);
    if (!module) {
      std::cerr << "error: failed to parse mlir input from stdin\n";
      return 1;
    }
  } else {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    ctx.appendDialectRegistry(registry);
    module = mlir::parseSourceFile<mlir::ModuleOp>(input, &ctx);
    if (!module) {
      std::cerr << "error: failed to parse '" << input << "'\n";
      return 1;
    }
  }
  return compileAndWrite(module, options);
}

int runAtomMode(const AtomOptions &options, const CompileOptions &compile) {
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

  if (options.emitMlir || compile.enabled) {
    mlir::MLIRContext ctx;
    if (compile.enabled) {
      // The vendor pipelines create LLVM/NVVM/ROCDL ops during lowering, so
      // the compile path registers the full MLIR dialect + extension set (the
      // atom dialect is appended by the emitter itself).
      mlir::DialectRegistry registry;
      mlir::registerAllDialects(registry);
      mlir::registerAllExtensions(registry);
      ctx.appendDialectRegistry(registry);
      ctx.loadAllAvailableDialects();
    }
    auto gpu = kernelh::compiler::emitGpuModule(module, ctx);
    if (!gpu) {
      std::cerr << "error: " << gpu.error() << "\n";
      return 1;
    }
    if (options.emitMlir) {
      (**gpu).print(llvm::outs());
      llvm::outs() << "\n";
    }
    if (compile.enabled) {
      return compileAndWrite(*gpu, compile);
    }
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  CompileOptions compileOptions;
  scanCompileFlags(argc, argv, compileOptions);

  AtomOptions atomOptions;
  switch (scanAtomOptions(argc, argv, atomOptions)) {
  case ScanResult::UsageError:
    return 1;
  case ScanResult::AtomMode:
    if (compileOptions.enabled && !compileOptions.hasOutDir) {
      std::cerr << "error: --compile-target requires --out <dir>\n";
      return 1;
    }
    return runAtomMode(atomOptions, compileOptions);
  case ScanResult::NoAtom:
    break;
  }

  if (compileOptions.enabled) {
    if (!compileOptions.hasOutDir) {
      std::cerr << "error: --compile-target requires --out <dir>\n";
      return 1;
    }
    return runMlirCompileMode(argc, argv, compileOptions);
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
