// kernelh - the kernel.h command-line driver (plan todos 12 + 24).
//
// Subcommands (hand-rolled argument parsing, no CLI dependency):
//   compile --input <file.atom> | --kernel <name> [--shape k=v,k=v,...]
//           --target <amdgcn-gfx1101|nvptx-sm_80> [--target ...] --out <dir>
//           [--dump-dir <dir>]
//   explore / bench / tune - stubs until plan todos 37 / 35 / 34.
//
// The tool links the kernelh libraries IN-PROCESS (plan Metis G37): it NEVER
// shells out to kernelh-opt, which stays the debug/test driver. `compile` is
// the PRODUCTION end-to-end driver (todo 24), wired exactly to the upstream
// API surface:
//   - --input parses the .atom file via kernelh::ir::parseModule; --kernel
//     emits the named template from the todo-11 DSL registry
//     (kernelh::program().kernel(name).shape(...).emit()).
//   - the module is verified against the device-limits table for EVERY
//     --target (amdgcn-gfx1101 -> "gfx1101", nvptx-sm_80 -> "sm_80"); an
//     unknown target surfaces device_limits_for's error plus the CLI's valid
//     --target listing.
//   - the pre-pipeline module is printed to <out>/<name>.atom (todo-12
//     contract: a round-trippable input artifact).
//   - kernelh::passes::runDefaultPipeline runs the pinned canonical order
//     (todo 19); --dump-dir writes 00-input.atom + 01..07 per-pass dumps
//     (todo 9 hooks); the post-pipeline module is printed to
//     <out>/<name>.atom.final.
//   - kernelh::compiler::emitGpuModule lowers the pipelined module to a
//     gpu.module (todo 21), printed to <out>/<name>.mlir.
//   - the vendor backends (todo 22) produce the full artifact family per
//     --target: nvptx -> <name>.ptx + <name>.cubin, amdgcn -> <name>.isa +
//     <name>.hsaco, each compiled from a fresh clone (the pipeline mutates).
//
// Exit codes: 0 success; 1 operational failure (missing file, parse error,
// verify failure, unknown kernel/target, backend compile error); 2 CLI usage
// error.

#include "kernelh/compiler/backends.hpp"
#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/emitter.hpp"
#include "kernelh/Dialect/Atom/IR/AtomDialect.h"
#include "kernelh/dsl/program.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"
#include "kernelh/passes/pipeline.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitOperational = 1; // compile-time failures
constexpr int kExitUsage = 2;       // CLI usage errors

void printUsage(std::ostream &os) {
  os << "kernelh - kernel.h command-line driver (compile/explore/bench/tune)\n"
     << "\n"
     << "usage: kernelh <compile|explore|bench|tune> [options]\n"
     << "\n"
     << "compile options:\n"
     << "  --input <file.atom>   parse the .atom file (input source;\n"
     << "                        alternative to --kernel)\n"
     << "  --kernel <name>       emit the named kernel template from the DSL\n"
     << "                        registry (input source; alternative to\n"
     << "                        --input)\n"
      << "  --shape k=v,k=v,...   problem dims for --kernel (e.g. n=1024)\n"
      << "  --target <t>          device-limits target for verification and\n"
      << "                        vendor backend (repeatable; one artifact\n"
      << "                        family per target):\n"
      << "                        amdgcn-gfx1101 -> isa + hsaco,\n"
      << "                        nvptx-sm_80 -> ptx + cubin\n"
      << "  --out <dir>           write <dir>/<kernel-or-stem>.{atom,atom.final,\n"
      << "                        mlir,ptx,cubin,isa,hsaco} (dir is created if\n"
      << "                        missing)\n"
      << "  --dump-dir <dir>      per-pass IR dumps (00-input.atom + 01..07\n"
      << "                        with the default pipeline)\n"
     << "\n"
     << "explore, bench, tune: not implemented yet (plan todos 37/35/34); they\n"
     << "exit 2 and ignore any arguments after a warning.\n"
     << "\n"
     << "--help prints this usage and exits 0.\n";
}

// Extracts the value of `--name=<value>` or `--name <value>` into `out`.
// Returns false (with an error printed) when the value is missing.
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

// Parses "k=v,k=v,..." into a shape map; false on any malformed segment
// (empty key/value, non-integer value).
bool parseShape(const std::string &text, kernelh::ShapeMap &shape) {
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    const size_t end = comma == std::string::npos ? text.size() : comma;
    const std::string item = text.substr(start, end - start);
    const size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= item.size()) {
      return false;
    }
    const std::string value = item.substr(eq + 1);
    errno = 0;
    char *valueEnd = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &valueEnd, 10);
    if (errno != 0 || valueEnd == value.c_str() || *valueEnd != '\0') {
      return false;
    }
    shape[item.substr(0, eq)] = parsed;
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return true;
}

struct CompileOptions {
  bool hasInput = false;
  bool hasKernel = false;
  bool hasShape = false;
  bool hasOut = false;
  bool hasDumpDir = false;
  std::string inputFile;
  std::string kernelName;
  kernelh::ShapeMap shape;
  std::vector<std::string> targets;
  std::string outDir;
  std::string dumpDir;
};

// Scans argv[2..] for the compile flags. On a malformed invocation an error is
// printed and false is returned (caller prints usage and exits 2). --help
// prints usage to stdout and exits 0 (conventional).
bool scanCompileArgs(int argc, char **argv, CompileOptions &options) {
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(std::cout);
      std::exit(kExitOk);
    }
    if (arg == "--input" || arg.rfind("--input=", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--input", options.inputFile)) {
        return false;
      }
      options.hasInput = true;
    } else if (arg == "--kernel" || arg.rfind("--kernel=", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--kernel", options.kernelName)) {
        return false;
      }
      options.hasKernel = true;
    } else if (arg == "--shape" || arg.rfind("--shape=", 0) == 0) {
      std::string shapeText;
      if (!takeValue(argc, argv, i, arg, "--shape", shapeText)) {
        return false;
      }
      if (!parseShape(shapeText, options.shape)) {
        std::cerr << "error: malformed --shape '" << shapeText
                  << "' (expected k=v,k=v,...)\n";
        return false;
      }
      options.hasShape = true;
    } else if (arg == "--target" || arg.rfind("--target=", 0) == 0) {
      std::string target;
      if (!takeValue(argc, argv, i, arg, "--target", target)) {
        return false;
      }
      options.targets.push_back(target);
    } else if (arg == "--out" || arg.rfind("--out=", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--out", options.outDir)) {
        return false;
      }
      options.hasOut = true;
    } else if (arg == "--dump-dir" || arg.rfind("--dump-dir=", 0) == 0) {
      if (!takeValue(argc, argv, i, arg, "--dump-dir", options.dumpDir)) {
        return false;
      }
      options.hasDumpDir = true;
    } else {
      std::cerr << "error: unknown flag '" << arg << "' for compile\n";
      return false;
    }
  }

  if (options.hasInput && options.hasKernel) {
    std::cerr << "error: --input and --kernel are conflicting input sources\n";
    return false;
  }
  if (!options.hasInput && !options.hasKernel) {
    std::cerr << "error: compile requires --input <file.atom> or --kernel "
                 "<registry-name>\n";
    return false;
  }
  if (options.hasShape && !options.hasKernel) {
    std::cerr << "error: --shape requires --kernel\n";
    return false;
  }
  if (options.targets.empty()) {
    std::cerr << "error: compile requires at least one --target "
                 "<amdgcn-gfx1101|nvptx-sm_80>\n";
    return false;
  }
  if (!options.hasOut) {
    std::cerr << "error: compile requires --out <dir>\n";
    return false;
  }
  return true;
}

// Reads an entire stream into a string.
std::string readAll(std::istream &in) {
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Maps a CLI target token to a device-limits table token; nullptr for an
// unknown token (passed through so device_limits_for reports the known ones).
const char *deviceTokenFor(const std::string &target) {
  if (target == "amdgcn-gfx1101") {
    return "gfx1101";
  }
  if (target == "nvptx-sm_80") {
    return "sm_80";
  }
  return nullptr;
}

// The CLI's accepted --target set (todo 12 pin: the device-limits table only
// holds gfx1101 and sm_80). Listed on target errors so the user sees the
// exact tokens instead of only the raw table tokens.
void printValidTargets(std::ostream &os) {
  os << "error: valid --target values: amdgcn-gfx1101, nvptx-sm_80\n";
}

bool writeTextFile(const std::filesystem::path &path,
                   const std::string &text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "error: cannot write '" << path.string() << "'\n";
    return false;
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!out) {
    std::cerr << "error: failed writing '" << path.string() << "'\n";
    return false;
  }
  return true;
}

bool writeBlobFile(const std::filesystem::path &path,
                   const std::vector<char> &bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "error: cannot write '" << path.string() << "'\n";
    return false;
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    std::cerr << "error: failed writing '" << path.string() << "'\n";
    return false;
  }
  return true;
}

// Registers the dialect set the emitter touches (mirror of emitter.cpp's
// ensureDialectsLoaded): the atom dialect plus gpu/scf/memref/vector/
// arith/math. The vendor backends append their own dialect set internally.
void registerCompileDialects(mlir::MLIRContext &ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::kernelh::atom::AtomDialect, mlir::gpu::GPUDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::vector::VectorDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
}

int runCompile(const CompileOptions &options) {
  // 1. Input: parse the .atom file, or emit the registry kernel (todo 11).
  kernelh::ir::Module module;
  std::string outName;
  if (options.hasInput) {
    std::ifstream in(options.inputFile);
    if (!in) {
      std::cerr << "error: cannot open input file '" << options.inputFile
                << "'\n";
      return kExitOperational;
    }
    auto parsed = kernelh::ir::parseModule(readAll(in));
    if (!parsed) {
      std::cerr << "error: " << parsed.error() << "\n";
      return kExitOperational;
    }
    module = std::move(*parsed);
    outName = std::filesystem::path(options.inputFile).stem().string();
  } else {
    try {
      kernelh::ProgramBuilder builder = kernelh::program();
      builder.kernel(options.kernelName).shape(options.shape);
      kernelh::EmittedKernel emitted = builder.emit();
      module = std::move(emitted.module);
    } catch (const std::exception &e) {
      std::cerr << "error: " << e.what() << "\n";
      return kExitOperational;
    }
    outName = module.kernels.front()->name;
  }

  // 2. Target-aware structural verification against EVERY --target (todo 10a
  //    API): the module must be valid for all requested devices. An unknown
  //    target surfaces device_limits_for's std::invalid_argument, which names
  //    the offending token and lists the known table tokens; the CLI then
  //    lists its own accepted --target values too.
  for (const std::string &target : options.targets) {
    kernelh::DeviceLimits limits;
    try {
      const char *token = deviceTokenFor(target);
      limits =
          kernelh::device_limits_for(token ? token : target.c_str());
    } catch (const std::invalid_argument &e) {
      std::cerr << "error: " << e.what() << "\n";
      printValidTargets(std::cerr);
      return kExitOperational;
    }
    const kernelh::ir::Diagnostics diags =
        kernelh::ir::verifyModule(module, limits);
    bool failed = false;
    for (const kernelh::ir::Diagnostic &d : diags) {
      std::cerr << d.toString() << "\n";
      failed |= (d.severity == kernelh::ir::Diagnostic::Severity::Error);
    }
    if (failed) {
      return kExitOperational;
    }
  }

  // 3. Write the pre-pipeline ".atom" text to <out>/<name>.atom (todo-12
  //    contract: a round-trippable input artifact; --kernel names the file
  //    after the emitted kernel, --input after the input stem). --out is
  //    created if missing.
  std::error_code ec;
  std::filesystem::create_directories(options.outDir, ec);
  if (ec) {
    std::cerr << "error: cannot create output dir '" << options.outDir
              << "': " << ec.message() << "\n";
    return kExitOperational;
  }
  const std::filesystem::path outDir(options.outDir);
  const std::filesystem::path atomFile = outDir / (outName + ".atom");
  if (!writeTextFile(atomFile, kernelh::ir::printModule(module))) {
    return kExitOperational;
  }
  std::cout << "wrote " << atomFile.string() << "\n";

  // 4. Default pipeline (todo 19): the pinned canonical order through the
  //    global registry (which picks up every self-registering pass at static
  //    init). --dump-dir writes 00-input.atom + 01..07 per-pass dumps through
  //    the todo-9 dump hook. The post-pipeline module is printed to
  //    <out>/<name>.atom.final.
  const auto pipeline = kernelh::passes::runDefaultPipeline(
      module, options.hasDumpDir ? std::string_view(options.dumpDir)
                                 : std::string_view());
  if (!pipeline) {
    std::cerr << "error: " << pipeline.error() << "\n";
    return kExitOperational;
  }
  const std::filesystem::path finalFile = outDir / (outName + ".atom.final");
  if (!writeTextFile(finalFile, kernelh::ir::printModule(module))) {
    return kExitOperational;
  }
  std::cout << "wrote " << finalFile.string() << "\n";

  // 5. Lowering (todo 21): custom atom IR -> gpu.module, printed to
  //    <out>/<name>.mlir.
  mlir::MLIRContext ctx;
  registerCompileDialects(ctx);
  auto gpu = kernelh::compiler::emitGpuModule(module, ctx);
  if (!gpu) {
    std::cerr << "error: " << gpu.error() << "\n";
    return kExitOperational;
  }
  std::string mlirText;
  {
    llvm::raw_string_ostream os(mlirText);
    (**gpu).print(os);
    os.flush();
  }
  const std::filesystem::path mlirFile = outDir / (outName + ".mlir");
  if (!writeTextFile(mlirFile, mlirText)) {
    return kExitOperational;
  }
  std::cout << "wrote " << mlirFile.string() << "\n";

  // 6. Vendor backends (todo 22): each --target produces its full artifact
  //    family. compileToBlob/compileToText mutate the module they are given
  //    (gpu.module -> gpu.binary), so every artifact compiles from a fresh
  //    clone of the shared gpu module.
  const auto cloneOf = [](mlir::OwningOpRef<mlir::ModuleOp> &src) {
    return mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::cast<mlir::ModuleOp>(src->clone()));
  };
  for (const std::string &target : options.targets) {
    auto spec = kernelh::compiler::parseTargetSpec(target);
    if (!spec) {
      std::cerr << "error: " << spec.error() << "\n";
      printValidTargets(std::cerr);
      return kExitOperational;
    }
    if (spec->isNVPTX()) {
      auto ptxModule = cloneOf(*gpu);
      auto ptx = kernelh::compiler::compileToText(ptxModule, *spec, "ptx");
      if (!ptx) {
        std::cerr << "error: " << ptx.error() << "\n";
        return kExitOperational;
      }
      const std::filesystem::path ptxFile = outDir / (outName + ".ptx");
      if (!writeTextFile(ptxFile, *ptx)) {
        return kExitOperational;
      }
      std::cout << "wrote " << ptxFile.string() << "\n";

      auto cubinModule = cloneOf(*gpu);
      auto cubin = kernelh::compiler::compileToBlob(cubinModule, *spec,
                                                    "cubin");
      if (!cubin) {
        std::cerr << "error: " << cubin.error() << "\n";
        return kExitOperational;
      }
      const std::filesystem::path cubinFile = outDir / (outName + ".cubin");
      if (!writeBlobFile(cubinFile, *cubin)) {
        return kExitOperational;
      }
      std::cout << "wrote " << cubinFile.string() << "\n";
    } else {
      auto isaModule = cloneOf(*gpu);
      auto isa = kernelh::compiler::compileToText(isaModule, *spec, "isa");
      if (!isa) {
        std::cerr << "error: " << isa.error() << "\n";
        return kExitOperational;
      }
      const std::filesystem::path isaFile = outDir / (outName + ".isa");
      if (!writeTextFile(isaFile, *isa)) {
        return kExitOperational;
      }
      std::cout << "wrote " << isaFile.string() << "\n";

      auto hsacoModule = cloneOf(*gpu);
      auto hsaco = kernelh::compiler::compileToBlob(hsacoModule, *spec,
                                                    "hsaco");
      if (!hsaco) {
        std::cerr << "error: " << hsaco.error() << "\n";
        return kExitOperational;
      }
      const std::filesystem::path hsacoFile = outDir / (outName + ".hsaco");
      if (!writeBlobFile(hsacoFile, *hsaco)) {
        return kExitOperational;
      }
      std::cout << "wrote " << hsacoFile.string() << "\n";
    }
  }
  std::cout << "compile ok\n";
  return kExitOk;
}

// explore/bench/tune stubs: accepted so the CLI shape is stable, but they only
// report the plan todo that implements them (37/35/34) and exit 2. Extra
// arguments are ignored after a warning (their real flags arrive with their
// todos).
int notImplemented(const char *subcommand, int todo, int argc, char **argv) {
  std::cerr << subcommand << ": not implemented yet (plan todo " << todo
            << ")\n";
  if (argc > 2) {
    std::cerr << "warning: ignoring arguments for '" << subcommand << "':";
    for (int i = 2; i < argc; ++i) {
      std::cerr << " " << argv[i];
    }
    std::cerr << "\n";
  }
  return kExitUsage;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage(std::cerr);
    return kExitUsage;
  }
  const std::string subcommand = argv[1];
  if (subcommand == "--help" || subcommand == "-h") {
    printUsage(std::cout);
    return kExitOk;
  }
  if (subcommand == "compile") {
    CompileOptions options;
    if (!scanCompileArgs(argc, argv, options)) {
      printUsage(std::cerr);
      return kExitUsage;
    }
    return runCompile(options);
  }
  if (subcommand == "explore") {
    return notImplemented("explore", 37, argc, argv);
  }
  if (subcommand == "bench") {
    return notImplemented("bench", 35, argc, argv);
  }
  if (subcommand == "tune") {
    return notImplemented("tune", 34, argc, argv);
  }
  std::cerr << "error: unknown subcommand '" << subcommand << "'\n";
  printUsage(std::cerr);
  return kExitUsage;
}
