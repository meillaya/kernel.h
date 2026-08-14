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

#include "kernelh/compiler/autotuner.hpp"
#include "kernelh/compiler/backends.hpp"
#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/emitter.hpp"
#include "kernelh/compiler/launch_verifier.hpp"
#include "kernelh/compiler/resource_reporter.hpp"
#include "kernelh/Dialect/Atom/IR/AtomDialect.h"
#include "kernelh/dsl/program.hpp"
#include "kernelh/ir/parser.hpp"
#include "kernelh/ir/pass_manager.hpp"
#include "kernelh/ir/printer.hpp"
#include "kernelh/ir/verifier.hpp"
#include "kernelh/passes/pipeline.hpp"
#include "kernelh/runtime/backend.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/event_timer.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

// The todo-35 benchmark harness (`kernelh bench`): G32 results.csv writer,
// metric formulas, roofline microbenchmarks, suite registry, orchestration.
#include "run_bench.hpp"

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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/wait.h>
#include <unistd.h>
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
      << "  --passes-no-fuse      pipeline override: run the default pipeline\n"
      << "                        WITHOUT the fuse pass (proves fusion fired\n"
      << "                        via the loop-nest-count differential)\n"
      << "  --resources           (todo 33) write resources.json next to the\n"
      << "                        artifacts with per-kernel resource usage\n"
      << "                        (vgpr/sgpr/lds/wavefront/occupancy); the\n"
      << "                        contract consumed by the autotuner\n"
      << "\n"
      << "tune options (plan todo 34; GEMM only):\n"
      << "  --kernel <name>       the registry kernel to tune (default: gemm;\n"
      << "                        only gemm is supported)\n"
      << "  --shape k=v,k=v,...   problem dims, e.g. m=512,n=512,k=512\n"
      << "                        (required)\n"
      << "  --target <t>          device target (default: amdgcn-gfx1101; the\n"
      << "                        local execution target)\n"
      << "  --out <dir>           write <dir>/tune-results.csv (the full CSV of\n"
      << "                        evaluated configs; required)\n"
      << "  --max-candidates <n>  hard cap on evaluated configs (1..80,\n"
      << "                        default 80; Metis G22)\n"
      << "  --sample <n>          fixed-seed sample size (1..216, default 64)\n"
      << "  --refine-budget <n>   max neighbors of the best to refine (0..80,\n"
      << "                        default 16)\n"
      << "  --cache <path>        JSON cache file (default build/tune-cache.json;\n"
      << "                        keyed sha256(kernel+shape+device_name); a\n"
      << "                        second run with the same key logs cache-hit)\n"
      << "  --timeout <seconds>   wall clock per candidate (1..3600, default 30;\n"
       << "                        a hung candidate is skipped with reason timeout)\n"
       << "\n"
       << "bench options (plan todo 35):\n"
       << "  --suite <name>        benchmark suite: roofline, kernelh, rocblas,\n"
       << "                        hipref, cuda-cross, all (default: all)\n"
       << "  --out <path>          results.csv destination: a .csv file path OR a\n"
       << "                        directory (results.csv is written inside)\n"
       << "                        (required)\n"
       << "\n"
       << "explore: not implemented yet (plan todo 37); it exits 2 and ignores\n"
       << "any arguments after a warning.\n"
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
  bool passesNoFuse = false;
  bool resources = false;
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
    } else if (arg == "--passes-no-fuse") {
      options.passesNoFuse = true;
    } else if (arg == "--resources") {
      options.resources = true;
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
  //    the todo-9 dump hook. --passes-no-fuse (todo 30 fusion-proof override)
  //    reruns the SAME order minus the fuse pass, so a fused_gelu compile
  //    produces ONE loop nest vs TWO - the differential the e2e test asserts.
  //    The post-pipeline module is printed to <out>/<name>.atom.final.
  const std::string_view dumpView =
      options.hasDumpDir ? std::string_view(options.dumpDir)
                         : std::string_view();
  const auto pipeline =
      options.passesNoFuse
          ? kernelh::ir::runPasses(module, "canonicalize,cse,tile,promote-shared,"
                                           "vectorize,unroll",
                                   dumpView)
          : kernelh::passes::runDefaultPipeline(module, dumpView);
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
  // 7. Resource reporter (plan todo 33, flag ownership pinned here): with
  //    --resources, invoke the reporter for EVERY --target and write
  //    resources.json next to the artifacts. The reporter follows the pinned
  //    source priority (rocprofv3 when available, else the amdhsa notes /
  //    ptxas -v static parsing); a truncated/corrupt code object fails the
  //    compile cleanly.
  if (options.resources) {
    for (const std::string &target : options.targets) {
      auto resources =
          kernelh::compiler::resource_report(outName, target, outDir);
      if (!resources) {
        std::cerr << "error: " << resources.error() << "\n";
        return kExitOperational;
      }
      std::cout << "wrote " << (outDir / "resources.json").string() << " ("
                << target << ")\n";
    }
  }
  std::cout << "compile ok\n";
  return kExitOk;
}

// ===========================================================================
// `kernelh tune` (plan todo 34): bounded GEMM autotune.
//
// The SEARCH logic (fixed-seed sampling, pruning policy, local refinement,
// whole-run cache, CSV) lives in lib/compiler/autotuner.cpp and is MLIR-free.
// This driver supplies the EVALUATOR for each candidate:
//   1. compile the DSL schedule in-process (registry emit with explicit
//      tile/map_blocks/map_warps/use_shared_memory/vectorize/unroll schedule
//      attrs -> default pipeline -> emitter -> HSACO),
//   2. todo-32 launch-config verification (analytic occupancy; the
//      hipOccupancy API is broken on this box - analytic only),
//   3. todo-33 resource check (resources.json lds_bytes / vgpr_count vs the
//      todo-10 table budget; the profiler-feedback seam, Metis G23),
//   4. numeric re-verification on a SMALL shape (capped at 128 per dim; the
//      todo-26 tolerance: relative Frobenius <= 1e-5) - candidates that fail
//      are discarded,
//   5. EventTimer timing (warmup 3, median of 10) on the tuning shape.
//
// Every candidate runs in a FORKED child with a wall-clock deadline
// (per-config timeout, default 30s) so a hung compile or hung GPU launch can
// never hang the tune run: on timeout the child is SIGKILLed and the candidate
// is recorded with reason "timeout". The parent never initializes MLIR/HIP;
// every child re-initializes them fresh after fork.
// ===========================================================================

namespace {

// CLI flags for `kernelh tune` (see the usage text).
struct TuneFlags {
  std::string kernelName = "gemm";
  kernelh::ShapeMap shape;
  std::string target = "amdgcn-gfx1101";
  std::string outDir;
  std::string cachePath = "build/tune-cache.json";
  int sampleCap = 64;
  int refineBudget = 16;
  int maxCandidates = 80;
  int timeoutSeconds = 30;
};

bool scanTuneArgs(int argc, char **argv, TuneFlags &flags) {
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(std::cout);
      std::exit(kExitOk);
    }
    std::string value;
    if (takeValue(argc, argv, i, arg, "--kernel", value)) {
      flags.kernelName = value;
    } else if (takeValue(argc, argv, i, arg, "--shape", value)) {
      if (!parseShape(value, flags.shape)) {
        std::cerr << "error: malformed --shape '" << value
                  << "' (expected k=v,k=v,...)\n";
        return false;
      }
    } else if (takeValue(argc, argv, i, arg, "--target", value)) {
      flags.target = value;
    } else if (takeValue(argc, argv, i, arg, "--out", value)) {
      flags.outDir = value;
    } else if (takeValue(argc, argv, i, arg, "--cache", value)) {
      flags.cachePath = value;
    } else if (takeValue(argc, argv, i, arg, "--sample", value)) {
      long v = std::strtol(value.c_str(), nullptr, 10);
      if (v < 1 || v > 216) {
        std::cerr << "error: --sample must be an integer in [1,216]\n";
        return false;
      }
      flags.sampleCap = static_cast<int>(v);
    } else if (takeValue(argc, argv, i, arg, "--refine-budget", value)) {
      long v = std::strtol(value.c_str(), nullptr, 10);
      if (v < 0 || v > 80) {
        std::cerr << "error: --refine-budget must be an integer in [0,80]\n";
        return false;
      }
      flags.refineBudget = static_cast<int>(v);
    } else if (takeValue(argc, argv, i, arg, "--max-candidates", value)) {
      long v = std::strtol(value.c_str(), nullptr, 10);
      if (v < 1 || v > 80) {
        std::cerr << "error: --max-candidates must be an integer in [1,80]\n";
        return false;
      }
      flags.maxCandidates = static_cast<int>(v);
    } else if (takeValue(argc, argv, i, arg, "--timeout", value)) {
      long v = std::strtol(value.c_str(), nullptr, 10);
      if (v < 1 || v > 3600) {
        std::cerr << "error: --timeout must be an integer in [1,3600]\n";
        return false;
      }
      flags.timeoutSeconds = static_cast<int>(v);
    } else {
      std::cerr << "error: unknown flag '" << arg << "' for tune\n";
      return false;
    }
  }
  if (flags.shape.empty()) {
    std::cerr << "error: tune requires --shape m=...,n=...,k=...\n";
    return false;
  }
  if (flags.outDir.empty()) {
    std::cerr << "error: tune requires --out <dir>\n";
    return false;
  }
  return true;
}

std::string formatDouble(double v) {
  std::ostringstream os;
  os << std::setprecision(10) << v;
  return os.str();
}

// Reason strings land in the CSV and in the fork-result record, so strip the
// record separator '|' and any newlines; keep it short.
std::string sanitizeReason(std::string reason) {
  for (char &c : reason) {
    if (c == '|' || c == '\n' || c == '\r') {
      c = ' ';
    }
  }
  while (!reason.empty() && reason.front() == ' ') {
    reason.erase(reason.begin());
  }
  while (!reason.empty() && reason.back() == ' ') {
    reason.pop_back();
  }
  if (reason.size() > 200) {
    reason.resize(200);
  }
  return reason;
}

// "m=512,n=512,k=512" (ShapeMap is std::map -> sorted keys).
std::string shapeToString(const kernelh::ShapeMap &shape) {
  std::string out;
  for (const auto &[key, value] : shape) {
    if (!out.empty()) {
      out += ",";
    }
    out += key + "=" + std::to_string(value);
  }
  return out;
}

// What the fork child must know to evaluate one candidate.
struct TuneEvalRequest {
  kernelh::compiler::GemmCandidate config;
  std::string kernelName;
  kernelh::ShapeMap shape; // tuning shape (m, n, k)
  std::string target;      // CLI token, "amdgcn-gfx1101"
  std::string deviceToken; // "gfx1101" (device-limits table token)
};

// One candidate compilation: DSL emit with the candidate's schedule attrs ->
// structural verify -> default pipeline -> emitter -> HSACO blob. Returns the
// in-memory blob + the pipelined module's static shared bytes (LDS) or an
// error. `shape` is the problem dims baked into the kernel (the tuning shape,
// or the small numeric-verify shape).
struct CandidateCompile {
  std::vector<char> hsaco;
  int64_t shared_bytes = 0;
  std::string error;
};

CandidateCompile compileCandidate(const TuneEvalRequest &req,
                                  const kernelh::ShapeMap &shape) {
  CandidateCompile out;
  kernelh::ir::Module module;
  try {
    kernelh::ProgramBuilder builder = kernelh::program();
    builder.kernel(req.kernelName)
        .shape(shape)
        .tile(req.config.bm, req.config.bn, req.config.bk)
        .map_blocks()
        .map_warps()
        .use_shared_memory()
        .vectorize(req.config.vec)
        .unroll(req.config.unroll);
    kernelh::EmittedKernel emitted = builder.emit();
    module = std::move(emitted.module);
  } catch (const std::exception &e) {
    out.error = e.what();
    return out;
  }
  kernelh::DeviceLimits limits;
  try {
    limits = kernelh::device_limits_for(req.deviceToken.c_str());
  } catch (const std::invalid_argument &e) {
    out.error = e.what();
    return out;
  }
  const kernelh::ir::Diagnostics diags = kernelh::ir::verifyModule(module, limits);
  for (const kernelh::ir::Diagnostic &d : diags) {
    if (d.severity == kernelh::ir::Diagnostic::Severity::Error) {
      out.error = d.toString();
      return out;
    }
  }
  auto pipeline = kernelh::passes::runDefaultPipeline(module, {});
  if (!pipeline) {
    out.error = pipeline.error();
    return out;
  }
  out.shared_bytes = kernelh::ir::kernelTotalSharedBytes(*module.kernels.front());
  mlir::MLIRContext ctx;
  registerCompileDialects(ctx);
  auto gpu = kernelh::compiler::emitGpuModule(module, ctx);
  if (!gpu) {
    out.error = gpu.error();
    return out;
  }
  auto spec = kernelh::compiler::parseTargetSpec(req.target);
  if (!spec) {
    out.error = spec.error();
    return out;
  }
  auto hsaco = kernelh::compiler::compileToBlob(*gpu, *spec, "hsaco");
  if (!hsaco) {
    out.error = hsaco.error();
    return out;
  }
  out.hsaco = std::move(*hsaco);
  return out;
}

// One compiled + loaded gemm kernel ready to launch (used for both the
// numeric re-verification and the timing). Owns its buffers + host data so
// the launch's memref arguments stay valid across repeated launches.
struct GemmLaunch {
  std::shared_ptr<kernelh::runtime::Backend> backend;
  std::optional<kernelh::runtime::GpuModule> module;
  kernelh::runtime::KernelFunction fn;
  std::optional<kernelh::runtime::KernelLaunch> launch;
  std::unique_ptr<kernelh::runtime::Buffer> a, b, c;
  std::vector<float> h_a, h_b, h_c;
  long m = 0, n = 0, k = 0;
};

// Loads the HSACO, allocates buffers, fills exact-in-f32 inputs, and configures
// the launch (grid = ceil(m / bm), block = waves * wavefront_size). Returns a
// diagnostic on failure; `gl.launch` is then ready for repeated launch() calls.
std::string setupGemmLaunch(const TuneEvalRequest &req,
                            const std::vector<char> &hsaco,
                            const kernelh::ShapeMap &shape, GemmLaunch &gl) {
  gl.m = shape.at("m");
  gl.n = shape.at("n");
  gl.k = shape.at("k");
  gl.backend = kernelh::runtime::make_hip_backend();
  if (!gl.backend) {
    return "no HIP backend available";
  }
  auto ctx = kernelh::runtime::DeviceContext::create(gl.backend);
  if (!ctx) {
    return "device unavailable: " + ctx.error().message;
  }
  kernelh::DeviceLimits limits = kernelh::device_limits_for(req.deviceToken.c_str());
  const int wavefront = limits.wavefront_size > 0 ? limits.wavefront_size : 32;

  auto module = kernelh::runtime::GpuModule::load(gl.backend, hsaco.data(),
                                                  hsaco.size());
  if (!module) {
    return "module load failed: " + module.error().message;
  }
  auto fn = module->get_function(req.kernelName.c_str());
  if (!fn) {
    return "get_function failed: " + fn.error().message;
  }

  const size_t aCount = static_cast<size_t>(gl.m) * static_cast<size_t>(gl.k);
  const size_t bCount = static_cast<size_t>(gl.k) * static_cast<size_t>(gl.n);
  const size_t cCount = static_cast<size_t>(gl.m) * static_cast<size_t>(gl.n);
  gl.h_a.resize(aCount);
  gl.h_b.resize(bCount);
  gl.h_c.assign(cCount, 0.0f);
  for (long i = 0; i < gl.m; ++i) {
    for (long kk = 0; kk < gl.k; ++kk) {
      gl.h_a[static_cast<size_t>(i * gl.k + kk)] =
          static_cast<float>((i * 131 + kk * 17 + 7) & 63);
    }
  }
  for (long kk = 0; kk < gl.k; ++kk) {
    for (long j = 0; j < gl.n; ++j) {
      gl.h_b[static_cast<size_t>(kk * gl.n + j)] =
          static_cast<float>((kk * 97 + j * 41 + 3) & 63);
    }
  }
  auto a = kernelh::runtime::Buffer::allocate(gl.backend, aCount * sizeof(float));
  auto b = kernelh::runtime::Buffer::allocate(gl.backend, bCount * sizeof(float));
  auto c = kernelh::runtime::Buffer::allocate(gl.backend, cCount * sizeof(float));
  if (!a || !b || !c) {
    return "buffer allocation failed";
  }
  if (!a->copy_from_host(gl.h_a.data(), aCount * sizeof(float)) ||
      !b->copy_from_host(gl.h_b.data(), bCount * sizeof(float)) ||
      !c->copy_from_host(gl.h_c.data(), cCount * sizeof(float))) {
    return "buffer upload failed";
  }

  gl.a = std::make_unique<kernelh::runtime::Buffer>(std::move(*a));
  gl.b = std::make_unique<kernelh::runtime::Buffer>(std::move(*b));
  gl.c = std::make_unique<kernelh::runtime::Buffer>(std::move(*c));
  gl.module.emplace(std::move(*module));
  gl.fn = *fn;
  gl.launch.emplace(gl.backend);
  const long grid = (gl.m + req.config.bm - 1) / req.config.bm;
  gl.launch->grid(static_cast<unsigned>(grid)).block(
      static_cast<unsigned>(req.config.waves * wavefront));
  gl.launch->memref_arg(gl.a->device_ptr(), std::vector<int64_t>{gl.m, gl.k});
  gl.launch->memref_arg(gl.b->device_ptr(), std::vector<int64_t>{gl.k, gl.n});
  gl.launch->memref_arg(gl.c->device_ptr(), std::vector<int64_t>{gl.m, gl.n});
  return "";
}

// fp64 CPU reference cast to fp32 (rank-1 update over contiguous B rows; exact
// -in-f32 integer inputs, so the accumulation order does not matter).
void gemmFillReference(const std::vector<float> &a, const std::vector<float> &b,
                       std::vector<float> &ref, long m, long n, long k) {
  std::vector<double> acc(static_cast<size_t>(n), 0.0);
  for (long i = 0; i < m; ++i) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (long kk = 0; kk < k; ++kk) {
      const double av =
          static_cast<double>(a[static_cast<size_t>(i * k + kk)]);
      const float *bRow = &b[static_cast<size_t>(kk * n)];
      for (long j = 0; j < n; ++j) {
        acc[static_cast<size_t>(j)] += av * static_cast<double>(bRow[j]);
      }
    }
    for (long j = 0; j < n; ++j) {
      ref[static_cast<size_t>(i * n + j)] =
          static_cast<float>(acc[static_cast<size_t>(j)]);
    }
  }
}

// Relative Frobenius norm of c - ref over the M x N outputs (todo-26 G11).
double gemmRelativeFrobenius(const std::vector<float> &c,
                             const std::vector<float> &ref, size_t count) {
  double diffSq = 0.0;
  double refSq = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double d = static_cast<double>(c[i]) - static_cast<double>(ref[i]);
    diffSq += d * d;
    const double r = static_cast<double>(ref[i]);
    refSq += r * r;
  }
  return std::sqrt(diffSq / refSq);
}

// Launches once, copies C back, and returns the relative Frobenius vs the
// fp64 reference (the todo-26 tolerance check); -1 on any runtime failure.
double verifyGemmNumerically(GemmLaunch &gl) {
  auto result = gl.launch->launch(gl.fn);
  if (!result) {
    return -1.0;
  }
  if (!gl.backend->device_synchronize()) {
    return -1.0;
  }
  const size_t cCount = static_cast<size_t>(gl.m) * static_cast<size_t>(gl.n);
  if (!gl.c->copy_to_host(gl.h_c.data(), cCount * sizeof(float))) {
    return -1.0;
  }
  std::vector<float> ref(cCount);
  gemmFillReference(gl.h_a, gl.h_b, ref, gl.m, gl.n, gl.k);
  return gemmRelativeFrobenius(gl.h_c, ref, cCount);
}

// EventTimer median-of-N (warmup `warmup` untimed launches first). Returns -1
// on any runtime failure.
double timeGemmMedian(GemmLaunch &gl, int warmup, int iterations) {
  auto timer = kernelh::runtime::EventTimer::create(gl.backend);
  if (!timer) {
    return -1.0;
  }
  for (int i = 0; i < warmup; ++i) {
    if (!gl.launch->launch(gl.fn)) {
      return -1.0;
    }
  }
  if (!gl.backend->device_synchronize()) {
    return -1.0;
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    if (!timer->record_start()) {
      return -1.0;
    }
    if (!gl.launch->launch(gl.fn)) {
      return -1.0;
    }
    if (!timer->record_end()) {
      return -1.0;
    }
    if (!timer->synchronize()) {
      return -1.0;
    }
    auto ms = timer->elapsed_milliseconds();
    if (!ms) {
      return -1.0;
    }
    samples.push_back(static_cast<double>(*ms));
  }
  std::sort(samples.begin(), samples.end());
  if (samples.size() % 2 == 1) {
    return samples[samples.size() / 2];
  }
  return 0.5 * (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]);
}

// Runs in the FORKED child: full per-candidate pipeline (compile -> launch
// verify -> resource check -> numeric re-verify -> timing).
kernelh::compiler::EvaluatedCandidate evaluateCandidateCore(const TuneEvalRequest &req) {
  kernelh::compiler::EvaluatedCandidate result;
  result.config = req.config;

  const std::filesystem::path scratch =
      std::filesystem::temp_directory_path() /
      ("kernelh-t34-cand-" + std::to_string(static_cast<long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  const auto finish = [&](const std::string &reason, bool accepted = false,
                          double ms = 0.0) {
    result.accepted = accepted;
    result.ms_median = ms;
    result.reason = sanitizeReason(reason);
    std::filesystem::remove_all(scratch, ec);
    return result;
  };

  // 1. Compile the candidate for the TUNING shape.
  CandidateCompile compiled = compileCandidate(req, req.shape);
  if (!compiled.error.empty()) {
    // The todo-26 LDS-overflow path: the pipeline's promote-shared pass
    // rejects oversized tiles with an LDS diagnostic before any launch -
    // surface it as the resource prune (the QA "pruned: lds" probe).
    if (compiled.error.find("LDS") != std::string::npos ||
        compiled.error.find("shared memory") != std::string::npos) {
      return finish("pruned: lds");
    }
    return finish("compile-error: " + compiled.error);
  }

  // 2. todo-32 launch-config verification (analytic; no loaded function).
  kernelh::DeviceLimits limits;
  try {
    limits = kernelh::device_limits_for(req.deviceToken.c_str());
  } catch (const std::invalid_argument &) {
    return finish("compile-error: unknown device target");
  }
  const int wavefront = limits.wavefront_size > 0 ? limits.wavefront_size : 32;
  kernelh::compiler::LaunchConfig launchConfig;
  launchConfig.threads_per_workgroup = req.config.waves * wavefront;
  launchConfig.grid_x =
      static_cast<int>((req.shape.at("m") + req.config.bm - 1) / req.config.bm);
  launchConfig.static_shared_bytes = static_cast<size_t>(compiled.shared_bytes);
  kernelh::compiler::LaunchVerifier verifier;
  auto verified = verifier.verify(launchConfig, limits);
  if (!verified) {
    return finish("pruned: launch");
  }

  // 3. todo-33 resource check (the profiler-feedback seam): resources.json is
  //    written next to the tuning-shape artifacts; configs whose lds_bytes
  //    exceed the device LDS or whose vgpr_count exceeds the register-file
  //    budget are pruned without ever being launched.
  {
    std::ofstream hsacoFile(scratch / (req.kernelName + ".hsaco"),
                            std::ios::binary);
    hsacoFile.write(compiled.hsaco.data(),
                    static_cast<std::streamsize>(compiled.hsaco.size()));
  }
  auto resources =
      kernelh::compiler::resource_report(req.kernelName, req.target, scratch);
  if (resources) {
    const std::string prune =
        kernelh::compiler::resource_prune_reason(*resources, limits);
    if (!prune.empty()) {
      return finish(prune);
    }
  }

  // 4. Numeric re-verification on a SMALL shape (capped at 128 per dim): the
  //    todo-26 tolerance (relative Frobenius <= 1e-5). Candidates that fail
  //    are discarded.
  {
    kernelh::ShapeMap verifyShape = req.shape;
    for (const char *dim : {"m", "n", "k"}) {
      verifyShape[dim] = std::min(verifyShape[dim], 128L);
    }
    CandidateCompile small = compileCandidate(req, verifyShape);
    if (!small.error.empty()) {
      return finish("compile-error: " + small.error);
    }
    GemmLaunch gl;
    const std::string setupErr = setupGemmLaunch(req, small.hsaco, verifyShape, gl);
    if (!setupErr.empty()) {
      return finish("degraded: timing skipped");
    }
    const double rel = verifyGemmNumerically(gl);
    if (rel < 0.0) {
      return finish("reverify: FAIL (launch error)");
    }
    if (rel > 1e-5) {
      return finish("reverify: FAIL (rel-frobenius " + formatDouble(rel) +
                    " > 1e-5)");
    }
  }

  // 5. EventTimer timing on the TUNING shape: warmup 3, median of 10.
  GemmLaunch gl;
  const std::string setupErr = setupGemmLaunch(req, compiled.hsaco, req.shape, gl);
  if (!setupErr.empty()) {
    return finish("degraded: timing skipped");
  }
  const double ms = timeGemmMedian(gl, /*warmup=*/3, /*iterations=*/10);
  if (ms < 0.0) {
    return finish("timing-error");
  }
  return finish("ok", /*accepted=*/true, ms);
}

// Runs `evaluateCandidateCore` in a forked child with a wall-clock deadline;
// on timeout the child is SIGKILLed and the candidate is recorded as
// "timeout". The child writes its result record to a pipe as
// "<accepted>|<ms_median>|<reason>".
kernelh::compiler::EvaluatedCandidate evaluateCandidateWithTimeout(
    const TuneEvalRequest &req, int timeoutSeconds) {
  kernelh::compiler::EvaluatedCandidate out;
  out.config = req.config;

  int fds[2];
  if (::pipe(fds) != 0) {
    out.reason = "compile-error: pipe failed";
    return out;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    out.reason = "compile-error: fork failed";
    return out;
  }
  if (pid == 0) {
    // Child: evaluate; write the record to fd 3; _exit without running any
    // C++ static destructors (MLIR/LLVM static teardown after fork is unsafe).
    ::close(fds[0]);
    if (fds[1] != 3) {
      ::dup2(fds[1], 3);
    }
    ::close(fds[1]);
    const int devnull = ::open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      ::dup2(devnull, 1);
      ::dup2(devnull, 2);
      ::close(devnull);
    }
    const kernelh::compiler::EvaluatedCandidate result =
        evaluateCandidateCore(req);
    const std::string record =
        (result.accepted ? "1|" : "0|") + formatDouble(result.ms_median) +
        "|" + sanitizeReason(result.reason);
    (void)::write(3, record.data(), record.size());
    ::_exit(0);
  }

  // Parent: read the record with a deadline; SIGKILL the child on timeout.
  ::close(fds[1]);
  std::string record;
  bool timedOut = false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeoutSeconds);
  char buf[4096];
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      timedOut = true;
      break;
    }
    const auto remainingMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
    struct pollfd pfd = {fds[0], static_cast<short>(POLLIN), 0};
    const int pr = ::poll(&pfd, 1, static_cast<int>(remainingMs.count()));
    if (pr < 0) {
      if (errno == EINTR) {
        continue;
      }
      timedOut = true;
      break;
    }
    if (pr == 0) {
      timedOut = true;
      break;
    }
    const ssize_t n = ::read(fds[0], buf, sizeof buf);
    if (n == 0) {
      break; // child exited and closed the pipe
    }
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      timedOut = true;
      break;
    }
    record.append(buf, static_cast<size_t>(n));
  }
  ::close(fds[0]);
  if (timedOut) {
    ::kill(pid, SIGKILL);
  }
  int status = 0;
  ::waitpid(pid, &status, 0);

  if (timedOut) {
    out.accepted = false;
    out.reason = "timeout";
    return out;
  }
  const size_t p1 = record.find('|');
  const size_t p2 = p1 == std::string::npos ? std::string::npos
                                            : record.find('|', p1 + 1);
  if (p1 == std::string::npos || p2 == std::string::npos) {
    out.accepted = false;
    out.reason = "compile-error: child produced no result";
    return out;
  }
  out.accepted = record.substr(0, p1) == "1";
  out.ms_median = std::strtod(record.substr(p1 + 1, p2 - p1 - 1).c_str(), nullptr);
  out.reason = sanitizeReason(record.substr(p2 + 1));
  return out;
}

// Queries the LIVE device name via a forked child (the parent never
// initializes MLIR/HIP; children do that fresh). "degraded" when no device.
std::string queryDeviceName() {
  int fds[2];
  if (::pipe(fds) != 0) {
    return "degraded";
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    return "degraded";
  }
  if (pid == 0) {
    ::close(fds[0]);
    if (fds[1] != 1) {
      ::dup2(fds[1], 1);
    }
    ::close(fds[1]);
    const int devnull = ::open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      ::dup2(devnull, 2);
      ::close(devnull);
    }
    std::string name;
    auto backend = kernelh::runtime::make_hip_backend();
    if (backend) {
      auto ctx = kernelh::runtime::DeviceContext::create(backend);
      if (ctx) {
        name = ctx->name();
      }
    }
    if (!name.empty()) {
      (void)::write(1, name.data(), name.size());
    }
    ::_exit(0);
  }
  ::close(fds[1]);
  std::string out;
  char buf[1024];
  ssize_t n;
  while ((n = ::read(fds[0], buf, sizeof buf)) > 0) {
    out.append(buf, static_cast<size_t>(n));
  }
  ::close(fds[0]);
  int status = 0;
  ::waitpid(pid, &status, 0);
  return out.empty() ? "degraded" : out;
}

int runTune(const TuneFlags &flags) {
  if (flags.kernelName != "gemm") {
    std::cerr << "error: tune currently supports only the 'gemm' kernel (the "
                 "config space is GEMM-shaped)\n";
    return kExitOperational;
  }
  for (const char *dim : {"m", "n", "k"}) {
    const auto it = flags.shape.find(dim);
    if (it == flags.shape.end() || it->second <= 0) {
      std::cerr << "error: tune requires a positive '" << dim << "' in --shape\n";
      return kExitOperational;
    }
  }
  if (flags.target != "amdgcn-gfx1101") {
    std::cerr << "error: tune requires --target amdgcn-gfx1101 (the local "
                 "execution target)\n";
    return kExitOperational;
  }

  // The cache key is sha256(kernel+shape+device_name) (Metis note 27: the
  // device name is IN the key - a different GPU never reuses the cached run).
  const std::string deviceName = queryDeviceName();
  const std::string shapeText = shapeToString(flags.shape);
  const std::string cacheKey = kernelh::compiler::sha256_hex(
      flags.kernelName + "|" + shapeText + "|" + deviceName);
  std::cout << "device: " << deviceName << "\n";
  std::cout << "cache-key: " << cacheKey << "\n";

  kernelh::compiler::TuneOptions options;
  options.sample_cap = flags.sampleCap;
  options.refine_budget = flags.refineBudget;
  options.max_candidates = flags.maxCandidates;

  TuneEvalRequest baseReq;
  baseReq.kernelName = flags.kernelName;
  baseReq.shape = flags.shape;
  baseReq.target = flags.target;
  const char *token = deviceTokenFor(flags.target.c_str());
  baseReq.deviceToken = token != nullptr ? token : "gfx1101";

  kernelh::compiler::TuneEvaluator evaluator =
      [&](const kernelh::compiler::GemmCandidate &config) {
        TuneEvalRequest req = baseReq;
        req.config = config;
        kernelh::compiler::EvaluatedCandidate result =
            evaluateCandidateWithTimeout(req, flags.timeoutSeconds);
        if (result.accepted) {
          std::cout << "reverify: PASS " << config.to_string() << "\n";
        }
        return result;
      };

  std::error_code ec;
  std::filesystem::create_directories(flags.outDir, ec);
  if (ec) {
    std::cerr << "error: cannot create --out dir '" << flags.outDir
              << "': " << ec.message() << "\n";
    return kExitOperational;
  }

  const kernelh::compiler::TuneRun run = kernelh::compiler::autotune(
      options, evaluator, cacheKey, flags.cachePath, std::cout);

  const std::filesystem::path csvPath =
      std::filesystem::path(flags.outDir) / "tune-results.csv";
  if (!kernelh::compiler::write_tune_csv(csvPath, flags.kernelName, shapeText,
                                         run.evaluated)) {
    std::cerr << "error: cannot write '" << csvPath.string() << "'\n";
    return kExitOperational;
  }
  std::cout << "wrote " << csvPath.string() << " (" << run.evaluated.size()
            << " rows)\n";

  if (run.best_index >= 0) {
    const kernelh::compiler::EvaluatedCandidate &best =
        run.evaluated[static_cast<size_t>(run.best_index)];
    std::cout << "best config: " << best.config.to_string()
              << " ms_median=" << best.ms_median << "\n";
  } else {
    std::cout << "best config: none (no candidate passed all gates)\n";
  }
  if (run.cache_hit) {
    std::cout << "tune ok (served from cache)\n";
  } else {
    std::cout << "tune ok\n";
  }
  return kExitOk;
}

} // namespace


// explore/bench stubs: accepted so the CLI shape is stable, but they only
// report the plan todo that implements them (37/35) and exit 2. Extra
// arguments are ignored after a warning (their real flags arrive with their
// todos). `tune` is fully implemented above (plan todo 34). `bench` is fully
// implemented below (plan todo 35) via the bench harness library.
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

// Scans argv[2..] for the bench flags (--suite/--out; --help handled in the
// loop). On a malformed invocation an error is printed and false is returned
// (caller prints usage and exits 2).
bool scanBenchArgs(int argc, char **argv, kernelh::bench::BenchFlags &flags) {
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(std::cout);
      std::exit(kExitOk);
    }
    std::string value;
    if (takeValue(argc, argv, i, arg, "--suite", value)) {
      flags.suite = value;
    } else if (takeValue(argc, argv, i, arg, "--out", value)) {
      flags.out = value;
    } else {
      std::cerr << "error: unknown flag '" << arg << "' for bench\n";
      return false;
    }
  }
  if (flags.out.empty()) {
    std::cerr << "error: bench requires --out <path> (a .csv file or a dir)\n";
    return false;
  }
  return true;
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
    kernelh::bench::BenchFlags flags;
    if (!scanBenchArgs(argc, argv, flags)) {
      printUsage(std::cerr);
      return kExitUsage;
    }
    return kernelh::bench::run_bench(flags);
  }
  if (subcommand == "tune") {
    TuneFlags flags;
    if (!scanTuneArgs(argc, argv, flags)) {
      printUsage(std::cerr);
      return kExitUsage;
    }
    return runTune(flags);
  }
  std::cerr << "error: unknown subcommand '" << subcommand << "'\n";
  printUsage(std::cerr);
  return kExitUsage;
}
