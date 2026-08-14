#include "kernelh_backend.hpp"

#include "metrics.hpp"
#include "shape_text.hpp"
#include "timer.hpp"
#include "util.hpp"

#include "kernelh/compiler/backends.hpp"
#include "kernelh/compiler/device_limits.hpp"
#include "kernelh/compiler/emitter.hpp"
#include "kernelh/compiler/occupancy.hpp"
#include "kernelh/compiler/resource_reporter.hpp"
#include "kernelh/Dialect/Atom/IR/AtomDialect.h"
#include "kernelh/dsl/program.hpp"
#include "kernelh/ir/verifier.hpp"
#include "kernelh/passes/pipeline.hpp"
#include "kernelh/runtime/buffer.hpp"
#include "kernelh/runtime/device_context.hpp"
#include "kernelh/runtime/gpu_module.hpp"
#include "kernelh/runtime/kernel_launch.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kernelh::bench {

namespace {

// Dialect set the emitter touches (mirror of the CLI's registerCompileDialects
// - the bench harness is a separate TU and cannot see main.cpp's copy).
void registerCompileDialects(mlir::MLIRContext &ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::kernelh::atom::AtomDialect, mlir::gpu::GPUDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::vector::VectorDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
}

// In-process compile of one DSL kernel for the given CLI --target token.
struct CompiledKernel {
  std::vector<char> hsaco; // amdgcn blob (empty for nvptx targets)
  std::string ptx;         // nvptx text (empty for amdgcn targets)
  std::string error;
};

// device-limits token for a CLI --target token ("amdgcn-gfx1101" -> "gfx1101",
// "nvptx-sm_80" -> "sm_80"); bare tokens pass through.
std::string device_token(const std::string &target) {
  if (target == "amdgcn-gfx1101") {
    return "gfx1101";
  }
  if (target == "nvptx-sm_80") {
    return "sm_80";
  }
  return target;
}

CompiledKernel compileKernelFor(const std::string &name,
                                const std::string &shapeText,
                                const std::string &target) {
  CompiledKernel out;
  const ShapeMap shape = parse_shape_text(shapeText);
  if (shape.empty()) {
    out.error = "malformed shape '" + shapeText + "'";
    return out;
  }
  kernelh::ir::Module module;
  try {
    kernelh::ProgramBuilder builder = kernelh::program();
    builder.kernel(name).shape(kernelh::ShapeMap(shape.begin(), shape.end()));
    kernelh::EmittedKernel emitted = builder.emit();
    module = std::move(emitted.module);
  } catch (const std::exception &e) {
    out.error = e.what();
    return out;
  }
  kernelh::DeviceLimits limits;
  try {
    limits = kernelh::device_limits_for(device_token(target).c_str());
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
  mlir::MLIRContext ctx;
  registerCompileDialects(ctx);
  auto gpu = kernelh::compiler::emitGpuModule(module, ctx);
  if (!gpu) {
    out.error = gpu.error();
    return out;
  }
  auto spec = kernelh::compiler::parseTargetSpec(target);
  if (!spec) {
    out.error = spec.error();
    return out;
  }
  if (spec->isNVPTX()) {
    auto ptx = kernelh::compiler::compileToText(*gpu, *spec, "ptx");
    if (!ptx) {
      out.error = ptx.error();
      return out;
    }
    out.ptx = std::move(*ptx);
  } else {
    auto hsaco = kernelh::compiler::compileToBlob(*gpu, *spec, "hsaco");
    if (!hsaco) {
      out.error = hsaco.error();
      return out;
    }
    out.hsaco = std::move(*hsaco);
  }
  return out;
}

CompiledKernel compileKernel(const std::string &name,
                             const std::string &shapeText) {
  return compileKernelFor(name, shapeText, "amdgcn-gfx1101");
}

struct LoadedKernel {
  std::optional<kernelh::runtime::GpuModule> module;
  kernelh::runtime::KernelFunction fn;
};

// Loads `blob` and resolves `symbol`; "" on success.
std::string loadKernel(const std::shared_ptr<kernelh::runtime::Backend> &backend,
                       const std::vector<char> &blob, const char *symbol,
                       LoadedKernel &out) {
  auto module = kernelh::runtime::GpuModule::load(backend, blob.data(), blob.size());
  if (!module) {
    return "module load failed: " + module.error().message;
  }
  auto fn = module->get_function(symbol);
  if (!fn) {
    return std::string("get_function(\"") + symbol +
           "\") failed: " + fn.error().message;
  }
  out.module.emplace(std::move(*module));
  out.fn = *fn;
  return "";
}

// Convenience: compiles + loads a kernel, returns the compiled blob too (the
// caller keeps the blob alive only as long as needed - the module owns the
// loaded image).
std::string compileAndLoad(const std::string &name, const std::string &shapeText,
                           const std::shared_ptr<kernelh::runtime::Backend> &backend,
                           LoadedKernel &out) {
  CompiledKernel compiled = compileKernel(name, shapeText);
  if (!compiled.error.empty()) {
    return compiled.error;
  }
  return loadKernel(backend, compiled.hsaco, name.c_str(), out);
}

// Times `warmup` + `iterations` repeated launches of a one-shot closure.
double timeIt(const std::shared_ptr<kernelh::runtime::Backend> &backend,
              const std::function<kernelh::runtime::Result<void>()> &one_launch) {
  return time_launch_median(backend, one_launch, /*warmup=*/5,
                            /*iterations=*/20);
}

BenchRow makeRow(const RowSpec &spec, const std::string &deviceName, double ms,
                 double rate) {
  BenchRow row;
  row.kernel = spec.kernel;
  row.shape = spec.shapeText;
  row.backend = spec.backend;
  row.ms_median = ms;
  row.gflops_or_gbps = rate;
  row.timestamp = now_timestamp();
  row.device = deviceName;
  return row;
}

std::optional<BenchRow> runGemm(const RowSpec &spec,
                                const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                const std::string &deviceName,
                                std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long m = shape.at("m"), n = shape.at("n"), k = shape.at("k");
  LoadedKernel lk;
  error = compileAndLoad("gemm", spec.shapeText, backend, lk);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto a = kernelh::runtime::Buffer::allocate(backend, m * k * 4);
  auto b = kernelh::runtime::Buffer::allocate(backend, k * n * 4);
  auto c = kernelh::runtime::Buffer::allocate(backend, m * n * 4);
  if (!a || !b || !c) {
    error = "gemm buffer allocation failed";
    return std::nullopt;
  }
  // Default template tile is 32; block = 256 (todo-26 launch config).
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>((m + 31) / 32)).block(256u);
  launch.memref_arg(a->device_ptr(), std::vector<int64_t>{m, k});
  launch.memref_arg(b->device_ptr(), std::vector<int64_t>{k, n});
  launch.memref_arg(c->device_ptr(), std::vector<int64_t>{m, n});
  const auto oneLaunch = [&]() { return launch.launch(lk.fn); };
  const double ms = timeIt(backend, oneLaunch);
  if (ms < 0.0) {
    error = "gemm timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms, gflops_from_ms(gemm_flops(m, n, k), ms));
}

std::optional<BenchRow> runSoftmax(const RowSpec &spec,
                                   const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                   const std::string &deviceName,
                                   std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long rows = shape.at("rows"), cols = shape.at("cols");
  LoadedKernel lk;
  error = compileAndLoad("softmax", spec.shapeText, backend, lk);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  if (!x || !y) {
    error = "softmax buffer allocation failed";
    return std::nullopt;
  }
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(rows))
      .block(static_cast<unsigned>(std::min(cols, 256L)));
  launch.memref_arg(x->device_ptr(), std::vector<int64_t>{rows, cols});
  launch.memref_arg(y->device_ptr(), std::vector<int64_t>{rows, cols});
  const auto oneLaunch = [&]() { return launch.launch(lk.fn); };
  const double ms = timeIt(backend, oneLaunch);
  if (ms < 0.0) {
    error = "softmax timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms,
                 gbps_from_ms(elementwise_bytes(rows, cols), ms));
}

std::optional<BenchRow> runLayernorm(const RowSpec &spec,
                                     const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                     const std::string &deviceName,
                                     std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long rows = shape.at("rows"), cols = shape.at("cols");
  LoadedKernel lk;
  error = compileAndLoad("layernorm", spec.shapeText, backend, lk);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  auto gamma = kernelh::runtime::Buffer::allocate(backend, cols * 4);
  auto beta = kernelh::runtime::Buffer::allocate(backend, cols * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, rows * cols * 4);
  if (!x || !gamma || !beta || !y) {
    error = "layernorm buffer allocation failed";
    return std::nullopt;
  }
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(rows)).block(256u);
  launch.memref_arg(x->device_ptr(), std::vector<int64_t>{rows, cols});
  launch.memref_arg(gamma->device_ptr(), std::vector<int64_t>{cols});
  launch.memref_arg(beta->device_ptr(), std::vector<int64_t>{cols});
  launch.memref_arg(y->device_ptr(), std::vector<int64_t>{rows, cols});
  const auto oneLaunch = [&]() { return launch.launch(lk.fn); };
  const double ms = timeIt(backend, oneLaunch);
  if (ms < 0.0) {
    error = "layernorm timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms,
                 gbps_from_ms(layernorm_bytes(rows, cols), ms));
}

std::optional<BenchRow> runScan(const RowSpec &spec,
                                const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                const std::string &deviceName,
                                std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long n = shape.at("n");
  const long t = (n + 255) / 256; // 256-thread workgroups (todo-29 template)
  LoadedKernel lk1, lk2;
  error = compileAndLoad("scan", spec.shapeText, backend, lk1);
  if (!error.empty()) {
    return std::nullopt;
  }
  error = compileAndLoad("scan_totals", spec.shapeText, backend, lk2);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto ylo = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto thi = kernelh::runtime::Buffer::allocate(backend, t * 4);
  auto tlo = kernelh::runtime::Buffer::allocate(backend, t * 4);
  auto offh = kernelh::runtime::Buffer::allocate(backend, t * 4);
  auto offl = kernelh::runtime::Buffer::allocate(backend, t * 4);
  auto off2 = kernelh::runtime::Buffer::allocate(backend, t * 4);
  if (!x || !y || !ylo || !thi || !tlo || !offh || !offl || !off2) {
    error = "scan buffer allocation failed";
    return std::nullopt;
  }
  // Pass 1: grid = ceil(n/256), 5 rank-1 memrefs (todo-29 layout).
  kernelh::runtime::KernelLaunch l1(backend);
  l1.grid(static_cast<unsigned>(t)).block(256u);
  l1.memref_arg(x->device_ptr(), std::vector<int64_t>{n});
  l1.memref_arg(y->device_ptr(), std::vector<int64_t>{n});
  l1.memref_arg(ylo->device_ptr(), std::vector<int64_t>{n});
  l1.memref_arg(thi->device_ptr(), std::vector<int64_t>{t});
  l1.memref_arg(tlo->device_ptr(), std::vector<int64_t>{t});
  // Pass 2: grid = 1, 7 rank-1 memrefs.
  kernelh::runtime::KernelLaunch l2(backend);
  l2.grid(1u).block(256u);
  l2.memref_arg(thi->device_ptr(), std::vector<int64_t>{t});
  l2.memref_arg(tlo->device_ptr(), std::vector<int64_t>{t});
  l2.memref_arg(offh->device_ptr(), std::vector<int64_t>{t});
  l2.memref_arg(offl->device_ptr(), std::vector<int64_t>{t});
  l2.memref_arg(off2->device_ptr(), std::vector<int64_t>{t});
  l2.memref_arg(y->device_ptr(), std::vector<int64_t>{n});
  l2.memref_arg(ylo->device_ptr(), std::vector<int64_t>{n});
  const auto oneLaunch = [&]() -> kernelh::runtime::Result<void> {
    auto r1 = l1.launch(lk1.fn);
    if (!r1) {
      return r1;
    }
    return l2.launch(lk2.fn);
  };
  const double ms = timeIt(backend, oneLaunch);
  if (ms < 0.0) {
    error = "scan timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms, gbps_from_ms(scan_bytes(n), ms));
}

std::optional<BenchRow> runFusedGelu(const RowSpec &spec,
                                     const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                     const std::string &deviceName,
                                     std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long n = shape.at("n");
  LoadedKernel lk;
  error = compileAndLoad("fused_gelu", spec.shapeText, backend, lk);
  if (!error.empty()) {
    return std::nullopt;
  }
  auto x = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto tmp = kernelh::runtime::Buffer::allocate(backend, n * 4);
  auto y = kernelh::runtime::Buffer::allocate(backend, n * 4);
  if (!x || !tmp || !y) {
    error = "fused_gelu buffer allocation failed";
    return std::nullopt;
  }
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>((n + 255) / 256)).block(256u);
  launch.memref_arg(x->device_ptr(), std::vector<int64_t>{n});
  launch.memref_arg(tmp->device_ptr(), std::vector<int64_t>{n});
  launch.memref_arg(y->device_ptr(), std::vector<int64_t>{n});
  const auto oneLaunch = [&]() { return launch.launch(lk.fn); };
  const double ms = timeIt(backend, oneLaunch);
  if (ms < 0.0) {
    error = "fused_gelu timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms, gbps_from_ms(fused_gelu_bytes(n), ms));
}

std::optional<BenchRow> runAttention(const RowSpec &spec,
                                     const std::shared_ptr<kernelh::runtime::Backend> &backend,
                                     const std::string &deviceName,
                                     std::string &error) {
  const ShapeMap shape = parse_shape_text(spec.shapeText);
  const long batch = shape.at("batch"), heads = shape.at("heads");
  const long seq = shape.at("seq"), headDim = shape.at("head_dim");
  LoadedKernel lk;
  error = compileAndLoad("attention", spec.shapeText, backend, lk);
  if (!error.empty()) {
    return std::nullopt;
  }
  const size_t bytes = batch * heads * seq * headDim * 4;
  auto q = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto k = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto v = kernelh::runtime::Buffer::allocate(backend, bytes);
  auto o = kernelh::runtime::Buffer::allocate(backend, bytes);
  if (!q || !k || !v || !o) {
    error = "attention buffer allocation failed";
    return std::nullopt;
  }
  kernelh::runtime::KernelLaunch launch(backend);
  launch.grid(static_cast<unsigned>(seq), static_cast<unsigned>(heads),
              static_cast<unsigned>(batch))
      .block(64u);
  const std::vector<int64_t> shape4{batch, heads, seq, headDim};
  launch.memref_arg(q->device_ptr(), shape4);
  launch.memref_arg(k->device_ptr(), shape4);
  launch.memref_arg(v->device_ptr(), shape4);
  launch.memref_arg(o->device_ptr(), shape4);
  const auto oneLaunch = [&]() { return launch.launch(lk.fn); };
  const double ms = timeIt(backend, oneLaunch);
  if (ms < 0.0) {
    error = "attention timing failed";
    return std::nullopt;
  }
  return makeRow(spec, deviceName, ms,
                 gflops_from_ms(attention_flops(batch, heads, seq, headDim), ms));
}

// ---------------------------------------------------------------------------
// Plan todo 36 occupancy generation (consumed by `kernelh bench --occupancy`).
// ---------------------------------------------------------------------------

// Per-kernel default shapes for the occupancy-only mode: the DSL templates
// REQUIRE shape keys, and the todo-36 acceptance `bench --occupancy --kernel
// gemm --target amdgcn-gfx1101` passes none, so these defaults kick in. The
// occupancy does not depend strongly on the shape (the default schedule's
// resources are what matter), and the config text is echoed into the report.
std::string default_shape_for(const std::string &kernel) {
  if (kernel == "gemm") {
    return "m=512,n=512,k=512";
  }
  if (kernel == "softmax") {
    return "rows=4096,cols=1024";
  }
  if (kernel == "layernorm") {
    return "rows=4096,cols=1024";
  }
  if (kernel == "scan") {
    return "n=100000";
  }
  if (kernel == "fused_gelu") {
    return "n=1048576";
  }
  if (kernel == "attention") {
    return "batch=2,heads=8,seq=512,head_dim=64";
  }
  if (kernel == "vecadd") {
    return "n=1024";
  }
  return "";
}

// Default launch workgroup size for the occupancy-only mode (mirrors the
// todo-35 launch configs in the run* functions).
int default_threads_for(const std::string &kernel) {
  if (kernel == "attention") {
    return 64;
  }
  return 256;
}

// The launch workgroup size the todo-35 bench uses for a kernelh row (mirrors
// the run* functions' block() calls in this file).
int launch_threads_for(const RowSpec &spec) {
  if (spec.kernel == "attention") {
    return 64;
  }
  if (spec.kernel == "softmax") {
    const ShapeMap shape = parse_shape_text(spec.shapeText);
    const long cols = shape.count("cols") ? shape.at("cols") : 256;
    return static_cast<int>(std::min(cols, 256L));
  }
  return 256;
}

// Core: builds the occupancy report for one kernel+config+target. Writes the
// compiled artifact into `scratch` (the todo-33 reporter reads
// <scratch>/<kernel>.{hsaco,ptx}). `backend` may be null (analytic only); the
// measured path loads the HSACO and hands the function handle to the todo-32
// weak hook (which returns -1 on this box - the vendor API is broken - so
// measured_pct degrades to 0 with a note).
std::optional<kernelh::compiler::OccupancyReport> build_occupancy_report(
    const std::string &kernel, const std::string &target,
    const std::string &shapeText, int threads,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::filesystem::path &scratch, std::string &error) {
  std::string shape =
      shapeText.empty() ? default_shape_for(kernel) : shapeText;
  if (shape.empty()) {
    error = "unknown kernel '" + kernel +
            "' for --occupancy (no default shape); use --shape k=v,...";
    return std::nullopt;
  }
  const ShapeMap shapeMap = parse_shape_text(shape);
  if (shapeMap.empty()) {
    error = "malformed shape '" + shape + "'";
    return std::nullopt;
  }

  CompiledKernel compiled = compileKernelFor(kernel, shape, target);
  if (!compiled.error.empty()) {
    error = compiled.error;
    return std::nullopt;
  }
  if (target.rfind("amdgcn", 0) == 0) {
    const std::filesystem::path hsaco = scratch / (kernel + ".hsaco");
    std::ofstream out(hsaco, std::ios::binary);
    if (!out) {
      error = "cannot write '" + hsaco.string() + "'";
      return std::nullopt;
    }
    out.write(compiled.hsaco.data(),
              static_cast<std::streamsize>(compiled.hsaco.size()));
  } else {
    const std::filesystem::path ptx = scratch / (kernel + ".ptx");
    std::ofstream out(ptx, std::ios::binary);
    if (!out) {
      error = "cannot write '" + ptx.string() + "'";
      return std::nullopt;
    }
    out.write(compiled.ptx.data(),
              static_cast<std::streamsize>(compiled.ptx.size()));
  }

  auto resources =
      kernelh::compiler::resource_report(kernel, target, scratch);
  if (!resources) {
    error = resources.error();
    return std::nullopt;
  }

  kernelh::DeviceLimits limits;
  try {
    limits = kernelh::device_limits_for(device_token(target).c_str());
  } catch (const std::invalid_argument &e) {
    error = e.what();
    return std::nullopt;
  }

  kernelh::compiler::OccupancyInput input;
  input.kernel = kernel;
  input.target = device_token(target);
  input.config = shape;
  input.vgpr_count = resources->vgpr_count;
  input.lds_bytes = resources->lds_bytes;
  input.threads_per_workgroup =
      threads > 0 ? threads : default_threads_for(kernel);

  void *function = nullptr;
  if (backend && target.rfind("amdgcn", 0) == 0 && !compiled.hsaco.empty()) {
    auto module = kernelh::runtime::GpuModule::load(
        backend, compiled.hsaco.data(), compiled.hsaco.size());
    if (module) {
      auto fn = module->get_function(kernel.c_str());
      if (fn) {
        function = fn->handle;
      }
    }
  }
  return kernelh::compiler::occupancy_report(input, limits, function);
}

} // namespace

std::optional<BenchRow> run_kernelh_row(
    const RowSpec &spec,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &deviceName, std::string &error) {
  if (spec.kernel == "gemm") {
    return runGemm(spec, backend, deviceName, error);
  }
  if (spec.kernel == "softmax") {
    return runSoftmax(spec, backend, deviceName, error);
  }
  if (spec.kernel == "layernorm") {
    return runLayernorm(spec, backend, deviceName, error);
  }
  if (spec.kernel == "scan") {
    return runScan(spec, backend, deviceName, error);
  }
  if (spec.kernel == "fused_gelu") {
    return runFusedGelu(spec, backend, deviceName, error);
  }
  if (spec.kernel == "attention") {
    return runAttention(spec, backend, deviceName, error);
  }
  error = "kernelh backend has no runner for kernel '" + spec.kernel + "'";
  return std::nullopt;
}

int write_occupancy_for_kernel(const std::string &kernel,
                               const std::string &target,
                               const std::string &shapeText, int threads,
                               const std::filesystem::path &outDir,
                               std::string &error) {
  std::error_code ec;
  std::filesystem::create_directories(outDir, ec);
  if (ec) {
    error = "cannot create output dir '" + outDir.string() +
            "': " + ec.message();
    return 1;
  }

  const std::filesystem::path scratch =
      std::filesystem::temp_directory_path() / ("kernelh-t36-" + kernel);
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  // Best-effort device for the measured path; the analytic occupancy works
  // without one (and the vendor API is broken on this box anyway, so the
  // measured part degrades to a note).
  std::shared_ptr<kernelh::runtime::Backend> backend;
  if (target.rfind("amdgcn", 0) == 0) {
    backend = kernelh::runtime::make_hip_backend();
    if (backend) {
      auto ctx = kernelh::runtime::DeviceContext::create(backend);
      if (!ctx) {
        backend.reset();
      }
    }
  }

  auto report = build_occupancy_report(kernel, target, shapeText, threads,
                                       backend, scratch, error);
  std::filesystem::remove_all(scratch, ec);
  if (!report) {
    return 1;
  }
  if (!kernelh::compiler::write_occupancy_json(outDir / "occupancy.json",
                                               *report) ||
      !kernelh::compiler::write_occupancy_markdown(outDir / "occupancy.md",
                                                   {*report})) {
    error = "cannot write occupancy files in '" + outDir.string() + "'";
    return 1;
  }
  return 0;
}

int write_occupancy_for_rows(
    const std::vector<RowSpec> &rows,
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::filesystem::path &outDir, std::string &error) {
  std::error_code ec;
  std::filesystem::create_directories(outDir, ec);
  if (ec) {
    error = "cannot create output dir '" + outDir.string() +
            "': " + ec.message();
    return 1;
  }

  std::vector<kernelh::compiler::OccupancyReport> reports;
  reports.reserve(rows.size());
  int rowIndex = 0;
  for (const RowSpec &spec : rows) {
    if (spec.backend != "kernelh") {
      continue;
    }
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() /
        ("kernelh-t36-row-" + std::to_string(rowIndex++));
    std::filesystem::remove_all(scratch, ec);
    std::filesystem::create_directories(scratch, ec);
    auto report =
        build_occupancy_report(spec.kernel, "amdgcn-gfx1101", spec.shapeText,
                               launch_threads_for(spec), backend, scratch,
                               error);
    std::filesystem::remove_all(scratch, ec);
    if (!report) {
      return 1;
    }
    reports.push_back(*report);
  }
  if (reports.empty()) {
    error = "no kernelh rows in the suite for occupancy generation";
    return 1;
  }
  if (!kernelh::compiler::write_occupancy_json_all(outDir / "occupancy.json",
                                                   reports) ||
      !kernelh::compiler::write_occupancy_markdown(outDir / "occupancy.md",
                                                   reports)) {
    error = "cannot write occupancy files in '" + outDir.string() + "'";
    return 1;
  }
  return 0;
}

} // namespace kernelh::bench
