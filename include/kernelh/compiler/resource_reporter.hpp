// kernel.h compiler - resource reporter (plan todo 33).
//
// Produces a per-kernel resources.json next to the compiled artifacts:
//
//   {kernel, target, vgpr_count, sgpr_count, lds_bytes, wavefront_size,
//    occupancy_estimate}
//
// The JSON schema is the CONTRACT consumed by todo 34 (the autotuner's
// profiler-feedback seam, Metis G23) and todo 36 (analytic occupancy). It is
// therefore written from a single hand-rolled JSON writer here - the project
// has no JSON library and must not gain one (todo-33 MUST NOT).
//
// SOURCE PRIORITY (Metis G7, pinned):
//   1. rocprofv3 kernel-trace, when installed (`command -v rocprofv3`) AND a
//      micro-launch binary is configured (KERNELH_MICRO_LAUNCH env var, a
//      binary/script that loads the HSACO and launches the kernel). Expected
//      to be absent in the devenv; the reporter then falls back silently.
//   2. Static artifact parsing (the expected path):
//        AMD     `llvm-readobj --notes <kernel>.hsaco` -> the amdhsa kernel
//                descriptor keys .vgpr_count / .sgpr_count /
//                .group_segment_fixed_size / .wavefront_size.
//        NVIDIA  `$CUDA_TOOLKIT_ROOT/bin/ptxas -v <kernel>.ptx` as a SEPARATE
//                explicit step (Oracle N5: in-process serialization does NOT
//                expose ptxas stderr), parsing "Used N registers, M bytes
//                smem"; M is complemented by a static-shared scan of the PTX
//                text because ptxas does not report module-scope `.shared`
//                arrays (verified empirically on LLVM-emitted PTX).
//
// Occupancy is the analytic VGPR/register-limited waves-per-CU from the
// todo-10 device-limits table (the full theoretical occupancy lands with todo
// 36): floor(reg_file_per_cu / (vgpr_count x wavefront_size)) capped by
// waves_per_cu.
//
// kernelh_compiler stays MLIR-free (the todo-23 weak-default link depends on
// it), so this file uses only POSIX fork/exec/poll - no MLIR, no runtime.

#ifndef KERNELH_COMPILER_RESOURCE_REPORTER_HPP
#define KERNELH_COMPILER_RESOURCE_REPORTER_HPP

#include <expected>
#include <filesystem>
#include <string>
#include <string_view>

namespace kernelh::compiler {

// Resource usage of ONE compiled kernel (the resources.json payload schema,
// consumed by todo 34/36). Zeros mean "not available for this source path".
struct KernelResources {
  std::string kernel;
  std::string target; // CLI token, e.g. "amdgcn-gfx1101" or "nvptx-sm_80"
  int vgpr_count = 0;
  int sgpr_count = 0;
  long lds_bytes = 0;
  int wavefront_size = 0;
  int occupancy_estimate = 0;
};

// Parses a captured `llvm-readobj --notes <hsaco>` dump (the amdhsa metadata
// YAML). Pure text parsing - no subprocess. `kernel_hint` selects the
// matching amdhsa.kernels entry by its .name field; an empty hint takes the
// first entry. Returns an error containing "unparseable code object" when the
// text is truncated/corrupt or carries no amdhsa kernel descriptor.
std::expected<KernelResources, std::string>
parse_amdhsa_notes(std::string_view notes, std::string_view kernel_hint);

// Parses ptxas -v output ("Used N registers, M bytes smem") for the register
// count + reported smem, AND scans the PTX text for module-scope `.shared
// .align N .b8 name[bytes];` static shared arrays (ptxas does NOT report
// those - verified on LLVM-emitted PTX). lds_bytes = max(ptxas smem, static
// shared sum) so both nvcc-style (entry-scope shared, reported) and
// MLIR-style (module-scope shared, unreported) kernels give the right value.
// Registers map to vgpr_count; sgpr_count/wavefront_size stay 0 for NVIDIA
// (ptxas does not expose them).
std::expected<KernelResources, std::string>
parse_ptxas_report(std::string_view ptxas_output, std::string_view ptx_text);

// Runtime probe for source-priority-1: is rocprofv3 installed and on PATH in
// the current environment? (`command -v rocprofv3`.) Expected false in the
// devenv; the [gpu] test SKIPs when it is false.
bool rocprofv3_available();

// Parses a rocprofv3 kernel_trace.csv (ROCm 7.x `--output-format csv`:
// header contains Kernel_Name, VGPR_Count, SGPR_Count, LDS_Block_Size, ...).
// Selects the row whose Kernel_Name equals `kernel` and fills vgpr/sgpr/lds;
// columns are located BY NAME, so ordering does not matter. Errors when the
// CSV lacks the required columns or the kernel's row.
std::expected<KernelResources, std::string>
parse_kernel_trace_csv(std::string_view csv, std::string_view kernel);

// Full pipeline (todo-33 source priority). `target` is a CLI --target token
// ("amdgcn-gfx1101" / "nvptx-sm_80"); the artifacts live in `artifact_dir`
// as <artifact_dir>/<kernel>.hsaco (AMD) or <artifact_dir>/<kernel>.ptx
// (NVIDIA). Writes <artifact_dir>/resources.json with the single kernel's
// resources and returns them. Errors (missing artifact, unparseable code
// object, ptxas failure) are non-fatal to the CALLER's exit status only when
// the caller decides so - the CLI fails the compile; the NVIDIA ptxas-missing
// case fills zeros with a warning per the todo-33 contract.
std::expected<KernelResources, std::string>
resource_report(const std::string &kernel, const std::string &target,
                const std::filesystem::path &artifact_dir);

// Hand-rolled JSON writer for the resources.json schema (no JSON dependency).
// Returns false only on a write failure.
bool write_resources_json(const std::filesystem::path &artifact_dir,
                          const KernelResources &res);

} // namespace kernelh::compiler

#endif // KERNELH_COMPILER_RESOURCE_REPORTER_HPP
