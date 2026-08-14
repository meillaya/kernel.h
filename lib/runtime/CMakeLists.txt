# kernelh_runtime - the HOST-side GPU runtime (plan todo 23): HIP loader /
# launcher / buffers / events (local gfx1101 execution, todo 6 gate) plus a
# CUDA Driver API backend bound at RUNTIME via dlopen("libcuda.so.1")
# (cross-ready, plan G10 - libcuda is NEVER linked at build time).
#
# Build config (plan Metis G39): plain clang from llvmPackages_21 + the clr
# include dir + -lamdhip64 (host-only API; hipcc NOT needed). Both paths are
# discovered from the devenv environment variable HIP_PATH (HIP_PATH =
# <nix-store>/clr-7.2.3 -> include/ + lib/libamdhip64.so).
#
# Memory is plain hipMalloc/hipFree/hipMemcpy ONLY (plan D11: hipMallocAsync /
# hipMallocFromPoolAsync are banned - in-tree ROCm driver regression
# ROCm/ROCM#6178); the todo-23 acceptance grep on this archive must find zero
# hipMallocAsync references.
#
# STRONG WEAK-HOOK OVERRIDES: lib/runtime/query_cu_count.cpp defines the strong
# kernelh::query_cu_count_runtime() (todo-10 weak-hook contract) and
# lib/runtime/query_occupancy.cpp the strong
# kernelh::compiler::query_occupancy_max_blocks_runtime() (todo-32
# launch-verifier contract) that must win over the weak defaults in
# lib/compiler at FINAL LINK time. Plain archive extraction keeps the WEAK
# definitions (a weak def satisfies the reference, so the member is never
# pulled in - verified empirically, see the todo-23 evidence), so every
# consumer must force this library into the final link with
# $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_runtime> (kernelh_unit does).

set(KERNELH_HIP_PATH "$ENV{HIP_PATH}")
if(KERNELH_HIP_PATH STREQUAL "")
  message(FATAL_ERROR
    "HIP_PATH env var is empty; configure inside `devenv shell` "
    "(devenv.nix exports it as the clr package path).")
endif()

add_library(kernelh_runtime STATIC
  backend.cpp
  backend_hip.cpp
  backend_cuda.cpp
  query_cu_count.cpp
  query_occupancy.cpp
  descriptor_packing.cpp
  device_context.cpp
  buffer.cpp
  gpu_module.cpp
  kernel_launch.cpp
  event_timer.cpp
)

target_include_directories(kernelh_runtime PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(kernelh_runtime PRIVATE
  "${KERNELH_HIP_PATH}/include")

# The launch path (kernel_launch.cpp) pre-verifies every launch against the
# device-limits table through the MLIR-free launch verifier (plan todo 32).
# kernelh_compiler stays HIP-free (its occupancy hook is resolved through the
# weak symbol defined in this library).
target_link_libraries(kernelh_runtime PRIVATE kernelh_compiler)

# hipcc auto-defines __HIP_PLATFORM_AMD__ only for HIP sources; plain clang
# needs it manually or <hip/hip_runtime_api.h> #errors (todo-6 recipe).
target_compile_definitions(kernelh_runtime PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(kernelh_runtime PRIVATE
  "${KERNELH_HIP_PATH}/lib/libamdhip64.so")

# dlopen/dlsym live in libc on glibc >= 2.34 (the devenv glibc is 2.42).
