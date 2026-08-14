# kernelh_bench - the benchmark harness (plan todo 35): G32 CSV writer,
# metric formulas, warmup+median timing, roofline microbenchmarks, suite
# registry, the kernelh/rocblas/hipref backends, and the `kernelh bench`
# orchestration. Linked into the kernelh CLI and the unit tests.
add_library(kernelh_bench STATIC
  clock_pin.cpp
  csv.cpp
  hipref_backend.cpp
  kernelh_backend.cpp
  metrics.cpp
  refs_loader.cpp
  rocblas_backend.cpp
  roofline.cpp
  run_bench.cpp
  shape_text.cpp
  suites.cpp
  timer.cpp
  util.cpp
)

target_include_directories(kernelh_bench PUBLIC
  "${CMAKE_SOURCE_DIR}/bench/harness"
  "${CMAKE_SOURCE_DIR}/include")

# Timing + launch go through the runtime abstraction (EventTimer /
# Backend / Buffer / GpuModule - no HIP headers needed in the harness).
# The kernelh backend additionally links the compile chain (DSL registry ->
# default pipeline -> emitter -> HSACO), the same set as the CLI. WHOLE_ARCHIVE
# on kernelh_passes so every self-registering pass initializer runs.
target_link_libraries(kernelh_bench PUBLIC
  kernelh_runtime
  $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_passes>
  kernelh_dsl kernelh_ir kernelh_compiler kernelh_emitter kernelh_backends)

# rocblas (plan todo 35 backend (b)): the LOCAL GEMM baseline via
# rocblas_gemm_ex + rocblas_gemm_algo_standard. Linked when present in the
# devenv (rocmPackages.rocblas); without it rocblas_backend.cpp compiles to
# an honest per-row error. The devenv puts rocblas on the nix CMAKE prefix
# path; HIP_PATH is an extra hint for non-nix layouts.
find_path(KERNELH_ROCBLAS_INCLUDE_DIR rocblas/rocblas.h
  HINTS "$ENV{HIP_PATH}/../../rocblas/include" "$ENV{HIP_PATH}/include")
find_library(KERNELH_ROCBLAS_LIBRARY NAMES rocblas
  HINTS "$ENV{HIP_PATH}/../../rocblas/lib")
if(KERNELH_ROCBLAS_INCLUDE_DIR AND KERNELH_ROCBLAS_LIBRARY)
  target_compile_definitions(kernelh_bench PUBLIC KERNELH_HAVE_ROCBLAS=1)
  target_include_directories(kernelh_bench PUBLIC ${KERNELH_ROCBLAS_INCLUDE_DIR})
  target_link_libraries(kernelh_bench PUBLIC ${KERNELH_ROCBLAS_LIBRARY})
  message(STATUS "kernelh_bench: rocblas found (${KERNELH_ROCBLAS_LIBRARY})")
else()
  message(WARNING
    "kernelh_bench: rocblas NOT found - rocblas GEMM baseline disabled")
endif()

# The hipcc-compiled refs build dir baked in as KERNELH_REFS_DIR (public so
# consumers of kernelh_bench get it too). Set by bench/CMakeLists.txt.
if(KERNELH_REFS_DIR)
  target_compile_definitions(kernelh_bench PUBLIC
    KERNELH_REFS_DIR="${KERNELH_REFS_DIR}")
endif()

# The nvcc-built cross harness path (cuBLAS + CUTLASS), set by
# bench/CMakeLists.txt when nvcc is available.
if(KERNELH_CROSS_BIN)
  target_compile_definitions(kernelh_bench PUBLIC
    KERNELH_CROSS_BIN="${KERNELH_CROSS_BIN}")
endif()
