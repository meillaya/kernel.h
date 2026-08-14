# kernelh_bench - the benchmark harness (plan todo 35): G32 CSV writer,
# metric formulas, warmup+median timing, roofline microbenchmarks, suite
# registry, and the `kernelh bench` orchestration. Linked into the kernelh CLI
# and the unit tests.
add_library(kernelh_bench STATIC
  clock_pin.cpp
  csv.cpp
  metrics.cpp
  refs_loader.cpp
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
target_link_libraries(kernelh_bench PUBLIC kernelh_runtime)

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
