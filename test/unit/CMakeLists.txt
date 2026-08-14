# Catch2 v3 unit tests (own main() in main.cpp).
include(Catch)

add_executable(kernelh_unit
  main.cpp
  gpu_gate_test.cpp
  device_limits_test.cpp
  ir_types_test.cpp
  ir_verifier_test.cpp
  ir_printer_test.cpp
  ir_parser_test.cpp
  ir_pass_manager_test.cpp
  dsl_test.cpp
  pass_canonicalize_test.cpp
  pass_cse_test.cpp
  pass_tiling_test.cpp
  pass_fusion_test.cpp
  pass_vectorize_test.cpp
)
target_link_libraries(kernelh_unit PRIVATE Catch2::Catch2
  # WHOLE_ARCHIVE: force-load every pass object so its self-registration
  # initializer runs (see lib/passes/CMakeLists.txt), or the pass tests would
  # find an empty global registry.
  $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_passes>
  kernelh_compiler kernelh_ir kernelh_dsl)

# Bake the degraded-mode flag path into the binary (path resolved in
# test/CMakeLists.txt from $ENV{KERNELH_DEGRADED_FLAG}) and the repo root (for
# the printer fixture-comparison tests reading test/fixtures).
target_compile_definitions(kernelh_unit PRIVATE
  KERNELH_DEGRADED_FLAG="${KERNELH_DEGRADED_FLAG_PATH}"
  KERNELH_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# The gpu_gate cases mutate the shared flag file; serialize them under -j.
catch_discover_tests(kernelh_unit
  PROPERTIES RESOURCE_LOCK kernelh_degraded_flag)
