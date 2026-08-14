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
  pass_promote_shared_test.cpp
  pass_unroll_test.cpp
  pass_pipeline_test.cpp
  runtime_error_test.cpp
  runtime_descriptor_test.cpp
  runtime_device_test.cpp
  runtime_integration_test.cpp
  launch_verifier_test.cpp
  emitter_test.cpp
  backend_test.cpp
  resource_reporter_test.cpp
  autotuner_test.cpp
  canary_gpu_test.cpp
  e2e_vecadd_test.cpp
  e2e_gemm_test.cpp
  e2e_layernorm_test.cpp
  e2e_softmax_test.cpp
  e2e_scan_test.cpp
  e2e_fused_gelu_test.cpp
  e2e_attention_test.cpp
  # Plan todo 35 benchmark harness tests (G32 CSV schema + metric formulas).
  bench_csv_test.cpp
  bench_metrics_test.cpp
  bench_shape_text_test.cpp
  # Plan todo 36 occupancy tests (hand-computed analytic + [gpu] validation).
  occupancy_test.cpp
  # Plan todo 38 charts script test (shells out to scripts/charts.py on a
  # hand-computed fixture; asserts the >10KB PNG gate + AI formulas).
  charts_script_test.cpp
)
target_link_libraries(kernelh_unit PRIVATE Catch2::Catch2
  # WHOLE_ARCHIVE: force-load every pass object so its self-registration
  # initializer runs (see lib/passes/CMakeLists.txt), or the pass tests would
  # find an empty global registry.
  $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_passes>
  kernelh_compiler kernelh_ir kernelh_dsl
  # The MLIR emitter (plan todo 21): its PUBLIC MLIR/atom-dialect deps
  # propagate transitively.
  kernelh_emitter
  # The vendor backends (plan todo 22): NVVM/ROCDL lowering + blob extraction.
  kernelh_backends
  # WHOLE_ARCHIVE on kernelh_runtime too: its strong
  # query_cu_count_runtime() override (todo-10 weak-hook contract) must win
  # over the weak default in lib/compiler; plain archive extraction keeps the
  # weak definition (verified empirically, see todo-23 evidence).
  $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_runtime>
  # The todo-35 benchmark harness (G32 CSV writer, metric formulas, shape-text
  # parsing - the bench_* tests exercise it).
  kernelh_bench)

# Bake the degraded-mode flag path into the binary (path resolved in
# test/CMakeLists.txt from $ENV{KERNELH_DEGRADED_FLAG}) and the repo root (for
# the printer fixture-comparison tests reading test/fixtures).
target_compile_definitions(kernelh_unit PRIVATE
  KERNELH_DEGRADED_FLAG="${KERNELH_DEGRADED_FLAG_PATH}"
  KERNELH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  # The production compile driver (todo 24): the todo-25 e2e/canary tests
  # shell out to it via std::system, then load the produced HSACO through
  # kernelh_runtime (see kernelh_compile_util.hpp).
  KERNELH_BIN="${CMAKE_BINARY_DIR}/tools/kernelh/kernelh")

# The gpu_gate cases mutate the shared flag file; serialize them under -j.
catch_discover_tests(kernelh_unit
  PROPERTIES RESOURCE_LOCK kernelh_degraded_flag)

# Weak-hook default contract (plan todo 10 + todo 23): a runtime-free binary
# must keep the weak query_cu_count_runtime() default of 0. kernelh_unit
# cannot verify this (it links the strong override), so it runs here.
add_executable(kernelh_weak_default_test
  main.cpp
  weak_default_test.cpp
)
target_link_libraries(kernelh_weak_default_test PRIVATE
  Catch2::Catch2 kernelh_compiler)
catch_discover_tests(kernelh_weak_default_test)
