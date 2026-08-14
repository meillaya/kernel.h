# Catch2 v3 unit tests (own main() in main.cpp).
include(Catch)

add_executable(kernelh_unit main.cpp gpu_gate_test.cpp device_limits_test.cpp)
target_link_libraries(kernelh_unit PRIVATE Catch2::Catch2 kernelh_compiler)

# Bake the degraded-mode flag path into the binary (path resolved in
# test/CMakeLists.txt from $ENV{KERNELH_DEGRADED_FLAG}).
target_compile_definitions(kernelh_unit PRIVATE
  KERNELH_DEGRADED_FLAG="${KERNELH_DEGRADED_FLAG_PATH}")

# The gpu_gate cases mutate the shared flag file; serialize them under -j.
catch_discover_tests(kernelh_unit
  PROPERTIES RESOURCE_LOCK kernelh_degraded_flag)
