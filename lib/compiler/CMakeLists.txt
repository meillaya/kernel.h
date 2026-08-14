# kernelh_compiler - compiler-side helpers that do NOT depend on the custom IR
# or on MLIR. The device-limits table (plan todo 10) lives here because it is
# consumed by the passes (lib/passes), the launch-config verifier, and the
# occupancy calculations without dragging in MLIR.
#
# IMPORTANT: this library intentionally never links HIP/ROCm. The runtime CU
# count is resolved at FINAL LINK time through the weak-symbol hook
# kernelh::query_cu_count_runtime() (see
# include/kernelh/compiler/device_limits.hpp for the contract); plan todo 23's
# runtime library provides the strong override.
add_library(kernelh_compiler STATIC device_limits.cpp)

# Header search: repo-root include/ (kernelh/compiler/device_limits.hpp).
target_include_directories(kernelh_compiler PUBLIC "${CMAKE_SOURCE_DIR}/include")
