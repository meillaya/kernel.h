# kernelh_dsl - the fluent schedule builder + kernel template registry
# (plan todo 11). Builds atom IR through kernelh_ir; kernel template bodies
# live under kernels/ (vecadd is the first registration; gemm/softmax/...
# land in todos 25-31).
add_library(kernelh_dsl STATIC
  program.cpp
  registry.cpp
  kernels/vecadd.cpp
  kernels/gemm.cpp
  kernels/layernorm.cpp
  kernels/softmax.cpp
  kernels/scan.cpp
  kernels/fused_gelu.cpp
  kernels/attention.cpp
)

target_link_libraries(kernelh_dsl PUBLIC kernelh_ir)
target_include_directories(kernelh_dsl PUBLIC "${CMAKE_SOURCE_DIR}/include")
# Internal headers (kernels/kernels.hpp) are reachable from within the lib.
target_include_directories(kernelh_dsl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
