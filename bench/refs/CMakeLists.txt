# Reference HIP kernels (plan todo 35): hand-written .hip files compiled with
# `hipcc --genco --offload-arch=gfx1101` into COF code objects that the bench
# harness loads via hipModuleLoadData (GpuModule::load) and launches with the
# raw-pointer ABI. roofline.hip is the todo-35 minimum (Oracle N8 microbenches);
# the softmax/layernorm/scan/fused_gelu naive + vectorized-4 refs and the naive
# tiled GEMM land in later phases of todo 35.
find_program(HIPCC hipcc)

if(HIPCC)
  set(KERNELH_REF_ARCH "gfx1101")
  set(KERNELH_REFS_OUT "${CMAKE_BINARY_DIR}/bench/refs")

  add_custom_command(
    OUTPUT "${KERNELH_REFS_OUT}/roofline.o"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${KERNELH_REFS_OUT}"
    COMMAND "${HIPCC}" --genco --offload-arch=${KERNELH_REF_ARCH}
            "${CMAKE_CURRENT_SOURCE_DIR}/roofline.hip"
            -o "${KERNELH_REFS_OUT}/roofline.o"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/roofline.hip"
    COMMENT "hipcc --genco roofline.hip -> roofline.o")

  add_custom_target(bench_refs ALL DEPENDS "${KERNELH_REFS_OUT}/roofline.o")
else()
  message(WARNING
    "hipcc not found in the devenv env; bench refs (roofline) will not build")
endif()
