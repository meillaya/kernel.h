# Reference HIP kernels (plan todo 35): hand-written .hip files compiled with
# `hipcc --genco --offload-arch=gfx1101` into COF code objects that the bench
# harness loads via hipModuleLoadData (GpuModule::load) and launches with the
# raw-pointer ABI.
#
# One COF per kernel family (each file carries the naive + vectorized-4
# variants as separate entry points): softmax/layernorm/scan/fused_gelu
# (hipref-naive + hipref-vec) and gemm (naive tiled). roofline.hip carries
# the todo-35 roofline microbenches (Oracle N8 ceilings for todo 38).
find_program(HIPCC hipcc)

if(HIPCC)
  set(KERNELH_REF_ARCH "gfx1101")
  set(KERNELH_REFS_OUT "${CMAKE_BINARY_DIR}/bench/refs")

  set(KERNELH_REF_SOURCES
    roofline.hip
    softmax.hip
    layernorm.hip
    scan.hip
    fused_gelu.hip
    gemm.hip)

  set(KERNELH_REF_OBJECTS "")
  foreach(src ${KERNELH_REF_SOURCES})
    get_filename_component(stem ${src} NAME_WE)
    set(out "${KERNELH_REFS_OUT}/${stem}.o")
    list(APPEND KERNELH_REF_OBJECTS "${out}")
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${KERNELH_REFS_OUT}"
      COMMAND "${HIPCC}" --genco --offload-arch=${KERNELH_REF_ARCH}
              "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
              -o "${out}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
      COMMENT "hipcc --genco ${src} -> ${stem}.o")
  endforeach()

  add_custom_target(bench_refs ALL DEPENDS ${KERNELH_REF_OBJECTS})
else()
  message(WARNING
    "hipcc not found in the devenv env; bench refs will not build")
endif()
