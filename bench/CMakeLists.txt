# Bench harness (plan todo 35).
# The hipcc-compiled refs land under the build tree; the harness bakes the
# path in as KERNELH_REFS_DIR (see bench/harness/CMakeLists.txt).
set(KERNELH_REFS_DIR "${CMAKE_BINARY_DIR}/bench/refs")

# ---------------------------------------------------------------------------
# Cross harness (cuBLAS + CUTLASS, nvcc sm_80): built at configure time with
# the PINNED flags `-arch=sm_80 -std=c++17 --expt-relaxed-constexpr`. The
# build is incremental (skipped when the binary is newer than the source).
# Oracle N6 fallback: if the CUTLASS compile fails, retry with
# -DKERNELH_CROSS_NO_CUTLASS (cuBLAS-only; the binary records
# `cutlass,build-failed` rows at runtime). Missing nvcc/cublas leaves
# KERNELH_CROSS_BIN unset - `kernelh bench --suite cuda-cross` then reports
# the 77 skip path (this box has no NVIDIA driver either way).
# ---------------------------------------------------------------------------
find_program(KERNELH_NVCC nvcc)
find_library(KERNELH_CUBLAS_LIBRARY cublas)

if(KERNELH_NVCC AND KERNELH_CUBLAS_LIBRARY)
  set(KERNELH_CROSS_OUT "${CMAKE_BINARY_DIR}/bench/cross")
  set(KERNELH_CROSS_BIN "${KERNELH_CROSS_OUT}/cross_harness")
  set(KERNELH_CROSS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cross/cross_harness.cu")
  set(KERNELH_CROSS_COMMON_FLAGS -arch=sm_80 -std=c++17
      --expt-relaxed-constexpr)
  set(KERNELH_CROSS_WITH_CUTLASS_FLAGS ${KERNELH_CROSS_COMMON_FLAGS}
      -DKERNELH_CROSS_HAS_CUTLASS)
  if(KERNELH_CUTLASS_INCLUDE_DIR)
    list(APPEND KERNELH_CROSS_WITH_CUTLASS_FLAGS
         "-I${KERNELH_CUTLASS_INCLUDE_DIR}")
  endif()

  if(NOT EXISTS "${KERNELH_CROSS_BIN}" OR
     "${KERNELH_CROSS_SRC}" IS_NEWER_THAN "${KERNELH_CROSS_BIN}")
    file(MAKE_DIRECTORY "${KERNELH_CROSS_OUT}")
    execute_process(
      COMMAND "${KERNELH_NVCC}" ${KERNELH_CROSS_WITH_CUTLASS_FLAGS}
              "${KERNELH_CROSS_SRC}" -o "${KERNELH_CROSS_BIN}"
              -lcublas -lcudart
      RESULT_VARIABLE KERNELH_CROSS_RC
      OUTPUT_VARIABLE KERNELH_CROSS_LOG
      ERROR_VARIABLE KERNELH_CROSS_LOG)
    if(NOT KERNELH_CROSS_RC EQUAL 0)
      # Oracle N6: CUTLASS build failed - record the fact and retry
      # cuBLAS-only (the binary emits `cutlass,build-failed` rows).
      message(WARNING
        "cross harness: CUTLASS compile failed; building cuBLAS-only "
        "(cutlass,build-failed rows will be recorded):\n${KERNELH_CROSS_LOG}")
      file(REMOVE "${KERNELH_CROSS_BIN}")
      execute_process(
        COMMAND "${KERNELH_NVCC}" ${KERNELH_CROSS_COMMON_FLAGS}
                -DKERNELH_CROSS_NO_CUTLASS
                "${KERNELH_CROSS_SRC}" -o "${KERNELH_CROSS_BIN}"
                -lcublas -lcudart
        RESULT_VARIABLE KERNELH_CROSS_RC2
        OUTPUT_VARIABLE KERNELH_CROSS_LOG2
        ERROR_VARIABLE KERNELH_CROSS_LOG2)
      if(NOT KERNELH_CROSS_RC2 EQUAL 0)
        message(WARNING
          "cross harness: cuBLAS-only build ALSO failed; cross harness "
          "disabled (suite reports the 77 skip):\n${KERNELH_CROSS_LOG2}")
        unset(KERNELH_CROSS_BIN)
      endif()
    endif()
  endif()
else()
  message(WARNING
    "cross harness: nvcc or cublas not found; cuda-cross suite reports the "
    "77 skip path")
endif()

add_subdirectory(harness)
add_subdirectory(refs)
