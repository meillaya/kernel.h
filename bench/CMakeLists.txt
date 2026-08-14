# Bench harness (plan todo 35).
# The hipcc-compiled refs land under the build tree; the harness bakes the
# path in as KERNELH_REFS_DIR (see bench/harness/CMakeLists.txt).
set(KERNELH_REFS_DIR "${CMAKE_BINARY_DIR}/bench/refs")
add_subdirectory(harness)
add_subdirectory(refs)
