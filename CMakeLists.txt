cmake_minimum_required(VERSION 3.20)
# C is enabled in addition to CXX because HandleLLVMOptions (part of the
# required out-of-tree MLIR recipe) probes C compiler flags; the devenv build
# script passes both -DCMAKE_C_COMPILER and -DCMAKE_CXX_COMPILER.
project(kernelh C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Out-of-tree MLIR consumer recipe (verify lane B, empirically proven):
# MLIR_DIR / LLVM_DIR / MLIR_TABLEGEN_EXE are provided by the devenv
# environment (scripts.build.exec passes them explicitly).
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Installed LLVM exports do not set LLVM_BUILD_TOOLS; without this,
# add_mlir_tool marks tools EXCLUDE_FROM_ALL and `cmake --build` skips them.
set(LLVM_BUILD_TOOLS ON)

# Include trees shared by all targets: the kernel.h MLIR dialect sources,
# their TableGen outputs under <build>/mlir/include, and the MLIR/LLVM
# headers (the mlir_tblgen runs in mlir/include/... also pick these up via
# the INCLUDE_DIRECTORIES directory property).
include_directories(${CMAKE_SOURCE_DIR}/mlir/include)
include_directories(${CMAKE_BINARY_DIR}/mlir/include)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

enable_testing()
find_package(Catch2 3 REQUIRED)

add_subdirectory(lib)
add_subdirectory(mlir)
add_subdirectory(tools)
add_subdirectory(test)
# Benchmark harness + hipcc-compiled reference kernels (plan todo 35).
add_subdirectory(bench)
