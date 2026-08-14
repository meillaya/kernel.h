# kernelh_compiler - compiler-side helpers that do NOT depend on the custom IR
# or on MLIR. The device-limits table (plan todo 10), the launch-config
# verifier (plan todo 32), and the resource reporter (plan todo 33) live here
# because they are consumed by the passes (lib/passes), the runtime launch
# path (lib/runtime), the autotuner (todo 34), and the occupancy calculations
# without dragging in MLIR.
#
# IMPORTANT: this library intentionally never links HIP/ROCm. The runtime CU
# count is resolved at FINAL LINK time through the weak-symbol hook
# kernelh::query_cu_count_runtime() (see
# include/kernelh/compiler/device_limits.hpp for the contract); plan todo 23's
# runtime library provides the strong override.
add_library(kernelh_compiler STATIC device_limits.cpp launch_verifier.cpp resource_reporter.cpp)

# Header search: repo-root include/ (kernelh/compiler/device_limits.hpp).
target_include_directories(kernelh_compiler PUBLIC "${CMAKE_SOURCE_DIR}/include")

# kernelh_emitter (plan todo 21): the MLIR emitter (custom IR -> atom dialect
# -> gpu/scf/vector) is a SEPARATE static library so kernelh_compiler itself
# stays MLIR-free. This keeps the no-MLIR link of kernelh_weak_default_test
# (todo 23) intact: that binary links only kernelh_compiler and never sees
# MLIR. Consumers of the emitter link kernelh_emitter + kernelh_ir +
# kernelh_atom_mlir + the MLIR libs below.
#
# Only lib/compiler/device_limits.cpp must remain MLIR-free; emitter.cpp lives
# in this separate library on purpose.
add_library(kernelh_emitter STATIC emitter.cpp)
target_include_directories(kernelh_emitter PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(kernelh_emitter PUBLIC
  kernelh_ir
  kernelh_atom_mlir
  MLIRGPUDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRDialect
  MLIRSupport
  LLVM
)

# kernelh_backends (plan todo 22): the in-process vendor lowering pipelines
# (NVVM PTX/cubin + ROCDL HSACO/ISA + blob extraction) over a gpu.module.
# Like kernelh_emitter this is a SEPARATE static library so kernelh_compiler
# itself stays MLIR-free (the no-MLIR link of kernelh_weak_default_test only
# pulls kernelh_compiler). The exact pass pipelines from todos 7/8 are driven
# through mlir::PassManager - no shelling out to mlir-opt, no bare-ptr options.
add_library(kernelh_backends STATIC backends.cpp)
target_include_directories(kernelh_backends PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(kernelh_backends PUBLIC
  kernelh_ir
  kernelh_atom_mlir
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRGPUToNVVMTransforms
  MLIRGPUToROCDLTransforms
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUPipelines
  MLIRNVVMTarget
  MLIRROCDLTarget
  MLIRTargetLLVM
  MLIRFuncToLLVM
  MLIRControlFlowToLLVM
  MLIRIndexToLLVM
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRComplexToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRSPIRVToLLVMIRTranslation
  MLIRVectorToLLVMPass
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRROCDLDialect
  MLIRAMDGPUDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRDialect
  MLIRSupport
  LLVM
)
