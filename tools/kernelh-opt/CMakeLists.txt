# kernelh-opt: mlir-opt-style driver.
#
# In LLVM 21, mlir::registerAllDialects() / registerAllExtensions() /
# registerAllPasses() are `inline` header functions that directly reference
# every dialect / extension / pass registration symbol. Linking therefore
# requires the FULL set of MLIR static libraries (exactly like the in-tree
# mlir-opt). We collect every exported static MLIR target from
# MLIRTargets.cmake rather than hard-coding ~300 names, so this stays correct
# across LLVM point releases. The MLIR static archives are built against
# libLLVM-21.so, so we also link the `LLVM` imported target (the dylib).

file(STRINGS "${MLIR_CMAKE_DIR}/MLIRTargets.cmake" _kernelh_mlir_target_lines
     REGEX "^add_library\\(MLIR[A-Za-z0-9_]+ STATIC IMPORTED\\)")

set(KERNELH_MLIR_STATIC_LIBS "")
foreach(_line IN LISTS _kernelh_mlir_target_lines)
  if(_line MATCHES "^add_library\\((MLIR[A-Za-z0-9_]+) STATIC IMPORTED\\)")
    # Capture immediately: any further MATCHES evaluates clobbers CMAKE_MATCH_1.
    set(_kernelh_tgt "${CMAKE_MATCH_1}")
    # Skip the C API wrappers; the C++ tool does not need them.
    if(NOT _kernelh_tgt MATCHES "^MLIRCAPI")
      list(APPEND KERNELH_MLIR_STATIC_LIBS "${_kernelh_tgt}")
    endif()
  endif()
endforeach()

add_mlir_tool(kernelh-opt main.cpp)
target_link_libraries(kernelh-opt PRIVATE
  kernelh_ir
  ${KERNELH_MLIR_STATIC_LIBS}
  LLVM
)
