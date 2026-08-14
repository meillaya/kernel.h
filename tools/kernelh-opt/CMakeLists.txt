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
# kernelh-opt catches std::invalid_argument from device_limits_for() to turn
# an unknown --atom-target into a clean error; the LLVM/MLIR toolchain builds
# with -fno-exceptions, so re-enable exceptions for this one TU only.
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")
target_link_libraries(kernelh-opt PRIVATE
  # kernelh_passes is force-loaded (WHOLE_ARCHIVE) so every pass object file is
  # present at startup and its self-registration initializer runs - a plain
  # archive link would drop every pass (nothing references its symbols) and
  # --run-passes would always report "registered passes: (none)".
  $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_passes>
  kernelh_ir
  kernelh_compiler
  # The atom MLIR dialect (plan todo 20). main.cpp calls
  # kernelh::atom::registerAtomDialect directly, so a plain link is enough.
  kernelh_atom_mlir
  # The MLIR emitter (plan todo 21): --emit-mlir lowers the atom module to a
  # device-only gpu.module. kernelh-opt already links the full MLIR static set
  # below, so the emitter's own MLIR deps are satisfied.
  kernelh_emitter
  # The vendor backends (plan todo 22): --compile-target runs the NVVM/ROCDL
  # pipelines in-process and writes PTX/cubin/HSACO/ISA artifacts to --out.
  kernelh_backends
  ${KERNELH_MLIR_STATIC_LIBS}
  LLVM
)
