# kernelh - the kernel.h command-line driver (plan todos 12 + 24).
#
# This tool links the kernelh libraries IN-PROCESS (plan Metis G37): it never
# shells out to kernelh-opt, which stays the debug/test driver. `compile` is
# the production driver (todo 24): kernelh_dsl / kernelh_ir / kernelh_compiler
# handle input/verify, kernelh_passes the default pipeline, and
# kernelh_emitter + kernelh_backends (which are PUBLIC-linked and pull their
# MLIR dependency set transitively) the MLIR lowering and the vendor
# artifacts.
add_mlir_tool(kernelh main.cpp)

# main.cpp catches std::invalid_argument from device_limits_for() and
# std::runtime_error from the DSL registry emit(); the LLVM/MLIR toolchain
# builds with -fno-exceptions, so re-enable exceptions for this one TU only
# (same pattern as tools/kernelh-opt).
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")

target_link_libraries(kernelh PRIVATE
  # WHOLE_ARCHIVE: force-load every pass object so its self-registration
  # initializer runs (see lib/passes/CMakeLists.txt). A plain archive link
  # would drop all passes and leave the global registry empty.
  $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_passes>
  kernelh_dsl
  kernelh_ir
  kernelh_compiler
  # The todo-21 emitter (custom IR -> gpu.module) and the todo-22 vendor
  # backends (NVVM PTX/cubin + ROCDL HSACO/ISA). Both list their MLIR library
  # dependencies as PUBLIC, so this single link line pulls everything they
  # need (MLIRIR/Pass/Transforms, the dialect libs, the LLVM dylib) without
  # naming the ~40 MLIR targets by hand.
  kernelh_emitter
  kernelh_backends
)
