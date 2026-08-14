# kernelh - the kernel.h command-line driver (plan todo 12).
#
# This tool links the kernelh libraries IN-PROCESS (plan Metis G37): it never
# shells out to kernelh-opt, which stays the debug/test driver. It is pure
# C++23 with NO MLIR dependency - kernelh_dsl / kernelh_ir / kernelh_compiler
# are self-contained static libs (the todo-11 standalone-probe recipe links
# exactly these three and nothing else).
add_mlir_tool(kernelh main.cpp)

# main.cpp catches std::invalid_argument from device_limits_for() and
# std::runtime_error from the DSL registry emit(); the LLVM/MLIR toolchain
# builds with -fno-exceptions, so re-enable exceptions for this one TU only
# (same pattern as tools/kernelh-opt).
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")

target_link_libraries(kernelh PRIVATE
  kernelh_dsl
  kernelh_ir
  kernelh_compiler
)
