# kernelh_ir - the custom structured-SSA "atom" IR (plan todo 9).
#
# Pure C++23, NO MLIR / HIP dependency: types + ops + verifier + printer +
# parser + pass-manager skeleton. Later todos (passes in lib/passes, the DSL
# in lib/dsl, the MLIR bridge in todo 20) link this library.
add_library(kernelh_ir STATIC
  types.cpp
  ops.cpp
  module.cpp
  verifier.cpp
  printer.cpp
  parser.cpp
  pass_manager.cpp
)

target_include_directories(kernelh_ir PUBLIC "${CMAKE_SOURCE_DIR}/include")
