# kernel.h core libraries (IR, DSL, passes, compiler, runtime).
# lib/ir holds the custom structured-SSA "atom" IR (todo 9).
# lib/compiler holds the device-limits table (todo 10) - the single source of
# truth for device constants; DSL/passes/runtime arrive with later todos.
add_subdirectory(ir)
add_subdirectory(compiler)
