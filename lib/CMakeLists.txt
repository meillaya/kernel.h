# kernel.h core libraries (IR, DSL, passes, compiler, runtime).
# lib/compiler holds the device-limits table (todo 10) - the single source of
# truth for device constants; IR/DSL/passes/runtime arrive with later todos.
add_subdirectory(compiler)
