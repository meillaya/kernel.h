# kernel.h core libraries (IR, DSL, passes, compiler, runtime).
# lib/ir holds the custom structured-SSA "atom" IR (todo 9).
# lib/compiler holds the device-limits table (todo 10) - the single source of
# truth for device constants; lib/passes holds the transformation passes
# (todo 13+; each pass is a self-registering .cpp, see its CMakeLists).
add_subdirectory(ir)
add_subdirectory(compiler)
add_subdirectory(passes)
add_subdirectory(dsl)
add_subdirectory(runtime)
