# kernel.h MLIR dialects (tblgen "atom" dialect + lowerings).
#
# The kernel.h dialect include trees (source + generated .inc) and the
# MLIR/LLVM include dirs are set globally in the top-level CMakeLists.txt so
# tools/ consumers (kernelh-opt) see them too.
add_subdirectory(include/kernelh/Dialect/Atom/IR)
add_subdirectory(lib/Dialect/Atom/IR)
