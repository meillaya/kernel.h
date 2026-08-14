# kernelh_atom_mlir: the "atom" MLIR dialect library (plan todo 20).
#
# The dialect is linked into kernelh-opt, whose main.cpp references
# kernelh::atom::registerAtomDialect directly, so a plain (non-WHOLE_ARCHIVE)
# link pulls the objects in. The dialect's registration symbols are what make
# `kernelh-opt --help` list the atom dialect and let .mlir files containing
# atom.* ops parse.
add_mlir_dialect_library(kernelh_atom_mlir
  AtomDialect.cpp
  AtomOps.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  )

# Ensure the TableGen outputs exist before the sources compile (the generated
# .inc files land in <build>/mlir/include/kernelh/Dialect/Atom/IR/).
add_dependencies(kernelh_atom_mlir kernelh_atom_ops_inc_gen)
