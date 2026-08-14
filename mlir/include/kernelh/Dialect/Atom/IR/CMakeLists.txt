# TableGen for the "atom" dialect (plan todo 20).
#
# The standard mlir_tablegen + add_public_tablegen_target recipe (verify-lane-B
# pattern, plan todo 4): generates the op/dialect/enum declarations and
# definitions that AtomDialect.h/.cpp and AtomOps.cpp consume. MLIR_TABLEGEN_EXE
# comes from the devenv environment.
set(LLVM_TARGET_DEFINITIONS AtomDialect.td)

mlir_tablegen(AtomOpsDialect.h.inc -gen-dialect-decls -dialect=atom)
mlir_tablegen(AtomOpsDialect.cpp.inc -gen-dialect-defs -dialect=atom)
mlir_tablegen(AtomOps.h.inc -gen-op-decls)
mlir_tablegen(AtomOps.cpp.inc -gen-op-defs)
mlir_tablegen(AtomOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(AtomOpsEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(kernelh_atom_ops_inc_gen)
