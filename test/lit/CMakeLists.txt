# lit + FileCheck test infrastructure (plan todo 5).
#
# Standard MLIR exec-config pattern: CMake configures lit.site.cfg.py.in into
# the BUILD dir; that site config is lit's entry point and loads lit.cfg.py
# from the source dir. Running `lit test/lit` on the source dir also works:
# lit.cfg.py discovers the site config under <repo>/build (pinned by devenv
# scripts.build.exec, which always configures with `-B build`).

find_program(KERNELH_LIT_BINARY NAMES lit REQUIRED)
# LLVM's native FileCheck (llvmPackages_21.llvm) resolves first; the nix
# python `filecheck` package is the fallback.
find_program(KERNELH_FILECHECK_BINARY NAMES FileCheck filecheck REQUIRED)

# kernelh-opt output location (add_mlir_tool in tools/kernelh-opt, layout
# verified in todo 4: build/tools/kernelh-opt/kernelh-opt).
set(KERNELH_OPT_BIN "${CMAKE_BINARY_DIR}/tools/kernelh-opt/kernelh-opt")
# kernelh CLI driver output location (tools/kernelh, plan todo 12).
set(KERNELH_BIN "${CMAKE_BINARY_DIR}/tools/kernelh/kernelh")
set(KERNELH_LIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(KERNELH_LIT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(check-kernelh
  COMMAND ${KERNELH_LIT_BINARY} ${CMAKE_CURRENT_BINARY_DIR} -v
  DEPENDS kernelh-opt
  COMMENT "Running kernel.h lit tests"
  USES_TERMINAL)

add_test(NAME lit-tests
  COMMAND ${KERNELH_LIT_BINARY} ${CMAKE_CURRENT_BINARY_DIR} -v)
