# kernelh_passes - transformation passes on the atom IR (plan todo 13).
#
# SELF-REGISTRATION CONTRACT (todos 14-18, read before adding a pass):
#
#   A pass is ONE self-contained .cpp file in this directory plus ONE lit test
#   under test/lit/passes/. It touches NOTHING else: no CMake edits, no changes
#   to pass_manager, no changes to the tools. Source files are collected by
#   GLOB (below), so a new canonicalize-style file is picked up on the next
#   configure (CONFIGURE_DEPENDS re-globs on every build), and it registers
#   itself into the GLOBAL pass registry from a file-scope static initializer:
#
#       static const bool kRegistered =
#           (kernelh::ir::registerPass("mypass", myPass), true);
#
#   where `myPass` is `std::expected<void, std::string>(Module&)` (the PassFn
#   signature). The registry map is a function-local static (Meyers) in
#   pass_manager.cpp, so the registration is safe against static-init order.
#
#   Two rules make the self-registration actually visible to the binaries:
#
#   1. WHOLE_ARCHIVE: the archive is force-loaded into kernelh-opt / kernelh /
#      kernelh_unit (see those CMakeLists). Without it the linker would drop
#      every pass object file (nothing references its symbols) and the
#      registration initializers would never run.
#
#   2. GLOB: add_library uses file(GLOB_RECURSE ... CONFIGURE_DEPENDS), so a
#      new .cpp is compiled and linked with zero edits to this file.
#
# Pipeline order is set by the caller, never by this library.

file(GLOB_RECURSE KERNELH_PASS_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(kernelh_passes STATIC ${KERNELH_PASS_SOURCES})

target_include_directories(kernelh_passes PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(kernelh_passes PUBLIC kernelh_ir kernelh_compiler)
