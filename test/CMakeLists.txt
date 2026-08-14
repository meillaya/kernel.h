# Degraded-mode flag path (plan todo 6 gate): baked into every test target as
# the KERNELH_DEGRADED_FLAG compile definition. test/unit/gpu_gate.hpp reads
# this file at test start; a missing file means degraded=true (fail-safe).
set(KERNELH_DEGRADED_FLAG_PATH "$ENV{KERNELH_DEGRADED_FLAG}")
if(KERNELH_DEGRADED_FLAG_PATH STREQUAL "")
  message(FATAL_ERROR
    "KERNELH_DEGRADED_FLAG env var is empty; configure inside `devenv shell` "
    "(devenv.nix sets it to <repo>/.omo/evidence/degraded.flag).")
endif()

add_subdirectory(unit)
add_subdirectory(lit)
