# Catch2 v3 unit tests (own main() via catch_session.hpp).
include(Catch)

add_executable(kernelh_unit main.cpp)
target_link_libraries(kernelh_unit PRIVATE Catch2::Catch2)

catch_discover_tests(kernelh_unit)
