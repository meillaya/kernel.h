// Strong override of the device-limits weak hook (plan todo 10 contract; see
// include/kernelh/compiler/device_limits.hpp): returns the live CU count of
// device 0 via hipDeviceGetAttribute(hipDeviceAttributeMultiprocessorCount).
//
// The weak default in lib/compiler/device_limits.cpp returns 0 ("no GPU
// runtime linked / no device visible"). This strong definition wins at FINAL
// LINK time whenever this library is linked together with lib/compiler - so
// binaries linking both automatically get the real CU count, and binaries
// without a runtime keep the weak default. lib/compiler itself never links
// HIP.
//
// GOTCHAS (both verified empirically, see todo-23 evidence):
//  1. Do NOT include device_limits.hpp here: its prior declaration carries
//     __attribute__((weak)), and C++ propagates the attribute from a prior
//     declaration to the definition - the "strong" override would compile
//     weak too and lose the link. The function is therefore declared locally,
//     WITHOUT the attribute.
//  2. Plain archive extraction keeps the WEAK definition (a weak def
//     satisfies the reference, so the archive member is never pulled in).
//     Consumers must force this TU into the final link (WHOLE_ARCHIVE,
//     exactly like the pass-registration archives in lib/passes). kernelh_unit
//     does this via $<LINK_LIBRARY:WHOLE_ARCHIVE,kernelh_runtime>.
//
// Never throws and never does more than necessary: returns 0 when no device
// is present, matching the weak default exactly.

#include <hip/hip_runtime_api.h>

namespace kernelh {

// Forward declaration WITHOUT __attribute__((weak)) - see gotcha 1 above.
long query_cu_count_runtime();

long query_cu_count_runtime() {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
    return 0;
  }
  int cu_count = 0;
  hipError_t err =
      hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount, 0);
  return err == hipSuccess && cu_count > 0 ? static_cast<long>(cu_count) : 0;
}

} // namespace kernelh
