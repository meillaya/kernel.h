// kernelh-opt — mlir-opt-style driver for kernel.h.
//
// LLVM 21 API (verified against the installed mlir.dev headers):
//   - mlir/Tools/mlir-opt/MlirOptMain.h provides MlirOptMain() and
//     asMainReturnCode(); the 4-arg overload registers/parses all CLI options
//     (incl. --pass-pipeline) itself.
//   - registerAllDialects / registerAllExtensions / registerAllPasses are
//     `inline` in mlir/InitAll*.h and reference every upstream registration
//     symbol, which is why this tool links the full MLIR static lib set.
//
// Registers all upstream MLIR dialects (incl. gpu, nvvm, rocdl, amdgpu,
// vector, scf, arith, func, memref, LLVM) and all passes so later todos can
// run GPU lowering pipelines through this tool. No custom "atom" dialect yet.

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "kernelh-opt: MLIR optimizer driver for kernel.h", registry));
}
