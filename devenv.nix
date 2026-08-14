{ pkgs, config, lib, ... }:
let
  llvm = pkgs.llvmPackages_21;
in
{
  packages = with pkgs; [
    llvm.llvm llvm.llvm.dev llvm.mlir llvm.mlir.dev llvm.tblgen llvm.clang
    cmake ninja catch2_3 lit filecheck
    (python3.withPackages (ps: [ ps.matplotlib ]))
    rocmPackages.clr rocmPackages.rocm-runtime rocmPackages.rocblas
    # NOTE: rocmPackages.llvm is a package *set* in this nixpkgs (eval error if used
    # directly); use the actual derivations .llvm and .lld (deviation recorded in evidence)
    rocmPackages.rocm-device-libs rocmPackages.llvm.llvm rocmPackages.llvm.lld rocmPackages.rocminfo
    cudaPackages_12_9.cuda_nvcc cudaPackages_12_9.cuda_cudart cudaPackages_12_9.libcublas
    # CUTLASS headers: try cudaPackages_12_9.cutlass FIRST; if that attribute does not
    # exist (nix eval error), use top-level pkgs.cutlass instead, and record which was used
    cudaPackages_12_9.cutlass
  ];

  env = {
    KERNELH_ROOT = builtins.toString ./.;
    MLIR_DIR = "${llvm.mlir.dev}/lib/cmake/mlir";
    LLVM_DIR = "${llvm.llvm.dev}/lib/cmake/llvm";
    MLIR_TABLEGEN_EXE = "${llvm.tblgen}/bin/mlir-tblgen";
    KERNELH_DEGRADED_FLAG = "${config.env.KERNELH_ROOT}/.omo/evidence/degraded.flag";
  };

  scripts.build.exec = "cmake -G Ninja -B build -S . -DCMAKE_C_COMPILER=$(dirname $(which clang))/clang -DCMAKE_CXX_COMPILER=$(dirname $(which clang++))/clang++ -DCMAKE_CXX_STANDARD=23 -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR -DMLIR_TABLEGEN_EXE=$MLIR_TABLEGEN_EXE && cmake --build build -j$(nproc)";
  scripts.test.exec = "ctest --test-dir build --output-on-failure";
}
