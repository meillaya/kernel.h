{ pkgs, config, lib, ... }:
let
  llvm = pkgs.llvmPackages_21;

  # Toolkit-root linkFarms for MLIR gpu-module-to-binary (plan todo 3).
  # ROCDL serializer hard-requires: <toolkit>/llvm/bin/ld.lld + <toolkit>/amdgcn/bitcode/*.bc
  # NVVM cubin path wants:          <toolkit>/bin/ptxas + <toolkit>/nvvm/libdevice/libdevice.10.bc
  rocmDeviceLibs = pkgs.rocmPackages.rocm-device-libs;
  # Verified layout (todo 3 discovery): $out/amdgcn/bitcode/*.bc (ocml, ockl, hip,
  # opencl + oclc_* controls). Enumerate every regular *.bc deterministically.
  # NOTE: builtins.readDir needs the store path present at eval time - guaranteed
  # here by the packages list below + devenv.lock pin (materialized in todo 2).
  deviceLibBcs = lib.attrNames (lib.filterAttrs
    (name: type: type == "regular" && lib.hasSuffix ".bc" name)
    (builtins.readDir "${rocmDeviceLibs}/amdgcn/bitcode"));
  rocmToolkitRoot = pkgs.linkFarm "rocm-toolkit-root" ([
    # rocmPackages.llvm is a package SET; .lld is the derivation (todo 2 finding)
    { name = "llvm/bin/ld.lld"; path = "${pkgs.rocmPackages.llvm.lld}/bin/ld.lld"; }
  ] ++ map (bc: {
    name = "amdgcn/bitcode/${bc}";
    path = "${rocmDeviceLibs}/amdgcn/bitcode/${bc}";
  }) deviceLibBcs);
  cudaToolkitRoot = pkgs.linkFarm "cuda-toolkit-root" [
    { name = "bin/ptxas"; path = "${pkgs.cudaPackages_12_9.cuda_nvcc}/bin/ptxas"; }
    # libdevice lives inside cuda_nvcc (no separate package); exact name verified
    { name = "nvvm/libdevice/libdevice.10.bc"; path = "${pkgs.cudaPackages_12_9.cuda_nvcc}/nvvm/libdevice/libdevice.10.bc"; }
  ];
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
    # TeX Live for the LaTeX technical report (docs/report.tex). Focused combine:
    # scheme-small provides the pdflatex engine + core LaTeX; the individual packages
    # add TikZ diagrams (tikz-cd), PGFPlots charts, biblatex+biber bibliography, and
    # the table/listings styling used by the report. All attribute names verified to
    # exist in this nixpkgs (devenv-nixpkgs-patched) via nix eval probes.
    (texlive.combine {
      inherit (texlive)
        scheme-small latexmk biber biblatex pgfplots tikz-cd pgf
        booktabs listings hyperref xcolor geometry amsmath amsfonts
        csquotes microtype multirow float cleveref;
    })
  ];

  env = {
    KERNELH_ROOT = builtins.toString ./.;
    MLIR_DIR = "${llvm.mlir.dev}/lib/cmake/mlir";
    LLVM_DIR = "${llvm.llvm.dev}/lib/cmake/llvm";
    MLIR_TABLEGEN_EXE = "${llvm.tblgen}/bin/mlir-tblgen";
    KERNELH_DEGRADED_FLAG = "${config.env.KERNELH_ROOT}/.omo/evidence/degraded.flag";
    ROCM_TOOLKIT_ROOT = builtins.toString rocmToolkitRoot;
    CUDA_TOOLKIT_ROOT = builtins.toString cudaToolkitRoot;
  };

  scripts.build.exec = "cmake -G Ninja -B build -S . -DCMAKE_C_COMPILER=$(dirname $(which clang))/clang -DCMAKE_CXX_COMPILER=$(dirname $(which clang++))/clang++ -DCMAKE_CXX_STANDARD=23 -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR -DMLIR_TABLEGEN_EXE=$MLIR_TABLEGEN_EXE && cmake --build build -j$(nproc)";
  scripts.test.exec = "ctest --test-dir build --output-on-failure";
}
