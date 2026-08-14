/* test/fixtures/xchg_kernel.c
 * Plain-C rendition of the todo-7 cross-thread exchange kernel for amdgcn
 * (atomgpu plan todo 7, step 6 - D8 fallback-path probe).
 *
 * Compiled freestanding for the AMDGPU target. DEVIATION (todo 7 D-C1):
 * plain-C tentative definitions are always zeroinitialized, but the AMDGPU
 * backend only lowers addrspace(3) globals with UNDEF initializers
 * (llvmorg-21.1.8 AMDGPUMemoryUtils.cpp isLDSVariableToLower: "Initializers
 * are unimplemented for LDS address space"; AMDGPUAsmPrinter.cpp
 * emitGlobalVariable errors otherwise). The MLIR path emits exactly undef.
 * So the build goes through LLVM IR with a one-line initializer patch:
 *
 *   NIX_HARDENING_ENABLE="" clang -target amdgcn-amd-amdhsa -mcpu=gfx1101 \
 *       -O2 -emit-llvm -c xchg_kernel.c -o xchg_kernel.ll
 *       # (D-C0: NIX_HARDENING_ENABLE cleared - the nix clang wrapper injects
 *       #  -fzero-call-used-regs=used-gpr, which amdgcn rejects)
 *   sed -i '/@shared_buf/s/zeroinitializer/undef/' xchg_kernel.ll
 *   NIX_HARDENING_ENABLE="" clang -target amdgcn-amd-amdhsa -mcpu=gfx1101 \
 *       -O2 -c xchg_kernel.ll -o xchg_kernel.o
 *   $ROCM_TOOLKIT_ROOT/llvm/bin/ld.lld -shared xchg_kernel.o \
 *       -o xchg_c.hsaco
 *
 * then loaded + executed by build/fixtures/load_module (todo-6 harness).
 *
 * Semantics (per workgroup of 256 threads; matches w2_vecadd_gpu.mlir):
 *   shared[tid] = in[gid];  s_barrier;
 *   out[gid]    = shared[(tid + 1) % block_dim];
 * where gid = workgroup_id_x * block_dim + tid (GLOBAL index).
 *
 * DEVIATION (todo 7 D-X2, same as the MLIR fixture): in/out are indexed by
 * the GLOBAL id, not the block-local tid. The harness launches N/256
 * workgroups and verifies ALL N elements (out[i] == (i+1) % 256); block-local
 * indexing would only exchange out[0..255]. The exchange itself stays
 * block-local in LDS, which is exactly the per-block rotation the harness
 * expects. The MLIR fixture's redundant Phase-1 vector copy is omitted here:
 * the exchange alone produces the verified result.
 *
 * Builtins used (clang 21, amdgcn):
 *   __builtin_amdgcn_workitem_id_x()   -> thread id in workgroup
 *   __builtin_amdgcn_workgroup_id_x()  -> workgroup id
 *   __builtin_amdgcn_workgroup_size_x()-> block dim (adds hidden_group_size_x
 *                                         kernel arg, filled by the runtime)
 *   __builtin_amdgcn_s_barrier()       -> workgroup barrier (s_barrier)
 * Static LDS via __attribute__((address_space(3))) on a file-scope buffer.
 */

static float shared_buf[256] __attribute__((address_space(3)));

__attribute__((amdgpu_kernel))
void xchg_kernel(const float* in, float* out) {
  const unsigned tid = __builtin_amdgcn_workitem_id_x();
  const unsigned bdim = __builtin_amdgcn_workgroup_size_x();
  const unsigned gid = __builtin_amdgcn_workgroup_id_x() * bdim + tid;

  shared_buf[tid] = in[gid];
  __builtin_amdgcn_s_barrier();
  out[gid] = shared_buf[(tid + 1) % bdim];
}
