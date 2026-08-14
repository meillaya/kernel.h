// AMD MLIR pipeline gate fixture (plan todo 7): cross-thread exchange kernel,
// gfx1101. ROCDL twin of w2_xchg_gpu_nv.mlir (todo 8 NV fixture).
//
// Semantics (per workgroup, launch with blockDim.x <= 256):
//   Phase 1: vectorized grid-stride copy %in -> %out in vector<4xf32> chunks
//            (exercises scf.for -> convert-scf-to-cf and vector.load/store ->
//            convert-vector-to-llvm).
//   gpu.barrier: all vector stores visible before the exchange overwrites.
//   Phase 2: cross-thread exchange through STATIC shared memory represented as
//            an ADDRESS-SPACE-3 GLOBAL symbol (Oracle B4 mechanism - NOT an
//            in-kernel addrspace(3) alloca):
//              shared[tid] = in[tid]; gpu.barrier;
//              out[tid] = shared[(tid + 1) % block_dim].
//
// ABI: DYNAMIC memref<?xf32> kernel arguments = memref descriptors at the
//      LLVM level (plan decision B3; NO bare-pointer convention).
//
// DEVIATION (todo 7 D-ABI, harness shim): the todo-6 load_module.c harness is
// frozen and passes EXACTLY 2 kernelParams ({&d_in, &d_out}). ROCm 7.2.3
// hipModuleLaunchKernel copies kernelParams[i] for EVERY kernel argument
// listed in the HSACO metadata (clr hipamd/src/hip_module.cpp,
// ihipLaunchKernelCommand -> kernel->parameters().set(i, desc.size_,
// kernelParams[i], ...)), so a 10-argument descriptor-ABI kernel cannot be
// launched by it (params 2..9 would be out-of-bounds harness-stack reads).
// AMDGPU also forbids kernel -> kernel calls. This module therefore provides:
//   - @xchg_kernel: the PRODUCTION descriptor-ABI kernel (plan B3 ABI) and
//     static-verification target (ELF/notes/symbol/ISA assertions).
//   - @xchg_impl:   shared device function holding the exchange body.
//   - @xchg_kernel_harness: 2-bare-pointer kernel entry (matches the harness
//     ABI) that rebuilds both memref descriptors ON DEVICE with
//     n = gridDim.x * blockDim.x (exact for the harness launch of
//     N/256 blocks x 256 threads) and tail-calls @xchg_impl. This is the
//     symbol load_module executes; it runs the SAME MLIR-lowered body.

gpu.module @xchg_amd {
  // Static shared buffer. addrspace 3 = per-workgroup LDS.
  // NOTE (todo 8 deviation D2, applies to ROCDL too): `= uninitialized` is
  // REQUIRED - a memref.global WITHOUT initial_value is an EXTERNAL
  // declaration in LLVM 21 MLIR; `= uninitialized` yields a defined
  // per-workgroup allocation (1024-byte .group segment on amdgcn).
  memref.global @shared_buf : memref<256xf32, 3> = uninitialized

  // Shared device implementation of the exchange (called by both entries).
  func.func @xchg_impl(%in: memref<?xf32>, %out: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %tid = gpu.thread_id x
    %bid = gpu.block_id x
    %bdim = gpu.block_dim x
    %base = arith.muli %bid, %bdim : index
    %gid = arith.addi %base, %tid : index

    // Phase 1: vectorized copy in -> out (4 x f32 per step) of THIS block's
    // segment. DEVIATION (todo 7 D-X1, vs the NV twin): the twin grid-strides
    // the WHOLE array with block-local indices, so every block redundantly
    // copies the entire buffer and those stores race with Phase 2's exchange
    // stores across wave-scheduled workgroups (no grid sync exists in v1).
    // Copying only this workgroup's segment keeps every Phase-1 store inside
    // the region this workgroup's Phase 2 overwrites, ordered by the barrier
    // below -> deterministic. Still exercises scf.for + vector.load/store.
    %seg_nvec = arith.divsi %bdim, %c4 : index
    scf.for %i = %tid to %seg_nvec step %bdim {
      %off_rel = arith.muli %i, %c4 : index
      %off = arith.addi %base, %off_rel : index
      %v = vector.load %in[%off] : memref<?xf32>, vector<4xf32>
      vector.store %v, %out[%off] : memref<?xf32>, vector<4xf32>
    }

    // Phase separator: every vector store is visible to all threads of this
    // workgroup before the exchange writes below.
    gpu.barrier

    // Phase 2: cross-thread exchange via the addrspace-3 global.
    // DEVIATION (todo 7 D-X2, vs the NV twin): in/out are indexed by the
    // GLOBAL id %gid, not the block-local %tid. The harness launches N/256
    // workgroups and verifies ALL N elements (out[i] == (i+1) % 256);
    // block-local indexing would only exchange out[0..255] in every workgroup
    // and leave the rest at the Phase-1 copy. Shared memory stays block-local
    // (%tid): each workgroup exchanges its own 256-element segment, which is
    // exactly the per-block rotation the harness expects.
    %shared = memref.get_global @shared_buf : memref<256xf32, 3>
    %val = memref.load %in[%gid] : memref<?xf32>
    memref.store %val, %shared[%tid] : memref<256xf32, 3>
    gpu.barrier
    %next = arith.addi %tid, %c1 : index
    %peer = arith.remsi %next, %bdim : index
    %xval = memref.load %shared[%peer] : memref<256xf32, 3>
    memref.store %xval, %out[%gid] : memref<?xf32>
    return
  }

  // Production entry: descriptor ABI per plan decision B3 (todo 8 deviation
  // D3: the `kernel` unit attr is REQUIRED for kernel entry).
  gpu.func @xchg_kernel(%in: memref<?xf32>, %out: memref<?xf32>) kernel {
    func.call @xchg_impl(%in, %out) : (memref<?xf32>, memref<?xf32>) -> ()
    gpu.return
  }

  // Harness shim entry (todo 7 deviation D-ABI): bare pointers matching
  // load_module.c's void* args[] = {&d_in, &d_out}. Descriptors are rebuilt
  // on device; n = gridDim.x * blockDim.x equals the harness's N exactly.
  gpu.func @xchg_kernel_harness(%in: !llvm.ptr, %out: !llvm.ptr) kernel {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %gd = gpu.grid_dim x
    %bd = gpu.block_dim x
    %n = arith.muli %gd, %bd : index
    %n_i64 = arith.index_cast %n : index to i64

    // memref<?xf32> descriptor layout produced by the LLVM type converter:
    //   !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    //   = (allocated, aligned, offset, sizes[1], strides[1]).
    // Build it EXACTLY in that type so the unrealized_conversion_cast below
    // cancels against the memref->struct materialization at the @xchg_impl
    // call site during reconcile-unrealized-casts.
    %sizes0 = llvm.mlir.undef : !llvm.array<1 x i64>
    %sizes = llvm.insertvalue %n_i64, %sizes0[0] : !llvm.array<1 x i64>
    %strides0 = llvm.mlir.undef : !llvm.array<1 x i64>
    %strides = llvm.insertvalue %c1_i64, %strides0[0] : !llvm.array<1 x i64>

    %in_desc0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %in_desc1 = llvm.insertvalue %in, %in_desc0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %in_desc2 = llvm.insertvalue %in, %in_desc1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %in_desc3 = llvm.insertvalue %c0_i64, %in_desc2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %in_desc4 = llvm.insertvalue %sizes, %in_desc3[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %in_desc = llvm.insertvalue %strides, %in_desc4[4] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

    %out_desc0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %out_desc1 = llvm.insertvalue %out, %out_desc0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %out_desc2 = llvm.insertvalue %out, %out_desc1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %out_desc3 = llvm.insertvalue %c0_i64, %out_desc2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %out_desc4 = llvm.insertvalue %sizes, %out_desc3[3] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %out_desc = llvm.insertvalue %strides, %out_desc4[4] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

    %in_memref = builtin.unrealized_conversion_cast %in_desc : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf32>
    %out_memref = builtin.unrealized_conversion_cast %out_desc : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf32>
    func.call @xchg_impl(%in_memref, %out_memref) : (memref<?xf32>, memref<?xf32>) -> ()
    gpu.return
  }
}
