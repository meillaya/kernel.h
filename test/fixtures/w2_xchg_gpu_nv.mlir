// NVIDIA cross gate fixture (plan todo 8): cross-thread exchange kernel, sm_80.
// Cross-compilation only - there is no NVIDIA GPU on this machine.
//
// Semantics (per workgroup, launch with blockDim.x <= 256):
//   Phase 1: vectorized grid-stride copy %in -> %out in vector<4xf32> chunks
//            (exercises scf.for -> convert-scf-to-cf and vector.load/store ->
//            convert-vector-to-llvm).
//   gpu.barrier: all vector stores visible before the exchange overwrites.
//   Phase 2: cross-thread exchange through STATIC shared memory represented as
//            an ADDRESS-SPACE-3 GLOBAL symbol (Oracle B4 mechanism - NOT an
//            in-kernel addrspace(3) alloca, which NVPTX would lower to a
//            per-thread .local stack slot):
//              shared[tid] = in[tid]; gpu.barrier;
//              out[tid] = shared[(tid + 1) % block_dim].
//
// ABI: DYNAMIC memref<?xf32> kernel arguments = memref descriptors at the
//      LLVM level (plan decision B3; NO bare-pointer convention).

gpu.module @xchg_nv {
  // Static shared buffer. addrspace 3 = per-workgroup .shared memory.
  // Must lower to a DEFINED `.shared .align ... shared_buf[...]` PTX global,
  // never to a `.local` per-thread alloca (Oracle B4 mechanism).
  // NOTE (todo 8 deviation): `= uninitialized` is required - a memref.global
  // WITHOUT initial_value is an EXTERNAL declaration in LLVM 21 MLIR and
  // lowers to `.extern .shared` (dynamic-shared-style), not a static alloc.
  memref.global @shared_buf : memref<256xf32, 3> = uninitialized

  gpu.func @xchg_kernel(%in: memref<?xf32>, %out: memref<?xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %tid = gpu.thread_id x
    %bdim = gpu.block_dim x

    // Phase 1: vectorized grid-stride copy in -> out (4 x f32 per step).
    %n = memref.dim %in, %c0 : memref<?xf32>
    %nvec = arith.divsi %n, %c4 : index
    scf.for %i = %tid to %nvec step %bdim {
      %off = arith.muli %i, %c4 : index
      %v = vector.load %in[%off] : memref<?xf32>, vector<4xf32>
      vector.store %v, %out[%off] : memref<?xf32>, vector<4xf32>
    }

    // Phase separator: every vector store is visible to all threads before the
    // exchange writes below (keeps the overlap of out[tid] deterministic).
    gpu.barrier

    // Phase 2: cross-thread exchange via the addrspace-3 global.
    %shared = memref.get_global @shared_buf : memref<256xf32, 3>
    %val = memref.load %in[%tid] : memref<?xf32>
    memref.store %val, %shared[%tid] : memref<256xf32, 3>
    gpu.barrier
    %next = arith.addi %tid, %c1 : index
    %peer = arith.remsi %next, %bdim : index
    %xval = memref.load %shared[%peer] : memref<256xf32, 3>
    memref.store %xval, %out[%tid] : memref<?xf32>
    gpu.return
  }
}
