// Atom MLIR dialect round-trip test (plan todo 20): a small atom.mlir module
// is parsed, verified, and printed back by kernelh-opt. FileCheck locks the
// printed structure so the emitter (todo 21) and later todos can rely on a
// stable assembly form.

// RUN: kernelh-opt %s | FileCheck %s

module {
  atom.kernel @vecadd(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) attributes {map_blocks = true, tile = [256], vector_width = 4 : i64} {
    %0 = atom.const 0 : index
    %1 = atom.const 1 : index
    %2 = atom.const 1 : i32
    %3 = atom.thread_id x : index
    %4 = atom.block_id x : index
    %5 = atom.block_dim x : index
    %6 = atom.grid_dim y : index
    %7 = atom.add %4, %3 : index
    %8 = atom.const 0.000000e+00 : f32
    %9 = atom.const 1.000000e+00 : f32
    %10 = atom.const 0.000000e+00 : memref<64x16xf32>
    %11 = atom.loop_for %0 to %1 step %1 iter_args(%8) -> (f32) {
    ^bb0(%arg2: index, %arg3: f32):
      %25 = atom.load %arg0[%arg2] : memref<1024xf32>, f32
      %26 = atom.add %25, %arg3 : f32
      atom.yield %26 : f32
    }
    %12 = atom.cmp lt %7, %5 : index
    %13 = atom.select %12, %11, %9 : f32
    %14 = atom.math sqrt %13 : f32
    %15 = atom.max %13, %9 : f32
    %16 = atom.min %13, %9 : f32
    %17 = atom.div %13, %9 : f32
    %18 = atom.fma %13, %9, %9 : f32
    %19 = atom.sub %13, %9 : f32
    %20 = atom.mul %13, %9 : f32
    %21 = atom.select %2, %9, %8 : f32
    atom.store %14, %arg1[%7] : memref<1024xf32>, f32
    %22 = atom.alloc_shared : memref<64x64xf32, 3>
    atom.barrier
    %23 = atom.reduce sum %arg0, dim = 0 : memref<1024xf32>, f32
    %24 = atom.load_vec %arg0[%0] : memref<1024xf32>, vector<4xf32>
    atom.store_vec %24, %arg0[%0] : memref<1024xf32>, vector<4xf32>
    atom.copy %arg0, %arg1 : memref<1024xf32> to memref<1024xf32>
  }
}

// CHECK-LABEL: module {
// CHECK-LABEL: atom.kernel @vecadd(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) attributes {map_blocks = true, tile = [256], vector_width = 4 : i64} {
// CHECK: %0 = atom.const 0 : index
// CHECK: %3 = atom.thread_id x : index
// CHECK: %4 = atom.block_id x : index
// CHECK: %7 = atom.add %4, %3 : index
// CHECK: %8 = atom.const 0.000000e+00 : f32
// CHECK: %10 = atom.const 0.000000e+00 : memref<64x16xf32>
// CHECK: %11 = atom.loop_for %0 to %1 step %1 iter_args(%8) -> (f32) {
// CHECK: ^bb0(%arg2: index, %arg3: f32):
// CHECK: %25 = atom.load %arg0[%arg2] : memref<1024xf32>, f32
// CHECK: %26 = atom.add %25, %arg3 : f32
// CHECK: atom.yield %26 : f32
// CHECK: %12 = atom.cmp lt %7, %5 : index
// CHECK: %13 = atom.select %12, %11, %9 : f32
// CHECK: %14 = atom.math sqrt %13 : f32
// CHECK: atom.store %14, %arg1[%7] : memref<1024xf32>, f32
// CHECK: %22 = atom.alloc_shared : memref<64x64xf32, 3>
// CHECK: atom.barrier
// CHECK: %23 = atom.reduce sum %arg0, dim = 0 : memref<1024xf32>, f32
// CHECK: %24 = atom.load_vec %arg0[%0] : memref<1024xf32>, vector<4xf32>
// CHECK: atom.store_vec %24, %arg0[%0] : memref<1024xf32>, vector<4xf32>
// CHECK: atom.copy %arg0, %arg1 : memref<1024xf32> to memref<1024xf32>
