//===- AtomOps.td - kernel.h atom IR ops -------------------*- tablegen -*-===//
//
// Ops of the "atom" dialect, mirroring the todo-9 custom IR op inventory
// (include/kernelh/ir/ops.hpp) 1:1 so the emitter (plan todo 21) is a
// mechanical translation.
//
// This file is part of kernel.h.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for atom ops.
class Atom_Op<string mnemonic, list<Trait> traits = []>
    : Op<Atom_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def Atom_ConstOp : Atom_Op<"const", [Pure]> {
  let summary = "typed constant literal";
  let description = [{
    A constant literal value. The result type is a scalar (`f16/f32/i32/i64/
    index`) or a tensor-with-memory-space (mirrored as a `memref`); a
    non-scalar result is a splat of the single literal value.

    Example:
    ```mlir
    %c0 = atom.const 0 : index
    %cf = atom.const 1.000000e+00 : f32
    %cs = atom.const 0.000000e+00 : memref<64x16xf32>
    ```
  }];
  let arguments = (ins AnyAttr:$value);
  let results = (outs Atom_ElementwiseType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Elementwise arithmetic
//===----------------------------------------------------------------------===//

class Atom_ElementwiseOp<string mnemonic>
    : Atom_Op<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "out"]>]> {
  let arguments = (ins Atom_ElementwiseType:$lhs, Atom_ElementwiseType:$rhs);
  let results = (outs Atom_ElementwiseType:$out);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($out)";
}

def Atom_AddOp : Atom_ElementwiseOp<"add">;
def Atom_SubOp : Atom_ElementwiseOp<"sub">;
def Atom_MulOp : Atom_ElementwiseOp<"mul">;
def Atom_DivOp : Atom_ElementwiseOp<"div">;
def Atom_MaxOp : Atom_ElementwiseOp<"max">;
def Atom_MinOp : Atom_ElementwiseOp<"min">;

def Atom_FmaOp : Atom_Op<"fma",
    [Pure, AllTypesMatch<["a", "b", "c", "out"]>]> {
  let arguments = (ins Atom_ElementwiseType:$a, Atom_ElementwiseType:$b,
                       Atom_ElementwiseType:$c);
  let results = (outs Atom_ElementwiseType:$out);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($out)";
}

def Atom_CmpOp : Atom_Op<"cmp", [Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "scalar comparison (result is the i32 boolean carrier)";
  let arguments = (ins Atom_CmpPredAttr:$pred, Atom_Scalar:$lhs,
                       Atom_Scalar:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$pred $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Atom_SelectOp : Atom_Op<"select", [Pure,
    AllTypesMatch<["onTrue", "onFalse", "result"]>]> {
  let arguments = (ins I32:$condition, Atom_ElementwiseType:$onTrue,
                       Atom_ElementwiseType:$onFalse);
  let results = (outs Atom_ElementwiseType:$result);
  let assemblyFormat = "$condition `,` $onTrue `,` $onFalse attr-dict `:` type($result)";
}

def Atom_MathOp : Atom_Op<"math", [Pure, AllTypesMatch<["input", "result"]>]> {
  let arguments = (ins Atom_MathKindAttr:$kind, Atom_Scalar:$input);
  let results = (outs Atom_Scalar:$result);
  let assemblyFormat = "$kind $input attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Memory ops
//===----------------------------------------------------------------------===//

def Atom_LoadOp : Atom_Op<"load", [Pure]> {
  let arguments = (ins Atom_TensorType:$memref, Variadic<Index>:$indices);
  let results = (outs Atom_Scalar:$result);
  let assemblyFormat =
      "$memref `[` $indices `]` attr-dict `:` type($memref) `,` type($result)";
  let hasVerifier = 1;
}

def Atom_StoreOp : Atom_Op<"store", []> {
  let arguments = (ins Atom_Scalar:$value, Atom_TensorType:$memref,
                       Variadic<Index>:$indices);
  let assemblyFormat =
      "$value `,` $memref `[` $indices `]` attr-dict `:` type($memref) `,` type($value)";
  let hasVerifier = 1;
}

def Atom_LoadVecOp : Atom_Op<"load_vec", [Pure]> {
  let arguments = (ins Atom_TensorType:$memref, Variadic<Index>:$indices);
  let results = (outs Atom_VectorType:$result);
  let assemblyFormat =
      "$memref `[` $indices `]` attr-dict `:` type($memref) `,` type($result)";
  let hasVerifier = 1;
}

def Atom_StoreVecOp : Atom_Op<"store_vec", []> {
  let arguments = (ins Atom_VectorType:$value, Atom_TensorType:$memref,
                       Variadic<Index>:$indices);
  let assemblyFormat =
      "$value `,` $memref `[` $indices `]` attr-dict `:` type($memref) `,` type($value)";
  let hasVerifier = 1;
}

def Atom_AllocSharedOp : Atom_Op<"alloc_shared", []> {
  let summary = "static shared-memory allocation";
  let arguments = (ins);
  let results = (outs Atom_SharedMemRef:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Atom_BarrierOp : Atom_Op<"barrier", []> {
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Thread/block ids and dims
//===----------------------------------------------------------------------===//

class Atom_IdOp<string mnemonic> : Atom_Op<mnemonic, [Pure]> {
  let arguments = (ins Atom_IdDimAttr:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

def Atom_ThreadIdOp : Atom_IdOp<"thread_id">;
def Atom_BlockIdOp  : Atom_IdOp<"block_id">;
def Atom_BlockDimOp : Atom_IdOp<"block_dim">;
def Atom_GridDimOp  : Atom_IdOp<"grid_dim">;

//===----------------------------------------------------------------------===//
// Structured control flow
//===----------------------------------------------------------------------===//

def Atom_LoopForOp : Atom_Op<"loop_for",
    [SingleBlockImplicitTerminator<"::mlir::kernelh::atom::YieldOp">,
     AllTypesMatch<["iter_args", "results"]>]> {
  let summary = "structured loop with iter_args";
  let description = [{
    A structured loop mirroring the custom IR's atom.loop_for: three control
    operands (`lb`, `ub`, `step`), loop-carried `iter_args` whose final values
    are returned as `results`, and a single-block body region whose entry
    arguments are the induction variable (an `index`) followed by one argument
    per iter_arg.  The region must terminate with `atom.yield`.

    Example:
    ```mlir
    %r = atom.loop_for %lb to %ub step %step iter_args(%acc) -> (f32) {
      ^bb0(%iv: index, %acc_it: f32):
        ...
        atom.yield %next : f32
    }
    ```
  }];
  let arguments = (ins Index:$lb, Index:$ub, Index:$step,
                       Variadic<AnyType>:$iter_args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $lb `to` $ub `step` $step `iter_args` `(` $iter_args `)` `->` custom<AtomLoopResults>(type($results)) $region attr-dict
  }];
  let hasVerifier = 1;
}

def Atom_YieldOp : Atom_Op<"yield",
    [Pure, Terminator, ParentOneOf<["LoopForOp"]>]> {
  let summary = "loop yield and termination operation";
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def Atom_ReduceOp : Atom_Op<"reduce", [Pure]> {
  let arguments = (ins Atom_TensorType:$tensor, Atom_ReduceKindAttr:$kind,
                       I64Attr:$dim);
  let results = (outs Atom_Scalar:$result);
  let assemblyFormat =
      "$kind $tensor `,` `dim` `=` $dim attr-dict `:` type($tensor) `,` type($result)";
  let hasVerifier = 1;
}

def Atom_CopyOp : Atom_Op<"copy", []> {
  let arguments = (ins Atom_TensorType:$src, Atom_TensorType:$dst);
  let assemblyFormat =
      "$src `,` $dst attr-dict `:` type($src) `to` type($dst)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Kernel container
//===----------------------------------------------------------------------===//

def Atom_KernelOp : Atom_Op<"kernel",
    [Symbol, IsolatedFromAbove, NoTerminator]> {
  let summary = "kernel container";
  let description = [{
    A kernel container mirroring the custom IR's atom.kernel: the region's
    single block arguments are the kernel parameters, and the schedule
    attributes (tile / map_blocks / map_warps / use_shared_memory /
    vector_width / unroll) plus the shape dictionary ride along as operation
    attributes.

    Example:
    ```mlir
    atom.kernel @vecadd(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>)
        attributes {tile = [256], map_blocks = true} {
      ...
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<I64ArrayAttr>:$tile,
                       OptionalAttr<BoolAttr>:$map_blocks,
                       OptionalAttr<BoolAttr>:$map_warps,
                       OptionalAttr<BoolAttr>:$use_shared_memory,
                       OptionalAttr<I64Attr>:$vector_width,
                       OptionalAttr<I64Attr>:$unroll,
                       OptionalAttr<DictionaryAttr>:$shape);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}
