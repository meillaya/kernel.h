//===- AtomDialect.td - kernel.h atom IR MLIR dialect ----*- tablegen -*-===//
//
// The "atom" dialect is the MLIR-side mirror of the custom structured-SSA
// atom IR (plan todo 9). It exists so the emitter (plan todo 21) can lower
// the custom IR 1:1 into MLIR, where later todos run MLIR passes and the
// gpu/nvvm/rocdl lowerings.
//
// TYPE MAPPING (pinned by plan todo 20):
//   scalar          -> f16 / f32 / i32 / i64 / index
//   tensor<...E,global> -> memref<...xE>            (default memory space)
//   tensor<...E,shared> -> memref<...xE, 3>         (address space 3)
//   vector<WxE>     -> vector<WxE>
//
// This file is part of kernel.h.
//===----------------------------------------------------------------------===//

#ifndef KERNELH_ATOM_DIALECT
#define KERNELH_ATOM_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Atom_Dialect : Dialect {
  let name = "atom";
  let cppNamespace = "::mlir::kernelh::atom";
  let summary = "kernel.h atom IR dialect";
  let description = [{
    The `atom` dialect mirrors the kernel.h custom atom IR (plan todo 9)
    op-for-op: it is the intermediate representation between the custom
    structured-SSA IR and the MLIR GPU dialect.

    Every construct the todo-9 IR can emit has a corresponding op here, so
    the emitter (plan todo 21) can construct them 1:1 from the custom IR.
    Tensor-with-memory-space values of the custom IR map to `memref` types:
    global tensors become plain memrefs (`memref<NxMxf32>`) and shared
    tensors become memrefs in address space 3 (`memref<NxMxf32, 3>`).
  }];
}

//===----------------------------------------------------------------------===//
// Enums (mirrors of include/kernelh/ir/types.hpp)
//===----------------------------------------------------------------------===//

def Atom_CmpPredAttr : I64EnumAttr<"CmpPred", "atom comparison predicate", [
  I64EnumAttrCase<"lt", 0>,
  I64EnumAttrCase<"le", 1>,
  I64EnumAttrCase<"gt", 2>,
  I64EnumAttrCase<"ge", 3>,
  I64EnumAttrCase<"eq", 4>,
  I64EnumAttrCase<"ne", 5>,
]> {
  let cppNamespace = "::mlir::kernelh::atom";
}

def Atom_MathKindAttr : I64EnumAttr<"MathKind", "atom unary math kind", [
  I64EnumAttrCase<"exp", 0>,
  I64EnumAttrCase<"log", 1>,
  I64EnumAttrCase<"sqrt", 2>,
  I64EnumAttrCase<"rsqrt", 3>,
  I64EnumAttrCase<"tanh", 4>,
]> {
  let cppNamespace = "::mlir::kernelh::atom";
}

def Atom_ReduceKindAttr : I64EnumAttr<"ReduceKind", "atom reduction kind", [
  I64EnumAttrCase<"sum", 0>,
  I64EnumAttrCase<"max", 1>,
]> {
  let cppNamespace = "::mlir::kernelh::atom";
}

def Atom_IdDimAttr : I64EnumAttr<"IdDim", "atom thread/block id dimension", [
  I64EnumAttrCase<"x", 0>,
  I64EnumAttrCase<"y", 1>,
  I64EnumAttrCase<"z", 2>,
]> {
  let cppNamespace = "::mlir::kernelh::atom";
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// The scalar types of the custom atom IR: f16, f32, i32, i64, index.
def Atom_Scalar : AnyTypeOf<[F16, F32, I32, I64, Index]>;

// "Tensor" (buffer) operands of the custom IR map to memref types.
def Atom_TensorType : AnyTypeOf<[AnyMemRef]>;

// Vector operands/results of the custom IR (vector<WxE>, load_vec/store_vec).
def Atom_VectorType : AnyTypeOf<[AnyVectorOfAnyRank]>;

// Shared tensors map to memrefs in address space 3 (checked in C++ via
// kernelh::atom::isSharedMemRefType, defined in AtomDialect.h/.cpp).
def Atom_SharedMemRef : Type<CPred<"kernelh::atom::isSharedMemRefType($_self)">,
                             "memref in the shared memory space (3)">;

// Elementwise ops accept scalars or tensors (buffers) of the same type.
def Atom_ElementwiseType : AnyTypeOf<[Atom_Scalar, Atom_TensorType]>;

include "kernelh/Dialect/Atom/IR/AtomOps.td"

#endif // KERNELH_ATOM_DIALECT
