// kernel.h "atom" IR - type system (plan todo 9).
//
// The atom IR is a structured-SSA custom IR the DSL emits and the passes
// transform. It has NO MLIR dependency: it is plain C++23 types that a later
// todo (the MLIR bridge) lowers into the gpu dialect.
//
// TYPES (pinned by the todo-9 plan):
//   scalars: f16, f32, i32, i64, index
//   tensor<Nx...xE, space> with space in {global, shared} - statically shaped
//     tiles; dynamic extents live in `index` VALUES, never in tensor dims.
//   vector<WxE> - only used by atom.load_vec / atom.store_vec.
//
// PLAN F2 NOTE: this header carries NO device constants (the LDS size lives
// in verifier.hpp as kDefaultLdsLimitBytes until todo 10a swaps it for the
// device-limits table lookup).

#ifndef KERNELH_IR_TYPES_HPP
#define KERNELH_IR_TYPES_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace kernelh::ir {

// ---------------------------------------------------------------------------
// Scalar types
// ---------------------------------------------------------------------------

enum class ScalarType : uint8_t { F16, F32, I32, I64, Index };

const char *toString(ScalarType type);
bool parseScalarType(std::string_view text, ScalarType &out);

// Byte size of one element (f16=2, f32=4, i32=4, i64=8, index=8).
int64_t sizeOfScalar(ScalarType type);

inline bool isFloatType(ScalarType type) {
  return type == ScalarType::F16 || type == ScalarType::F32;
}
inline bool isIntegerType(ScalarType type) {
  return type == ScalarType::I32 || type == ScalarType::I64 ||
         type == ScalarType::Index;
}

// ---------------------------------------------------------------------------
// Memory spaces
// ---------------------------------------------------------------------------

enum class MemSpace : uint8_t { Global, Shared };

const char *toString(MemSpace space);
bool parseMemSpace(std::string_view text, MemSpace &out);

// ---------------------------------------------------------------------------
// Small per-op enums
// ---------------------------------------------------------------------------

enum class CmpPred : uint8_t { Lt, Le, Gt, Ge, Eq, Ne };
const char *toString(CmpPred pred);
bool parseCmpPred(std::string_view text, CmpPred &out);

enum class MathKind : uint8_t { Exp, Log, Sqrt, Rsqrt, Tanh };
const char *toString(MathKind kind);
bool parseMathKind(std::string_view text, MathKind &out);

enum class ReduceKind : uint8_t { Sum, Max };
const char *toString(ReduceKind kind);
bool parseReduceKind(std::string_view text, ReduceKind &out);

enum class IdDim : uint8_t { X, Y, Z };
const char *toString(IdDim dim);
bool parseIdDim(std::string_view text, IdDim &out);

// ---------------------------------------------------------------------------
// Composite types
// ---------------------------------------------------------------------------

// Statically-shaped tensor tile: extents then element type and memory space.
// shape.size() == rank; every extent must be positive (verified).
struct TensorType {
  std::vector<int64_t> shape;
  ScalarType elem = ScalarType::F32;
  MemSpace space = MemSpace::Global;

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
  // Total element count (product of extents; 1 for rank-0).
  int64_t numElements() const;
  // Size in bytes = numElements * sizeOfScalar(elem).
  int64_t sizeBytes() const;

  bool operator==(const TensorType &other) const;
  bool operator!=(const TensorType &other) const { return !(*this == other); }
};

// vector<WxE>: W lanes of element type E (used by load_vec/store_vec only).
struct VectorType {
  int64_t width = 1;
  ScalarType elem = ScalarType::F32;

  bool operator==(const VectorType &other) const;
  bool operator!=(const VectorType &other) const { return !(*this == other); }
};

// A value type: scalar, tensor, or vector. Defaults to scalar f32 (used by
// Value::type() for an invalid value - callers must check Value::valid()).
class Type {
public:
  Type() = default;

  static Type scalar(ScalarType type);
  static Type tensor(const TensorType &type);
  static Type vector(const VectorType &type);

  bool isScalar() const;
  bool isTensor() const;
  bool isVector() const;

  ScalarType asScalar() const; // requires isScalar()
  const TensorType &asTensor() const; // requires isTensor()
  const VectorType &asVector() const; // requires isVector()

  // MLIR-like text: "f32", "tensor<16x16xf32, global>", "vector<4xf32>".
  std::string toString() const;

  bool operator==(const Type &other) const;
  bool operator!=(const Type &other) const { return !(*this == other); }

private:
  std::variant<ScalarType, TensorType, VectorType> impl_{ScalarType::F32};
};

} // namespace kernelh::ir

#endif // KERNELH_IR_TYPES_HPP
