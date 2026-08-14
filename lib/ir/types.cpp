// Type system definitions for the atom IR (plan todo 9).

#include "kernelh/ir/types.hpp"

#include <cassert>
#include <numeric>
#include <string>

namespace kernelh::ir {

// ---------------------------------------------------------------------------
// Scalar types
// ---------------------------------------------------------------------------

const char *toString(ScalarType type) {
  switch (type) {
  case ScalarType::F16:
    return "f16";
  case ScalarType::F32:
    return "f32";
  case ScalarType::I32:
    return "i32";
  case ScalarType::I64:
    return "i64";
  case ScalarType::Index:
    return "index";
  }
  return "?";
}

bool parseScalarType(std::string_view text, ScalarType &out) {
  if (text == "f16") {
    out = ScalarType::F16;
  } else if (text == "f32") {
    out = ScalarType::F32;
  } else if (text == "i32") {
    out = ScalarType::I32;
  } else if (text == "i64") {
    out = ScalarType::I64;
  } else if (text == "index") {
    out = ScalarType::Index;
  } else {
    return false;
  }
  return true;
}

int64_t sizeOfScalar(ScalarType type) {
  switch (type) {
  case ScalarType::F16:
    return 2;
  case ScalarType::F32:
  case ScalarType::I32:
    return 4;
  case ScalarType::I64:
  case ScalarType::Index:
    return 8;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Memory spaces
// ---------------------------------------------------------------------------

const char *toString(MemSpace space) {
  return space == MemSpace::Global ? "global" : "shared";
}

bool parseMemSpace(std::string_view text, MemSpace &out) {
  if (text == "global") {
    out = MemSpace::Global;
    return true;
  }
  if (text == "shared") {
    out = MemSpace::Shared;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Small per-op enums
// ---------------------------------------------------------------------------

const char *toString(CmpPred pred) {
  switch (pred) {
  case CmpPred::Lt:
    return "lt";
  case CmpPred::Le:
    return "le";
  case CmpPred::Gt:
    return "gt";
  case CmpPred::Ge:
    return "ge";
  case CmpPred::Eq:
    return "eq";
  case CmpPred::Ne:
    return "ne";
  }
  return "?";
}

bool parseCmpPred(std::string_view text, CmpPred &out) {
  if (text == "lt") {
    out = CmpPred::Lt;
  } else if (text == "le") {
    out = CmpPred::Le;
  } else if (text == "gt") {
    out = CmpPred::Gt;
  } else if (text == "ge") {
    out = CmpPred::Ge;
  } else if (text == "eq") {
    out = CmpPred::Eq;
  } else if (text == "ne") {
    out = CmpPred::Ne;
  } else {
    return false;
  }
  return true;
}

const char *toString(MathKind kind) {
  switch (kind) {
  case MathKind::Exp:
    return "exp";
  case MathKind::Log:
    return "log";
  case MathKind::Sqrt:
    return "sqrt";
  case MathKind::Rsqrt:
    return "rsqrt";
  case MathKind::Tanh:
    return "tanh";
  }
  return "?";
}

bool parseMathKind(std::string_view text, MathKind &out) {
  if (text == "exp") {
    out = MathKind::Exp;
  } else if (text == "log") {
    out = MathKind::Log;
  } else if (text == "sqrt") {
    out = MathKind::Sqrt;
  } else if (text == "rsqrt") {
    out = MathKind::Rsqrt;
  } else if (text == "tanh") {
    out = MathKind::Tanh;
  } else {
    return false;
  }
  return true;
}

const char *toString(ReduceKind kind) {
  return kind == ReduceKind::Sum ? "sum" : "max";
}

bool parseReduceKind(std::string_view text, ReduceKind &out) {
  if (text == "sum") {
    out = ReduceKind::Sum;
    return true;
  }
  if (text == "max") {
    out = ReduceKind::Max;
    return true;
  }
  return false;
}

const char *toString(IdDim dim) {
  switch (dim) {
  case IdDim::X:
    return "x";
  case IdDim::Y:
    return "y";
  case IdDim::Z:
    return "z";
  }
  return "?";
}

bool parseIdDim(std::string_view text, IdDim &out) {
  if (text == "x") {
    out = IdDim::X;
  } else if (text == "y") {
    out = IdDim::Y;
  } else if (text == "z") {
    out = IdDim::Z;
  } else {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Composite types
// ---------------------------------------------------------------------------

int64_t TensorType::numElements() const {
  int64_t n = 1;
  for (int64_t extent : shape) {
    n *= extent;
  }
  return n;
}

int64_t TensorType::sizeBytes() const { return numElements() * sizeOfScalar(elem); }

bool TensorType::operator==(const TensorType &other) const {
  return shape == other.shape && elem == other.elem && space == other.space;
}

bool VectorType::operator==(const VectorType &other) const {
  return width == other.width && elem == other.elem;
}

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

Type Type::scalar(ScalarType type) {
  Type t;
  t.impl_ = type;
  return t;
}

Type Type::tensor(const TensorType &type) {
  Type t;
  t.impl_ = type;
  return t;
}

Type Type::vector(const VectorType &type) {
  Type t;
  t.impl_ = type;
  return t;
}

bool Type::isScalar() const { return std::holds_alternative<ScalarType>(impl_); }
bool Type::isTensor() const { return std::holds_alternative<TensorType>(impl_); }
bool Type::isVector() const { return std::holds_alternative<VectorType>(impl_); }

ScalarType Type::asScalar() const {
  assert(isScalar() && "Type::asScalar on non-scalar type");
  return std::get<ScalarType>(impl_);
}

const TensorType &Type::asTensor() const {
  assert(isTensor() && "Type::asTensor on non-tensor type");
  return std::get<TensorType>(impl_);
}

const VectorType &Type::asVector() const {
  assert(isVector() && "Type::asVector on non-vector type");
  return std::get<VectorType>(impl_);
}

std::string Type::toString() const {
  if (isScalar()) {
    return ::kernelh::ir::toString(asScalar());
  }
  if (isTensor()) {
    const TensorType &t = asTensor();
    std::string s = "tensor<";
    for (size_t i = 0; i < t.shape.size(); ++i) {
      if (i != 0) {
        s += "x";
      }
      s += std::to_string(t.shape[i]);
    }
    s += "x";
    s += ::kernelh::ir::toString(t.elem);
    s += ", ";
    s += ::kernelh::ir::toString(t.space);
    s += ">";
    return s;
  }
  const VectorType &v = asVector();
  return "vector<" + std::to_string(v.width) + "x" +
         ::kernelh::ir::toString(v.elem) + ">";
}

bool Type::operator==(const Type &other) const { return impl_ == other.impl_; }

} // namespace kernelh::ir
