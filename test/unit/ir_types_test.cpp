// Type-system unit tests for the atom IR (plan todo 9).

#include "kernelh/ir/types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using kernelh::ir::MemSpace;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::VectorType;
using kernelh::ir::parseMemSpace;
using kernelh::ir::parseScalarType;
using kernelh::ir::sizeOfScalar;
using kernelh::ir::toString;

TEST_CASE("ir_types_scalar_to_string", "[unit][ir]") {
  CHECK(std::string(toString(ScalarType::F16)) == "f16");
  CHECK(std::string(toString(ScalarType::F32)) == "f32");
  CHECK(std::string(toString(ScalarType::I32)) == "i32");
  CHECK(std::string(toString(ScalarType::I64)) == "i64");
  CHECK(std::string(toString(ScalarType::Index)) == "index");
  CHECK(std::string(toString(MemSpace::Global)) == "global");
  CHECK(std::string(toString(MemSpace::Shared)) == "shared");
}

TEST_CASE("ir_types_parse_scalar", "[unit][ir]") {
  ScalarType out = ScalarType::F32;
  CHECK((parseScalarType("f16", out) && out == ScalarType::F16));
  CHECK((parseScalarType("f32", out) && out == ScalarType::F32));
  CHECK((parseScalarType("i32", out) && out == ScalarType::I32));
  CHECK((parseScalarType("i64", out) && out == ScalarType::I64));
  CHECK((parseScalarType("index", out) && out == ScalarType::Index));
  CHECK_FALSE(parseScalarType("i1", out));
  CHECK_FALSE(parseScalarType("", out));
  CHECK_FALSE(parseScalarType("f64", out));
}

TEST_CASE("ir_types_parse_memspace", "[unit][ir]") {
  MemSpace out = MemSpace::Global;
  CHECK((parseMemSpace("global", out) && out == MemSpace::Global));
  CHECK((parseMemSpace("shared", out) && out == MemSpace::Shared));
  CHECK_FALSE(parseMemSpace("host", out));
}

TEST_CASE("ir_types_scalar_type_tostring", "[unit][ir]") {
  const Type t = Type::scalar(ScalarType::Index);
  CHECK(t.isScalar());
  CHECK_FALSE(t.isTensor());
  CHECK_FALSE(t.isVector());
  CHECK(t.toString() == "index");
}

TEST_CASE("ir_types_tensor_to_string", "[unit][ir]") {
  const Type t = Type::tensor(TensorType{{16, 16}, ScalarType::F32, MemSpace::Global});
  CHECK(t.isTensor());
  CHECK(t.toString() == "tensor<16x16xf32, global>");
  CHECK(t.asTensor().rank() == 2);
  CHECK(t.asTensor().numElements() == 256);
  CHECK(t.asTensor().sizeBytes() == 1024);
}

TEST_CASE("ir_types_tensor_shared_to_string", "[unit][ir]") {
  const Type t = Type::tensor(TensorType{{64}, ScalarType::F32, MemSpace::Shared});
  CHECK(t.toString() == "tensor<64xf32, shared>");
}

TEST_CASE("ir_types_vector_to_string", "[unit][ir]") {
  const Type t = Type::vector(VectorType{4, ScalarType::F32});
  CHECK(t.isVector());
  CHECK(t.toString() == "vector<4xf32>");
  CHECK(t.asVector().width == 4);
}

TEST_CASE("ir_types_equality", "[unit][ir]") {
  const Type a = Type::scalar(ScalarType::F32);
  const Type b = Type::scalar(ScalarType::F32);
  const Type c = Type::scalar(ScalarType::I32);
  const Type t1 = Type::tensor(TensorType{{4, 4}, ScalarType::F32, MemSpace::Global});
  const Type t2 = Type::tensor(TensorType{{4, 4}, ScalarType::F32, MemSpace::Global});
  const Type t3 = Type::tensor(TensorType{{4, 4}, ScalarType::F32, MemSpace::Shared});
  const Type v1 = Type::vector(VectorType{2, ScalarType::F32});
  const Type v2 = Type::vector(VectorType{4, ScalarType::F32});
  CHECK(a == b);
  CHECK(a != c);
  CHECK(t1 == t2);
  CHECK(t1 != t3);
  CHECK(v1 != v2);
}

TEST_CASE("ir_types_sizeof_scalar", "[unit][ir]") {
  CHECK(sizeOfScalar(ScalarType::F16) == 2);
  CHECK(sizeOfScalar(ScalarType::F32) == 4);
  CHECK(sizeOfScalar(ScalarType::I32) == 4);
  CHECK(sizeOfScalar(ScalarType::I64) == 8);
  CHECK(sizeOfScalar(ScalarType::Index) == 8);
}

TEST_CASE("ir_types_tensor_size_bytes", "[unit][ir]") {
  // 128x128 f32 = 65536 bytes (the todo-9 LDS budget boundary).
  const TensorType big{{128, 128}, ScalarType::F32, MemSpace::Shared};
  CHECK(big.sizeBytes() == 65536);
  const TensorType half{{64, 64}, ScalarType::F32, MemSpace::Shared};
  CHECK(half.sizeBytes() == 16384);
  const TensorType f16tile{{128, 128}, ScalarType::F16, MemSpace::Shared};
  CHECK(f16tile.sizeBytes() == 32768);
}

TEST_CASE("ir_types_default_is_scalar_f32", "[unit][ir]") {
  const Type t;
  CHECK(t.isScalar());
  CHECK(t.asScalar() == ScalarType::F32);
}
