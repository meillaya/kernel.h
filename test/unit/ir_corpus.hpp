// Corpus of structurally-valid atom kernels shared by the IR unit tests
// (plan todo 9). The corpus exercises EVERY pinned op kind:
//   vecadd   - const, block_id/block_dim/thread_id, mul/add, load, store,
//              cmp, select, sub, min
//   gemm     - alloc_shared x2, store/load shared, barrier, fma, loop_for
//              + yield, load_vec/store_vec, reduce, copy, math, tensor const,
//              div
//   softmax  - reduce, loop_for iter_args (reduction), math exp, div,
//              load/store
//   scan     - alloc_shared, barrier, load_vec/store_vec, loop_for, max
//   minimal  - float consts + add
//
// The same kernels are committed as .atom fixtures under test/fixtures
// (printed by this IR's printer); the printer tests assert they match exactly.

#ifndef KERNELH_TEST_UNIT_IR_CORPUS_HPP
#define KERNELH_TEST_UNIT_IR_CORPUS_HPP

#include "kernelh/ir/module.hpp"
#include "kernelh/ir/ops.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kernelh::test {

using kernelh::ir::Block;
using kernelh::ir::BlockArg;
using kernelh::ir::CmpPred;
using kernelh::ir::IdDim;
using kernelh::ir::Kernel;
using kernelh::ir::MathKind;
using kernelh::ir::MemSpace;
using kernelh::ir::Module;
using kernelh::ir::Op;
using kernelh::ir::OpKind;
using kernelh::ir::ReduceKind;
using kernelh::ir::ScalarType;
using kernelh::ir::TensorType;
using kernelh::ir::Type;
using kernelh::ir::Value;
namespace builder = kernelh::ir::builder;

namespace {

inline Type tensorType(std::vector<int64_t> shape, ScalarType elem,
                       MemSpace space) {
  return Type::tensor(TensorType{std::move(shape), elem, space});
}
inline Type scalarType(ScalarType elem) { return Type::scalar(elem); }
inline Type indexType() { return Type::scalar(ScalarType::Index); }

inline Value kernelArg(Block &body, int64_t index) {
  return Value::blockArg(&body, index);
}

} // namespace

// ---------------------------------------------------------------------------
// Corpus kernels
// ---------------------------------------------------------------------------

inline Module buildMinimal() {
  Module module;
  auto kernel = std::make_unique<Kernel>("minimal");
  Block &b = kernel->body;
  const Value c0 =
      builder::makeConst(b, 0.5, scalarType(ScalarType::F32))->result();
  const Value c1 =
      builder::makeConst(b, 0.25, scalarType(ScalarType::F32))->result();
  builder::makeBin(b, OpKind::Add, c0, c1);
  module.kernels.push_back(std::move(kernel));
  return module;
}

inline Module buildVecadd() {
  Module module;
  auto kernel = std::make_unique<Kernel>("vecadd");
  kernel->schedule.tile = {256};
  kernel->schedule.map_blocks = true;
  kernel->shape["n"] = 1048576;

  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensorType({1024}, ScalarType::F32, MemSpace::Global)});
  b.args.push_back(BlockArg{tensorType({1024}, ScalarType::F32, MemSpace::Global)});
  const Value a = kernelArg(b, 0);
  const Value bb = kernelArg(b, 1);

  const Value c0 = builder::makeConst(b, 0, indexType())->result();
  const Value bid = builder::makeId(b, OpKind::BlockId, IdDim::X)->result();
  const Value bdim = builder::makeId(b, OpKind::BlockDim, IdDim::X)->result();
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value base = builder::makeBin(b, OpKind::Mul, bid, bdim)->result();
  const Value gid = builder::makeBin(b, OpKind::Add, base, tid)->result();
  const Value x = builder::makeLoad(b, a, {gid})->result();
  const Value y = builder::makeLoad(b, bb, {gid})->result();
  const Value sum = builder::makeBin(b, OpKind::Add, x, y)->result();
  builder::makeStore(b, sum, bb, {gid});
  const Value lt = builder::makeCmp(b, CmpPred::Lt, sum, x)->result();
  const Value sel = builder::makeSelect(b, lt, sum, x)->result();
  builder::makeStore(b, sel, bb, {gid});
  const Value sub = builder::makeBin(b, OpKind::Sub, sum, x)->result();
  builder::makeBin(b, OpKind::Min, sub, sum);
  (void)c0;

  module.kernels.push_back(std::move(kernel));
  return module;
}

inline Module buildGemm() {
  Module module;
  auto kernel = std::make_unique<Kernel>("gemm");
  kernel->schedule.tile = {64, 64, 32};
  kernel->schedule.map_blocks = true;
  kernel->schedule.map_warps = true;
  kernel->schedule.use_shared_memory = true;
  kernel->schedule.vector_width = 4;
  kernel->schedule.unroll = 2;
  kernel->shape["m"] = 1024;
  kernel->shape["n"] = 1024;
  kernel->shape["k"] = 1024;

  Block &b = kernel->body;
  for (int i = 0; i < 3; ++i) {
    b.args.push_back(
        BlockArg{tensorType({1024, 1024}, ScalarType::F32, MemSpace::Global)});
  }
  const Value a = kernelArg(b, 0);
  const Value bb = kernelArg(b, 1);
  const Value c = kernelArg(b, 2);

  const Value c0 = builder::makeConst(b, 0, indexType())->result();
  const Value c32 = builder::makeConst(b, 32, indexType())->result();
  const Value c1 = builder::makeConst(b, 1, indexType())->result();
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value bid = builder::makeId(b, OpKind::BlockId, IdDim::X)->result();
  const Value bdim = builder::makeId(b, OpKind::BlockDim, IdDim::X)->result();
  const Value base = builder::makeBin(b, OpKind::Mul, bid, bdim)->result();
  const Value gid = builder::makeBin(b, OpKind::Add, base, tid)->result();
  const Value as = builder::makeAllocShared(
                       b, TensorType{{64, 64}, ScalarType::F32, MemSpace::Shared})
                       ->result();
  const Value bs = builder::makeAllocShared(
                       b, TensorType{{64, 16}, ScalarType::F32, MemSpace::Shared})
                       ->result();
  const Value loaded = builder::makeLoad(b, a, {gid, c0})->result();
  builder::makeStore(b, loaded, as, {tid, c0});
  builder::makeBarrier(b);
  const Value v2 = builder::makeLoad(b, as, {tid, c0})->result();
  const Value one = builder::makeConst(b, 1.0, scalarType(ScalarType::F32))->result();
  const Value v3 = builder::makeFma(b, v2, one, one)->result();
  builder::makeStore(b, v3, c, {gid, c0});

  Op *loop = builder::makeLoopFor(b, c0, c32, c1, {v3},
                                  {scalarType(ScalarType::F32)});
  Block &body = *loop->region;
  const Value ind = kernelArg(body, 0);
  const Value acc = kernelArg(body, 1);
  const Value lv = builder::makeLoad(body, as, {tid, ind})->result();
  const Value lsum = builder::makeBin(body, OpKind::Add, lv, acc)->result();
  builder::makeYield(body, {lsum});

  const Value v4 = builder::makeLoadVec(b, a, {base, c0}, 4)->result();
  builder::makeStoreVec(b, v4, bb, {base, c0});
  builder::makeReduce(b, ReduceKind::Max, a, 1);
  builder::makeCopy(b, a, bb);
  const Value sq = builder::makeMath(b, MathKind::Sqrt, v2)->result();
  builder::makeConst(b, 0.0,
                     tensorType({64, 16}, ScalarType::F32, MemSpace::Global));
  builder::makeBin(b, OpKind::Div, sq, v3);
  (void)bs;

  module.kernels.push_back(std::move(kernel));
  return module;
}

inline Module buildSoftmax() {
  Module module;
  auto kernel = std::make_unique<Kernel>("softmax");
  kernel->schedule.tile = {128};
  kernel->schedule.map_blocks = true;
  kernel->schedule.map_warps = true;
  kernel->shape["rows"] = 1024;
  kernel->shape["cols"] = 1024;

  Block &b = kernel->body;
  b.args.push_back(
      BlockArg{tensorType({1024, 1024}, ScalarType::F32, MemSpace::Global)});
  b.args.push_back(
      BlockArg{tensorType({1024, 1024}, ScalarType::F32, MemSpace::Global)});
  const Value x = kernelArg(b, 0);
  const Value y = kernelArg(b, 1);

  const Value c0 = builder::makeConst(b, 0, indexType())->result();
  const Value c1024 = builder::makeConst(b, 1024, indexType())->result();
  const Value c1 = builder::makeConst(b, 1, indexType())->result();
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value red = builder::makeReduce(b, ReduceKind::Sum, x, 1)->result();

  Op *loop = builder::makeLoopFor(b, c0, c1024, c1, {red},
                                  {scalarType(ScalarType::F32)});
  Block &body = *loop->region;
  const Value ind = kernelArg(body, 0);
  const Value acc = kernelArg(body, 1);
  const Value lv = builder::makeLoad(body, x, {tid, ind})->result();
  const Value exp = builder::makeMath(body, MathKind::Exp, lv)->result();
  const Value nsum = builder::makeBin(body, OpKind::Add, acc, exp)->result();
  builder::makeYield(body, {nsum});

  const Value first = builder::makeLoad(b, x, {tid, c0})->result();
  const Value expFirst = builder::makeMath(b, MathKind::Exp, first)->result();
  const Value out = builder::makeBin(b, OpKind::Div, expFirst, loop->result())->result();
  builder::makeStore(b, out, y, {tid, c0});

  module.kernels.push_back(std::move(kernel));
  return module;
}

inline Module buildScan() {
  Module module;
  auto kernel = std::make_unique<Kernel>("scan");
  kernel->schedule.tile = {256};
  kernel->schedule.map_blocks = true;
  kernel->schedule.use_shared_memory = true;
  kernel->schedule.vector_width = 4;
  kernel->shape["n"] = 1024;

  Block &b = kernel->body;
  b.args.push_back(BlockArg{tensorType({1024}, ScalarType::F32, MemSpace::Global)});
  b.args.push_back(BlockArg{tensorType({1024}, ScalarType::F32, MemSpace::Global)});
  const Value in = kernelArg(b, 0);
  const Value out = kernelArg(b, 1);

  const Value c0 = builder::makeConst(b, 0, indexType())->result();
  const Value c256 = builder::makeConst(b, 256, indexType())->result();
  const Value c1 = builder::makeConst(b, 1, indexType())->result();
  const Value tid = builder::makeId(b, OpKind::ThreadId, IdDim::X)->result();
  const Value bid = builder::makeId(b, OpKind::BlockId, IdDim::X)->result();
  const Value bdim = builder::makeId(b, OpKind::BlockDim, IdDim::X)->result();
  const Value base = builder::makeBin(b, OpKind::Mul, bid, bdim)->result();
  const Value gid = builder::makeBin(b, OpKind::Add, base, tid)->result();
  const Value sh = builder::makeAllocShared(
                       b, TensorType{{256}, ScalarType::F32, MemSpace::Shared})
                       ->result();
  const Value v = builder::makeLoad(b, in, {gid})->result();
  builder::makeStore(b, v, sh, {tid});
  builder::makeBarrier(b);
  const Value v2 = builder::makeLoad(b, sh, {tid})->result();

  Op *loop = builder::makeLoopFor(b, c0, c256, c1, {v2},
                                  {scalarType(ScalarType::F32)});
  Block &body = *loop->region;
  const Value ind = kernelArg(body, 0);
  const Value acc = kernelArg(body, 1);
  const Value lv = builder::makeLoad(body, sh, {ind})->result();
  const Value mx = builder::makeBin(body, OpKind::Max, lv, acc)->result();
  builder::makeYield(body, {mx});

  const Value v4 = builder::makeLoadVec(b, in, {base}, 4)->result();
  builder::makeStoreVec(b, v4, out, {base});
  builder::makeStore(b, loop->result(), out, {gid});

  module.kernels.push_back(std::move(kernel));
  return module;
}

// All five corpus kernels in one module (tests the multi-kernel path).
inline Module buildCorpusModule() {
  Module module;
  Module vecadd = buildVecadd();
  Module gemm = buildGemm();
  Module softmax = buildSoftmax();
  Module scan = buildScan();
  Module minimal = buildMinimal();
  for (auto &k : minimal.kernels) {
    module.kernels.push_back(std::move(k));
  }
  for (auto &k : vecadd.kernels) {
    module.kernels.push_back(std::move(k));
  }
  for (auto &k : gemm.kernels) {
    module.kernels.push_back(std::move(k));
  }
  for (auto &k : softmax.kernels) {
    module.kernels.push_back(std::move(k));
  }
  for (auto &k : scan.kernels) {
    module.kernels.push_back(std::move(k));
  }
  return module;
}

} // namespace kernelh::test

#endif // KERNELH_TEST_UNIT_IR_CORPUS_HPP
