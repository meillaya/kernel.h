// kernel.h "atom" IR - values, block arguments, and basic blocks (plan todo 9).
//
// The atom IR is STRUCTURED SSA: values dominate within their defining scope
// and every scope nested inside it. There are NO arbitrary gotos and NO phi
// nodes (plan decision D10). A Value is a lightweight handle to either an op
// result or a block argument; blocks are owned by their parent op's region
// (loop_for) or by a kernel, so raw pointers stay valid for the module's
// lifetime (the IR is built then only traversed/verified; passes rebuild).

#ifndef KERNELH_IR_VALUE_HPP
#define KERNELH_IR_VALUE_HPP

#include "kernelh/ir/types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace kernelh::ir {

struct Op; // defined in ops.hpp

// A block argument: identified by its position in Block::args.
struct BlockArg {
  Type type;
};

// A basic block. Blocks are single-entry, single-exit scopes:
//   - the kernel body block (no parent),
//   - a loop_for body block (parent op kind == LoopFor, one per region).
// args: block arguments (kernel parameters for the kernel body; induction
// var + iter args for a loop body). ops: the operations in textual order.
// parentBlock / parentOp describe the lexical nesting (used by printer,
// parser, and the dominance analysis).
struct Block {
  std::vector<BlockArg> args;
  std::vector<std::unique_ptr<Op>> ops;
  Block *parentBlock = nullptr;
  Op *parentOp = nullptr;

  Block() = default;
  // Out-of-line (defined in ops.hpp after Op is complete): the ops vector
  // holds unique_ptr<Op> which needs the complete type to destroy.
  ~Block();
  Block(Block &&) noexcept = default;
  Block &operator=(Block &&) noexcept = default;
  Block(const Block &) = delete;
  Block &operator=(const Block &) = delete;
};

// A handle to an SSA value: an op result or a block argument. Values do not
// carry their printed name ("%vN"); names are assigned deterministically by
// the printer from structural position (per-scope numbering) and resolved
// back by the parser.
class Value {
public:
  enum class Kind : uint8_t { None, OpResult, BlockArg };

  Value() = default;

  static Value none() { return Value(); }
  static Value opResult(Op *op, int64_t index) {
    Value v;
    v.kind_ = Kind::OpResult;
    v.op_ = op;
    v.index_ = index;
    return v;
  }
  static Value blockArg(Block *block, int64_t index) {
    Value v;
    v.kind_ = Kind::BlockArg;
    v.block_ = block;
    v.index_ = index;
    return v;
  }

  Kind kind() const { return kind_; }
  bool valid() const { return kind_ != Kind::None; }

  Op *op() const { return op_; }           // when kind() == OpResult
  Block *block() const { return block_; }  // when kind() == BlockArg
  int64_t index() const { return index_; } // result index / arg index

  // Resolved type from the definition site. Defined in ops.hpp (needs Op).
  Type type() const;

  bool operator==(const Value &other) const {
    return kind_ == other.kind_ && op_ == other.op_ &&
           block_ == other.block_ && index_ == other.index_;
  }
  bool operator!=(const Value &other) const { return !(*this == other); }

private:
  Kind kind_ = Kind::None;
  Op *op_ = nullptr;
  Block *block_ = nullptr;
  int64_t index_ = 0;
};

} // namespace kernelh::ir

#endif // KERNELH_IR_VALUE_HPP
