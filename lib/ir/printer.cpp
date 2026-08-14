// MLIR-like text printer for the atom IR (plan todo 9). Deterministic output:
// per-scope %vN numbering, fixed attribute order, sorted shape map, and
// shortest round-trip float literals. See printer.hpp for the format summary.

#include "kernelh/ir/printer.hpp"

#include <charconv>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kernelh::ir {

namespace {

// Shortest round-trip float text via std::to_chars (chars_format::general with
// no precision = shortest representation).
std::string formatFloat(double value) {
  char buffer[64];
  const std::to_chars_result res =
      std::to_chars(buffer, buffer + sizeof(buffer), value,
                    std::chars_format::general);
  return std::string(buffer, res.ptr);
}

// Deterministic per-scope SSA numbering with shadow avoidance. Each scope
// names its block arguments first (in order), then every op result in textual
// order, with a per-scope counter that restarts at 0; a number is SKIPPED when
// that name is already active from an enclosing scope (otherwise a reference
// printed inside a nested scope could resolve to the wrong definition - the
// classic `%v0` shadowing bug). A scope's names are freed when the scope
// exits, so sibling scopes can reuse them. Names are "%v<N>"; the table
// stores "v<N>".
class NameTable {
public:
  void assign(const Block &block) {
    int64_t counter = 0;
    std::vector<std::string> argNames;
    argNames.reserve(block.args.size());
    for (size_t i = 0; i < block.args.size(); ++i) {
      argNames.push_back(freshName(counter));
    }
    argNames_[&block] = std::move(argNames);
    for (const auto &op : block.ops) {
      std::vector<std::string> resultNames;
      resultNames.reserve(op->resultTypes.size());
      for (size_t i = 0; i < op->resultTypes.size(); ++i) {
        resultNames.push_back(freshName(counter));
      }
      opNames_[op.get()] = std::move(resultNames);
      if (op->region != nullptr) {
        assign(*op->region);
      }
    }
    releaseNames(block);
  }

  const std::string &blockArgName(const Block *block, int64_t index) const {
    return argNames_.at(block)[static_cast<size_t>(index)];
  }
  const std::string &opResultName(const Op *op, int64_t index) const {
    return opNames_.at(op)[static_cast<size_t>(index)];
  }

  std::string valueName(const Value &v) const {
    return "%" + (v.kind() == Value::Kind::BlockArg
                      ? blockArgName(v.block(), v.index())
                      : opResultName(v.op(), v.index()));
  }

private:
  std::string freshName(int64_t &counter) {
    for (;;) {
      const std::string candidate = "v" + std::to_string(counter++);
      if (activeNames_.insert(candidate).second) {
        return candidate;
      }
    }
  }

  void releaseNames(const Block &block) {
    for (const std::string &name : argNames_[&block]) {
      activeNames_.erase(name);
    }
    for (const auto &op : block.ops) {
      for (const std::string &name : opNames_[op.get()]) {
        activeNames_.erase(name);
      }
    }
  }

  std::unordered_set<std::string> activeNames_;
  std::unordered_map<const Op *, std::vector<std::string>> opNames_;
  std::unordered_map<const Block *, std::vector<std::string>> argNames_;
};

class Printer {
public:
  Printer(const Module &module) {
    for (const auto &kernel : module.kernels) {
      names_.assign(kernel->body);
    }
  }

  std::string run(const Module &module) {
    std::string out;
    for (const auto &kernel : module.kernels) {
      printKernel(*kernel, out);
    }
    return out;
  }

private:
  NameTable names_;

  void printKernel(const Kernel &kernel, std::string &out) {
    out += "atom.kernel @" + kernel.name;
    const Block &body = kernel.body;
    if (!body.args.empty()) {
      out += "(";
      for (size_t i = 0; i < body.args.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        out += "%" + names_.blockArgName(&body, static_cast<int64_t>(i));
        out += ": " + body.args[i].type.toString();
      }
      out += ")";
    }
    out += " {\n";
    printScheduleAttrs(kernel, out);
    printBlock(body, /*indent=*/1, out);
    out += "}\n";
  }

  // Fixed attribute order (deterministic). Shape map prints key-sorted
  // because std::map iterates in key order.
  void printScheduleAttrs(const Kernel &kernel, std::string &out) {
    const std::string pad = "  ";
    out += pad + "tile = [";
    for (size_t i = 0; i < kernel.schedule.tile.size(); ++i) {
      if (i != 0) {
        out += ", ";
      }
      out += std::to_string(kernel.schedule.tile[i]);
    }
    out += "]\n";
    out += pad + "map_blocks = ";
    out += kernel.schedule.map_blocks ? "true\n" : "false\n";
    out += pad + "map_warps = ";
    out += kernel.schedule.map_warps ? "true\n" : "false\n";
    out += pad + "use_shared_memory = ";
    out += kernel.schedule.use_shared_memory ? "true\n" : "false\n";
    out += pad + "vector_width = " +
           std::to_string(kernel.schedule.vector_width) + "\n";
    out += pad + "unroll = " + std::to_string(kernel.schedule.unroll) + "\n";
    out += pad + "shape = {";
    bool first = true;
    for (const auto &[key, value] : kernel.shape) {
      if (!first) {
        out += ", ";
      }
      first = false;
      out += key + " = " + std::to_string(value);
    }
    out += "}\n";
  }

  void printBlock(const Block &block, int indent, std::string &out) {
    for (const auto &op : block.ops) {
      printOp(*op, indent, out);
    }
  }

  void printOp(const Op &op, int indent, std::string &out) {
    const std::string pad(static_cast<size_t>(indent) * 2, ' ');
    std::string prefix;
    if (!op.resultTypes.empty()) {
      std::string resultList;
      for (size_t i = 0; i < op.resultTypes.size(); ++i) {
        if (i != 0) {
          resultList += ", ";
        }
        resultList +=
            "%" + names_.opResultName(&op, static_cast<int64_t>(i));
      }
      prefix = resultList + " = ";
    }
    out += pad + prefix + "atom." + toString(op.kind);

    auto operand = [&](int64_t i) { return names_.valueName(op.operands[i]); };

    switch (op.kind) {
    case OpKind::Const: {
      out += " ";
      if (op.intValue.has_value()) {
        out += std::to_string(*op.intValue);
      } else {
        out += formatFloat(*op.floatValue);
      }
      out += " : " + op.resultTypes[0].toString();
      break;
    }
    case OpKind::Add:
    case OpKind::Sub:
    case OpKind::Mul:
    case OpKind::Div:
    case OpKind::Max:
    case OpKind::Min:
      out += " " + operand(0) + ", " + operand(1);
      out += " : " + op.resultTypes[0].toString();
      break;
    case OpKind::Fma:
      out += " " + operand(0) + ", " + operand(1) + ", " + operand(2);
      out += " : " + op.resultTypes[0].toString();
      break;
    case OpKind::Cmp:
      out += " " + std::string(toString(op.pred)) + " " + operand(0) +
             ", " + operand(1);
      out += " : " + op.operands[0].type().toString();
      break;
    case OpKind::Select:
      out += " " + operand(0) + ", " + operand(1) + ", " + operand(2);
      out += " : " + op.resultTypes[0].toString();
      break;
    case OpKind::Math:
      out += " " + std::string(toString(op.mathKind)) + " " + operand(0);
      out += " : " + op.resultTypes[0].toString();
      break;
    case OpKind::Load: {
      out += " " + operand(0);
      printIndices(out, op, 1, static_cast<int64_t>(op.operands.size()));
      out += " : " + op.operands[0].type().toString();
      break;
    }
    case OpKind::Store: {
      out += " " + operand(0) + ", " + operand(1);
      printIndices(out, op, 2, static_cast<int64_t>(op.operands.size()));
      out += " : " + op.operands[1].type().toString();
      break;
    }
    case OpKind::LoadVec: {
      out += " " + operand(0);
      printIndices(out, op, 1, static_cast<int64_t>(op.operands.size()));
      out += " : " + op.operands[0].type().toString() + ", " +
             op.resultTypes[0].toString();
      break;
    }
    case OpKind::StoreVec: {
      out += " " + operand(0) + ", " + operand(1);
      printIndices(out, op, 2, static_cast<int64_t>(op.operands.size()));
      out += " : " + op.operands[1].type().toString() + ", " +
             op.operands[0].type().toString();
      break;
    }
    case OpKind::AllocShared:
      out += " : " + op.resultTypes[0].toString();
      break;
    case OpKind::Barrier:
      break;
    case OpKind::ThreadId:
    case OpKind::BlockId:
    case OpKind::BlockDim:
    case OpKind::GridDim:
      out += " " + std::string(toString(op.idDim));
      out += " : " + op.resultTypes[0].toString();
      break;
    case OpKind::LoopFor:
      printLoopFor(op, pad, out);
      break;
    case OpKind::Yield: {
      if (!op.operands.empty()) {
        out += " ";
        for (size_t i = 0; i < op.operands.size(); ++i) {
          if (i != 0) {
            out += ", ";
          }
          out += operand(static_cast<int64_t>(i));
        }
        out += " : ";
        for (size_t i = 0; i < op.operands.size(); ++i) {
          if (i != 0) {
            out += ", ";
          }
          out += op.operands[i].type().toString();
        }
      }
      break;
    }
    case OpKind::Reduce:
      out += " " + std::string(toString(op.reduceKind)) + " " + operand(0);
      out += ", dim = " + std::to_string(op.reduceDim);
      out += " : " + op.operands[0].type().toString();
      break;
    case OpKind::Copy:
      out += " " + operand(0) + ", " + operand(1);
      out += " : " + op.operands[0].type().toString() + " to " +
             op.operands[1].type().toString();
      break;
    }
    out += "\n";
  }

  // Prints "[%i0, %i1]" for operands [first, last) of a memory op.
  void printIndices(std::string &out, const Op &op, int64_t first,
                    int64_t last) {
    out += "[";
    for (int64_t i = first; i < last; ++i) {
      if (i != first) {
        out += ", ";
      }
      out += names_.valueName(op.operands[static_cast<size_t>(i)]);
    }
    out += "]";
  }

  void printLoopFor(const Op &op, const std::string &pad, std::string &out) {
    auto operand = [&](int64_t i) { return names_.valueName(op.operands[i]); };
    const size_t numIter = op.resultTypes.size();
    out += " " + operand(0) + " to " + operand(1) + " step " + operand(2);
    out += " iter_args(";
    for (size_t i = 0; i < numIter; ++i) {
      if (i != 0) {
        out += ", ";
      }
      out += names_.valueName(op.operands[3 + i]);
    }
    out += ") -> (";
    for (size_t i = 0; i < numIter; ++i) {
      if (i != 0) {
        out += ", ";
      }
      out += op.resultTypes[i].toString();
    }
    out += ") {\n";

    const Block &body = *op.region;
    const std::string bodyPad = pad + "  ";
    // Block header: ^bb0(%vN: index, %vM: f32): - printed names ARE the
    // per-scope numbering (registered by assign() before the body ops).
    out += bodyPad + "^bb0(";
    for (size_t i = 0; i < body.args.size(); ++i) {
      if (i != 0) {
        out += ", ";
      }
      out += "%" + names_.blockArgName(&body, static_cast<int64_t>(i));
      out += ": " + body.args[i].type.toString();
    }
    out += "):\n";
    printBlock(body, /*indent=*/static_cast<int>(pad.size() / 2) + 2, out);
    // No trailing newline here: printOp appends one after every op.
    out += pad + "}";
  }
};

} // namespace

std::string printModule(const Module &module) {
  return Printer(module).run(module);
}

} // namespace kernelh::ir
