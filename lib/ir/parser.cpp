// Hand-written recursive-descent parser for the atom IR (plan todo 9).
//
// Round-trip contract: the parser accepts exactly the printer's ".atom"
// output, so print(parse(print(M))) == print(M). Errors carry line/column and
// are reported through std::expected (no exceptions across the public API;
// a private ParseError exception is converted at the boundary).

#include "kernelh/ir/parser.hpp"

#include <charconv>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kernelh::ir {

namespace {

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

enum class TokKind : uint8_t { Ident, Number, ValueRef, Punct, Eof };

struct Token {
  TokKind kind = TokKind::Eof;
  std::string text;  // Ident: name; Number: raw literal text; Punct: chars
  int64_t num = 0;   // ValueRef: the N in %vN
  int line = 1;
  int col = 1;
};

// Private parse failure; converted to a message at the parseModule boundary.
class ParseError {
public:
  ParseError(int line, int col, std::string message)
      : line_(line), col_(col), message_(std::move(message)) {}

  std::string fullMessage() const {
    return "parse error at line " + std::to_string(line_) + ", col " +
           std::to_string(col_) + ": " + message_;
  }

private:
  int line_;
  int col_;
  std::string message_;
};

class Lexer {
public:
  explicit Lexer(std::string_view text) : text_(text) {}

  // Produces the next token; throws ParseError on malformed input.
  Token next() {
    skipTrivia();
    Token tok;
    tok.line = line_;
    tok.col = col_;
    const char c = peek();
    if (c == '\0') {
      tok.kind = TokKind::Eof;
      prevWasNumber_ = false;
      return tok;
    }
    if (c == '%') {
      lexValueRef(tok);
      prevWasNumber_ = false;
      return tok;
    }
    if (isDigit(c) || (c == '-' && isDigit(peekNext()))) {
      tok.kind = TokKind::Number;
      tok.text = lexNumber();
      prevWasNumber_ = true;
      return tok;
    }
    if (isAlpha(c)) {
      // The tensor-dimension separator 'x' is ambiguous with identifiers
      // (`16x16xf32`); disambiguate CONTEXTUALLY: a single 'x' directly after
      // a NUMBER is the separator punctuation. Everywhere else x/y/z lex as
      // ordinary identifiers (dims of the id intrinsics, "yield", ...).
      if (c == 'x' && prevWasNumber_) {
        advance();
        tok.kind = TokKind::Punct;
        tok.text = std::string(1, c);
        prevWasNumber_ = false;
        return tok;
      }
      tok.kind = TokKind::Ident;
      tok.text = lexIdent();
      prevWasNumber_ = false;
      return tok;
    }
    if (c == '-' && peekNext() == '>') {
      advance(); // '-'
      advance(); // '>'
      tok.kind = TokKind::Punct;
      tok.text = "->";
      prevWasNumber_ = false;
      return tok;
    }
    // Single-character punctuation.
    advance();
    tok.kind = TokKind::Punct;
    tok.text = std::string(1, c);
    prevWasNumber_ = false;
    return tok;
  }

private:
  std::string_view text_;
  size_t pos_ = 0;
  int line_ = 1;
  int col_ = 1;
  bool prevWasNumber_ = false;

  char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }
  char peekNext() const {
    return pos_ + 1 < text_.size() ? text_[pos_ + 1] : '\0';
  }
  char advance() {
    const char c = text_[pos_++];
    if (c == '\n') {
      ++line_;
      col_ = 1;
    } else {
      ++col_;
    }
    return c;
  }

  static bool isDigit(char c) { return c >= '0' && c <= '9'; }
  static bool isAlpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
  }

  void fail(int line, int col, std::string message) {
    throw ParseError(line, col, std::move(message));
  }

  void skipTrivia() {
    for (;;) {
      while (peek() == ' ' || peek() == '\t' || peek() == '\r' ||
             peek() == '\n') {
        advance();
      }
      // "//" line comments (never emitted by the printer; tolerated input).
      if (peek() == '/' && peekNext() == '/') {
        while (peek() != '\n' && peek() != '\0') {
          advance();
        }
        continue;
      }
      break;
    }
  }

  void lexValueRef(Token &tok) {
    const int line = tok.line;
    const int col = tok.col;
    advance(); // '%'
    if (peek() != 'v') {
      fail(line, col, "expected value reference '%v<N>'");
    }
    advance();
    if (!isDigit(peek())) {
      fail(line, col, "expected digits after '%v'");
    }
    int64_t num = 0;
    while (isDigit(peek())) {
      num = num * 10 + (advance() - '0');
    }
    tok.kind = TokKind::ValueRef;
    tok.num = num;
  }

  std::string lexNumber() {
    const int line = line_;
    const int col = col_;
    std::string text;
    if (peek() == '-') {
      text += advance();
    }
    while (isDigit(peek())) {
      text += advance();
    }
    if (peek() == '.') {
      text += advance();
      while (isDigit(peek())) {
        text += advance();
      }
    }
    if (peek() == 'e' || peek() == 'E') {
      text += advance();
      if (peek() == '+' || peek() == '-') {
        text += advance();
      }
      if (!isDigit(peek())) {
        fail(line, col, "invalid number literal: exponent has no digits");
      }
      while (isDigit(peek())) {
        text += advance();
      }
    }
    return text;
  }

  // Identifiers are letters followed by letters/digits (e.g. "f32", "bb0").
  // x/y/z never reach here - the lexer emits them as punctuation.
  std::string lexIdent() {
    std::string text;
    while (isAlpha(peek()) || isDigit(peek())) {
      text += advance();
    }
    return text;
  }
};

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

// A result/block-arg name together with its source position (for duplicate
// diagnostics that point at the offending token).
struct NameInfo {
  std::string name; // without the leading '%'
  int line = 1;
  int col = 1;
};

class Parser {
public:
  static std::expected<Module, std::string> parse(std::string_view text) {
    try {
      return Parser(text).parseModule();
    } catch (const ParseError &e) {
      return std::unexpected(e.fullMessage());
    }
  }

private:
  Lexer lexer_;
  Token cur_;
  std::vector<std::map<std::string, Value>> scopes_;

  explicit Parser(std::string_view text) : lexer_(text) { cur_ = lexer_.next(); }

  // -------------------------------------------------------------------------
  // Token helpers
  // -------------------------------------------------------------------------

  [[noreturn]] void fail(std::string message) const {
    throw ParseError(cur_.line, cur_.col, std::move(message));
  }
  [[noreturn]] void failAt(int line, int col, std::string message) const {
    throw ParseError(line, col, std::move(message));
  }

  void advance() { cur_ = lexer_.next(); }

  bool isPunct(char c) const {
    return cur_.kind == TokKind::Punct && cur_.text.size() == 1 &&
           cur_.text[0] == c;
  }
  bool isPunctStr(std::string_view s) const {
    return cur_.kind == TokKind::Punct && cur_.text == s;
  }
  bool isIdent(std::string_view s) const {
    return cur_.kind == TokKind::Ident && cur_.text == s;
  }

  void expectPunct(char c) {
    if (!isPunct(c)) {
      fail(std::string("expected '") + c + "'");
    }
    advance();
  }
  void expectPunctStr(std::string_view s) {
    if (!isPunctStr(s)) {
      fail(std::string("expected '") + std::string(s) + "'");
    }
    advance();
  }
  std::string expectIdent() {
    if (cur_.kind != TokKind::Ident) {
      fail("expected identifier");
    }
    const std::string text = cur_.text;
    advance();
    return text;
  }
  void expectIdent(std::string_view want) {
    if (!isIdent(want)) {
      fail("expected '" + std::string(want) + "'");
    }
    advance();
  }

  Token expectNumber() {
    if (cur_.kind != TokKind::Number) {
      fail("expected number literal");
    }
    const Token tok = cur_;
    advance();
    return tok;
  }

  int64_t expectInt() {
    const Token tok = expectNumber();
    int64_t value = 0;
    const std::from_chars_result res =
        std::from_chars(tok.text.data(), tok.text.data() + tok.text.size(),
                        value);
    if (res.ec != std::errc() || res.ptr != tok.text.data() + tok.text.size()) {
      failAt(tok.line, tok.col,
             "invalid integer literal '" + tok.text + "'");
    }
    return value;
  }

  NameInfo expectValueName() {
    if (cur_.kind != TokKind::ValueRef) {
      fail("expected value reference '%v<N>'");
    }
    NameInfo info;
    info.name = "v" + std::to_string(cur_.num);
    info.line = cur_.line;
    info.col = cur_.col;
    advance();
    return info;
  }

  Value parseValueRef() {
    if (cur_.kind != TokKind::ValueRef) {
      fail("expected value reference '%v<N>'");
    }
    const std::string name = "v" + std::to_string(cur_.num);
    const int line = cur_.line;
    const int col = cur_.col;
    advance();
    // Lookup innermost scope first (structured SSA: defs dominate uses).
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      const auto found = it->find(name);
      if (found != it->end()) {
        return found->second;
      }
    }
    failAt(line, col,
           "use of undefined value '%" + name +
               "' (SSA dominance violation: the value is not defined in an "
               "enclosing scope before this use)");
  }

  // -------------------------------------------------------------------------
  // Scope management
  // -------------------------------------------------------------------------

  void pushScope() { scopes_.emplace_back(); }
  void popScope() { scopes_.pop_back(); }

  void registerName(const NameInfo &info, Value value) {
    auto &scope = scopes_.back();
    if (scope.count(info.name) != 0) {
      failAt(info.line, info.col,
             "duplicate value name '%" + info.name + "'");
    }
    scope[info.name] = value;
  }

  // -------------------------------------------------------------------------
  // Types
  // -------------------------------------------------------------------------

  Type parseType() {
    if (cur_.kind != TokKind::Ident) {
      fail("expected type");
    }
    const std::string first = cur_.text;
    if (first == "tensor") {
      return parseTensorType();
    }
    if (first == "vector") {
      return parseVectorType();
    }
    ScalarType scalar;
    if (!parseScalarType(first, scalar)) {
      fail("unknown type '" + first + "'");
    }
    advance();
    return Type::scalar(scalar);
  }

  Type parseTensorType() {
    expectIdent("tensor");
    expectPunct('<');
    std::vector<int64_t> shape;
    shape.push_back(expectInt());
    // `16x16xf32`: each 'x' that is followed by a NUMBER starts another dim;
    // the final 'x' (followed by the element type) is left consumed so the
    // element-type parse below picks up exactly where it stopped.
    while (isPunct('x') || isIdent("x")) {
      advance();
      if (cur_.kind == TokKind::Number) {
        shape.push_back(expectInt());
      } else {
        break;
      }
    }
    if (cur_.kind != TokKind::Ident) {
      fail("expected element type in tensor type");
    }
    ScalarType elem;
    if (!parseScalarType(cur_.text, elem)) {
      fail("unknown element type '" + cur_.text + "'");
    }
    advance();
    expectPunct(',');
    const std::string spaceText = expectIdent();
    MemSpace space;
    if (!parseMemSpace(spaceText, space)) {
      fail("unknown memory space '" + spaceText + "'");
    }
    expectPunct('>');
    return Type::tensor(TensorType{shape, elem, space});
  }

  Type parseVectorType() {
    expectIdent("vector");
    expectPunct('<');
    const int64_t width = expectInt();
    if (!isPunct('x') && !isIdent("x")) {
      fail("expected 'x' between vector width and element type");
    }
    advance();
    if (cur_.kind != TokKind::Ident) {
      fail("expected element type in vector type");
    }
    ScalarType elem;
    if (!parseScalarType(cur_.text, elem)) {
      fail("unknown element type '" + cur_.text + "'");
    }
    advance();
    expectPunct('>');
    return Type::vector(VectorType{width, elem});
  }

  // -------------------------------------------------------------------------
  // Module / kernel structure
  // -------------------------------------------------------------------------

  Module parseModule() {
    Module module;
    while (cur_.kind != TokKind::Eof) {
      module.kernels.push_back(parseKernel());
    }
    return module;
  }

  std::unique_ptr<Kernel> parseKernel() {
    expectIdent("atom");
    expectPunct('.');
    expectIdent("kernel");
    expectPunct('@');
    const std::string name = expectIdent();

    auto kernel = std::make_unique<Kernel>(name);
    Block &body = kernel->body;

    // Kernel parameters = body block args.
    std::vector<NameInfo> argNames;
    if (isPunct('(')) {
      advance();
      while (!isPunct(')')) {
        NameInfo argName = expectValueName();
        expectPunct(':');
        body.args.push_back(BlockArg{parseType()});
        argNames.push_back(std::move(argName));
        if (isPunct(',')) {
          advance();
        } else {
          break;
        }
      }
      expectPunct(')');
    }

    expectPunct('{');
    pushScope();
    for (size_t i = 0; i < argNames.size(); ++i) {
      registerName(argNames[i], Value::blockArg(&body, static_cast<int64_t>(i)));
    }
    parseAttrs(*kernel);
    parseOps(body); // consumes the closing '}'
    popScope();
    return kernel;
  }

  void parseAttrs(Kernel &kernel) {
    while (cur_.kind == TokKind::Ident && cur_.text != "atom") {
      parseAttr(kernel);
    }
  }

  void parseAttr(Kernel &kernel) {
    const std::string name = expectIdent();
    expectPunct('=');
    if (name == "tile") {
      expectPunct('[');
      while (!isPunct(']')) {
        kernel.schedule.tile.push_back(expectInt());
        if (isPunct(',')) {
          advance();
        } else {
          break;
        }
      }
      expectPunct(']');
    } else if (name == "map_blocks") {
      kernel.schedule.map_blocks = expectBool();
    } else if (name == "map_warps") {
      kernel.schedule.map_warps = expectBool();
    } else if (name == "use_shared_memory") {
      kernel.schedule.use_shared_memory = expectBool();
    } else if (name == "vector_width") {
      kernel.schedule.vector_width = expectInt();
    } else if (name == "unroll") {
      kernel.schedule.unroll = expectInt();
    } else if (name == "shape") {
      expectPunct('{');
      while (!isPunct('}')) {
        const std::string key = expectIdent();
        expectPunct('=');
        kernel.shape[key] = expectInt();
        if (isPunct(',')) {
          advance();
        } else {
          break;
        }
      }
      expectPunct('}');
    } else {
      fail("unknown kernel attribute '" + name + "'");
    }
  }

  bool expectBool() {
    if (cur_.kind != TokKind::Ident) {
      fail("expected 'true' or 'false'");
    }
    if (cur_.text == "true") {
      advance();
      return true;
    }
    if (cur_.text == "false") {
      advance();
      return false;
    }
    fail("expected 'true' or 'false'");
  }

  // Parses ops until and including the closing '}'.
  void parseOps(Block &block) {
    while (!isPunct('}')) {
      if (cur_.kind == TokKind::Eof) {
        fail("unexpected end of input: expected '}'");
      }
      parseOp(block);
    }
    expectPunct('}');
  }

  // Block header: ^bb0(%vN: type, ...):
  void parseBlockHeader(Block &block) {
    expectPunct('^');
    expectIdent(); // block label ("bb0") - the label is not significant
    expectPunct('(');
    int64_t index = 0;
    while (!isPunct(')')) {
      NameInfo name = expectValueName();
      expectPunct(':');
      block.args.push_back(BlockArg{parseType()});
      registerName(name, Value::blockArg(&block, index));
      ++index;
      if (isPunct(',')) {
        advance();
      } else {
        break;
      }
    }
    expectPunct(')');
    expectPunct(':');
  }

  // -------------------------------------------------------------------------
  // Ops
  // -------------------------------------------------------------------------

  void parseOp(Block &block) {
    std::vector<NameInfo> resultNames;
    if (cur_.kind == TokKind::ValueRef) {
      resultNames.push_back(expectValueName());
      while (isPunct(',')) {
        advance();
        resultNames.push_back(expectValueName());
      }
      expectPunct('=');
    }

    expectIdent("atom");
    expectPunct('.');
    const std::string opname = expectIdent();

    if (opname == "const") {
      parseConst(block, resultNames);
    } else if (opname == "add" || opname == "sub" || opname == "mul" ||
               opname == "div" || opname == "max" || opname == "min") {
      parseBin(block, opname, resultNames);
    } else if (opname == "fma") {
      parseFma(block, resultNames);
    } else if (opname == "cmp") {
      parseCmp(block, resultNames);
    } else if (opname == "select") {
      parseSelect(block, resultNames);
    } else if (opname == "math") {
      parseMath(block, resultNames);
    } else if (opname == "load") {
      parseLoad(block, resultNames);
    } else if (opname == "store") {
      parseStore(block, resultNames);
    } else if (opname == "load_vec") {
      parseLoadVec(block, resultNames);
    } else if (opname == "store_vec") {
      parseStoreVec(block, resultNames);
    } else if (opname == "alloc_shared") {
      parseAllocShared(block, resultNames);
    } else if (opname == "barrier") {
      parseBarrier(block, resultNames);
    } else if (opname == "thread_id" || opname == "block_id" ||
               opname == "block_dim" || opname == "grid_dim") {
      parseIdOp(block, opname, resultNames);
    } else if (opname == "loop_for") {
      parseLoopFor(block, resultNames);
    } else if (opname == "yield") {
      parseYield(block, resultNames);
    } else if (opname == "reduce") {
      parseReduce(block, resultNames);
    } else if (opname == "copy") {
      parseCopy(block, resultNames);
    } else {
      fail("unknown atom operation 'atom." + opname + "'");
    }
  }

  // Registers `count` results of `op` from the captured names and validates
  // the expected result count.
  void finishOp(Op *op, const std::vector<NameInfo> &names, size_t expected,
                const char *what) {
    if (names.size() != expected) {
      fail(std::string(what) + " expects " + std::to_string(expected) +
           " result(s), got " + std::to_string(names.size()));
    }
    for (size_t i = 0; i < names.size(); ++i) {
      registerName(names[i], op->result(static_cast<int64_t>(i)));
    }
  }

  std::vector<Value> parseIndexList() {
    expectPunct('[');
    std::vector<Value> indices;
    while (!isPunct(']')) {
      indices.push_back(parseValueRef());
      if (isPunct(',')) {
        advance();
      } else {
        break;
      }
    }
    expectPunct(']');
    return indices;
  }

  void parseConst(Block &block, const std::vector<NameInfo> &resultNames) {
    const Token lit = expectNumber();
    expectPunct(':');
    const Type type = parseType();
    if (type.isVector()) {
      fail("atom.const result cannot be a vector type");
    }
    const ScalarType elem =
        type.isTensor() ? type.asTensor().elem : type.asScalar();
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Const;
    op->resultTypes.push_back(type);
    if (isFloatType(elem)) {
      double value = 0.0;
      const std::from_chars_result res =
          std::from_chars(lit.text.data(),
                          lit.text.data() + lit.text.size(), value);
      if (res.ec != std::errc() ||
          res.ptr != lit.text.data() + lit.text.size()) {
        failAt(lit.line, lit.col,
               "invalid floating-point literal '" + lit.text + "'");
      }
      op->floatValue = value;
    } else {
      int64_t value = 0;
      const std::from_chars_result res =
          std::from_chars(lit.text.data(),
                          lit.text.data() + lit.text.size(), value);
      if (res.ec != std::errc() ||
          res.ptr != lit.text.data() + lit.text.size()) {
        failAt(lit.line, lit.col,
               "invalid integer literal '" + lit.text + "'");
      }
      op->intValue = value;
    }
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.const");
  }

  static OpKind binKind(std::string_view opname) {
    if (opname == "add") {
      return OpKind::Add;
    }
    if (opname == "sub") {
      return OpKind::Sub;
    }
    if (opname == "mul") {
      return OpKind::Mul;
    }
    if (opname == "div") {
      return OpKind::Div;
    }
    if (opname == "max") {
      return OpKind::Max;
    }
    return OpKind::Min;
  }

  void parseBin(Block &block, std::string_view opname,
                const std::vector<NameInfo> &resultNames) {
    const Value a = parseValueRef();
    expectPunct(',');
    const Value b = parseValueRef();
    expectPunct(':');
    const Type type = parseType();
    auto op = std::make_unique<Op>();
    op->kind = binKind(opname);
    op->operands = {a, b};
    op->resultTypes.push_back(type);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, ("atom." + std::string(opname)).c_str());
  }

  void parseFma(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value a = parseValueRef();
    expectPunct(',');
    const Value b = parseValueRef();
    expectPunct(',');
    const Value c = parseValueRef();
    expectPunct(':');
    const Type type = parseType();
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Fma;
    op->operands = {a, b, c};
    op->resultTypes.push_back(type);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.fma");
  }

  void parseCmp(Block &block, const std::vector<NameInfo> &resultNames) {
    const std::string predText = expectIdent();
    CmpPred pred;
    if (!parseCmpPred(predText, pred)) {
      fail("unknown comparison predicate '" + predText + "'");
    }
    const Value a = parseValueRef();
    expectPunct(',');
    const Value b = parseValueRef();
    expectPunct(':');
    const Type operandType = parseType(); // printed operand type
    (void)operandType;
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Cmp;
    op->pred = pred;
    op->operands = {a, b};
    // i32 is the documented boolean carrier of this IR.
    op->resultTypes.push_back(Type::scalar(ScalarType::I32));
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.cmp");
  }

  void parseSelect(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value cond = parseValueRef();
    expectPunct(',');
    const Value a = parseValueRef();
    expectPunct(',');
    const Value b = parseValueRef();
    expectPunct(':');
    const Type type = parseType();
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Select;
    op->operands = {cond, a, b};
    op->resultTypes.push_back(type);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.select");
  }

  void parseMath(Block &block, const std::vector<NameInfo> &resultNames) {
    const std::string kindText = expectIdent();
    MathKind mathKind;
    if (!parseMathKind(kindText, mathKind)) {
      fail("unknown math function '" + kindText + "'");
    }
    const Value a = parseValueRef();
    expectPunct(':');
    const Type type = parseType();
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Math;
    op->mathKind = mathKind;
    op->operands = {a};
    op->resultTypes.push_back(type);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.math");
  }

  void parseLoad(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value tensor = parseValueRef();
    const std::vector<Value> indices = parseIndexList();
    expectPunct(':');
    const Type tensorType = parseType();
    if (!tensorType.isTensor()) {
      fail("atom.load requires a tensor type");
    }
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Load;
    op->operands.push_back(tensor);
    for (Value i : indices) {
      op->operands.push_back(i);
    }
    op->resultTypes.push_back(
        Type::scalar(tensorType.asTensor().elem));
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.load");
  }

  void parseStore(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value value = parseValueRef();
    expectPunct(',');
    const Value tensor = parseValueRef();
    const std::vector<Value> indices = parseIndexList();
    expectPunct(':');
    const Type tensorType = parseType();
    if (!tensorType.isTensor()) {
      fail("atom.store requires a tensor type");
    }
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Store;
    op->operands.push_back(value);
    op->operands.push_back(tensor);
    for (Value i : indices) {
      op->operands.push_back(i);
    }
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 0, "atom.store");
  }

  void parseLoadVec(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value tensor = parseValueRef();
    const std::vector<Value> indices = parseIndexList();
    expectPunct(':');
    const Type tensorType = parseType();
    if (!tensorType.isTensor()) {
      fail("atom.load_vec requires a tensor type");
    }
    expectPunct(',');
    const Type vectorType = parseType();
    if (!vectorType.isVector()) {
      fail("atom.load_vec requires a vector<WxE> result type");
    }
    auto op = std::make_unique<Op>();
    op->kind = OpKind::LoadVec;
    op->operands.push_back(tensor);
    for (Value i : indices) {
      op->operands.push_back(i);
    }
    op->resultTypes.push_back(vectorType);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.load_vec");
  }

  void parseStoreVec(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value vec = parseValueRef();
    expectPunct(',');
    const Value tensor = parseValueRef();
    const std::vector<Value> indices = parseIndexList();
    expectPunct(':');
    const Type tensorType = parseType();
    if (!tensorType.isTensor()) {
      fail("atom.store_vec requires a tensor type");
    }
    expectPunct(',');
    const Type vectorType = parseType();
    if (!vectorType.isVector()) {
      fail("atom.store_vec requires a vector<WxE> operand type");
    }
    auto op = std::make_unique<Op>();
    op->kind = OpKind::StoreVec;
    op->operands.push_back(vec);
    op->operands.push_back(tensor);
    for (Value i : indices) {
      op->operands.push_back(i);
    }
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 0, "atom.store_vec");
  }

  void parseAllocShared(Block &block,
                        const std::vector<NameInfo> &resultNames) {
    expectPunct(':');
    const Type type = parseType();
    auto op = std::make_unique<Op>();
    op->kind = OpKind::AllocShared;
    op->resultTypes.push_back(type);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.alloc_shared");
  }

  void parseBarrier(Block &block, const std::vector<NameInfo> &resultNames) {
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Barrier;
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 0, "atom.barrier");
  }

  static OpKind idKind(std::string_view opname) {
    if (opname == "thread_id") {
      return OpKind::ThreadId;
    }
    if (opname == "block_id") {
      return OpKind::BlockId;
    }
    if (opname == "block_dim") {
      return OpKind::BlockDim;
    }
    return OpKind::GridDim;
  }

  IdDim expectIdDim() {
    if (isPunct('x') || isIdent("x")) {
      advance();
      return IdDim::X;
    }
    if (isIdent("y")) {
      advance();
      return IdDim::Y;
    }
    if (isIdent("z")) {
      advance();
      return IdDim::Z;
    }
    fail("expected dimension 'x', 'y', or 'z'");
  }

  void parseIdOp(Block &block, std::string_view opname,
                 const std::vector<NameInfo> &resultNames) {
    const IdDim dim = expectIdDim();
    expectPunct(':');
    const Type type = parseType();
    auto op = std::make_unique<Op>();
    op->kind = idKind(opname);
    op->idDim = dim;
    op->resultTypes.push_back(type);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, ("atom." + std::string(opname)).c_str());
  }

  void parseLoopFor(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value lb = parseValueRef();
    expectIdent("to");
    const Value ub = parseValueRef();
    expectIdent("step");
    const Value step = parseValueRef();
    expectIdent("iter_args");
    expectPunct('(');
    std::vector<Value> iterArgs;
    while (!isPunct(')')) {
      iterArgs.push_back(parseValueRef());
      if (isPunct(',')) {
        advance();
      } else {
        break;
      }
    }
    expectPunct(')');
    expectPunctStr("->");
    expectPunct('(');
    std::vector<Type> resultTypes;
    while (!isPunct(')')) {
      resultTypes.push_back(parseType());
      if (isPunct(',')) {
        advance();
      } else {
        break;
      }
    }
    expectPunct(')');
    expectPunct('{');

    if (resultNames.size() != resultTypes.size()) {
      fail("atom.loop_for result count (" + std::to_string(resultNames.size()) +
           ") does not match iter_args count (" +
           std::to_string(resultTypes.size()) + ")");
    }

    auto op = std::make_unique<Op>();
    op->kind = OpKind::LoopFor;
    op->operands.push_back(lb);
    op->operands.push_back(ub);
    op->operands.push_back(step);
    for (Value arg : iterArgs) {
      op->operands.push_back(arg);
    }
    op->resultTypes = std::move(resultTypes);

    auto body = std::make_unique<Block>();
    body->parentBlock = &block;
    body->parentOp = op.get();
    Op *raw = op.get();
    block.ops.push_back(std::move(op));

    pushScope();
    parseBlockHeader(*body);
    parseOps(*body); // consumes '}'
    popScope();
    raw->region = std::move(body);

    for (size_t i = 0; i < resultNames.size(); ++i) {
      registerName(resultNames[i], raw->result(static_cast<int64_t>(i)));
    }
  }

  void parseYield(Block &block, const std::vector<NameInfo> &resultNames) {
    std::vector<Value> values;
    while (cur_.kind == TokKind::ValueRef) {
      values.push_back(parseValueRef());
      if (isPunct(',')) {
        advance();
      } else {
        break;
      }
    }
    if (isPunct(':')) {
      advance();
      for (size_t i = 0; i < values.size(); ++i) {
        const Type printed = parseType();
        if (printed != values[i].type()) {
          fail("atom.yield type mismatch: expected " +
               values[i].type().toString() + ", got " + printed.toString());
        }
        if (i + 1 < values.size()) {
          expectPunct(',');
        }
      }
    }
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Yield;
    op->operands = std::move(values);
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 0, "atom.yield");
  }

  void parseReduce(Block &block, const std::vector<NameInfo> &resultNames) {
    const std::string kindText = expectIdent();
    ReduceKind reduceKind;
    if (!parseReduceKind(kindText, reduceKind)) {
      fail("unknown reduction kind '" + kindText + "'");
    }
    const Value tensor = parseValueRef();
    expectPunct(',');
    expectIdent("dim");
    expectPunct('=');
    const int64_t dim = expectInt();
    expectPunct(':');
    const Type tensorType = parseType();
    if (!tensorType.isTensor()) {
      fail("atom.reduce requires a tensor type");
    }
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Reduce;
    op->reduceKind = reduceKind;
    op->reduceDim = dim;
    op->operands.push_back(tensor);
    op->resultTypes.push_back(Type::scalar(tensorType.asTensor().elem));
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 1, "atom.reduce");
  }

  void parseCopy(Block &block, const std::vector<NameInfo> &resultNames) {
    const Value src = parseValueRef();
    expectPunct(',');
    const Value dst = parseValueRef();
    expectPunct(':');
    const Type srcType = parseType();
    if (!srcType.isTensor()) {
      fail("atom.copy requires tensor types");
    }
    expectIdent("to");
    const Type dstType = parseType();
    if (!dstType.isTensor()) {
      fail("atom.copy requires tensor types");
    }
    auto op = std::make_unique<Op>();
    op->kind = OpKind::Copy;
    op->operands = {src, dst};
    Op *raw = op.get();
    block.ops.push_back(std::move(op));
    finishOp(raw, resultNames, 0, "atom.copy");
  }
};

} // namespace

std::expected<Module, std::string> parseModule(std::string_view text) {
  return Parser::parse(text);
}

} // namespace kernelh::ir
