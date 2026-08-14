// kernel.h compiler - autotuner search logic (plan todo 34). See
// include/kernelh/compiler/autotuner.hpp for the API contract.
//
// This TU is deliberately MLIR-free / GPU-free / dependency-free (it only
// includes the kernelh_compiler headers): candidate generation, pruning
// policy, refinement, the JSON cache, and the CSV writer all live here so the
// search is unit-testable with a fake evaluator and kernelh_compiler keeps the
// no-MLIR weak-default link contract (todo 23).
//
// Cache format (build/tune-cache.json, hand-rolled JSON - the project has no
// JSON library):
//
//   {
//     "version": 1,
//     "entries": {
//       "<sha256(kernel+shape+device_name)>": {
//         "best_index": 3,
//         "evaluated": [
//           {"bm":64,"bn":128,"bk":32,"waves":4,"unroll":1,"vec":2,
//            "ms_median":1.234,"accepted":true,"reason":"ok"},
//           ...
//         ]
//       }
//     }
//   }

#include "kernelh/compiler/autotuner.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <ostream>
#include <random>
#include <sstream>
#include <utility>

namespace kernelh::compiler {

namespace {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), self-contained. Produces the hex digest used for the
// cache key. ~70 lines; keeps kernelh_compiler free of any external hash lib.
// ---------------------------------------------------------------------------
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr u32 rotr(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha256 {
  u32 h[8];
  u64 bitLen = 0;
  std::array<unsigned char, 64> buf{};
  size_t bufLen = 0;

  Sha256() {
    h[0] = 0x6a09e667;
    h[1] = 0xbb67ae85;
    h[2] = 0x3c6ef372;
    h[3] = 0xa54ff53a;
    h[4] = 0x510e527f;
    h[5] = 0x9b05688c;
    h[6] = 0x1f83d9ab;
    h[7] = 0x5be0cd19;
  }

  void block(const unsigned char *p) {
    u32 w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<u32>(p[i * 4]) << 24) |
             (static_cast<u32>(p[i * 4 + 1]) << 16) |
             (static_cast<u32>(p[i * 4 + 2]) << 8) | static_cast<u32>(p[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    u32 a = h[0], b = h[1], c = h[2], d = h[3];
    u32 e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const u32 s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const u32 ch = (e & f) ^ (~e & g);
      const u32 t1 = hh + s1 + ch + kSha256K[i] + w[i];
      const u32 s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const u32 maj = (a & b) ^ (a & c) ^ (b & c);
      const u32 t2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  void update(const void *data, size_t len) {
    const auto *p = static_cast<const unsigned char *>(data);
    bitLen += static_cast<u64>(len) * 8;
    while (len > 0) {
      const size_t take = std::min(len, 64 - bufLen);
      std::copy(p, p + take, buf.begin() + static_cast<std::ptrdiff_t>(bufLen));
      bufLen += take;
      p += take;
      len -= take;
      if (bufLen == 64) {
        block(buf.data());
        bufLen = 0;
      }
    }
  }

  std::string finalHex() {
    // Padding: 0x80, then enough zeros to leave room for the 64-bit big-endian
    // bit length at the end of the final 64-byte block.
    const u64 bitLenCopy = bitLen;
    const unsigned char pad = 0x80;
    update(&pad, 1);
    std::array<unsigned char, 8> lenBytes{};
    for (int i = 0; i < 8; ++i) {
      lenBytes[static_cast<size_t>(7 - i)] =
          static_cast<unsigned char>(bitLenCopy >> (i * 8));
    }
    const size_t zerosNeeded = (64 - bufLen - 8) % 64;
    std::array<unsigned char, 64> zeros{};
    update(zeros.data(), zerosNeeded);
    update(lenBytes.data(), 8);
    std::string out;
    out.reserve(64);
    static const char *kHex = "0123456789abcdef";
    for (u32 v : h) {
      for (int i = 28; i >= 0; i -= 4) {
        out += kHex[(v >> i) & 0xf];
      }
    }
    return out;
  }
};

// ---------------------------------------------------------------------------
// Minimal JSON value tree (read side) + writer. The project has no JSON
// library (todo-33 rule); this is the second, cache-specific reader/writer.
// ---------------------------------------------------------------------------
struct JsonValue {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<std::unique_ptr<JsonValue>> array;
  std::vector<std::pair<std::string, std::unique_ptr<JsonValue>>> object;

  const JsonValue *find(const std::string &key) const {
    if (kind != Kind::Object) {
      return nullptr;
    }
    for (const auto &[k, v] : object) {
      if (k == key) {
        return v.get();
      }
    }
    return nullptr;
  }
};

class JsonReader {
public:
  explicit JsonReader(std::string_view text) : text_(text) {}

  bool parse(JsonValue &out) {
    skipWs();
    if (!parseValue(out)) {
      return false;
    }
    skipWs();
    return pos_ == text_.size(); // trailing garbage -> malformed
  }

  const std::string &error() const { return error_; }

private:
  std::string_view text_;
  size_t pos_ = 0;
  std::string error_;

  bool fail(const std::string &msg) {
    if (error_.empty()) {
      error_ = msg + " at offset " + std::to_string(pos_);
    }
    return false;
  }

  void skipWs() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool parseValue(JsonValue &out) {
    skipWs();
    if (pos_ >= text_.size()) {
      return fail("unexpected end of input");
    }
    const char c = text_[pos_];
    if (c == '{') {
      return parseObject(out);
    }
    if (c == '[') {
      return parseArray(out);
    }
    if (c == '"') {
      std::string s;
      if (!parseString(s)) {
        return false;
      }
      out.kind = JsonValue::Kind::String;
      out.str = std::move(s);
      return true;
    }
    if (c == 't' || c == 'f') {
      if (text_.substr(pos_, 4) == "true") {
        pos_ += 4;
        out.kind = JsonValue::Kind::Bool;
        out.boolean = true;
        return true;
      }
      if (text_.substr(pos_, 5) == "false") {
        pos_ += 5;
        out.kind = JsonValue::Kind::Bool;
        out.boolean = false;
        return true;
      }
      return fail("malformed literal");
    }
    if (c == 'n') {
      if (text_.substr(pos_, 4) == "null") {
        pos_ += 4;
        out.kind = JsonValue::Kind::Null;
        return true;
      }
      return fail("malformed literal");
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      return parseNumber(out);
    }
    return fail("unexpected character");
  }

  bool parseString(std::string &out) {
    if (pos_ >= text_.size() || text_[pos_] != '"') {
      return fail("expected string");
    }
    ++pos_;
    out.clear();
    while (pos_ < text_.size()) {
      const unsigned char c = static_cast<unsigned char>(text_[pos_]);
      if (c == '"') {
        ++pos_;
        return true;
      }
      if (c == '\\') {
        ++pos_;
        if (pos_ >= text_.size()) {
          return fail("truncated escape");
        }
        const char e = text_[pos_];
        switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u':
          // Keep it simple: parse 4 hex digits as a code point (>= U+0080 is
          // encoded UTF-8); the cache only carries ASCII, so anything else is
          // a tolerated best-effort.
          if (pos_ + 4 >= text_.size()) {
            return fail("truncated \\u escape");
          }
          {
            std::uint32_t cp = 0;
            for (int i = 1; i <= 4; ++i) {
              const char hx = text_[pos_ + i];
              cp = static_cast<std::uint32_t>(cp * 16) +
                   static_cast<std::uint32_t>(hx >= '0' && hx <= '9'
                                                  ? hx - '0'
                                                  : (hx >= 'a' && hx <= 'f'
                                                         ? hx - 'a' + 10
                                                         : (hx >= 'A' && hx <= 'F'
                                                                ? hx - 'A' + 10
                                                                : 16)));
              if (hx >= '0' && hx <= '9') {
              } else if (hx >= 'a' && hx <= 'f') {
              } else if (hx >= 'A' && hx <= 'F') {
              } else {
                return fail("bad \\u hex");
              }
            }
            pos_ += 4;
            if (cp < 0x80) {
              out += static_cast<char>(cp);
            } else if (cp < 0x800) {
              out += static_cast<char>(0xC0 | (cp >> 6));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (cp >> 12));
              out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            }
          }
          break;
        default:
          return fail("unknown escape");
        }
        ++pos_;
        continue;
      }
      out += static_cast<char>(c);
      ++pos_;
    }
    return fail("unterminated string");
  }

  bool parseNumber(JsonValue &out) {
    const size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ == start) {
      return fail("malformed number");
    }
    const std::string num(text_.substr(start, pos_ - start));
    char *end = nullptr;
    const double v = std::strtod(num.c_str(), &end);
    if (end != num.c_str() + num.size()) {
      return fail("malformed number");
    }
    out.kind = JsonValue::Kind::Number;
    out.number = v;
    return true;
  }

  bool parseArray(JsonValue &out) {
    ++pos_; // '['
    out.kind = JsonValue::Kind::Array;
    skipWs();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (true) {
      skipWs();
      JsonValue item;
      if (!parseValue(item)) {
        return false;
      }
      out.array.push_back(std::make_unique<JsonValue>(std::move(item)));
      skipWs();
      if (pos_ >= text_.size()) {
        return fail("unterminated array");
      }
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return fail("expected ',' or ']'");
    }
  }

  bool parseObject(JsonValue &out) {
    ++pos_; // '{'
    out.kind = JsonValue::Kind::Object;
    skipWs();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (true) {
      skipWs();
      std::string key;
      if (!parseString(key)) {
        return false;
      }
      skipWs();
      if (pos_ >= text_.size() || text_[pos_] != ':') {
        return fail("expected ':'");
      }
      ++pos_;
      JsonValue value;
      if (!parseValue(value)) {
        return false;
      }
      out.object.emplace_back(std::move(key),
                              std::make_unique<JsonValue>(std::move(value)));
      skipWs();
      if (pos_ >= text_.size()) {
        return fail("unterminated object");
      }
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return fail("expected ',' or '}'");
    }
  }
};

void jsonWriteString(std::ostream &os, const std::string &s) {
  os << '"';
  for (const unsigned char c : s) {
    switch (c) {
    case '"': os << "\\\""; break;
    case '\\': os << "\\\\"; break;
    case '\n': os << "\\n"; break;
    case '\r': os << "\\r"; break;
    case '\t': os << "\\t"; break;
    default:
      if (c < 0x20) {
        os << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(c) << std::dec << std::setfill(' ');
      } else {
        os << static_cast<char>(c);
      }
    }
  }
  os << '"';
}

std::string jsonNumber(double v) {
  if (std::isfinite(v)) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
  }
  return v > 0 ? "1e308" : "-1e308"; // JSON has no NaN/Inf; saturate
}

// ---------------------------------------------------------------------------
// Cache schema (see the file header).
// ---------------------------------------------------------------------------
struct CacheEntry {
  int best_index = -1;
  std::vector<EvaluatedCandidate> evaluated;
};
using TuneCache = std::map<std::string, CacheEntry>;

JsonValue candidateToJson(const EvaluatedCandidate &e) {
  JsonValue o;
  o.kind = JsonValue::Kind::Object;
  const auto addInt = [&](const char *k, long long v) {
    JsonValue num;
    num.kind = JsonValue::Kind::Number;
    num.number = static_cast<double>(v);
    o.object.emplace_back(k, std::make_unique<JsonValue>(std::move(num)));
  };
  const auto addStr = [&](const char *k, const std::string &v) {
    JsonValue s;
    s.kind = JsonValue::Kind::String;
    s.str = v;
    o.object.emplace_back(k, std::make_unique<JsonValue>(std::move(s)));
  };
  const auto addBool = [&](const char *k, bool v) {
    JsonValue b;
    b.kind = JsonValue::Kind::Bool;
    b.boolean = v;
    o.object.emplace_back(k, std::make_unique<JsonValue>(std::move(b)));
  };
  addInt("bm", e.config.bm);
  addInt("bn", e.config.bn);
  addInt("bk", e.config.bk);
  addInt("waves", e.config.waves);
  addInt("unroll", e.config.unroll);
  addInt("vec", e.config.vec);
  JsonValue ms;
  ms.kind = JsonValue::Kind::Number;
  ms.number = e.ms_median;
  o.object.emplace_back("ms_median", std::make_unique<JsonValue>(std::move(ms)));
  addBool("accepted", e.accepted);
  addStr("reason", e.reason);
  return o;
}

bool candidateFromJson(const JsonValue &o, EvaluatedCandidate &out) {
  if (o.kind != JsonValue::Kind::Object) {
    return false;
  }
  const auto intField = [&](const char *k) -> long long {
    const JsonValue *v = o.find(k);
    if (v == nullptr || v->kind != JsonValue::Kind::Number) {
      return 0;
    }
    return static_cast<long long>(std::llround(v->number));
  };
  const auto strField = [&](const char *k) -> std::string {
    const JsonValue *v = o.find(k);
    return (v != nullptr && v->kind == JsonValue::Kind::String) ? v->str : "";
  };
  const JsonValue *ms = o.find("ms_median");
  const JsonValue *accepted = o.find("accepted");
  out.config.bm = static_cast<int>(intField("bm"));
  out.config.bn = static_cast<int>(intField("bn"));
  out.config.bk = static_cast<int>(intField("bk"));
  out.config.waves = static_cast<int>(intField("waves"));
  out.config.unroll = static_cast<int>(intField("unroll"));
  out.config.vec = static_cast<int>(intField("vec"));
  out.ms_median =
      (ms != nullptr && ms->kind == JsonValue::Kind::Number) ? ms->number : 0.0;
  out.accepted = (accepted != nullptr && accepted->kind == JsonValue::Kind::Bool)
                     ? accepted->boolean
                     : false;
  out.reason = strField("reason");
  return true;
}

// Reads the whole cache file; an absent/unreadable/malformed file yields an
// EMPTY cache (the first run just rebuilds it). The file's "entries" object
// maps cache keys to CacheEntry payloads.
TuneCache readCacheFile(const std::filesystem::path &path) {
  TuneCache cache;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return cache;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  JsonValue root;
  JsonReader reader(text); // text must outlive the view
  if (!reader.parse(root) || root.kind != JsonValue::Kind::Object) {
    return cache;
  }
  const JsonValue *entries = root.find("entries");
  if (entries == nullptr || entries->kind != JsonValue::Kind::Object) {
    return cache;
  }
  for (const auto &[key, entryValue] : entries->object) {
    if (entryValue->kind != JsonValue::Kind::Object) {
      continue;
    }
    CacheEntry entry;
    const JsonValue *best = entryValue->find("best_index");
    if (best != nullptr && best->kind == JsonValue::Kind::Number) {
      entry.best_index = static_cast<int>(std::llround(best->number));
    }
    const JsonValue *evaluated = entryValue->find("evaluated");
    if (evaluated != nullptr && evaluated->kind == JsonValue::Kind::Array) {
      for (const std::unique_ptr<JsonValue> &item : evaluated->array) {
        EvaluatedCandidate cand;
        if (candidateFromJson(*item, cand)) {
          entry.evaluated.push_back(std::move(cand));
        }
      }
    }
    cache.emplace(key, std::move(entry));
  }
  return cache;
}

void writeCacheFile(const std::filesystem::path &path, const TuneCache &cache) {
  if (path.empty()) {
    return;
  }
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return;
  }
  out << "{\n  \"version\": 1,\n  \"entries\": {\n";
  bool firstKey = true;
  for (const auto &[key, entry] : cache) {
    if (!firstKey) {
      out << ",\n";
    }
    firstKey = false;
    out << "    ";
    jsonWriteString(out, key);
    out << ": {\n      \"best_index\": " << entry.best_index
        << ",\n      \"evaluated\": [\n";
    bool firstCand = true;
    for (const EvaluatedCandidate &cand : entry.evaluated) {
      if (!firstCand) {
        out << ",\n";
      }
      firstCand = false;
      const JsonValue o = candidateToJson(cand);
      out << "        {";
      bool firstField = true;
      for (const auto &[k, v] : o.object) {
        if (!firstField) {
          out << ", ";
        }
        firstField = false;
        jsonWriteString(out, k);
        out << ": ";
        switch (v->kind) {
        case JsonValue::Kind::String: jsonWriteString(out, v->str); break;
        case JsonValue::Kind::Bool: out << (v->boolean ? "true" : "false"); break;
        case JsonValue::Kind::Number: out << jsonNumber(v->number); break;
        default: out << "null"; break;
        }
      }
      out << "}";
    }
    out << "\n      ]\n    }";
  }
  out << "\n  }\n}\n";
}

// ---------------------------------------------------------------------------
// Search helpers
// ---------------------------------------------------------------------------
std::string csvEscape(const std::string &s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) {
    return s;
  }
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

int indexOfBest(const std::vector<EvaluatedCandidate> &evaluated) {
  int best = -1;
  for (size_t i = 0; i < evaluated.size(); ++i) {
    if (!evaluated[i].accepted) {
      continue;
    }
    if (best < 0 || evaluated[i].ms_median < evaluated[static_cast<size_t>(best)].ms_median) {
      best = static_cast<int>(i);
    }
  }
  return best;
}

bool containsConfig(const std::vector<EvaluatedCandidate> &evaluated,
                    const GemmCandidate &config) {
  for (const EvaluatedCandidate &e : evaluated) {
    if (e.config == config) {
      return true;
    }
  }
  return false;
}

// Single-field perturbations of `base` within the full config space, in
// deterministic field order, deduplicated against what was already evaluated.
std::vector<GemmCandidate> refinementNeighbors(
    const GemmCandidate &base,
    const std::vector<EvaluatedCandidate> &evaluated) {
  const std::vector<GemmCandidate> space = gemm_config_space();
  std::vector<GemmCandidate> neighbors;
  for (const GemmCandidate &candidate : space) {
    int diffs = 0;
    if (candidate.bm != base.bm) ++diffs;
    if (candidate.bn != base.bn) ++diffs;
    if (candidate.bk != base.bk) ++diffs;
    if (candidate.waves != base.waves) ++diffs;
    if (candidate.unroll != base.unroll) ++diffs;
    if (candidate.vec != base.vec) ++diffs;
    if (diffs != 1) {
      continue; // only direct neighbors (single-field step)
    }
    if (containsConfig(evaluated, candidate) ||
        std::find(neighbors.begin(), neighbors.end(), candidate) != neighbors.end()) {
      continue;
    }
    neighbors.push_back(candidate);
  }
  return neighbors;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string GemmCandidate::to_string() const {
  std::ostringstream os;
  os << "bm=" << bm << ",bn=" << bn << ",bk=" << bk << ",waves=" << waves
     << ",unroll=" << unroll << ",vec=" << vec;
  return os.str();
}

std::vector<GemmCandidate> gemm_config_space() {
  static const int kBM[] = {64, 128, 256};
  static const int kBN[] = {64, 128, 256};
  static const int kBK[] = {16, 32};
  static const int kWaves[] = {4, 8};
  static const int kUnroll[] = {1, 2};
  static const int kVec[] = {1, 2, 4};
  std::vector<GemmCandidate> space;
  space.reserve(216);
  for (int bm : kBM) {
    for (int bn : kBN) {
      for (int bk : kBK) {
        for (int waves : kWaves) {
          for (int unroll : kUnroll) {
            for (int vec : kVec) {
              space.push_back(GemmCandidate{bm, bn, bk, waves, unroll, vec});
            }
          }
        }
      }
    }
  }
  return space;
}

std::string resource_prune_reason(const KernelResources &res,
                                  const kernelh::DeviceLimits &limits) {
  // LDS budget (todo-33 lds_bytes vs the todo-10 per-workgroup LDS limit).
  if (limits.lds_bytes_per_workgroup > 0 && res.lds_bytes > 0 &&
      res.lds_bytes > limits.lds_bytes_per_workgroup) {
    return "pruned: lds";
  }
  // Register-file budget: ONE wavefront must fit in the register file
  // (reg_file_per_cu / wavefront_size VGPRs per wavefront). A config whose
  // kernel needs more VGPRs than that can never be resident -> prune.
  if (limits.reg_file_per_cu > 0 && limits.wavefront_size > 0 &&
      res.vgpr_count > 0) {
    const long budget = limits.reg_file_per_cu / limits.wavefront_size;
    if (res.vgpr_count > budget) {
      return "pruned: vgpr";
    }
  }
  // Belt-and-braces: the todo-33 occupancy_estimate (VGPR-limited waves) of 0
  // with a known vgpr_count means not even one wavefront can be resident.
  if (res.occupancy_estimate == 0 && res.vgpr_count > 0) {
    return "pruned: vgpr";
  }
  return "";
}

std::string sha256_hex(const std::string &data) {
  Sha256 sha;
  sha.update(data.data(), data.size());
  return sha.finalHex();
}

TuneRun autotune(const TuneOptions &options, const TuneEvaluator &evaluate,
                 const std::string &cache_key,
                 const std::filesystem::path &cache_path, std::ostream &log) {
  // 1. Whole-run cache lookup (Metis note 27: key includes the device name).
  TuneCache cache = readCacheFile(cache_path);
  const auto cachedIt = cache.find(cache_key);
  if (cachedIt != cache.end()) {
    log << "cache-hit: " << cache_key << "\n";
    TuneRun run;
    run.evaluated = cachedIt->second.evaluated;
    run.best_index = cachedIt->second.best_index;
    run.cache_hit = true;
    return run;
  }
  log << "cache-miss: " << cache_key << "\n";

  const int totalBudget = std::max(1, options.max_candidates);

  // 2. Fixed-seed sample of the config space (Metis G22 cap).
  std::vector<GemmCandidate> space = gemm_config_space();
  std::mt19937_64 rng(options.seed);
  std::shuffle(space.begin(), space.end(), rng);
  const size_t nSample =
      std::min<size_t>(space.size(), static_cast<size_t>(options.sample_cap));

  std::vector<EvaluatedCandidate> evaluated;
  evaluated.reserve(static_cast<size_t>(totalBudget));

  const auto evaluateOne = [&](const GemmCandidate &candidate, std::ostream &out) {
    EvaluatedCandidate result = evaluate(candidate);
    out << "candidate " << evaluated.size() << ": " << candidate.to_string()
        << " -> " << result.reason;
    if (result.accepted) {
      out << " (ms=" << result.ms_median << ")";
    }
    out << "\n";
    evaluated.push_back(std::move(result));
  };

  for (size_t i = 0; i < nSample &&
                     static_cast<int>(evaluated.size()) < totalBudget;
       ++i) {
    evaluateOne(space[i], log);
  }

  // 3. Local refinement around the best ACCEPTED config (neighbors in the
  //    full space, single-field steps, <= refine_budget extra, within the
  //    hard total cap).
  int bestIndex = indexOfBest(evaluated);
  if (bestIndex >= 0) {
    const std::vector<GemmCandidate> neighbors =
        refinementNeighbors(evaluated[static_cast<size_t>(bestIndex)].config,
                            evaluated);
    const int remaining =
        totalBudget - static_cast<int>(evaluated.size());
    const int refineN = std::min<int>(
        {options.refine_budget, static_cast<int>(neighbors.size()), remaining});
    log << "refining around best (index " << bestIndex << ": "
        << evaluated[static_cast<size_t>(bestIndex)].config.to_string()
        << ") with " << refineN << " neighbor(s)\n";
    for (int i = 0; i < refineN; ++i) {
      evaluateOne(neighbors[static_cast<size_t>(i)], log);
    }
    bestIndex = indexOfBest(evaluated);
  } else {
    log << "no accepted candidate; skipping refinement\n";
  }

  // 4. Persist the cache for the next run.
  CacheEntry entry;
  entry.best_index = bestIndex;
  entry.evaluated = evaluated;
  cache[cache_key] = std::move(entry);
  writeCacheFile(cache_path, cache);

  TuneRun run;
  run.evaluated = std::move(evaluated);
  run.best_index = bestIndex;
  run.cache_hit = false;
  return run;
}

bool write_tune_csv(const std::filesystem::path &path, const std::string &kernel,
                    const std::string &shape_text,
                    const std::vector<EvaluatedCandidate> &evaluated) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << "kernel,shape,bm,bn,bk,waves,unroll,vec,ms_median,accepted,reason\n";
  for (const EvaluatedCandidate &e : evaluated) {
    out << csvEscape(kernel) << ',' << csvEscape(shape_text) << ',' << e.config.bm
        << ',' << e.config.bn << ',' << e.config.bk << ',' << e.config.waves
        << ',' << e.config.unroll << ',' << e.config.vec << ','
        << jsonNumber(e.ms_median) << ',' << (e.accepted ? "true" : "false")
        << ',' << csvEscape(e.reason) << '\n';
  }
  return out.good();
}

} // namespace kernelh::compiler
