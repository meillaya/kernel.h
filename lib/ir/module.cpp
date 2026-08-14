// Module / kernel container definitions (plan todo 9).

#include "kernelh/ir/module.hpp"

namespace kernelh::ir {

namespace {

// Recursive walk that sums the static size of every atom.alloc_shared in the
// kernel's scopes. The result is compile-time because alloc_shared tensors
// are statically shaped.
void addSharedBytes(const Block &block, int64_t &total) {
  for (const auto &op : block.ops) {
    if (op->kind == OpKind::AllocShared && op->resultTypes.size() == 1 &&
        op->resultTypes[0].isTensor()) {
      total += op->resultTypes[0].asTensor().sizeBytes();
    }
    if (op->region != nullptr) {
      addSharedBytes(*op->region, total);
    }
  }
}

} // namespace

int64_t kernelTotalSharedBytes(const Kernel &kernel) {
  int64_t total = 0;
  addSharedBytes(kernel.body, total);
  return total;
}

} // namespace kernelh::ir
