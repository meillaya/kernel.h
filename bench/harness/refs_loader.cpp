#include "refs_loader.hpp"

#include <fstream>
#include <vector>

#ifndef KERNELH_REFS_DIR
#define KERNELH_REFS_DIR ""
#endif

namespace kernelh::bench {

std::string refs_artifact_path(const std::string &filename) {
  if (KERNELH_REFS_DIR[0] == '\0') {
    return "";
  }
  return std::string(KERNELH_REFS_DIR) + "/" + filename;
}

kernelh::runtime::Result<kernelh::runtime::GpuModule> load_refs_module(
    const std::shared_ptr<kernelh::runtime::Backend> &backend,
    const std::string &filename) {
  const std::string path = refs_artifact_path(filename);
  if (path.empty()) {
    return std::unexpected(kernelh::runtime::Error{
        1, "refs build dir unavailable (KERNELH_REFS_DIR not defined)"});
  }
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return std::unexpected(kernelh::runtime::Error{
        1, "cannot open refs artifact '" + path + "'"});
  }
  const auto size = in.tellg();
  if (size <= 0) {
    return std::unexpected(
        kernelh::runtime::Error{1, "refs artifact '" + path + "' is empty"});
  }
  std::vector<char> bytes(static_cast<size_t>(size));
  in.seekg(0);
  in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!in) {
    return std::unexpected(kernelh::runtime::Error{
        1, "failed reading refs artifact '" + path + "'"});
  }
  auto module = kernelh::runtime::GpuModule::load(backend, bytes.data(),
                                                  bytes.size());
  if (!module) {
    return std::unexpected(kernelh::runtime::Error{
        module.error().code,
        "refs artifact '" + path + "' failed to load: " +
            module.error().message});
  }
  return module;
}

} // namespace kernelh::bench
