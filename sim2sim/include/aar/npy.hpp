#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace aar {

struct FloatArray {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

FloatArray load_float32_npy(const std::filesystem::path& path);

}  // namespace aar
