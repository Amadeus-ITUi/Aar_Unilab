#include "aar/npy.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>

namespace aar {

FloatArray load_float32_npy(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open npy: " + path.string());
  char magic[6];
  input.read(magic, 6);
  if (std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
    throw std::runtime_error("invalid npy magic: " + path.string());
  }
  const auto major = static_cast<unsigned char>(input.get());
  static_cast<void>(input.get());
  std::uint32_t header_length = 0;
  if (major == 1) {
    std::uint8_t bytes[2];
    input.read(reinterpret_cast<char*>(bytes), 2);
    header_length = static_cast<std::uint32_t>(bytes[0]) |
                    (static_cast<std::uint32_t>(bytes[1]) << 8U);
  } else if (major == 2 || major == 3) {
    std::uint8_t bytes[4];
    input.read(reinterpret_cast<char*>(bytes), 4);
    header_length = static_cast<std::uint32_t>(bytes[0]) |
                    (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                    (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                    (static_cast<std::uint32_t>(bytes[3]) << 24U);
  } else {
    throw std::runtime_error("unsupported npy version");
  }
  std::string header(header_length, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (header.find("'<f4'") == std::string::npos &&
      header.find("'descr': '<f4'") == std::string::npos) {
    throw std::runtime_error("npy tensor must be little-endian float32: " + path.string());
  }
  if (header.find("'fortran_order': True") != std::string::npos) {
    throw std::runtime_error("Fortran-order npy is unsupported");
  }
  const std::regex shape_pattern("'shape'\\s*:\\s*\\(([^)]*)\\)");
  std::smatch shape_match;
  if (!std::regex_search(header, shape_match, shape_pattern)) {
    throw std::runtime_error("npy shape is missing");
  }
  FloatArray result;
  const std::regex dimension_pattern("([0-9]+)");
  const std::string dimensions = shape_match[1].str();
  for (std::sregex_iterator it(dimensions.begin(), dimensions.end(), dimension_pattern), end;
       it != end; ++it) {
    result.shape.push_back(std::stoll((*it)[1].str()));
  }
  if (result.shape.empty()) throw std::runtime_error("scalar npy is unsupported");
  const auto count = std::accumulate(result.shape.begin(), result.shape.end(), std::int64_t{1},
                                     std::multiplies<>());
  result.values.resize(static_cast<std::size_t>(count));
  input.read(reinterpret_cast<char*>(result.values.data()),
             static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!input) throw std::runtime_error("truncated npy tensor: " + path.string());
  return result;
}

}  // namespace aar
