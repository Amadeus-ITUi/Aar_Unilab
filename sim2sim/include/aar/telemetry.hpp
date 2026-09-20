#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <string>

namespace aar {

class TelemetryWriter {
 public:
  TelemetryWriter(const std::filesystem::path& csv_path,
                  const std::filesystem::path& jsonl_path);
  void write(double time_seconds, const std::map<std::string, double>& values);

 private:
  std::ofstream csv_;
  std::ofstream jsonl_;
  bool wrote_header_{false};
};

}  // namespace aar
