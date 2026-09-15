#include "aar/telemetry.hpp"

#include <iomanip>
#include <stdexcept>

namespace aar {

TelemetryWriter::TelemetryWriter(const std::filesystem::path& csv_path,
                                 const std::filesystem::path& jsonl_path)
    : csv_(csv_path), jsonl_(jsonl_path) {
  if (!csv_ || !jsonl_) throw std::runtime_error("cannot open telemetry outputs");
}

void TelemetryWriter::write(double time_seconds,
                            const std::map<std::string, double>& values) {
  if (!wrote_header_) {
    csv_ << "time_seconds";
    for (const auto& [name, value] : values) {
      static_cast<void>(value);
      csv_ << ',' << name;
    }
    csv_ << '\n';
    wrote_header_ = true;
  }
  csv_ << std::setprecision(17) << time_seconds;
  for (const auto& [name, value] : values) {
    static_cast<void>(name);
    csv_ << ',' << value;
  }
  csv_ << '\n';

  jsonl_ << "{\"time_seconds\":" << std::setprecision(17) << time_seconds;
  for (const auto& [name, value] : values) jsonl_ << ",\"" << name << "\":" << value;
  jsonl_ << "}\n";
}

}  // namespace aar
