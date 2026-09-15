#include "aar/deployment_contract.hpp"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace aar {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open manifest: " + path.string());
  std::ostringstream content;
  content << stream.rdbuf();
  return content.str();
}

std::string string_field(const std::string& document, const std::string& field) {
  const std::regex pattern("\\\"" + field + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  std::smatch match;
  if (!std::regex_search(document, match, pattern)) {
    throw std::runtime_error("missing manifest field: " + field);
  }
  return match[1].str();
}

std::filesystem::path relative_path(const std::string& value, const std::string& field) {
  std::filesystem::path path(value);
  if (path.is_absolute()) throw std::runtime_error(field + " must be release-relative");
  for (const auto& component : path) {
    if (component == "..") throw std::runtime_error(field + " escapes release directory");
  }
  return path;
}

}  // namespace

DeploymentContract load_contract(const std::filesystem::path& manifest_path) {
  const std::string document = read_file(manifest_path);
  const std::string schema = string_field(document, "schema");
  if (schema != "aar-unilab.actor.v1") {
    throw std::runtime_error("unsupported deployment contract: " + schema);
  }
  if (document.find("\"inputs\"") == std::string::npos ||
      document.find("\"outputs\"") == std::string::npos) {
    throw std::runtime_error("policy inputs and outputs are required");
  }
  return DeploymentContract{
      schema,
      string_field(document, "id"),
      string_field(document.substr(document.find("\"task\"")), "id"),
      relative_path(string_field(document, "policy_path"), "policy_path"),
      relative_path(string_field(document, "runtime_config_path"), "runtime_config_path"),
  };
}

}  // namespace aar
