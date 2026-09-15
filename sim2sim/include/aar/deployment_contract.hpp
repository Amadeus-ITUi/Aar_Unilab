#pragma once

#include <filesystem>
#include <string>

namespace aar {

struct DeploymentContract {
  std::string schema;
  std::string robot_id;
  std::string task_id;
  std::filesystem::path policy_path;
  std::filesystem::path runtime_config_path;
};

DeploymentContract load_contract(const std::filesystem::path& manifest_path);

}  // namespace aar
