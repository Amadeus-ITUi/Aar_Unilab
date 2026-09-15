#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace aar {

struct TensorContract {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
};

struct DeploymentContract {
  std::string schema;
  std::string robot_id;
  std::string task_id;
  std::filesystem::path policy_path;
  std::filesystem::path runtime_config_path;
  std::filesystem::path scene_path;
  std::vector<TensorContract> inputs;
  std::vector<TensorContract> outputs;
  std::string observation_builder;
  std::vector<std::string> joint_order;
  double physics_hz{0.0};
  double motor_hz{0.0};
  double policy_hz{0.0};
  double action_scale{1.0};
  double action_clip{1.0};
};

DeploymentContract load_contract(const std::filesystem::path& manifest_path);

}  // namespace aar
