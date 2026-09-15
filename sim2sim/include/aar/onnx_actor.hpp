#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "aar/deployment_contract.hpp"

namespace aar {

class OnnxActor {
 public:
  OnnxActor(const std::filesystem::path& policy_path, const DeploymentContract& contract);
  ~OnnxActor();
  OnnxActor(OnnxActor&&) noexcept;
  OnnxActor& operator=(OnnxActor&&) noexcept;
  OnnxActor(const OnnxActor&) = delete;
  OnnxActor& operator=(const OnnxActor&) = delete;

  std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aar
