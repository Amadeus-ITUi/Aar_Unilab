#pragma once

#include "aar/deployment_contract.hpp"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace aar {

// PE02 owns this observation layout independently of every other robot profile.
inline void update_pe02_inputs(const mjData* data, const DeploymentContract& contract,
                              std::vector<std::vector<float>>& inputs,
                              std::vector<float>& history) {
  const auto input = [&](const std::string& name) {
    for (std::size_t i = 0; i < contract.inputs.size(); ++i) {
      if (contract.inputs[i].name == name) return i;
    }
    throw std::runtime_error("pe02_v1 requires input: " + name);
  };
  const auto history_index = input("observation_history");
  const auto observation_index = input("observation");
  const auto count = contract.joint_order.size();
  std::vector<float> frame(inputs[observation_index].size(), 0.0F);
  if (frame.size() < 3 * count + 6 || history.empty() || history.size() % frame.size()) {
    throw std::runtime_error("pe02_v1 observation/history shape mismatch");
  }
  for (std::size_t i = 0; i < count; ++i) {
    frame[i] = static_cast<float>(data->qpos[7 + i]);
    frame[count + i] = static_cast<float>(data->qvel[6 + i]);
    frame[2 * count + 6 + i] = static_cast<float>(data->ctrl[i]);
  }
  for (std::size_t i = 0; i < 3; ++i) {
    frame[2 * count + i] = static_cast<float>(data->qvel[3 + i]);
    frame[2 * count + 3 + i] = static_cast<float>(data->qvel[i]);
  }
  std::move(history.begin() + frame.size(), history.end(), history.begin());
  std::copy(frame.begin(), frame.end(), history.end() - frame.size());
  inputs[history_index] = history;
  inputs[observation_index] = std::move(frame);
}

}  // namespace aar
