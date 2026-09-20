#pragma once

#include "aar/deployment_contract.hpp"
#include <mujoco/mujoco.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace aar {

// Versioned PE02 task/controller contract. No other robot uses this state.
class PE02Runtime {
 public:
  PE02Runtime(const std::filesystem::path& path, const mjModel* model,
              const DeploymentContract& contract) {
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(path.string(), tree);
    clock_gait_ = contract.observation_builder == "pe02_v2";
    const std::string expected_schema = clock_gait_ ? "pe02.runtime.v2" : "pe02.runtime.v3";
    if ((!clock_gait_ && contract.observation_builder != "pe02_v3") ||
        tree.get<std::string>("schema") != expected_schema) {
      throw std::runtime_error("unsupported PE02 runtime schema");
    }
    const auto array = [&](const std::string& key) {
      std::vector<double> result;
      for (const auto& value : tree.get_child(key)) result.push_back(value.second.get_value<double>());
      return result;
    };
    home_ = array("default_joint_position");
    kp_ = array("kp"); kd_ = array("kd"); limits_ = array("torque_limits");
    if (clock_gait_) gait_ = array("gait");
    const auto count = contract.joint_order.size();
    if (count != 6 || home_.size() != count || kp_.size() != count || kd_.size() != count ||
        limits_.size() != count || (clock_gait_ && gait_.size() != 4) || model->nu != static_cast<int>(count)) {
      throw std::runtime_error("PE02 runtime dimensions disagree with model/manifest");
    }
    delay_ = tree.get<int>("delay_steps");
    if (delay_ < 0 || delay_ > 10000) throw std::runtime_error("invalid PE02 action delay");
    difference_velocity_ = tree.get<bool>("position_difference");
    user_limit_ = tree.get<double>("user_torque_limit");
    q_scale_ = tree.get<double>("normalization.dof_pos");
    dq_scale_ = tree.get<double>("normalization.dof_vel");
    gyro_scale_ = tree.get<double>("normalization.ang_vel");
    lin_vel_scale_ = tree.get<double>("normalization.lin_vel");
    clip_obs_ = tree.get<double>("normalization.clip_observations");
    dt_ = 1.0 / contract.physics_hz;
    policy_dt_ = 1.0 / contract.policy_hz;
    if (contract.motor_hz != contract.physics_hz) throw std::runtime_error("pe02_v2 PD must run at physics frequency");
    previous_q_.resize(count); velocity_.assign(count, 0); action_.assign(count, 0);
    offset_.assign(count, 0); fifo_.assign(delay_ + 1, std::vector<double>(count, 0));
  }

  void normalize_gamepad_command(const DeploymentContract& contract,
                                 std::vector<std::vector<float>>& inputs) const {
    for (std::size_t i = 0; i < contract.inputs.size(); ++i) {
      if (contract.inputs[i].name == "command") {
        inputs[i].at(0) *= static_cast<float>(lin_vel_scale_);
        inputs[i].at(1) *= static_cast<float>(lin_vel_scale_);
        inputs[i].at(2) *= static_cast<float>(gyro_scale_);
      }
    }
  }

  void update_inputs(const mjData* data, const DeploymentContract& contract,
                     std::vector<std::vector<float>>& inputs, std::vector<float>& history) {
    const auto input = [&](const std::string& name) {
      for (std::size_t i = 0; i < contract.inputs.size(); ++i) if (contract.inputs[i].name == name) return i;
      throw std::runtime_error("missing PE02 input " + name);
    };
    std::vector<float> frame(clock_gait_ ? 30 : 24, 0);
    if (inputs[input("observation")].size() != frame.size() ||
        inputs[input("observation_history")].size() != history.size()) {
      throw std::runtime_error("PE02 observation version disagrees with ONNX inputs");
    }
    mjtNum inverse[4] = {data->qpos[3], -data->qpos[4], -data->qpos[5], -data->qpos[6]};
    mjtNum world_gravity[3] = {0, 0, -1}, gravity[3];
    mju_rotVecQuat(gravity, world_gravity, inverse);
    for (std::size_t i = 0; i < 3; ++i) {
      frame[i] = static_cast<float>(data->qvel[3+i]*gyro_scale_);
      frame[3+i] = static_cast<float>(gravity[i]);
    }
    for (std::size_t i = 0; i < home_.size(); ++i) {
      frame[6+i] = static_cast<float>((data->qpos[7+i]-home_[i])*q_scale_);
      frame[12+i] = static_cast<float>(velocity_[i]*dq_scale_);
      frame[18+i] = static_cast<float>(action_[i]);
    }
    if (clock_gait_) {
      constexpr double pi = 3.14159265358979323846;
      frame[24] = static_cast<float>(std::sin(2*pi*phase_));
      frame[25] = static_cast<float>(std::cos(2*pi*phase_));
      for (std::size_t i = 0; i < 4; ++i) frame[26+i] = static_cast<float>(gait_[i]);
    }
    for (float& value : frame) value = std::clamp(value, static_cast<float>(-clip_obs_), static_cast<float>(clip_obs_));
    if (history.empty() || history.size() % frame.size()) throw std::runtime_error("invalid PE02 history size");
    if (!initialized_) {
      for (std::size_t i = 0; i < history.size(); i += frame.size()) std::copy(frame.begin(), frame.end(), history.begin()+i);
      initialized_ = true;
    } else {
      std::move(history.begin()+frame.size(), history.end(), history.begin());
      std::copy(frame.begin(), frame.end(), history.end()-frame.size());
    }
    inputs[input("observation_history")] = history;
    inputs[input("observation")] = frame;
  }

  void prepare_action(const mjData* data, const std::vector<float>& action,
                      const DeploymentContract& contract) {
    const double kp_mean = std::accumulate(kp_.begin(), kp_.end(), 0.0)/kp_.size();
    const double kd_mean = std::accumulate(kd_.begin(), kd_.end(), 0.0)/kd_.size();
    for (std::size_t i = 0; i < home_.size(); ++i) {
      const double center = data->qpos[7+i]-home_[i] + kd_mean*velocity_[i]/kp_mean;
      const double raw = std::clamp(static_cast<double>(action[i]), -contract.action_clip, contract.action_clip)*contract.action_scale;
      offset_[i] = std::clamp(raw, center-user_limit_/kp_mean, center+user_limit_/kp_mean);
      action_[i] = offset_[i]/contract.action_scale;
    }
  }

  void before_step(const mjModel* model, mjData* data) {
    for (int i = delay_; i > 0; --i) fifo_[i] = fifo_[i-1];
    fifo_[0] = offset_;
    for (std::size_t i = 0; i < home_.size(); ++i) {
      previous_q_[i] = data->qpos[7+i];
      const double torque = kp_[i]*(home_[i]+fifo_[delay_][i]-previous_q_[i])-kd_[i]*velocity_[i];
      data->ctrl[i] = std::clamp(torque, -limits_[i], limits_[i])/model->actuator_gear[6*i];
    }
    // Batched training passes FULLPHYSICS and starts each substep without a warmstart.
    mju_zero(data->qacc_warmstart, model->nv);
  }

  void after_step(const mjData* data) {
    constexpr double pi = 3.14159265358979323846;
    for (std::size_t i = 0; i < home_.size(); ++i) {
      double difference = std::fmod(data->qpos[7+i]-previous_q_[i]+pi, 2*pi);
      if (difference < 0) difference += 2*pi;
      velocity_[i] = difference_velocity_ ? (difference-pi)/dt_ : data->qvel[6+i];
    }
  }

  void end_policy_step() {
    if (clock_gait_) phase_ = std::fmod(phase_ + policy_dt_*gait_[0], 1.0);
  }

 private:
  std::vector<double> home_, kp_, kd_, limits_, gait_, previous_q_, velocity_, action_, offset_;
  std::vector<std::vector<double>> fifo_;
  int delay_{0};
  bool difference_velocity_{true}, initialized_{false}, clock_gait_{true};
  double user_limit_{14}, q_scale_{1}, dq_scale_{0.1}, gyro_scale_{1}, clip_obs_{100};
  double lin_vel_scale_{1};
  double dt_{0.0025}, policy_dt_{0.02}, phase_{0};
};

}  // namespace aar
