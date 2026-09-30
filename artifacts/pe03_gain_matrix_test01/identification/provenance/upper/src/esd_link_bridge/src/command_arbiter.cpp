#include "esd_link_bridge/command_arbiter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace esd_link_bridge {
namespace {

bool valid_lengths(const esd_link_msgs::msg::ActuatorCommand & command)
{
  const auto size = command.port_id.size();
  return command.position_rad.size() == size && command.velocity_rad_s.size() == size &&
         command.kp.size() == size && command.kd.size() == size &&
         command.effort_nm.size() == size;
}

bool finite_nonnegative(float value)
{
  return std::isfinite(value) && value >= 0.0F;
}

bool position_target_is_safe(
  std::uint8_t mode,
  const esd_link_bridge::ActuatorProfile & actuator,
  float current,
  float target)
{
  if (!actuator.has_position_limits ||
    (target >= actuator.position_min_rad && target <= actuator.position_max_rad))
  {
    return true;
  }
  using Message = esd_link_msgs::msg::ActuatorCommand;
  if (mode != Message::MODE_SWEEP && mode != Message::MODE_MANUAL_TEST) {
    return false;
  }
  // A robot resting on the floor can enter maintenance just outside a soft
  // limit. Permit only a hold or an inward recovery; never permit the target
  // to move farther outward than the observed entry-side position.
  constexpr float tolerance = 0.01F;
  if (current > actuator.position_max_rad) {
    return target >= actuator.position_max_rad && target <= current + tolerance;
  }
  if (current < actuator.position_min_rad) {
    return target <= actuator.position_min_rad && target >= current - tolerance;
  }
  return false;
}

std::uint8_t expected_owner(const ActuatorProfile & actuator)
{
  if (actuator.policy_owner == PolicyOwner::POLICY) {
    return esd_link_msgs::msg::ActuatorCommand::OWNER_POLICY;
  }
  if (actuator.policy_owner == PolicyOwner::AUXILIARY) {
    return esd_link_msgs::msg::ActuatorCommand::OWNER_AUXILIARY;
  }
  return esd_link_msgs::msg::ActuatorCommand::OWNER_STANDBY;
}

}  // namespace

bool CommandArbiter::compose(
  std::uint8_t mode,
  const esd_link_msgs::msg::RobotState & state,
  const std::vector<esd_link_msgs::msg::ActuatorCommand> & contributions,
  esd_link_msgs::msg::ActuatorCommand & final_command,
  std::string & error) const
{
  using Message = esd_link_msgs::msg::ActuatorCommand;
  if (!profile_.control_allowed || !profile_.calibrated) {
    error = "Robot Profile 禁止控制";
    return false;
  }
  if (state.profile_id != profile_.profile_id || state.profile_hash != profile_.profile_hash ||
    state.active_port_mask != profile_.active_port_mask || state.offline_port_mask != 0U ||
    state.fault_flags != 0U)
  {
    error = "RobotState 与 Profile 不匹配或状态不健康";
    return false;
  }
  const auto state_count = state.port_id.size();
  if (state.position_rad.size() != state_count || state.velocity_rad_s.size() != state_count ||
    state.effort_nm.size() != state_count || state.valid_mask.size() != state_count)
  {
    error = "RobotState 电机数组长度不一致";
    return false;
  }
  std::unordered_map<std::uint8_t, std::size_t> state_index;
  for (std::size_t index = 0; index < state_count; ++index) {
    if (!state_index.emplace(state.port_id[index], index).second) {
      error = "RobotState 包含重复端口";
      return false;
    }
  }

  final_command = esd_link_msgs::msg::ActuatorCommand();
  final_command.header = state.header;
  final_command.owner = Message::OWNER_ARBITER;
  final_command.mode = mode;
  final_command.profile_id = profile_.profile_id;
  final_command.profile_hash = profile_.profile_hash;
  final_command.source_state_sample_seq = state.state_sample_seq;
  final_command.port_id.reserve(profile_.actuators.size());
  final_command.position_rad.reserve(profile_.actuators.size());
  final_command.velocity_rad_s.reserve(profile_.actuators.size());
  final_command.kp.reserve(profile_.actuators.size());
  final_command.kd.reserve(profile_.actuators.size());
  final_command.effort_nm.reserve(profile_.actuators.size());
  for (const auto & actuator : profile_.actuators) {
    const auto state_it = state_index.find(actuator.port_id);
    if (state_it == state_index.end()) {
      error = "RobotState 未覆盖全部 Profile 端口";
      return false;
    }
    const auto index = state_it->second;
    final_command.port_id.push_back(actuator.port_id);
    final_command.position_rad.push_back(state.position_rad[index]);
    final_command.velocity_rad_s.push_back(0.0F);
    final_command.kp.push_back(actuator.kp);
    final_command.kd.push_back(actuator.kd);
    final_command.effort_nm.push_back(0.0F);
  }

  std::unordered_map<std::uint8_t, std::size_t> final_index;
  for (std::size_t index = 0; index < final_command.port_id.size(); ++index) {
    final_index.emplace(final_command.port_id[index], index);
  }
  std::uint32_t contributed_mask = 0U;
  std::uint32_t source_state_sequence = 0U;
  bool have_source_state_sequence = false;
  for (const auto & contribution : contributions) {
    const auto source_lag = state.state_sample_seq - contribution.source_state_sample_seq;
    if (!valid_lengths(contribution) || contribution.profile_id != profile_.profile_id ||
      contribution.profile_hash != profile_.profile_hash || contribution.mode != mode ||
      source_lag > 10U)
    {
      error = "控制贡献长度、Profile、模式或源状态无效/陈旧";
      return false;
    }
    if (!have_source_state_sequence) {
      source_state_sequence = contribution.source_state_sample_seq;
      have_source_state_sequence = true;
    }
    for (std::size_t index = 0; index < contribution.port_id.size(); ++index) {
      const auto port = contribution.port_id[index];
      const auto * actuator = find_actuator(profile_, port);
      const auto target = final_index.find(port);
      if (!actuator || target == final_index.end() || (contributed_mask & (1U << port))) {
        error = "控制贡献包含未知或重复端口";
        return false;
      }
      if (mode == Message::MODE_POLICY && contribution.owner != expected_owner(*actuator)) {
        error = "POLICY 模式端口所有者越权";
        return false;
      }
      if ((mode == Message::MODE_SWEEP || mode == Message::MODE_MANUAL_TEST) &&
        contribution.owner != Message::OWNER_MAINTENANCE)
      {
        error = "维护模式只接受 maintenance 所有者";
        return false;
      }
      if (mode == Message::MODE_STANDBY && contribution.owner != Message::OWNER_STANDBY) {
        error = "STANDBY 模式只接受 standby 所有者";
        return false;
      }
      const auto position = contribution.position_rad[index];
      const auto velocity = contribution.velocity_rad_s[index];
      const auto kp = contribution.kp[index];
      const auto kd = contribution.kd[index];
      const auto effort = contribution.effort_nm[index];
      const auto state_position = state.position_rad[state_index.at(port)];
      if (!std::isfinite(position) || !std::isfinite(velocity) || !finite_nonnegative(kp) ||
        !finite_nonnegative(kd) || !std::isfinite(effort) ||
        std::abs(velocity) > actuator->command_velocity_max_rad_s ||
        std::abs(effort) > actuator->device_effort_max_nm ||
        !position_target_is_safe(mode, *actuator, state_position, position))
      {
        error = "控制目标包含非法值或超出 Profile 限位";
        return false;
      }
      const auto output = target->second;
      final_command.position_rad[output] = position;
      final_command.velocity_rad_s[output] = velocity;
      final_command.kp[output] = kp;
      final_command.kd[output] = kd;
      final_command.effort_nm[output] = effort;
      contributed_mask |= 1U << port;
    }
  }

  if (mode == Message::MODE_POLICY) {
    for (const auto & actuator : profile_.actuators) {
      if (actuator.policy_owner != PolicyOwner::HOLD &&
        (contributed_mask & (1U << actuator.port_id)) == 0U)
      {
        error = "POLICY 模式缺少端口所有者贡献";
        return false;
      }
    }
  }
  if (mode == Message::MODE_STANDBY && contributed_mask != profile_.active_port_mask) {
    error = "STANDBY 命令必须完整覆盖全部端口";
    return false;
  }
  if (have_source_state_sequence) {
    final_command.source_state_sample_seq = source_state_sequence;
  }
  error.clear();
  return true;
}

bool CommandArbiter::to_lower(
  const esd_link_msgs::msg::ActuatorCommand & final_command,
  CompleteLowerCommand & lower,
  std::string & error) const
{
  using Message = esd_link_msgs::msg::ActuatorCommand;
  if (final_command.owner != Message::OWNER_ARBITER ||
    final_command.profile_id != profile_.profile_id ||
    final_command.profile_hash != profile_.profile_hash || !valid_lengths(final_command) ||
    final_command.port_id.size() != profile_.actuators.size())
  {
    error = "最终命令不是当前 Profile 的完整 Arbiter 输出";
    return false;
  }
  lower = {};
  std::uint32_t seen = 0U;
  for (std::size_t index = 0; index < final_command.port_id.size(); ++index) {
    const auto port = final_command.port_id[index];
    const auto * actuator = find_actuator(profile_, port);
    if (!actuator || (seen & (1U << port))) {
      error = "最终命令包含未知或重复端口";
      return false;
    }
    seen |= 1U << port;
    auto & output = lower.ports[lower.count++];
    output.port_id = port;
    output.position = actuator->direction *
      (final_command.position_rad[index] + actuator->mechanical_zero_rad);
    output.velocity = actuator->direction * final_command.velocity_rad_s[index];
    output.kp = final_command.kp[index];
    output.kd = final_command.kd[index];
    output.effort = actuator->direction * final_command.effort_nm[index];
  }
  if (seen != profile_.active_port_mask) {
    error = "最终命令没有覆盖全部 active port";
    return false;
  }
  error.clear();
  return true;
}

}  // namespace esd_link_bridge
