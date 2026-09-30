#include "esd_link_bridge/robot_state_mapper.hpp"

#include <array>
#include <cmath>
#include <unordered_map>

namespace esd_link_bridge {
namespace {

using Quaternion = std::array<float, 4>;
using Vector3 = std::array<float, 3>;

Quaternion multiply(const Quaternion & a, const Quaternion & b)
{
  return {
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]};
}

Vector3 rotate(const Quaternion & q, const Vector3 & vector)
{
  const Quaternion v{vector[0], vector[1], vector[2], 0.0F};
  const Quaternion conjugate{-q[0], -q[1], -q[2], q[3]};
  const auto rotated = multiply(multiply(q, v), conjugate);
  return {rotated[0], rotated[1], rotated[2]};
}

bool finite(float value)
{
  return std::isfinite(value);
}

}  // namespace

bool RobotStateMapper::map(
  const esd_link_msgs::msg::LowerState & lower,
  esd_link_msgs::msg::RobotState & robot,
  std::string & error) const
{
  const auto count = lower.port_id.size();
  if (lower.valid_mask.size() != count || lower.position_rad.size() != count ||
    lower.velocity_rad_s.size() != count || lower.effort_nm.size() != count)
  {
    error = "LowerState 电机数组长度不一致";
    return false;
  }
  if (lower.active_port_mask != profile_.active_port_mask) {
    error = "LowerState active_port_mask 与 Profile 不一致";
    return false;
  }

  std::unordered_map<std::uint8_t, std::size_t> indices;
  for (std::size_t index = 0; index < count; ++index) {
    const auto port = lower.port_id[index];
    if (!find_actuator(profile_, port) || !indices.emplace(port, index).second) {
      error = "LowerState 包含未知或重复端口";
      return false;
    }
    if (!finite(lower.position_rad[index]) || !finite(lower.velocity_rad_s[index]) ||
      !finite(lower.effort_nm[index]))
    {
      error = "LowerState 包含非有限电机反馈";
      return false;
    }
  }
  if (indices.size() != profile_.actuators.size()) {
    error = "LowerState 未覆盖全部 Profile 端口";
    return false;
  }

  robot = esd_link_msgs::msg::RobotState();
  robot.header = lower.header;
  robot.header.frame_id = "base_link";
  robot.profile_id = profile_.profile_id;
  robot.profile_hash = profile_.profile_hash;
  robot.control_allowed = profile_.control_allowed;
  robot.calibrated = profile_.calibrated;
  robot.host_monotonic_ns = lower.host_monotonic_ns;
  robot.parse_done_monotonic_ns = lower.parse_done_monotonic_ns;
  robot.session_id = lower.session_id;
  robot.schema_id = lower.schema_id;
  robot.device_sample_time_us = lower.device_sample_time_us;
  robot.state_sample_seq = lower.state_sample_seq;
  robot.last_applied_command_seq = lower.last_applied_command_seq;
  robot.command_status_flags = lower.command_status_flags;
  robot.control_state = lower.control_state;
  robot.imu_valid_mask = lower.imu_valid_mask;
  robot.fault_flags = lower.fault_flags;
  robot.active_port_mask = lower.active_port_mask;
  robot.offline_port_mask = lower.offline_port_mask;

  robot.imu = lower.imu;
  robot.imu.header = robot.header;
  const auto rotation = profile_.imu_to_base_xyzw;
  const Quaternion orientation{
    static_cast<float>(lower.imu.orientation.x),
    static_cast<float>(lower.imu.orientation.y),
    static_cast<float>(lower.imu.orientation.z),
    static_cast<float>(lower.imu.orientation.w)};
  const Quaternion inverse_rotation{-rotation[0], -rotation[1], -rotation[2], rotation[3]};
  const auto base_orientation = multiply(orientation, inverse_rotation);
  robot.imu.orientation.x = base_orientation[0];
  robot.imu.orientation.y = base_orientation[1];
  robot.imu.orientation.z = base_orientation[2];
  robot.imu.orientation.w = base_orientation[3];
  const auto gyro = rotate(rotation, {
    static_cast<float>(lower.imu.angular_velocity.x),
    static_cast<float>(lower.imu.angular_velocity.y),
    static_cast<float>(lower.imu.angular_velocity.z)});
  robot.imu.angular_velocity.x = gyro[0];
  robot.imu.angular_velocity.y = gyro[1];
  robot.imu.angular_velocity.z = gyro[2];
  const auto acceleration = rotate(rotation, {
    static_cast<float>(lower.imu.linear_acceleration.x),
    static_cast<float>(lower.imu.linear_acceleration.y),
    static_cast<float>(lower.imu.linear_acceleration.z)});
  robot.imu.linear_acceleration.x = acceleration[0];
  robot.imu.linear_acceleration.y = acceleration[1];
  robot.imu.linear_acceleration.z = acceleration[2];

  robot.port_id.reserve(profile_.actuators.size());
  robot.joint_name.reserve(profile_.actuators.size());
  robot.valid_mask.reserve(profile_.actuators.size());
  robot.position_rad.reserve(profile_.actuators.size());
  robot.velocity_rad_s.reserve(profile_.actuators.size());
  robot.effort_nm.reserve(profile_.actuators.size());
  for (const auto & actuator : profile_.actuators) {
    const auto index = indices.at(actuator.port_id);
    robot.port_id.push_back(actuator.port_id);
    robot.joint_name.push_back(actuator.joint_name);
    robot.valid_mask.push_back(lower.valid_mask[index]);
    robot.position_rad.push_back(
      actuator.direction * lower.position_rad[index] - actuator.mechanical_zero_rad);
    robot.velocity_rad_s.push_back(actuator.direction * lower.velocity_rad_s[index]);
    robot.effort_nm.push_back(actuator.direction * lower.effort_nm[index]);
  }
  error.clear();
  return true;
}

}  // namespace esd_link_bridge
