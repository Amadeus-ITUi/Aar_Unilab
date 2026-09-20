// Copyright 2026 lzh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joy.hpp>

#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>

using namespace std::chrono_literals;

/**
 * @brief Xbox velocity publisher node
 *
 * Subscribes to /joy topic (Xbox controller data) and publishes
 * velocity commands to the configured output topic (/cmd_vel by default).
 */
class XboxVelPublisher : public rclcpp::Node
{
public:
  XboxVelPublisher()
  : Node("xbox_vel_publisher")
  {
    // Declare parameters with default values
    this->declare_parameter<double>("max_linear_speed", 1.0);
    this->declare_parameter<double>("max_angular_speed", 1.0);
    this->declare_parameter<double>("max_linear_accel", 3.0);
    this->declare_parameter<double>("deadzone", 0.05);
    this->declare_parameter<double>("command_timeout", 0.5);
    this->declare_parameter<std::string>("input_topic", "/joy");
    this->declare_parameter<std::string>("output_topic", "/cmd_vel");

    // Get parameter values
    this->get_parameter("max_linear_speed", max_linear_speed_);
    this->get_parameter("max_angular_speed", max_angular_speed_);
    this->get_parameter("max_linear_accel", max_linear_accel_);
    this->get_parameter("deadzone", deadzone_);
    this->get_parameter("command_timeout", command_timeout_);
    this->get_parameter("input_topic", input_topic_);
    this->get_parameter("output_topic", output_topic_);

    deadzone_ = std::clamp(deadzone_, 0.0, 0.99);
    max_linear_accel_ = std::max(max_linear_accel_, 0.0);
    command_timeout_ = std::max(command_timeout_, 0.0);

    RCLCPP_INFO(this->get_logger(), "Xbox Vel Publisher started");
    RCLCPP_INFO(this->get_logger(), "max_linear_speed: %.2f", max_linear_speed_);
    RCLCPP_INFO(this->get_logger(), "max_angular_speed: %.2f", max_angular_speed_);
    RCLCPP_INFO(this->get_logger(), "max_linear_accel: %.2f", max_linear_accel_);
    RCLCPP_INFO(this->get_logger(), "deadzone: %.2f", deadzone_);
    RCLCPP_INFO(
      this->get_logger(), "topics: %s -> %s", input_topic_.c_str(), output_topic_.c_str());

    // Create publisher for velocity commands
    publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      output_topic_, rclcpp::SensorDataQoS());

    // Create subscriber for joy messages
    subscriber_ = this->create_subscription<sensor_msgs::msg::Joy>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&XboxVelPublisher::joy_callback, this, std::placeholders::_1)
    );

    // Create timer for publishing at 50Hz
    timer_ = this->create_wall_timer(
      20ms,
      std::bind(&XboxVelPublisher::timer_callback, this)
    );
  }

private:
  /**
   * @brief Apply deadzone to joystick input
   * @param value Raw joystick input [-1.0, 1.0]
   * @return Processed value with deadzone applied
   */
  double apply_deadzone(double value)
  {
    if (std::abs(value) < deadzone_) {
      return 0.0;
    }
    // Rescale the value to maintain full range
    double sign = (value > 0) ? 1.0 : -1.0;
    return sign * ((std::abs(value) - deadzone_) / (1.0 - deadzone_));
  }

  double limit_rate(double target, double current, double limit, double dt)
  {
    if (limit <= 0.0 || dt <= 0.0) {
      return target;
    }
    const double max_delta = limit * dt;
    return current + std::clamp(target - current, -max_delta, max_delta);
  }

  /**
   * @brief Joystick callback - stores latest joystick data
   */
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    if (msg->axes.size() < 2) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Joy message has %zu axes; at least 2 are required", msg->axes.size());
      return;
    }
    latest_joy_ = msg;
    last_joy_time_ = this->now();
  }

  /**
   * @brief Timer callback - publishes velocity commands
   */
  void timer_callback()
  {
    // Check if we have received joy messages
    if (!latest_joy_) {
      return;
    }

    const auto now = this->now();
    double target_linear_x = 0.0;
    double target_linear_y = 0.0;
    double target_angular_z = 0.0;

    if (command_timeout_ > 0.0 &&
      (now - last_joy_time_).seconds() > command_timeout_)
    {
      const double dt = last_publish_time_.nanoseconds() > 0 ?
        std::max((now - last_publish_time_).seconds(), 0.0) : 0.02;
      last_command_linear_x_ = limit_rate(
        target_linear_x, last_command_linear_x_, max_linear_accel_, dt);

      geometry_msgs::msg::Twist twist;
      twist.linear.x = last_command_linear_x_;
      publisher_->publish(twist);
      last_publish_time_ = now;
      return;
    }

    auto joy = latest_joy_;

    // Xbox controller mapping:
    // Flydigi/xpad mapping, aligned with Play:
    // axes[1]: Left stick Y (up/down)    -> linear.x (forward/backward)
    // axes[0]: Left stick X (left/right) -> angular.z (yaw)
    // Lateral velocity is always zero. Right-stick Y is consumed directly by wing_motor_node.

    // Apply deadzone and scaling
    double left_stick_y = apply_deadzone(joy->axes[1]);  // Up/forward positive
    double left_stick_x = apply_deadzone(joy->axes[0]);  // Left/yaw positive

    target_linear_x = left_stick_y * max_linear_speed_;
    target_linear_y = 0.0;
    target_angular_z = left_stick_x * max_angular_speed_;

    const double dt = last_publish_time_.nanoseconds() > 0 ?
      std::max((now - last_publish_time_).seconds(), 0.0) : 0.02;
    last_command_linear_x_ = limit_rate(
      target_linear_x, last_command_linear_x_, max_linear_accel_, dt);

    // Map to velocity commands
    geometry_msgs::msg::Twist twist;
    twist.linear.x = last_command_linear_x_;
    twist.linear.y = target_linear_y;
    twist.linear.z = 0.0;
    twist.angular.x = 0.0;
    twist.angular.y = 0.0;
    twist.angular.z = target_angular_z;

    // Publish velocity command
    publisher_->publish(twist);
    last_publish_time_ = now;
  }

  // Parameters
  double max_linear_speed_;
  double max_angular_speed_;
  double max_linear_accel_;
  double deadzone_;
  double command_timeout_;
  std::string input_topic_;
  std::string output_topic_;

  // ROS interfaces
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr subscriber_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Store latest joy message
  sensor_msgs::msg::Joy::SharedPtr latest_joy_;
  rclcpp::Time last_joy_time_;
  rclcpp::Time last_publish_time_;
  double last_command_linear_x_ = 0.0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<XboxVelPublisher>());
  rclcpp::shutdown();
  return 0;
}
