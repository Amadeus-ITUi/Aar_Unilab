#include "motor_driver.hpp"
#include "wing_control_logic.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <motors/msg/wing_runtime_status.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class WingMotorNode : public rclcpp::Node {
public:
    WingMotorNode() : Node("wing_motor_node") {
        declare_parameter<std::string>("can_interface", "can0");
        declare_parameter<std::vector<int64_t>>("motor_ids", {7, 8});
        declare_parameter<std::string>("motor_type", "RS00");
        declare_parameter<int>("master_id", 0);
        declare_parameter<double>("publish_frequency", 200.0);
        declare_parameter<std::vector<double>>(
            "startup_target_positions_deg", {90.0, -90.0});
        declare_parameter<double>("startup_kp", 1.0);
        declare_parameter<double>("startup_kd", 0.1);
        declare_parameter<double>("startup_return_speed", 0.35);
        declare_parameter<double>("feedback_timeout_seconds", 2.0);
        declare_parameter<double>("runtime_feedback_timeout_seconds", 0.5);
        declare_parameter<int>("offline_threshold", 40);
        declare_parameter<bool>("disable_on_shutdown", true);
        declare_parameter<std::string>("publish_topic", "/policy/wing_angles");
        declare_parameter<bool>("joy_rc_enabled", true);
        declare_parameter<int>("joy_velocity_axis", 4);  // Flydigi right-stick Y
        declare_parameter<double>("joy_axis_deadzone", 0.1);
        declare_parameter<double>("joy_axis_sign", 1.0);
        declare_parameter<std::vector<double>>("wing_velocity_signs", {1.0, -1.0});
        declare_parameter<double>("wing_max_velocity_rad_s", kPi / 4.0);
        declare_parameter<double>("wing_rc_kp", 20.0);
        declare_parameter<double>("wing_rc_kd", 1.0);
        declare_parameter<double>("joy_command_timeout_seconds", 0.5);
        declare_parameter<std::vector<double>>(
            "position_limits_deg", {0.0, 90.0, -90.0, 0.0});
        declare_parameter<std::vector<double>>(
            "shutdown_position_limits_deg", {-10.0, 100.0, -100.0, 10.0});

        get_parameter("can_interface", can_interface_);
        get_parameter("motor_type", motor_type_);
        get_parameter("master_id", master_id_);
        get_parameter("publish_frequency", publish_frequency_);
        get_parameter("startup_kp", startup_kp_);
        get_parameter("startup_kd", startup_kd_);
        get_parameter("startup_return_speed", startup_return_speed_);
        get_parameter("feedback_timeout_seconds", feedback_timeout_seconds_);
        get_parameter("runtime_feedback_timeout_seconds", runtime_feedback_timeout_seconds_);
        get_parameter("offline_threshold", offline_threshold_);
        get_parameter("disable_on_shutdown", disable_on_shutdown_);
        get_parameter("publish_topic", publish_topic_);
        get_parameter("joy_rc_enabled", joy_rc_enabled_);
        get_parameter("joy_velocity_axis", joy_velocity_axis_);
        get_parameter("joy_axis_deadzone", joy_axis_deadzone_);
        get_parameter("joy_axis_sign", joy_axis_sign_);
        get_parameter("wing_max_velocity_rad_s", wing_max_velocity_rad_s_);
        get_parameter("wing_rc_kp", wing_rc_kp_);
        get_parameter("wing_rc_kd", wing_rc_kd_);
        get_parameter("joy_command_timeout_seconds", joy_command_timeout_seconds_);

        const auto ids = get_parameter("motor_ids").as_integer_array();
        if (ids.size() != kWingMotorCount) {
            throw std::runtime_error("wing_motor_node requires exactly motor_ids [7, 8]");
        }
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            motor_ids_[i] = static_cast<int>(ids[i]);
        }
        const auto position_limits_deg =
            get_parameter("position_limits_deg").as_double_array();
        const auto startup_target_positions_deg =
            get_parameter("startup_target_positions_deg").as_double_array();
        const auto shutdown_position_limits_deg =
            get_parameter("shutdown_position_limits_deg").as_double_array();
        const auto velocity_signs = get_parameter("wing_velocity_signs").as_double_array();
        if (position_limits_deg.size() != kWingMotorCount * 2) {
            throw std::runtime_error("position_limits_deg requires [min7,max7,min8,max8]");
        }
        if (startup_target_positions_deg.size() != kWingMotorCount) {
            throw std::runtime_error(
                "startup_target_positions_deg requires [motor7_target,motor8_target]");
        }
        if (shutdown_position_limits_deg.size() != kWingMotorCount * 2) {
            throw std::runtime_error(
                "shutdown_position_limits_deg requires [min7,max7,min8,max8]");
        }
        if (velocity_signs.size() != kWingMotorCount) {
            throw std::runtime_error("wing_velocity_signs requires [motor7_sign,motor8_sign]");
        }
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            position_min_[i] = static_cast<float>(position_limits_deg[i * 2] * kDegToRad);
            position_max_[i] = static_cast<float>(position_limits_deg[i * 2 + 1] * kDegToRad);
            startup_target_positions_[i] = static_cast<float>(
                startup_target_positions_deg[i] * kDegToRad);
            shutdown_position_min_[i] = static_cast<float>(
                shutdown_position_limits_deg[i * 2] * kDegToRad);
            shutdown_position_max_[i] = static_cast<float>(
                shutdown_position_limits_deg[i * 2 + 1] * kDegToRad);
            if (!std::isfinite(velocity_signs[i]) || std::abs(velocity_signs[i]) < 1.0e-6) {
                throw std::runtime_error("wing_velocity_signs entries must be finite and non-zero");
            }
            wing_velocity_signs_[i] = static_cast<float>(velocity_signs[i]);
        }
        if (publish_frequency_ <= 0.0 || startup_return_speed_ <= 0.0 ||
            !std::isfinite(startup_kp_) || startup_kp_ < 0.0 ||
            !std::isfinite(startup_kd_) || startup_kd_ < 0.0 ||
            feedback_timeout_seconds_ <= 0.0 || runtime_feedback_timeout_seconds_ <= 0.0 ||
            joy_axis_deadzone_ < 0.0 || joy_axis_deadzone_ >= 1.0 ||
            wing_max_velocity_rad_s_ <= 0.0 || wing_rc_kp_ < 0.0 || wing_rc_kd_ < 0.0 ||
            joy_command_timeout_seconds_ <= 0.0 || !std::isfinite(joy_axis_sign_) ||
            offline_threshold_ <= 0) {
            throw std::runtime_error("wing_motor_node timing and safety parameters must be positive");
        }
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            if (position_min_[i] >= position_max_[i] ||
                0.0f < position_min_[i] || 0.0f > position_max_[i]) {
                throw std::runtime_error("wing position limits must be ordered and contain 0 rad");
            }
            if (!std::isfinite(startup_target_positions_[i]) ||
                startup_target_positions_[i] < position_min_[i] ||
                startup_target_positions_[i] > position_max_[i]) {
                throw std::runtime_error("wing startup target must be inside software limits");
            }
            if (shutdown_position_min_[i] >= position_min_[i] ||
                shutdown_position_max_[i] <= position_max_[i]) {
                throw std::runtime_error(
                    "wing shutdown limits must strictly contain software position limits");
            }
        }

        auto qos = rclcpp::SensorDataQoS();
        publisher_ = create_publisher<sensor_msgs::msg::JointState>(publish_topic_, qos);
        status_publisher_ = create_publisher<motors::msg::WingRuntimeStatus>(
            "/wing/runtime_status", rclcpp::QoS(1).reliable().transient_local());
        enable_rc_service_ = create_service<std_srvs::srv::SetBool>(
            "/wing/enable_rc",
            std::bind(&WingMotorNode::enable_rc, this,
                      std::placeholders::_1, std::placeholders::_2));
        joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
            "/joy", qos, std::bind(&WingMotorNode::joy_callback, this, std::placeholders::_1));

        try {
            for (std::size_t i = 0; i < kWingMotorCount; ++i) {
                motors_[i] = MotorDriver::MotorCreate(
                    static_cast<std::uint16_t>(motor_ids_[i]), can_interface_.c_str(), motor_type_,
                    static_cast<std::uint16_t>(master_id_), 0);
            }
            for (std::size_t i = 0; i < kWingMotorCount; ++i) {
                if (!wait_for_feedback(motors_[i])) {
                    throw std::runtime_error(
                        "no startup feedback from wing motor " + std::to_string(motor_ids_[i]));
                }
            }
            for (std::size_t i = 0; i < kWingMotorCount; ++i) {
                motors_[i]->MotorInit();
                if (!wait_for_feedback(motors_[i])) {
                    throw std::runtime_error(
                        "no feedback after initialization from wing motor " +
                        std::to_string(motor_ids_[i]));
                }
                target_positions_[i] = motors_[i]->get_motor_pos();
                feedback_counts_[i] = motors_[i]->get_feedback_count();
                last_feedback_times_[i] = std::chrono::steady_clock::now();
            }
        } catch (...) {
            for (auto& motor : motors_) {
                if (motor) {
                    motor->MotorLock();
                }
            }
            throw;
        }

        state_enter_time_ = std::chrono::steady_clock::now();
        const auto period = std::chrono::duration<double>(1.0 / publish_frequency_);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&WingMotorNode::control_tick, this));
        startup_feedback_grace_until_ =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(feedback_timeout_seconds_));

        RCLCPP_WARN(
            get_logger(),
            "翼电机节点启动: CAN=%s motors=[%d,%d], /joy axes[%d] 控制翼速度，"
            "signs=[%.0f,%.0f]，最大=%.4frad/s，启动目标=[%.1f,%.1f]deg，发布%s",
            can_interface_.c_str(), motor_ids_[0], motor_ids_[1],
            joy_velocity_axis_, wing_velocity_signs_[0], wing_velocity_signs_[1],
            wing_max_velocity_rad_s_,
            startup_target_positions_[0] / static_cast<float>(kDegToRad),
            startup_target_positions_[1] / static_cast<float>(kDegToRad),
            publish_topic_.c_str());
    }

    ~WingMotorNode() override {
        if (!disable_on_shutdown_) {
            return;
        }
        for (auto& motor : motors_) {
            if (motor) {
                motor->MotorDeInit();
            }
        }
    }

private:
    static constexpr std::size_t kWingMotorCount = 2;
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kDegToRad = kPi / 180.0;

    enum class MotionState {
        POSITIONING,
        POSITION_HOLD,
        RC_CONTROL,
        FAULT,
    };

    float axis_to_rad_per_s(float raw) const {
        if (!std::isfinite(raw) || std::abs(raw) <= joy_axis_deadzone_) {
            return 0.0f;
        }
        const double magnitude =
            (std::abs(static_cast<double>(raw)) - joy_axis_deadzone_) /
            (1.0 - joy_axis_deadzone_);
        return static_cast<float>(
            std::copysign(magnitude, static_cast<double>(raw)) * joy_axis_sign_ *
            wing_max_velocity_rad_s_);
    }

    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg) {
        if (motion_state_ != MotionState::RC_CONTROL || !joy_rc_enabled_ || joy_velocity_axis_ < 0 ||
            static_cast<std::size_t>(joy_velocity_axis_) >= msg->axes.size()) {
            wing_vel_cmd_rad_s_.fill(0.0f);
            return;
        }
        const float velocity = axis_to_rad_per_s(
            msg->axes[static_cast<std::size_t>(joy_velocity_axis_)]);
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            wing_vel_cmd_rad_s_[i] = velocity * wing_velocity_signs_[i];
        }
        last_joy_command_time_ = std::chrono::steady_clock::now();
        joy_command_received_ = true;
    }

    void enter_state(MotionState state, const std::chrono::steady_clock::time_point& now) {
        motion_state_ = state;
        state_enter_time_ = now;
    }

    void enable_rc(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        if (request->data) {
            if (motion_state_ != MotionState::POSITION_HOLD) {
                response->success = false;
                response->message = "wing must be in POSITION_HOLD before RC can be enabled";
                return;
            }
            enter_state(MotionState::RC_CONTROL, std::chrono::steady_clock::now());
            response->message = "wing RC enabled";
        } else {
            if (motion_state_ == MotionState::FAULT) {
                response->success = false;
                response->message = "wing is faulted";
                return;
            }
            std::array<float, kWingMotorCount> actual_positions{};
            for (std::size_t i = 0; i < kWingMotorCount; ++i) {
                actual_positions[i] = motors_[i]->get_motor_pos();
            }
            wing_control_logic::hold_current_feedback(
                actual_positions, target_positions_, wing_vel_cmd_rad_s_);
            enter_state(MotionState::POSITION_HOLD, std::chrono::steady_clock::now());
            response->message = "wing RC disabled; holding current position";
        }
        response->success = true;
        publish_runtime_status();
    }

    bool advance_targets(const std::array<float, kWingMotorCount>& destinations) {
        const float max_step = static_cast<float>(startup_return_speed_ / publish_frequency_);
        return wing_control_logic::advance_targets(target_positions_, destinations, max_step);
    }

    bool positions_within_limits(
        const std::array<float, kWingMotorCount>& positions,
        float margin_rad = 0.0f) const {
        std::array<float, kWingMotorCount> minimum{};
        std::array<float, kWingMotorCount> maximum{};
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            minimum[i] = position_min_[i] - margin_rad;
            maximum[i] = position_max_[i] + margin_rad;
        }
        return wing_control_logic::positions_within_limits(positions, minimum, maximum);
    }

    bool positions_are_finite(
        const std::array<float, kWingMotorCount>& positions) const {
        for (const float position : positions) {
            if (!std::isfinite(position)) {
                return false;
            }
        }
        return true;
    }

    bool positions_within_shutdown_limits(
        const std::array<float, kWingMotorCount>& positions) const {
        return wing_control_logic::positions_within_limits(
            positions, shutdown_position_min_, shutdown_position_max_);
    }

    bool wait_for_feedback(const std::shared_ptr<MotorDriver>& motor) const {
        const auto initial_feedback_count = motor->get_feedback_count();
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration<double>(feedback_timeout_seconds_);
        while (std::chrono::steady_clock::now() < deadline) {
            motor->refresh_motor_status();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (motor->get_feedback_count() > initial_feedback_count) {
                return true;
            }
        }
        return false;
    }

    void fail_and_shutdown(const std::string& reason) {
        if (fatal_error_) {
            return;
        }
        fatal_error_ = true;
        motion_state_ = MotionState::FAULT;
        fault_code_ = reason;
        RCLCPP_FATAL(get_logger(), "%s；停止翼电机节点，部署启动必须退出", reason.c_str());
        for (auto& motor : motors_) {
            if (motor) {
                motor->MotorLock();
            }
        }
        publish_runtime_status();
        rclcpp::shutdown();
    }

    void control_tick() {
        if (fatal_error_) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        std::array<float, kWingMotorCount> actual_positions{};
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            actual_positions[i] = motors_[i]->get_motor_pos();
        }
        if (!positions_are_finite(actual_positions)) {
            fail_and_shutdown("7/8 号电机实际角度无效");
            return;
        }
        // 目标受软件限位约束；实际反馈只有越过更宽的停机边界才触发退出。
        // 两侧机械方向镜像：motor7 正角向上，motor8 负角向上。
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            if (actual_positions[i] < shutdown_position_min_[i] ||
                actual_positions[i] > shutdown_position_max_[i]) {
                fail_and_shutdown(
                    "翼电机 " + std::to_string(motor_ids_[i]) + " 实际角度 " +
                    std::to_string(actual_positions[i] / static_cast<float>(kDegToRad)) +
                    " deg 超出停机边界 [" +
                    std::to_string(shutdown_position_min_[i] / static_cast<float>(kDegToRad)) +
                    ", " +
                    std::to_string(shutdown_position_max_[i] / static_cast<float>(kDegToRad)) +
                    "] deg");
                return;
            }
        }
        bool send_position_command = true;
        float command_kp = static_cast<float>(startup_kp_);
        float command_kd = static_cast<float>(startup_kd_);

        switch (motion_state_) {
            case MotionState::POSITIONING:
                if (advance_targets(startup_target_positions_)) {
                    startup_target_commanded_ = true;
                    enter_state(MotionState::POSITION_HOLD, now);
                    RCLCPP_WARN(
                        get_logger(),
                        "7/8 号启动目标命令已完成: target=[%.2f,%.2f]deg "
                        "actual=[%.2f,%.2f]deg velocity=[%.4f,%.4f]rad/s；"
                        "按配置不判定实际到位误差",
                        startup_target_positions_[0] / static_cast<float>(kDegToRad),
                        startup_target_positions_[1] / static_cast<float>(kDegToRad),
                        actual_positions[0] / static_cast<float>(kDegToRad),
                        actual_positions[1] / static_cast<float>(kDegToRad),
                        motors_[0]->get_motor_spd(), motors_[1]->get_motor_spd());
                }
                break;
            case MotionState::POSITION_HOLD:
                break;
            case MotionState::RC_CONTROL: {
                if (!joy_command_received_ ||
                    std::chrono::duration<double>(now - last_joy_command_time_).count() >
                        joy_command_timeout_seconds_) {
                    wing_vel_cmd_rad_s_.fill(0.0f);
                }
                const float dt = static_cast<float>(1.0 / publish_frequency_);
                for (std::size_t i = 0; i < kWingMotorCount; ++i) {
                    float velocity = wing_vel_cmd_rad_s_[i];
                    if ((target_positions_[i] >= position_max_[i] && velocity > 0.0f) ||
                        (target_positions_[i] <= position_min_[i] && velocity < 0.0f)) {
                        velocity = 0.0f;
                    }
                    target_positions_[i] = std::clamp(
                        target_positions_[i] + velocity * dt,
                        position_min_[i], position_max_[i]);
                }
                command_kp = static_cast<float>(wing_rc_kp_);
                command_kd = static_cast<float>(wing_rc_kd_);
                break;
            }
            case MotionState::FAULT:
                return;
        }

        const bool target_is_safe = motion_state_ == MotionState::RC_CONTROL
            ? positions_within_limits(target_positions_)
            : positions_within_shutdown_limits(target_positions_);
        if (!positions_are_finite(target_positions_) || !target_is_safe) {
            fail_and_shutdown("7/8 号电机目标角度超出软件限位");
            return;
        }

        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            if (send_position_command) {
                motors_[i]->MotorMitModeCmd(
                    target_positions_[i], 0.0f, command_kp, command_kd, 0.0f);
            } else {
                motors_[i]->refresh_motor_status();
            }

            // MotorInit() 已确认电机在线，但启动后的第一批控制帧和反馈计数
            // 可能仍在接收线程中同步。给启动定位阶段一个反馈稳定窗口，
            // 避免 watchdog 在第一批回帧前误停机；窗口结束后恢复完整保护。
            if (now >= startup_feedback_grace_until_) {
                if (motors_[i]->get_response_count() > offline_threshold_) {
                    fail_and_shutdown(
                        "翼电机 " + std::to_string(motor_ids_[i]) + " 连续无有效反馈");
                    return;
                }
                const auto feedback_count = motors_[i]->get_feedback_count();
                if (feedback_count != feedback_counts_[i]) {
                    feedback_counts_[i] = feedback_count;
                    last_feedback_times_[i] = now;
                } else if (std::chrono::duration<double>(now - last_feedback_times_[i]).count() >
                           runtime_feedback_timeout_seconds_) {
                    fail_and_shutdown(
                        "翼电机 " + std::to_string(motor_ids_[i]) + " 真实反馈帧超时");
                    return;
                }
            }
        }

        // 启动定位期间也持续发布实际角度，供 supervisor 健康监控。
        publish_feedback();
        if (++status_counter_ % std::max(1, static_cast<int>(publish_frequency_ / 20.0)) == 0) {
            publish_runtime_status();
        }
    }

    void publish_runtime_status() {
        if (!status_publisher_) {
            return;
        }
        motors::msg::WingRuntimeStatus status;
        status.stamp = now();
        switch (motion_state_) {
            case MotionState::POSITIONING: status.state = status.POSITIONING; break;
            case MotionState::POSITION_HOLD: status.state = status.POSITION_HOLD; break;
            case MotionState::RC_CONTROL: status.state = status.RC_CONTROL; break;
            case MotionState::FAULT: status.state = status.FAULT; break;
        }
        status.motor_ids = {motor_ids_[0], motor_ids_[1]};
        status.position = {motors_[0]->get_motor_pos(), motors_[1]->get_motor_pos()};
        status.velocity = {motors_[0]->get_motor_spd(), motors_[1]->get_motor_spd()};
        status.feedback_ok = !fatal_error_;
        status.fault_code = fault_code_;
        status.message = startup_target_commanded_ ? "startup target commanded" : "";
        status_publisher_->publish(status);
    }

    void publish_feedback() {
        sensor_msgs::msg::JointState msg;
        msg.header.stamp = now();
        msg.name = {"motor_7", "motor_8"};
        msg.position.resize(kWingMotorCount);
        msg.velocity.resize(kWingMotorCount);
        msg.effort.resize(kWingMotorCount);
        for (std::size_t i = 0; i < kWingMotorCount; ++i) {
            msg.position[i] = motors_[i]->get_motor_pos();
            msg.velocity[i] = motors_[i]->get_motor_spd();
            msg.effort[i] = motors_[i]->get_motor_current();
        }
        publisher_->publish(msg);
    }

    std::string can_interface_ = "can0";
    std::string motor_type_ = "RS00";
    std::string publish_topic_ = "/policy/wing_angles";
    int master_id_ = 0;
    double publish_frequency_ = 200.0;
    double startup_kp_ = 1.0;
    double startup_kd_ = 0.1;
    double startup_return_speed_ = 0.35;
    double feedback_timeout_seconds_ = 2.0;
    double runtime_feedback_timeout_seconds_ = 0.5;
    int offline_threshold_ = 40;
    bool disable_on_shutdown_ = true;
    bool joy_rc_enabled_ = true;
    int joy_velocity_axis_ = 4;
    double joy_axis_deadzone_ = 0.1;
    double joy_axis_sign_ = 1.0;
    double wing_max_velocity_rad_s_ = kPi / 4.0;
    double wing_rc_kp_ = 20.0;
    double wing_rc_kd_ = 1.0;
    double joy_command_timeout_seconds_ = 0.5;
    std::array<int, kWingMotorCount> motor_ids_{7, 8};
    std::array<std::shared_ptr<MotorDriver>, kWingMotorCount> motors_{};
    std::array<float, kWingMotorCount> target_positions_{};
    std::array<float, kWingMotorCount> startup_target_positions_{};
    std::array<float, kWingMotorCount> wing_vel_cmd_rad_s_{};
    std::array<float, kWingMotorCount> wing_velocity_signs_{1.0f, -1.0f};
    std::array<float, kWingMotorCount> position_min_{};
    std::array<float, kWingMotorCount> position_max_{};
    std::array<float, kWingMotorCount> shutdown_position_min_{};
    std::array<float, kWingMotorCount> shutdown_position_max_{};
    std::array<std::uint64_t, kWingMotorCount> feedback_counts_{};
    std::array<std::chrono::steady_clock::time_point, kWingMotorCount> last_feedback_times_{};
    std::chrono::steady_clock::time_point state_enter_time_{};
    std::chrono::steady_clock::time_point startup_feedback_grace_until_{};
    MotionState motion_state_ = MotionState::POSITIONING;
    std::chrono::steady_clock::time_point last_joy_command_time_{};
    bool joy_command_received_ = false;
    bool startup_target_commanded_ = false;
    bool fatal_error_ = false;
    int status_counter_ = 0;
    std::string fault_code_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::Publisher<motors::msg::WingRuntimeStatus>::SharedPtr status_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_rc_service_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<WingMotorNode>());
    } catch (const std::exception& error) {
        RCLCPP_FATAL(rclcpp::get_logger("wing_motor_node"), "%s", error.what());
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
        return 1;
    }
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    return 0;
}
