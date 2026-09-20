// DR002 轮腿 Isaac Lab 策略推理节点。
//
// ESD 数据流: /lower/state + /cmd_vel
//   -> 构造 term-major actor_history 观测 (145 维, 29 单帧 x 5 帧)
//   -> MNN 推理 (lab_policy.mnn) 得 raw action (6 维)
//   -> 转 /policy/commands (6 维, data[i]=raw_clipped*scale)
//   -> motors_node 加默认角/翻转/限位, 腿位置控制 + 轮速度控制 (vel_only_mode)
//
// 与部署链路共用同一套 motors / 话题约定:
//   - /policy/joint_states.position 已是相对默认角 (motors 侧已减), 推理侧直接用。
//   - projected_gravity 直接取 IMU 节点算好的 angular_velocity_covariance[0..2]。
//   - /policy/commands 与 gym 版同布局 (6 维缩放后动作), motors 通吃。
#include "lab_deploy_utils.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <esd_link_msgs/msg/lower_state.hpp>
#include <esd_link_msgs/msg/policy_command.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class LabInferenceNode : public rclcpp::Node {
public:
    LabInferenceNode() : Node("lab_inference_node") {
        declare_parameter<std::string>("model_name", "lab_policy.mnn");
        declare_parameter<std::vector<int64_t>>("motor_ids", {1, 2, 3, 4, 5, 6});
        declare_parameter<std::vector<std::string>>(
            "joint_names",
            {"Left_thigh_joint", "Left_calf_joint", "Left_foot_joint",
             "Right_thigh_joint", "Right_calf_joint", "Right_foot_joint"});
        declare_parameter<float>("default_height_command", 0.24f);
        declare_parameter<float>("clip_observations", 100.0f);
        declare_parameter<int>("intra_threads", 1);
        declare_parameter<bool>("debug_mode", false);
        declare_parameter<float>("dt", 0.0025f);
        declare_parameter<int>("decimation", 8);
        declare_parameter<bool>("log_policy_outputs", true);
        declare_parameter<int>("policy_log_interval_ticks", 50);
        declare_parameter<bool>("dummy_observation_mode", false);
        declare_parameter<bool>("force_zero_policy_commands", false);
        declare_parameter<bool>("joy_policy_gate_enabled", false);
        declare_parameter<bool>("joy_policy_start_enabled", false);
        declare_parameter<int>("joy_policy_enable_button", 2);   // Xbox X
        declare_parameter<int>("joy_policy_standby_button", 3);  // Xbox Y
        declare_parameter<std::string>("wing_angle_topic", "/policy/wing_angles");
        declare_parameter<bool>("use_lower_state", true);
        declare_parameter<double>("wing_angle_timeout_seconds", 0.1);
        // These timeouts are retained only for the legacy split-topic backend.
        // Native ESD mode consumes one atomic LowerState and has no ROS age gate.
        declare_parameter<double>("imu_timeout_seconds", 0.2);
        // Fixed rotation from physical IMU frame to training/base_link frame.
        // Current mounting: IMU x points opposite base_link x; keep right-handed
        // frame by flipping y as well (180 deg yaw about z): [x,y,z] -> [-x,-y,z].
        declare_parameter<std::vector<double>>("imu_to_base_signs", {-1.0, -1.0, 1.0});

        get_parameter("model_name", model_name_);
        get_parameter("joint_names", joint_names_);
        get_parameter("default_height_command", default_height_command_);
        get_parameter("clip_observations", clip_observations_);
        get_parameter("intra_threads", intra_threads_);
        get_parameter("debug_mode", debug_mode_);
        get_parameter("dt", dt_);
        get_parameter("decimation", decimation_);
        get_parameter("log_policy_outputs", log_policy_outputs_);
        get_parameter("policy_log_interval_ticks", policy_log_interval_ticks_);
        get_parameter("dummy_observation_mode", dummy_observation_mode_);
        get_parameter("force_zero_policy_commands", force_zero_policy_commands_);
        get_parameter("joy_policy_gate_enabled", joy_policy_gate_enabled_);
        get_parameter("joy_policy_start_enabled", joy_policy_start_enabled_);
        get_parameter("joy_policy_enable_button", joy_policy_enable_button_);
        get_parameter("joy_policy_standby_button", joy_policy_standby_button_);
        get_parameter("wing_angle_topic", wing_angle_topic_);
        get_parameter("use_lower_state", use_lower_state_);
        get_parameter("wing_angle_timeout_seconds", wing_angle_timeout_seconds_);
        get_parameter("imu_timeout_seconds", imu_timeout_seconds_);
        if (!std::isfinite(wing_angle_timeout_seconds_) || wing_angle_timeout_seconds_ <= 0.0) {
            throw std::runtime_error("wing_angle_timeout_seconds must be finite and positive");
        }
        if (!std::isfinite(imu_timeout_seconds_) || imu_timeout_seconds_ <= 0.0) {
            throw std::runtime_error("imu_timeout_seconds must be finite and positive");
        }
        const auto imu_signs_tmp = get_parameter("imu_to_base_signs").as_double_array();
        if (imu_signs_tmp.size() == 3) {
            for (std::size_t i = 0; i < 3; ++i) {
                imu_to_base_signs_[i] = static_cast<float>(imu_signs_tmp[i] >= 0.0 ? 1.0 : -1.0);
            }
        } else {
            RCLCPP_WARN(
                get_logger(),
                "imu_to_base_signs must have 3 entries, using default [-1, -1, 1]");
        }
        policy_log_interval_ticks_ = std::max(1, policy_log_interval_ticks_);
        policy_output_enabled_ = joy_policy_start_enabled_;

        const auto motor_tmp = get_parameter("motor_ids").as_integer_array();
        motor_ids_.reserve(motor_tmp.size());
        for (auto id : motor_tmp) {
            motor_ids_.push_back(static_cast<int>(id));
        }

        command_[2] = default_height_command_;
        load_model();
        auto sensor_qos = rclcpp::SensorDataQoS();
        auto control_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        if (use_lower_state_) {
            lower_state_sub_ = create_subscription<esd_link_msgs::msg::LowerState>(
                "/lower/state", sensor_qos,
                std::bind(&LabInferenceNode::lower_state_callback, this, std::placeholders::_1));
        } else {
            imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
                "/IMU_data", sensor_qos, std::bind(&LabInferenceNode::imu_callback, this, std::placeholders::_1));
            joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
                "/policy/joint_states", sensor_qos,
                std::bind(&LabInferenceNode::joint_state_callback, this, std::placeholders::_1));
            wing_angle_sub_ = create_subscription<sensor_msgs::msg::JointState>(
                wing_angle_topic_, sensor_qos,
                std::bind(&LabInferenceNode::wing_angle_callback, this, std::placeholders::_1));
        }
        cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", sensor_qos, std::bind(&LabInferenceNode::cmd_callback, this, std::placeholders::_1));
        joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
            "/joy", sensor_qos, std::bind(&LabInferenceNode::joy_callback, this, std::placeholders::_1));
        command_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/policy/commands", control_qos);
        stamped_command_pub_ = create_publisher<esd_link_msgs::msg::PolicyCommand>(
            "/policy/stamped_commands", control_qos);
        mode_pub_ = create_publisher<std_msgs::msg::UInt8>(
            "/inference/runtime_mode", rclcpp::QoS(1).reliable().transient_local());
        set_policy_service_ = create_service<std_srvs::srv::SetBool>(
            "/inference/set_policy_enabled",
            std::bind(&LabInferenceNode::set_policy_enabled, this,
                      std::placeholders::_1, std::placeholders::_2));

        const auto policy_period =
            std::chrono::microseconds(static_cast<long long>(dt_ * decimation_ * 1000000.0f));
        timer_ = create_wall_timer(policy_period, std::bind(&LabInferenceNode::inference_tick, this));

        RCLCPP_INFO(
            get_logger(),
            "DR002 Lab 推理节点启动: model=%s input=%zu output=%zu lower_state=%s wing_topic=%s wing_timeout=%.3fs dt=%f decimation=%d period=%lldus -> stamped + legacy policy commands, log_outputs=%s every=%d ticks, dummy_observation=%s, force_zero_commands=%s",
            model_name_.c_str(), lab_deploy::kActorInputDim, lab_deploy::kActionDim,
            use_lower_state_ ? "native" : "legacy",
            wing_angle_topic_.c_str(), wing_angle_timeout_seconds_,
            dt_, decimation_, static_cast<long long>(policy_period.count()),
            log_policy_outputs_ ? "true" : "false", policy_log_interval_ticks_,
            dummy_observation_mode_ ? "true" : "false",
            force_zero_policy_commands_ ? "true" : "false");
        if (joy_policy_gate_enabled_) {
            RCLCPP_WARN(
                get_logger(),
                "手柄策略门控已启用: 当前模式=%s, X(button %d)=切换 standby/policy, Y(button %d)=回到 def-pos standby",
                policy_output_enabled_ ? "policy" : "standby",
                joy_policy_enable_button_, joy_policy_standby_button_);
        }
        if (!use_lower_state_) {
            RCLCPP_WARN(
                get_logger(),
                "legacy IMU->base_link sign transform: [%.0f, %.0f, %.0f]",
                imu_to_base_signs_[0], imu_to_base_signs_[1], imu_to_base_signs_[2]);
        }
        publish_mode();
    }

private:
    static std::uint64_t monotonic_ns() {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    void lower_state_callback(const esd_link_msgs::msg::LowerState::SharedPtr msg) {
        if (dummy_observation_mode_ || msg->schema_id != 1 ||
            msg->active_port_mask != 0x1FEU || msg->offline_port_mask != 0 ||
            msg->fault_flags != 0 || (msg->imu_valid_mask & 0x07U) != 0x07U) {
            return;
        }
        const auto &imu = msg->imu;
        const std::array<float, 3> angular{
            static_cast<float>(imu.angular_velocity.x),
            static_cast<float>(imu.angular_velocity.y),
            static_cast<float>(imu.angular_velocity.z)};
        const std::array<float, 3> gravity{
            static_cast<float>(imu.angular_velocity_covariance[0]),
            static_cast<float>(imu.angular_velocity_covariance[1]),
            static_cast<float>(imu.angular_velocity_covariance[2])};
        for (std::size_t i = 0; i < 3; ++i) {
            if (!std::isfinite(angular[i]) || !std::isfinite(gravity[i])) return;
        }
        std::array<float, lab_deploy::kWingAngleDim> wing_position{};
        std::array<float, lab_deploy::kWingVelDim> wing_velocity{};
        std::uint32_t found = 0;
        for (std::size_t i = 0; i < 8; ++i) {
            const auto id = msg->port_id[i];
            if (id < 1 || id > 8 || (msg->valid_mask[i] & 0x03U) != 0x03U ||
                !std::isfinite(msg->position_rad[i]) ||
                !std::isfinite(msg->velocity_rad_s[i])) return;
            found |= 1U << id;
            if (id <= 6) {
                joint_pos_[id - 1] = msg->position_rad[i];
                joint_vel_[id - 1] = msg->velocity_rad_s[i];
            } else {
                wing_position[id - 7] = msg->position_rad[i];
                wing_velocity[id - 7] = msg->velocity_rad_s[i];
            }
        }
        if (found != 0x1FEU) return;
        base_ang_vel_ = angular;
        projected_gravity_ = gravity;
        wing_angle_rad_ = wing_position;
        wing_ang_vel_rad_s_ = wing_velocity;
        wing_angle_obs_ = lab_deploy::motor_wing_angles_to_training_obs(wing_position);
        wing_vel_obs_ = lab_deploy::motor_wing_velocities_to_training_obs(wing_velocity);
        source_state_sample_seq_ = msg->state_sample_seq;
        lower_state_ready_ = true;
        last_imu_time_ = std::chrono::steady_clock::now();
        last_wing_angle_time_ = last_imu_time_;
        imu_received_ = joint_received_ = wing_angle_received_ = true;
    }

    void load_model() {
        std::string models_dir;
        try {
            models_dir = ament_index_cpp::get_package_share_directory("inference") + "/models/";
        } catch (const std::exception&) {
            models_dir = std::string(ROOT_DIR) + "models/";
        }
        model_path_ = models_dir + model_name_;
        if (!std::filesystem::exists(model_path_)) {
            // 退回到源码目录下的 models (与 gym 节点一致的查找习惯)
            model_path_ = std::string(ROOT_DIR) + "models/" + model_name_;
        }
        if (!std::filesystem::exists(model_path_)) {
            RCLCPP_ERROR(get_logger(), "Lab 策略模型不存在: %s", model_path_.c_str());
            return;
        }

        net_.reset(MNN::Interpreter::createFromFile(model_path_.c_str()));
        if (!net_) {
            RCLCPP_ERROR(get_logger(), "MNN 模型加载失败: %s", model_path_.c_str());
            return;
        }

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = std::max(1, intra_threads_);
        MNN::BackendConfig backend_config;
        backend_config.precision = MNN::BackendConfig::Precision_Low;
        backend_config.power = MNN::BackendConfig::Power_High;
        config.backendConfig = &backend_config;

        session_ = net_->createSession(config);
        if (!session_) {
            RCLCPP_ERROR(get_logger(), "MNN session 创建失败");
            return;
        }

        input_tensor_ = net_->getSessionInput(session_, nullptr);
        output_tensor_ = net_->getSessionOutput(session_, nullptr);
        if (!input_tensor_ || !output_tensor_) {
            RCLCPP_ERROR(get_logger(), "MNN 输入/输出 Tensor 获取失败");
            return;
        }

        if (input_tensor_->elementSize() != static_cast<int>(lab_deploy::kActorInputDim)) {
            RCLCPP_ERROR(
                get_logger(), "MNN 输入维度不匹配: model=%d expected=%zu",
                input_tensor_->elementSize(), lab_deploy::kActorInputDim);
            return;
        }
        if (output_tensor_->elementSize() != static_cast<int>(lab_deploy::kActionDim)) {
            RCLCPP_ERROR(
                get_logger(), "MNN 输出维度不匹配: model=%d expected=%zu",
                output_tensor_->elementSize(), lab_deploy::kActionDim);
            return;
        }
        model_ready_ = true;
    }

    void cmd_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        if (dummy_observation_mode_) {
            return;
        }
        // 与 gym 节点一致: vx=linear.x, yaw=angular.z, height=linear.z (无指令时保持默认高度)。
        command_[0] = static_cast<float>(msg->linear.x);
        command_[1] = static_cast<float>(msg->angular.z);
        const float height = static_cast<float>(msg->linear.z);
        command_[2] = (std::isfinite(height) && height > 1e-3f) ? height : default_height_command_;
    }

    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg) {
        if (!joy_policy_gate_enabled_) {
            return;
        }

        const bool enable_pressed = button_pressed(*msg, joy_policy_enable_button_);
        const bool standby_pressed = button_pressed(*msg, joy_policy_standby_button_);

        if (enable_pressed && !last_enable_button_pressed_) {
            if (policy_output_enabled_) {
                enter_standby("Xbox X");
            } else if (!try_enter_policy("Xbox X")) {
                // 保持 standby；拒绝原因已在 try_enter_policy 打日志。
            }
        }
        if (standby_pressed && !last_standby_button_pressed_) {
            enter_standby("Xbox Y");
        }

        last_enable_button_pressed_ = enable_pressed;
        last_standby_button_pressed_ = standby_pressed;
    }

    void enter_standby(const char* source) {
        policy_output_enabled_ = false;
        last_obs_action_.fill(0.0f);
        RCLCPP_WARN(
            get_logger(),
            "%s: 切换到 standby 模式；原生 PolicyCommand 停止发布，legacy /policy/commands 强制为 0",
            source);
        publish_mode();
    }

    bool try_enter_policy(const char* source) {
        std::string reason;
        if (!imu_ok_for_policy(&reason)) {
            RCLCPP_ERROR(
                get_logger(),
                "%s: 拒绝进入 policy（%s）；保持 standby。请确认原生 LowerState/IMU 有效后再试。",
                source, reason.c_str());
            return false;
        }
        last_obs_action_.fill(0.0f);
        reset_observation_history();
        policy_output_enabled_ = true;
        RCLCPP_WARN(
            get_logger(),
            "%s: 切换到 policy 模式，已重置 actor_history/last_action，/policy/commands 开始发布%s",
            source,
            force_zero_policy_commands_ ? "台架测试六维零动作" : " MNN 输出");
        publish_mode();
        return true;
    }

    void set_policy_enabled(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        if (!request->data) {
            enter_standby("WE11 supervisor");
            response->success = true;
            response->message = "inference is in STANDBY";
            return;
        }
        response->success = try_enter_policy("WE11 supervisor");
        response->message = response->success ?
            "inference entered POLICY" : "policy readiness gate rejected request";
    }

    void publish_mode() {
        if (!mode_pub_) {
            return;
        }
        std_msgs::msg::UInt8 mode;
        mode.data = policy_output_enabled_ ? 1 : 0;
        mode_pub_->publish(mode);
    }

    void reset_observation_history() {
        // 用当前传感器帧（last_action=0）填满 5 帧历史，避免把倾倒/standby 残差带进 policy。
        const auto frame = build_frame();
        history_.fill(frame);
    }

    bool imu_ok_for_policy(std::string* reason) const {
        if (dummy_observation_mode_ || debug_mode_) {
            return true;
        }
        if (use_lower_state_) {
            if (!lower_state_ready_) {
                if (reason) {
                    *reason = "尚未收到完整有效的 LowerState";
                }
                return false;
            }
            return true;
        }
        if (!imu_received_) {
            if (reason) {
                *reason = "尚未收到 IMU";
            }
            return false;
        }
        const double imu_age = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_imu_time_).count();
        if (imu_age > imu_timeout_seconds_) {
            if (reason) {
                *reason = "IMU 超时 " + std::to_string(imu_age) + "s（阈值 " +
                          std::to_string(imu_timeout_seconds_) + "s）；话题已停更，非姿态倾倒误判";
            }
            return false;
        }
        return true;
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        if (dummy_observation_mode_) {
            return;
        }
        const float wx = imu_to_base_signs_[0] * static_cast<float>(msg->angular_velocity.x);
        const float wy = imu_to_base_signs_[1] * static_cast<float>(msg->angular_velocity.y);
        const float wz = imu_to_base_signs_[2] * static_cast<float>(msg->angular_velocity.z);
        if (std::isfinite(wx) && std::isfinite(wy) && std::isfinite(wz)) {
            base_ang_vel_[0] = wx;
            base_ang_vel_[1] = wy;
            base_ang_vel_[2] = wz;
        }

        // projected_gravity 直接取 IMU 节点算好并滤波的重力投影 (机体系),
        // 存放在 angular_velocity_covariance[0..2]。
        const float gx = imu_to_base_signs_[0] * static_cast<float>(msg->angular_velocity_covariance[0]);
        const float gy = imu_to_base_signs_[1] * static_cast<float>(msg->angular_velocity_covariance[1]);
        const float gz = imu_to_base_signs_[2] * static_cast<float>(msg->angular_velocity_covariance[2]);
        if (std::isfinite(gx) && std::isfinite(gy) && std::isfinite(gz) &&
            (gx != 0.0f || gy != 0.0f || gz != 0.0f)) {
            projected_gravity_[0] = gx;
            projected_gravity_[1] = gy;
            projected_gravity_[2] = gz;
        }
        last_imu_time_ = std::chrono::steady_clock::now();
        imu_received_ = true;
    }

    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (dummy_observation_mode_) {
            return;
        }
        // motors_node 按 motor_ids 顺序发布 /policy/joint_states (name = "motor_<id>"),
        // 顺序即 [L_thigh, L_calf, L_foot, R_thigh, R_calf, R_foot], 与训练 6 维 joint 顺序一致。
        // position 已是相对默认角的策略坐标 (motors 侧已减默认角并翻转), 推理侧按下标直接用。
        const std::size_t n = std::min<std::size_t>(lab_deploy::kActionDim, msg->position.size());
        for (std::size_t i = 0; i < n; ++i) {
            joint_pos_[i] = static_cast<float>(msg->position[i]);
        }
        const std::size_t nv = std::min<std::size_t>(lab_deploy::kActionDim, msg->velocity.size());
        for (std::size_t i = 0; i < nv; ++i) {
            joint_vel_[i] = static_cast<float>(msg->velocity[i]);
        }
        joint_received_ = true;
    }

    void wing_angle_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (dummy_observation_mode_ || msg->position.size() < lab_deploy::kWingAngleDim) {
            return;
        }

        std::array<float, lab_deploy::kWingAngleDim> positions{};
        std::array<float, lab_deploy::kWingVelDim> velocities{};
        const bool have_velocity = msg->velocity.size() >= lab_deploy::kWingVelDim;
        bool mapped_by_name = false;
        if (msg->name.size() == msg->position.size()) {
            bool found_7 = false;
            bool found_8 = false;
            for (std::size_t i = 0; i < msg->name.size(); ++i) {
                if (msg->name[i] == "motor_7") {
                    positions[0] = static_cast<float>(msg->position[i]);
                    if (have_velocity && i < msg->velocity.size()) {
                        velocities[0] = static_cast<float>(msg->velocity[i]);
                    }
                    found_7 = true;
                } else if (msg->name[i] == "motor_8") {
                    positions[1] = static_cast<float>(msg->position[i]);
                    if (have_velocity && i < msg->velocity.size()) {
                        velocities[1] = static_cast<float>(msg->velocity[i]);
                    }
                    found_8 = true;
                }
            }
            mapped_by_name = found_7 && found_8;
        }
        if (!mapped_by_name) {
            positions[0] = static_cast<float>(msg->position[0]);
            positions[1] = static_cast<float>(msg->position[1]);
            if (have_velocity) {
                velocities[0] = static_cast<float>(msg->velocity[0]);
                velocities[1] = static_cast<float>(msg->velocity[1]);
            }
        }

        if (!std::isfinite(positions[0]) || !std::isfinite(positions[1])) {
            return;
        }
        if (!std::isfinite(velocities[0]) || !std::isfinite(velocities[1])) {
            velocities[0] = 0.0f;
            velocities[1] = 0.0f;
        }
        wing_angle_rad_ = positions;
        wing_ang_vel_rad_s_ = velocities;
        wing_angle_obs_ = lab_deploy::motor_wing_angles_to_training_obs(positions);
        wing_vel_obs_ = lab_deploy::motor_wing_velocities_to_training_obs(velocities);
        last_wing_angle_time_ = std::chrono::steady_clock::now();
        wing_angle_received_ = true;
    }

    lab_deploy::ObservationFrame build_frame() {
        lab_deploy::ObservationFrame frame{};
        std::size_t offset = 0;

        for (float v : base_ang_vel_) {
            frame[offset++] = v;
        }
        for (float v : projected_gravity_) {
            frame[offset++] = v;
        }
        // joint_pos_no_wheel: 仅腿关节 (idx 0,1,3,4) 的相对默认角; motors 已发相对值, 直接用。
        for (std::size_t index : lab_deploy::kLegActionIndices) {
            frame[offset++] = joint_pos_[index];
        }
        // joint_vel: 训练顺序为 [L_thigh, L_calf, R_thigh, R_calf, L_foot, R_foot]。
        const auto training_joint_vel = lab_deploy::motor_to_training_joint_order(joint_vel_);
        for (float v : training_joint_vel) {
            frame[offset++] = v * 0.1f;
        }
        // last_action: 上一帧 gym 裁剪后的 raw。
        for (float v : last_obs_action_) {
            frame[offset++] = v;
        }
        // wing_angle: 训练顺序 [left, right]。2026-09-09 实机辨向确认
        // bridge 输入为 [motor7=left(+), motor8=right(-)]；左翼取反后做 rad / pi。
        for (float v : wing_angle_obs_) {
            frame[offset++] = v;
        }
        // wing_vel: 与 wing_angle 相同的左翼取反映射，再乘 0.1。
        for (float v : wing_vel_obs_) {
            frame[offset++] = v;
        }
        // command: [lin_vel_x, ang_vel_z, height]。
        for (float v : command_) {
            frame[offset++] = v;
        }

        for (auto& v : frame) {
            v = std::clamp(v, -clip_observations_, clip_observations_);
        }
        return frame;
    }

    void inference_tick() {
        const auto inference_start_ns = monotonic_ns();
        if (!model_ready_) {
            return;
        }
        if (!debug_mode_ && !dummy_observation_mode_ &&
            (!imu_received_ || !joint_received_ || !wing_angle_received_)) {
            return;
        }
        const bool native_observation =
            use_lower_state_ && !debug_mode_ && !dummy_observation_mode_;
        if (native_observation &&
            !lab_deploy::has_new_sample_sequence(
                source_state_sample_seq_, last_inferred_state_sample_seq_,
                have_inferred_state_sample_seq_)) {
            return;
        }
        if (native_observation) {
            last_inferred_state_sample_seq_ = source_state_sample_seq_;
            have_inferred_state_sample_seq_ = true;
        }
        if (!use_lower_state_ && !debug_mode_ && !dummy_observation_mode_) {
            const double wing_age = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - last_wing_angle_time_).count();
            if (wing_age > wing_angle_timeout_seconds_) {
                if (!wing_angle_stale_latched_) {
                    RCLCPP_WARN(
                        get_logger(),
                        "翼角观测已超时 %.3fs（阈值 %.3fs）；保持当前模式并沿用最后一次翼角观测",
                        wing_age, wing_angle_timeout_seconds_);
                }
                wing_angle_stale_latched_ = true;
            }
            if (wing_angle_stale_latched_ && wing_age <= wing_angle_timeout_seconds_) {
                wing_angle_stale_latched_ = false;
                RCLCPP_WARN(
                    get_logger(),
                    "翼角观测已恢复；当前策略模式保持不变");
            }
        }

        std::move(history_.begin() + 1, history_.end(), history_.begin());
        history_.back() = build_frame();

        // term-major 展平: 每个观测项各自展平 5 帧历史 (最旧->最新), 再按项顺序拼接,
        // 与训练侧 IsaacLab ObservationManager + MlpAdaptModel 的布局一致。
        const auto input = lab_deploy::flatten_term_major_history(history_);

        MNN::Tensor input_host(input_tensor_, MNN::Tensor::CAFFE);
        std::memcpy(input_host.host<float>(), input.data(), input.size() * sizeof(float));
        input_tensor_->copyFromHostTensor(&input_host);
        net_->runSession(session_);

        MNN::Tensor output_host(output_tensor_, MNN::Tensor::CAFFE);
        output_tensor_->copyToHostTensor(&output_host);
        const float* output = output_host.host<float>();
        const std::size_t output_size = std::min<std::size_t>(output_host.elementSize(), lab_deploy::kActionDim);

        std::array<float, lab_deploy::kActionDim> raw{};
        for (std::size_t i = 0; i < output_size; ++i) {
            raw[i] = output[i];
        }

        // 观测裁剪后的 raw 用于下一帧 last_action 观测; 同口径缩放后下发 motors。
        const auto raw_clipped = lab_deploy::clip_obs_action(raw);
        const auto model_motor_cmd = lab_deploy::raw_to_motor_commands(raw_clipped);
        // POLICY authority decides whether the native command is published;
        // force_zero_policy_commands_ changes only its six values.  This lets
        // the bench test exercise POLICY -> bridge -> lower with an actual
        // zero command instead of silently starving the lower watchdog.
        const bool policy_authority_enabled =
            lab_deploy::should_publish_native_policy(policy_output_enabled_);
        const auto published_motor_cmd =
            lab_deploy::select_published_motor_commands(
                model_motor_cmd, policy_output_enabled_, force_zero_policy_commands_);
        if (!policy_authority_enabled || force_zero_policy_commands_) {
            last_obs_action_.fill(0.0f);
        } else {
            last_obs_action_ = raw_clipped;
        }

        ++policy_output_count_;
        maybe_log_policy_output(
            raw, raw_clipped, model_motor_cmd, published_motor_cmd,
            policy_authority_enabled);

        command_msg_.data.assign(published_motor_cmd.begin(), published_motor_cmd.end());
        command_pub_->publish(command_msg_);
        // The native command topic represents actual Policy authority. Keep the
        // legacy zero stream for readiness diagnostics, but do not make bridge
        // rejection counters grow while the supervisor is in Standby.
        if (policy_authority_enabled) {
            esd_link_msgs::msg::PolicyCommand stamped;
            stamped.header.stamp = now();
            stamped.header.frame_id = "base_link";
            stamped.source_state_sample_seq = source_state_sample_seq_;
            stamped.inference_start_monotonic_ns = inference_start_ns;
            stamped.inference_end_monotonic_ns = monotonic_ns();
            std::copy(published_motor_cmd.begin(), published_motor_cmd.end(), stamped.action.begin());
            stamped_command_pub_->publish(stamped);
        }
    }

    void maybe_log_policy_output(
        const std::array<float, lab_deploy::kActionDim>& raw,
        const std::array<float, lab_deploy::kActionDim>& raw_clipped,
        const std::array<float, lab_deploy::kActionDim>& model_motor_cmd,
        const std::array<float, lab_deploy::kActionDim>& published_motor_cmd,
        bool policy_output_allowed) {
        if (!log_policy_outputs_) {
            return;
        }
        if (policy_output_count_ > 5 &&
            policy_output_count_ % static_cast<std::uint64_t>(policy_log_interval_ticks_) != 0) {
            return;
        }

        RCLCPP_INFO(
            get_logger(),
            "[policy-output #%llu] mode=%s cmd_in=[vx=%.3f yaw=%.3f h=%.3f] "
            "base_ang_vel=[%.3f %.3f %.3f] "
            "gravity=[%.3f %.3f %.3f] "
            "wing_rad=[%.3f %.3f] wing_obs=[%.3f %.3f] "
            "wing_vel_rad_s=[%.3f %.3f] wing_vel_obs=[%.3f %.3f] "
            "joint_pos=[%.3f %.3f %.3f %.3f %.3f %.3f] "
            "raw=[%.3f %.3f %.3f %.3f %.3f %.3f] "
            "raw_clip=[%.3f %.3f %.3f %.3f %.3f %.3f] "
            "model_motor_cmd=[%.3f %.3f %.3f %.3f %.3f %.3f] "
            "published_motor_cmd=[%.3f %.3f %.3f %.3f %.3f %.3f]",
            static_cast<unsigned long long>(policy_output_count_),
            policy_output_allowed ? "policy" : "standby",
            command_[0], command_[1], command_[2],
            base_ang_vel_[0], base_ang_vel_[1], base_ang_vel_[2],
            projected_gravity_[0], projected_gravity_[1], projected_gravity_[2],
            wing_angle_rad_[0], wing_angle_rad_[1], wing_angle_obs_[0], wing_angle_obs_[1],
            wing_ang_vel_rad_s_[0], wing_ang_vel_rad_s_[1], wing_vel_obs_[0], wing_vel_obs_[1],
            joint_pos_[0], joint_pos_[1], joint_pos_[2], joint_pos_[3], joint_pos_[4], joint_pos_[5],
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
            raw_clipped[0], raw_clipped[1], raw_clipped[2],
            raw_clipped[3], raw_clipped[4], raw_clipped[5],
            model_motor_cmd[0], model_motor_cmd[1], model_motor_cmd[2],
            model_motor_cmd[3], model_motor_cmd[4], model_motor_cmd[5],
            published_motor_cmd[0], published_motor_cmd[1], published_motor_cmd[2],
            published_motor_cmd[3], published_motor_cmd[4], published_motor_cmd[5]);
    }

    bool button_pressed(const sensor_msgs::msg::Joy& msg, int button_index) const {
        return button_index >= 0 &&
               static_cast<std::size_t>(button_index) < msg.buttons.size() &&
               msg.buttons[static_cast<std::size_t>(button_index)] == 1;
    }

    std::string model_name_;
    std::string model_path_;
    std::string wing_angle_topic_ = "/policy/wing_angles";
    double wing_angle_timeout_seconds_ = 0.1;
    double imu_timeout_seconds_ = 0.2;
    std::vector<int> motor_ids_;
    std::vector<std::string> joint_names_;
    float default_height_command_ = 0.24f;
    float clip_observations_ = 100.0f;
    int intra_threads_ = 1;
    bool debug_mode_ = false;
    float dt_ = 0.0025f;
    int decimation_ = 8;
    bool log_policy_outputs_ = true;
    int policy_log_interval_ticks_ = 50;
    bool dummy_observation_mode_ = false;
    bool force_zero_policy_commands_ = false;
    bool joy_policy_gate_enabled_ = false;
    bool joy_policy_start_enabled_ = false;
    bool use_lower_state_ = true;
    int joy_policy_enable_button_ = 2;
    int joy_policy_standby_button_ = 3;
    std::array<float, 3> imu_to_base_signs_{-1.0f, -1.0f, 1.0f};
    bool policy_output_enabled_ = false;
    bool last_enable_button_pressed_ = false;
    bool last_standby_button_pressed_ = false;
    std::uint64_t policy_output_count_ = 0;
    std::uint32_t source_state_sample_seq_ = 0;
    std::uint32_t last_inferred_state_sample_seq_ = 0;
    bool have_inferred_state_sample_seq_ = false;
    bool lower_state_ready_ = false;

    std::unique_ptr<MNN::Interpreter> net_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
    MNN::Tensor* output_tensor_ = nullptr;
    bool model_ready_ = false;

    std::array<float, 3> base_ang_vel_{0.0f, 0.0f, 0.0f};
    std::array<float, 3> projected_gravity_{0.0f, 0.0f, -1.0f};
    std::array<float, lab_deploy::kActionDim> joint_pos_{};
    std::array<float, lab_deploy::kActionDim> joint_vel_{};
    std::array<float, lab_deploy::kActionDim> last_obs_action_{};
    std::array<float, lab_deploy::kWingAngleDim> wing_angle_rad_{};
    std::array<float, lab_deploy::kWingAngleDim> wing_angle_obs_{};
    std::array<float, lab_deploy::kWingVelDim> wing_ang_vel_rad_s_{};
    std::array<float, lab_deploy::kWingVelDim> wing_vel_obs_{};
    std::array<float, 3> command_{0.0f, 0.0f, 0.24f};
    lab_deploy::ObservationHistory history_{};
    bool imu_received_ = false;
    bool joint_received_ = false;
    bool wing_angle_received_ = false;
    bool wing_angle_stale_latched_ = false;
    std::chrono::steady_clock::time_point last_wing_angle_time_{};
    std::chrono::steady_clock::time_point last_imu_time_{};

    std_msgs::msg::Float32MultiArray command_msg_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<esd_link_msgs::msg::LowerState>::SharedPtr lower_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr wing_angle_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr command_pub_;
    rclcpp::Publisher<esd_link_msgs::msg::PolicyCommand>::SharedPtr stamped_command_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr mode_pub_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_policy_service_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LabInferenceNode>());
    rclcpp::shutdown();
    return 0;
}
