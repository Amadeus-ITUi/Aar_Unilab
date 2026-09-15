#pragma once
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <map>
#include <algorithm>
#include <future>
#include <cmath>
#include <cstdint>
#include <motors/srv/control_motor.hpp>
#include <motors/srv/read_motors.hpp>
#include <motors/srv/reset_motors.hpp>
#include <motors/srv/set_zeros.hpp>
#include <motors/srv/clear_errors.hpp>
#include <motors/srv/identify_motor_id.hpp>
#include <motors/srv/scan_motors.hpp>
#include <motors/srv/set_motor_gains.hpp>
#include <motors/srv/run_safety_check.hpp>
#include <motors/srv/set_operational_state.hpp>
#include <motors/msg/motor_runtime_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "motor_driver.hpp"
#include "timer.hpp"

// 关节指令结构
struct JointCommand {
    float position = 0.0f;
    float velocity = 0.0f;
    float kp = 0.0f;
    float kd = 0.0f;
    float effort = 0.0f;
};

// 指令插值器类（用于将低频策略指令平滑上采样到高频控制频率）
class CommandInterpolator {
public:
    CommandInterpolator() = default;

    // 更新新的目标指令（从当前命令插值到新目标）
    void update_target(const JointCommand& current, const JointCommand& target, int interpolation_steps) {
        start_cmd_ = current;
        target_cmd_ = target;
        total_steps_ = interpolation_steps;
        current_step_ = 0;
    }

    // 获取下一帧的插值指令（在200Hz控制循环中调用）
    // 注意：此函数会修改 current_step_，应在锁保护下调用
    JointCommand get_next_command() {
        if (current_step_ >= total_steps_) {
            // 插值完成，保持目标值（直到收到新的目标指令）
            return target_cmd_;
        }

        // 递增步数并计算插值参数 t ∈ [0, 1]
        current_step_++;
        float t = static_cast<float>(current_step_) / static_cast<float>(total_steps_);
        // 限制 t 不超过 1.0，防止数值误差
        t = std::min(t, 1.0f);
        
        JointCommand cmd;
        
        // 线性插值 (Linear Interpolation)
        cmd.position = start_cmd_.position + (target_cmd_.position - start_cmd_.position) * t;
        
        // 速度、力矩、增益直接使用目标值（简化处理，也可以做插值）
        cmd.velocity = target_cmd_.velocity;
        cmd.effort = target_cmd_.effort;
        cmd.kp = target_cmd_.kp;
        cmd.kd = target_cmd_.kd;
        
        return cmd;
    }

    // 检查是否完成插值
    bool is_complete() const {
        return current_step_ >= total_steps_;
    }

private:
    JointCommand start_cmd_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    JointCommand target_cmd_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int total_steps_{1};
    int current_step_{0};
};

class MotorsNode : public rclcpp::Node {
   public:
    MotorsNode() : Node("motors_node") {
        // 声明参数
        this->declare_parameter<std::vector<std::string>>(
            "can_interfaces",
            std::vector<std::string>{"can0", "can0", "can0", "can0", "can0", "can0"});
        this->declare_parameter<std::string>("motors_type", "RS00");
        this->declare_parameter<std::vector<int>>("motor_ids", std::vector<int>{1, 2, 3, 4, 5, 6});
        this->declare_parameter<std::vector<std::string>>("motor_types", 
            std::vector<std::string>{"RS05", "RS05", "RS00", "RS05", "RS05", "RS00"});
        this->declare_parameter<int>("master_id", 0);
        this->declare_parameter<std::vector<float>>("kp", 
            std::vector<float>{2.0, 8.0, 0.0, 2.0, 8.0, 0.0});
        this->declare_parameter<std::vector<float>>("kd", 
            std::vector<float>{0.1, 0.8, 0.05, 0.1, 0.8, 0.05});
        // Normal-deploy zero reference:
        // [-0.82, 1.07] + [-0.10020, -0.08662]
        // = [-0.92020, 0.98338]. Keep this fallback aligned with motors.yaml.
        this->declare_parameter<std::vector<float>>("joint_default_angle",
            std::vector<float>{-0.92020, 0.98338, 0.0,
                               -0.92020, 0.98338, 0.0});
        this->declare_parameter<std::vector<bool>>("flipped_motors",
            std::vector<bool>{true, false, true, false, true, false});
        this->declare_parameter<std::vector<bool>>("vel_only_mode",
            std::vector<bool>{false, false, true, false, false, true});  // true=推理 action 解读为速度目标（kp=0,kd=kd）
        // Match motors.yaml: MuJoCo-relative thigh limits and calf limits
        // shifted for the new normal-deploy zero reference.
        // 格式：每个电机两个值 [min1, max1, min2, max2, ...]
        this->declare_parameter<std::vector<float>>("joint_position_limits",
            std::vector<float>{-1.23, 0.77,         // Motor 1 (L_thigh)
                               -0.98338, 1.51662,   // Motor 2 (L_calf)
                               -6.28, 6.28,         // Motor 3 (L_foot)
                               -1.23, 0.77,         // Motor 4 (R_thigh)
                               -0.98338, 1.51662,   // Motor 5 (R_calf)
                               -6.28, 6.28});       // Motor 6 (R_foot)
        this->declare_parameter<bool>("debug_mode", false);
        this->declare_parameter<bool>("auto_zero_on_start", false);
        this->declare_parameter<bool>("start_disarmed", true);
        this->declare_parameter<double>("standby_ramp_seconds", 3.0);
        this->declare_parameter<double>("soft_disarm_seconds", 3.0);
        this->declare_parameter<double>("disarmed_query_frequency", 20.0);
        this->declare_parameter<bool>("enable_per_motor_topics", true);
        this->declare_parameter<bool>("publish_joint_commands", false);  // 是否发布动作指令到话题（用于Gazebo验证）
        this->declare_parameter<bool>("use_gazebo_states", false);  // 是否使用Gazebo的状态信息
        // Only disable the legacy A=init/deinit and B=reset shortcuts.
        // The /set_zeros service and the dedicated joystick zero script remain available.
        this->declare_parameter<bool>("enable_joy_motor_buttons", false);
        this->declare_parameter<int>("offline_threshold", 200);  // 连续多少次发送后未收到合法反馈才判定离线
        // 核心频率设置：腿轮与翼电机统一为 200Hz。
        this->declare_parameter<float>("control_frequency", 200.0f);  // 控制循环频率 (Hz)
        // publish_rate_policy_joint_states 与控制频率一致，推理节点每 20ms 使用最新状态。
        this->declare_parameter<float>("publish_rate_policy_joint_states", 200.0f);  // /policy/joint_states 发布频率 (Hz)
        // publish_rate_per_motor: 降低到 50Hz 以节省 CPU（仅用于调试/Rviz 显示）
        this->declare_parameter<float>("publish_rate_per_motor", 50.0f);  // /motor/motor_{id}/state 发布频率 (Hz)
        // publish_rate_joint_commands: 降低到 20Hz（通常不需要，用于 Gazebo 验证）
        this->declare_parameter<float>("publish_rate_joint_commands", 20.0f);  // /motor/joint_commands 发布频率 (Hz)
        // policy_frequency: 策略/推理节点发布指令的频率 (Hz)，用于计算插值步数
        this->declare_parameter<float>("policy_frequency", 50.0f);  // 策略频率 (Hz)，默认50Hz
        // enable_interpolation: 是否启用线性插值（50Hz->200Hz），设为false时直接使用策略指令
        this->declare_parameter<bool>("enable_interpolation", true);  // 是否启用插值，默认启用
        // 收到第一帧策略后，等待一小段时间再真正下发到电机，让策略历史观测先稳定。
        this->declare_parameter<float>("policy_motor_warmup_sec", 0.0f);

        // 获取参数
        motors_type_ = this->get_parameter("motors_type").as_string();
        std::vector<int64_t> motor_ids_tmp = this->get_parameter("motor_ids").as_integer_array();
        motor_ids_.assign(motor_ids_tmp.begin(), motor_ids_tmp.end());
        can_interfaces_ = this->get_parameter("can_interfaces").as_string_array();
        
        std::vector<std::string> motor_types = this->get_parameter("motor_types").as_string_array();
        if (motor_types.size() != motor_ids_.size()) {
            RCLCPP_WARN(this->get_logger(), "motor_types size mismatch, using default type");
            motor_types.resize(motor_ids_.size(), motors_type_);
        }
        
        std::vector<double> kp_tmp = this->get_parameter("kp").as_double_array();
        std::vector<double> kd_tmp = this->get_parameter("kd").as_double_array();
        std::vector<double> default_angle_tmp = this->get_parameter("joint_default_angle").as_double_array();
        std::vector<bool> flipped_tmp = this->get_parameter("flipped_motors").as_bool_array();
        std::vector<bool> torque_only_tmp = this->get_parameter("vel_only_mode").as_bool_array();
        std::vector<double> position_limits_tmp = this->get_parameter("joint_position_limits").as_double_array();
        debug_mode_ = this->get_parameter("debug_mode").as_bool();
        auto_zero_on_start_ = this->get_parameter("auto_zero_on_start").as_bool();
        start_disarmed_ = this->get_parameter("start_disarmed").as_bool();
        standby_ramp_seconds_ = this->get_parameter("standby_ramp_seconds").as_double();
        soft_disarm_seconds_ = this->get_parameter("soft_disarm_seconds").as_double();
        disarmed_query_frequency_ = this->get_parameter("disarmed_query_frequency").as_double();
        enable_per_motor_topics_ = this->get_parameter("enable_per_motor_topics").as_bool();
        publish_joint_commands_ = this->get_parameter("publish_joint_commands").as_bool();
        use_gazebo_states_ = this->get_parameter("use_gazebo_states").as_bool();
        enable_joy_motor_buttons_ = this->get_parameter("enable_joy_motor_buttons").as_bool();
        offline_threshold_ = std::max(1, static_cast<int>(this->get_parameter("offline_threshold").as_int()));
        this->get_parameter("control_frequency", control_frequency_);
        this->get_parameter("publish_rate_policy_joint_states", publish_rate_policy_joint_states_);
        this->get_parameter("publish_rate_per_motor", publish_rate_per_motor_);
        this->get_parameter("publish_rate_joint_commands", publish_rate_joint_commands_);
        this->get_parameter("policy_frequency", policy_frequency_);
        enable_interpolation_ = this->get_parameter("enable_interpolation").as_bool();
        this->get_parameter("policy_motor_warmup_sec", policy_motor_warmup_sec_);
        if (policy_motor_warmup_sec_ < 0.0f) {
            RCLCPP_WARN(this->get_logger(), "policy_motor_warmup_sec %.3f < 0, clamping to 0", policy_motor_warmup_sec_);
            policy_motor_warmup_sec_ = 0.0f;
        }
        
        // 参数验证
        if (can_interfaces_.size() != motor_ids_.size()) {
            RCLCPP_FATAL(this->get_logger(),
                         "can_interfaces size (%zu) must equal motor_ids size (%zu)",
                         can_interfaces_.size(), motor_ids_.size());
            rclcpp::shutdown();
            return;
        }
        for (size_t i = 0; i < can_interfaces_.size(); ++i) {
            if (can_interfaces_[i].empty()) {
                RCLCPP_FATAL(this->get_logger(), "can_interfaces[%zu] is empty", i);
                rclcpp::shutdown();
                return;
            }
        }
        if (control_frequency_ <= 0.0f) {
            RCLCPP_FATAL(this->get_logger(), "Invalid control_frequency: %.1f Hz", control_frequency_);
            rclcpp::shutdown();
            return;
        }
        if (!std::isfinite(standby_ramp_seconds_) || standby_ramp_seconds_ <= 0.0 ||
            !std::isfinite(soft_disarm_seconds_) || soft_disarm_seconds_ <= 0.0 ||
            !std::isfinite(disarmed_query_frequency_) || disarmed_query_frequency_ <= 0.0 ||
            disarmed_query_frequency_ > control_frequency_) {
            RCLCPP_FATAL(this->get_logger(),
                         "Invalid standby/disarmed timing: ramp=%.3fs soft_disarm=%.3fs query=%.3fHz",
                         standby_ramp_seconds_, soft_disarm_seconds_, disarmed_query_frequency_);
            rclcpp::shutdown();
            return;
        }
        if (publish_rate_policy_joint_states_ <= 0.0f || publish_rate_policy_joint_states_ > control_frequency_) {
            RCLCPP_WARN(this->get_logger(), "Invalid publish_rate_policy_joint_states (%.1f Hz), clamping to control_frequency (%.1f Hz)", 
                       publish_rate_policy_joint_states_, control_frequency_);
            publish_rate_policy_joint_states_ = control_frequency_;
        }
        if (publish_rate_per_motor_ <= 0.0f || publish_rate_per_motor_ > control_frequency_) {
            RCLCPP_WARN(this->get_logger(), "Invalid publish_rate_per_motor (%.1f Hz), clamping to control_frequency (%.1f Hz)", 
                       publish_rate_per_motor_, control_frequency_);
            publish_rate_per_motor_ = control_frequency_;
        }
        if (publish_rate_joint_commands_ <= 0.0f) {
            RCLCPP_WARN(this->get_logger(), "Invalid publish_rate_joint_commands (%.1f Hz), using default 20.0 Hz", 
                       publish_rate_joint_commands_);
            publish_rate_joint_commands_ = 20.0f;
        }
        
        // 预计算发布间隔（避免每次调用都计算）
        policy_interval_ = static_cast<int>(control_frequency_ / publish_rate_policy_joint_states_);
        if (policy_interval_ < 1) policy_interval_ = 1;
        per_motor_interval_ = static_cast<int>(control_frequency_ / publish_rate_per_motor_);
        if (per_motor_interval_ < 1) per_motor_interval_ = 1;
        
        master_id_ = this->get_parameter("master_id").as_int();
        
        // 转换参数
        kp_.resize(motor_ids_.size());
        kd_.resize(motor_ids_.size());
        joint_default_angle_.resize(motor_ids_.size());
        flipped_motors_.resize(motor_ids_.size());
        vel_only_mode_.resize(motor_ids_.size());
        joint_position_limits_.resize(motor_ids_.size());
        
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            kp_[i] = i < kp_tmp.size() ? static_cast<float>(kp_tmp[i]) : 5.5f;
            kd_[i] = i < kd_tmp.size() ? static_cast<float>(kd_tmp[i]) : 0.5f;
            joint_default_angle_[i] = i < default_angle_tmp.size() ? static_cast<float>(default_angle_tmp[i]) : 0.0f;
            flipped_motors_[i] = i < flipped_tmp.size() ? flipped_tmp[i] : false;
            vel_only_mode_[i] = i < torque_only_tmp.size() ? torque_only_tmp[i] : false;
            
            // 解析位置限制：每个电机两个值 [min, max]
            size_t limit_idx = i * 2;
            if (limit_idx + 1 < position_limits_tmp.size()) {
                float pos_min = static_cast<float>(position_limits_tmp[limit_idx]);
                float pos_max = static_cast<float>(position_limits_tmp[limit_idx + 1]);
                joint_position_limits_[i] = std::make_pair(pos_min, pos_max);
                RCLCPP_INFO(this->get_logger(), "电机 %d 位置限制: [%.3f, %.3f] rad", 
                           motor_ids_[i], pos_min, pos_max);
            } else {
                // 如果没有提供限制，使用默认值（无限制，但会被驱动器层面的限制约束）
                joint_position_limits_[i] = std::make_pair(-12.57f, 12.57f);
                RCLCPP_WARN(this->get_logger(), "电机 %d 未提供位置限制，使用默认值 (-12.57, 12.57)", motor_ids_[i]);
            }
        }
        default_kp_ = kp_;
        default_kd_ = kd_;

        RCLCPP_INFO(this->get_logger(), "CAN interfaces per motor: [%s]",
                    [&]() {
                        std::string s;
                        for (size_t i = 0; i < can_interfaces_.size(); ++i) {
                            if (i > 0) s += ", ";
                            s += "motor_" + std::to_string(motor_ids_[i]) + "->" + can_interfaces_[i];
                        }
                        return s;
                    }().c_str());
        RCLCPP_INFO(this->get_logger(), "Motor IDs: [%s]", 
                    [&]() {
                        std::string s;
                        for (size_t i = 0; i < motor_ids_.size(); ++i) {
                            if (i > 0) s += ", ";
                            s += std::to_string(motor_ids_[i]);
                        }
                        return s;
                    }().c_str());

        // 创建电机驱动（非debug模式）
        motors_.resize(motor_ids_.size());
        // 初始化互斥锁：使用智能指针（因为 mutex 不可复制/移动）
        motor_mutexes_.clear();
        motor_mutexes_.reserve(motor_ids_.size());
        // [新增] 初始化连接状态，默认先全部设为 false（等待扫描后更新）
        is_motor_connected_.resize(motor_ids_.size(), false);
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            motor_mutexes_.push_back(std::make_unique<std::shared_mutex>());
        }
        if (!debug_mode_) {
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            std::string motor_type = i < motor_types.size() ? motor_types[i] : motors_type_;
            motors_[i] = MotorDriver::MotorCreate(
                    motor_ids_[i], can_interfaces_[i].c_str(), motor_type, master_id_, 0);
            RCLCPP_INFO(this->get_logger(), "Created motor %d (type: %s, CAN: %s)",
                        motor_ids_[i], motor_type.c_str(), can_interfaces_[i].c_str());
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "debug_mode enabled: motors will be simulated with sine waves.");
        }

        // 初始化插值器和当前指令（用于50Hz推理 -> 200Hz控制）
        interpolators_.resize(motor_ids_.size());
        current_commands_.resize(motor_ids_.size());
        
        // 初始化 current_commands_ 为默认位置（已应用翻转）
        // 注意：这里需要应用翻转，因为这是最终发送给电机的绝对位置
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            float absolute_pos = joint_default_angle_[i];
            if (flipped_motors_[i]) {
                absolute_pos = -absolute_pos;
            }
            current_commands_[i].position = absolute_pos;
            current_commands_[i].velocity = 0.0f;
            current_commands_[i].effort = 0.0f;
            current_commands_[i].kp = (i < vel_only_mode_.size() && vel_only_mode_[i]) ? 0.0f : kp_[i];
            current_commands_[i].kd = (i < vel_only_mode_.size() && vel_only_mode_[i]) ? 0.0f : kd_[i];
            interpolators_[i].update_target(current_commands_[i], current_commands_[i], 1);
        }
        
        // ===== 滤波与插值系数打印 =====
        RCLCPP_INFO(this->get_logger(), "=== 电机控制滤波/插值参数 ===");
        if (enable_interpolation_) {
            RCLCPP_INFO(this->get_logger(), "插值器已启用：策略频率=%.1f Hz, 控制频率=%.1f Hz, 插值步数=%d",
                   policy_frequency_, control_frequency_, static_cast<int>(control_frequency_ / policy_frequency_));
            RCLCPP_INFO(this->get_logger(), "插值方法：线性插值（位置线性，速度/力矩/增益使用目标值）");
        } else {
            RCLCPP_INFO(this->get_logger(), "插值器已禁用：直接使用策略指令（无插值）");
        }
        RCLCPP_INFO(this->get_logger(), "电机增益 - Kp: %s, Kd: %s",
                   [&](){
                       std::string kp_str;
                       for(size_t i = 0; i < kp_.size(); i++){
                           if(i > 0) kp_str += ", ";
                           kp_str += std::to_string(kp_[i]);
                       }
                       return kp_str;
                   }().c_str(),
                   [&](){
                       std::string kd_str;
                       for(size_t i = 0; i < kd_.size(); i++){
                           if(i > 0) kd_str += ", ";
                           kd_str += std::to_string(kd_[i]);
                       }
                       return kd_str;
                   }().c_str());
        RCLCPP_INFO(this->get_logger(), "离线阈值：%d 次无响应", offline_threshold_);
        RCLCPP_INFO(this->get_logger(), "=======================================");

        // QoS配置
        auto sensor_data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        auto control_command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();

        // 订阅推理节点发布的Policy命令（Float32MultiArray格式）
        policy_command_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/policy/commands", control_command_qos,
            std::bind(&MotorsNode::policy_command_callback, this, std::placeholders::_1));
        inference_mode_subscription_ = this->create_subscription<std_msgs::msg::UInt8>(
            "/inference/runtime_mode", rclcpp::QoS(1).reliable().transient_local(),
            std::bind(&MotorsNode::inference_mode_callback, this, std::placeholders::_1));

        // 订阅手柄
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", sensor_data_qos, std::bind(&MotorsNode::subs_joy_callback, this, std::placeholders::_1));

        // 为每个电机创建状态发布器（可选，用于调试与单电机监控）
        if (enable_per_motor_topics_) {
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            std::string topic_name = "/motor/motor_" + std::to_string(motor_ids_[i]) + "/state";
            motor_state_publishers_[motor_ids_[i]] = 
                this->create_publisher<sensor_msgs::msg::JointState>(topic_name, sensor_data_qos);
            }
        }

        // 发布整合的关节状态（供推理节点使用）
        joint_states_policy_publisher_ = 
            this->create_publisher<sensor_msgs::msg::JointState>("/policy/joint_states", sensor_data_qos);

        // 发布动作指令到话题（用于Gazebo验证，在编码CAN之前）
        if (publish_joint_commands_) {
            joint_commands_publisher_ = 
                this->create_publisher<sensor_msgs::msg::JointState>("/motor/joint_commands", control_command_qos);
            RCLCPP_INFO(this->get_logger(), "Joint commands will be published to /motor/joint_commands for Gazebo verification");
        }

        // 订阅Gazebo中的机器人状态信息
        if (use_gazebo_states_) {
            gazebo_joint_states_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
                "/joint_states", sensor_data_qos,
                std::bind(&MotorsNode::gazebo_joint_states_callback, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Subscribed to /joint_states from Gazebo for robot state feedback");
        }

        // 服务
        control_motor_service_ = this->create_service<motors::srv::ControlMotor>(
            "control_motor",
            std::bind(&MotorsNode::control_motor_srv, this, std::placeholders::_1, std::placeholders::_2));
        reset_motors_service_ = this->create_service<motors::srv::ResetMotors>(
            "reset_motors",
            std::bind(&MotorsNode::reset_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_motors_service_ = this->create_service<motors::srv::ReadMotors>(
            "read_motors",
            std::bind(&MotorsNode::read_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_zeros_service_ = this->create_service<motors::srv::SetZeros>(
            "set_zeros",
            std::bind(&MotorsNode::set_zeros_srv, this, std::placeholders::_1, std::placeholders::_2));
        clear_errors_service_ = this->create_service<motors::srv::ClearErrors>(
            "clear_errors",
            std::bind(&MotorsNode::clear_errors_srv, this, std::placeholders::_1, std::placeholders::_2));
        identify_motor_id_service_ = this->create_service<motors::srv::IdentifyMotorID>(
            "identify_motor_id",
            std::bind(&MotorsNode::identify_motor_id_srv, this, std::placeholders::_1, std::placeholders::_2));
        scan_motors_service_ = this->create_service<motors::srv::ScanMotors>(
            "scan_motors",
            std::bind(&MotorsNode::scan_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_motor_gains_service_ = this->create_service<motors::srv::SetMotorGains>(
            "set_motor_gains",
            std::bind(&MotorsNode::set_motor_gains_srv, this, std::placeholders::_1, std::placeholders::_2));
        run_safety_check_service_ = this->create_service<motors::srv::RunSafetyCheck>(
            "run_safety_check",
            std::bind(&MotorsNode::run_safety_check_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_operational_state_service_ =
            this->create_service<motors::srv::SetOperationalState>(
                "/motors/set_operational_state",
                std::bind(&MotorsNode::set_operational_state_srv, this,
                          std::placeholders::_1, std::placeholders::_2));
        soft_disarm_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/motors/soft_disarm",
            std::bind(&MotorsNode::soft_disarm_srv, this,
                      std::placeholders::_1, std::placeholders::_2));
        runtime_status_publisher_ =
            this->create_publisher<motors::msg::MotorRuntimeStatus>(
                "/motors/runtime_status", rclcpp::QoS(1).reliable().transient_local());

        // 定时器：周期性控制循环（使用配置的频率）
        auto timer_period = std::chrono::microseconds(static_cast<long long>(1000000.0 / control_frequency_));
        timer_ = this->create_wall_timer(
            timer_period,
            std::bind(&MotorsNode::control_loop, this));
        
        RCLCPP_INFO(this->get_logger(), "Topic publish rates:");
        RCLCPP_INFO(this->get_logger(), "  Control frequency: %.1f Hz", control_frequency_);
        RCLCPP_INFO(this->get_logger(), "  /policy/joint_states: %.1f Hz", publish_rate_policy_joint_states_);
        RCLCPP_INFO(this->get_logger(), "  /motor/motor_*/state: %.1f Hz", publish_rate_per_motor_);
        if (publish_joint_commands_) {
            RCLCPP_INFO(this->get_logger(), "  /motor/joint_commands: %.1f Hz", publish_rate_joint_commands_);
        }
        
        // 执行自动初始化流程（debug_mode下会执行模拟流程，包括模拟的零点校准和安全检查）
        auto_init_sequence();
    }

    ~MotorsNode() {
        // 关闭节点时，确保所有电机失能（安全措施）
        RCLCPP_INFO(this->get_logger(), "正在关闭节点，失能所有电机...");
        disable_all_motors();
        
        if(is_init_.load()){
            deinit_motors();
        }
        RCLCPP_INFO(this->get_logger(), "节点已关闭");
    }

    void policy_command_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg);
    void enter_offline_safe_mode(size_t motor_index, int response_count, const JointCommand& last_command);
    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void gazebo_joint_states_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void inference_mode_callback(const std_msgs::msg::UInt8::SharedPtr msg);
    void publish_motor_state(int motor_index);
    void publish_joint_states_policy();
    void init_motors();
    void deinit_motors();
    bool disable_all_motors();  // 立即失能所有电机（急停/故障安全措施）
    bool set_zeros();
    void control_iteration();
    void clear_errors();
    void reset_motors();
    void control_loop();

    void reset_motors_srv(const std::shared_ptr<motors::srv::ResetMotors::Request> request,
                      std::shared_ptr<motors::srv::ResetMotors::Response> response);
    void read_motors_srv(const std::shared_ptr<motors::srv::ReadMotors::Request> request,
                     std::shared_ptr<motors::srv::ReadMotors::Response> response);
    void set_zeros_srv(const std::shared_ptr<motors::srv::SetZeros::Request> request,
                   std::shared_ptr<motors::srv::SetZeros::Response> response);
    void control_motor_srv(const std::shared_ptr<motors::srv::ControlMotor::Request> request,
                               std::shared_ptr<motors::srv::ControlMotor::Response> response);
    void clear_errors_srv(const std::shared_ptr<motors::srv::ClearErrors::Request> request,
                          std::shared_ptr<motors::srv::ClearErrors::Response> response);
    void identify_motor_id_srv(const std::shared_ptr<motors::srv::IdentifyMotorID::Request> request,
                               std::shared_ptr<motors::srv::IdentifyMotorID::Response> response);
    void scan_motors_srv(const std::shared_ptr<motors::srv::ScanMotors::Request> request,
                         std::shared_ptr<motors::srv::ScanMotors::Response> response);
    void set_motor_gains_srv(const std::shared_ptr<motors::srv::SetMotorGains::Request> request,
                             std::shared_ptr<motors::srv::SetMotorGains::Response> response);
    void run_safety_check_srv(const std::shared_ptr<motors::srv::RunSafetyCheck::Request> request,
                              std::shared_ptr<motors::srv::RunSafetyCheck::Response> response);
    void set_operational_state_srv(
        const std::shared_ptr<motors::srv::SetOperationalState::Request> request,
        std::shared_ptr<motors::srv::SetOperationalState::Response> response);
    void soft_disarm_srv(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void publish_runtime_status(const std::string& message = "");
    void disarmed_iteration();
    bool enter_standby_ramp(std::string* reason);
    bool enter_direct_policy_ready(std::string* reason);
    std::pair<float, float> get_motor_gains(size_t motor_index) const;
    
    // 自动初始化流程
    void auto_init_sequence();
    bool check_can_interface_exists(const std::string& interface);
    bool check_can_interface_up(const std::string& interface);
    void perform_zero_calibration(int times = 3);
    void perform_safety_check(bool wait_for_user = true);  // 安全性检查：以极低增益向默认位置附近±0.2rad范围运动
    
    // 辅助函数：等待用户按Enter键（支持从stdin或/dev/tty读取）
    void wait_for_enter_key();

    // Getter方法
    bool is_interpolation_enabled() const { return enable_interpolation_; }
    float get_control_frequency() const { return control_frequency_; }
    bool startup_failed() const { return startup_failed_; }
    bool joy_motor_buttons_enabled() const { return enable_joy_motor_buttons_; }

   private:
    std::atomic<bool> is_init_{false};
    bool startup_failed_{false};
    bool debug_mode_{false};
    bool auto_zero_on_start_{true};
    bool start_disarmed_{true};
    double standby_ramp_seconds_{3.0};
    double soft_disarm_seconds_{3.0};
    double disarmed_query_frequency_{20.0};
    bool enable_per_motor_topics_{true};
    bool publish_joint_commands_{false};  // 是否发布动作指令到话题（用于Gazebo验证）
    bool use_gazebo_states_{false};  // 是否使用Gazebo的状态信息
    bool enable_joy_motor_buttons_{false};  // 是否允许 motors_node 直接响应手柄 A/RB 初始化/复位
    int offline_threshold_ = 200;  // 连续发送后未收到合法反馈的最大次数，200次≈1秒@200Hz
    
    // 发布频率控制参数：腿轮与翼电机统一为 200Hz。
    float control_frequency_{200.0f};
    float publish_rate_policy_joint_states_{200.0f};
    float publish_rate_per_motor_{50.0f};  // /motor/motor_{id}/state 发布频率 (Hz)，降低到 50Hz 以节省 CPU（仅用于调试）
    float publish_rate_joint_commands_{20.0f};  // /motor/joint_commands 发布频率 (Hz)，降低到 20Hz（通常不需要，用于 Gazebo 验证）
    
    // 发布频率控制计数器
    int control_counter_{0};
    int disarmed_counter_{0};
    std::atomic<std::uint8_t> operational_state_{
        motors::msg::MotorRuntimeStatus::DISARMED};
    std::atomic<bool> operational_transitioning_{false};
    std::string runtime_fault_code_;
    std::mutex runtime_state_mutex_;
    std::mutex operational_service_mutex_;
    
    // 预计算的发布间隔（避免每次调用都计算）
    int policy_interval_{1};
    int per_motor_interval_{1};
    
    // 避免双重请求：一旦 Policy 指令开始更新，立即停止发送主动查询
    // MIT模式的控制指令会自动带回反馈，不需要额外查询
    std::atomic<bool> policy_is_active_{false};  // Policy 是否正在运行（一旦收到第一个指令就设为 true）
    std::atomic<bool> has_policy_command_{false};  // 标记是否已收到第一帧 Policy 指令
    std::atomic<bool> offline_latched_{false};  // 任一电机离线后锁存安全态，禁止策略消息自动恢复
    std::atomic<bool> zeroing_in_progress_{false};  // /set_zeros 同步执行期间禁止其他发控入口
    std::mutex motor_command_mutex_;  // 串行化 set zero 与实时控制/单电机服务的 CAN 发控
    std::atomic<std::chrono::steady_clock::time_point> last_policy_command_time_{
        std::chrono::steady_clock::now()};
    /** 超过该时长(ms)未收到策略消息则对各关节发送 0kp0kd；收到第一帧策略消息后才开始发控，此前不主动发控 */
    static constexpr int POLICY_COMMAND_TIMEOUT_MS = 200;
    float policy_motor_warmup_sec_{0.0f};
    std::atomic<bool> policy_warmup_active_{false};
    // ACTIVE is the no-ramp ground-start state. Ignore standby zero commands
    // until inference explicitly announces POLICY for the first time.
    std::atomic<bool> direct_policy_started_{false};
    std::atomic<std::chrono::steady_clock::time_point> policy_warmup_start_time_{
        std::chrono::steady_clock::now()};
    
    // 插值器相关（用于50Hz推理 -> 200Hz控制）
    float policy_frequency_{50.0f};  // 策略/推理频率 (Hz)，默认50Hz
    bool enable_interpolation_{false};  // 是否启用插值，默认关闭
    std::vector<CommandInterpolator> interpolators_;  // 每个电机的插值器
    std::vector<JointCommand> current_commands_;  // 当前实际下发给电机的指令（作为下一次插值的起点）
    
    std::string motors_type_;
    std::vector<std::string> can_interfaces_;
    std::vector<int> motor_ids_;
    std::vector<std::shared_ptr<MotorDriver>> motors_;
    // 使用 unique_ptr 包装 shared_mutex（因为 mutex 不可复制/移动）
    std::vector<std::unique_ptr<std::shared_mutex>> motor_mutexes_;
    std::vector<float> kp_, kd_, default_kp_, default_kd_, joint_default_angle_;
    mutable std::mutex gains_mutex_;
    std::vector<bool> flipped_motors_;
    std::vector<bool> vel_only_mode_;   // true=该电机从推理收到的 action 解读为速度目标
    std::vector<bool> is_motor_connected_;  // [新增] 标记电机是否在线（白名单）
    std::vector<std::pair<float, float>> joint_position_limits_;  // 每个电机的相对位置限制 [min, max]
    int master_id_;

    rclcpp::Service<motors::srv::ControlMotor>::SharedPtr control_motor_service_;
    rclcpp::Service<motors::srv::ResetMotors>::SharedPtr reset_motors_service_;
    rclcpp::Service<motors::srv::ReadMotors>::SharedPtr read_motors_service_;
    rclcpp::Service<motors::srv::SetZeros>::SharedPtr set_zeros_service_;
    rclcpp::Service<motors::srv::ClearErrors>::SharedPtr clear_errors_service_;
    rclcpp::Service<motors::srv::IdentifyMotorID>::SharedPtr identify_motor_id_service_;
    rclcpp::Service<motors::srv::ScanMotors>::SharedPtr scan_motors_service_;
    rclcpp::Service<motors::srv::SetMotorGains>::SharedPtr set_motor_gains_service_;
    rclcpp::Service<motors::srv::RunSafetyCheck>::SharedPtr run_safety_check_service_;
    rclcpp::Service<motors::srv::SetOperationalState>::SharedPtr set_operational_state_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr soft_disarm_service_;
    
    std::map<int, rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr> motor_state_publishers_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_policy_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_commands_publisher_;  // 发布动作指令（用于Gazebo）
    rclcpp::Publisher<motors::msg::MotorRuntimeStatus>::SharedPtr runtime_status_publisher_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr policy_command_subscription_;
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr inference_mode_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr gazebo_joint_states_subscription_;  // 订阅Gazebo状态
    rclcpp::TimerBase::SharedPtr timer_;
    
    int last_button0_ = 0, last_button5_ = 0;
};
