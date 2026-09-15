#include "motors_node.hpp"
#include "soft_disarm_logic.hpp"
#include "timer.hpp"
#include "SocketCAN.hpp"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <linux/can.h>

namespace {
// 将位置卷绕到 [pos_min, pos_max] 内（周期 = pos_max - pos_min），用于轮式电机。
// 编码器范围 -12.5~12.5 会映射到例如 -3.14~3.14，超过 3.14 则卷绕为 -3.14 侧。
inline float wrap_position_to_limits(float value, float pos_min, float pos_max) {
    float period = pos_max - pos_min;
    if (period <= 0.f) return value;
    float x = value - pos_min;
    x = std::fmod(x, period);
    if (x < 0.f) x += period;
    return pos_min + x;
}
// 判断是否为轮式电机（位置限制约 2*pi，如 [-3.14, 3.14]）
inline bool is_wheel_joint_limit(float pos_min, float pos_max) {
    return (pos_max - pos_min) >= (6.28f - 0.01f);
}
}
#include <cstring>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <net/if.h>
#include <cstdio>
#include <set>
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <cstdlib>

namespace {
std::string format_float_vector(const std::vector<float>& values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}
}

void MotorsNode::publish_motor_state(int motor_index) {
    // 如果未启用单电机话题，直接返回（节省不必要的消息构造）
    if (!enable_per_motor_topics_) {
        return;
    }

    if (motor_index < 0 || motor_index >= static_cast<int>(motor_ids_.size())) {
        return;
    }
    
    int motor_id = motor_ids_[motor_index];
    
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = this->now();
    msg.header.frame_id = "motor_" + std::to_string(motor_id);
    msg.name = {"motor_" + std::to_string(motor_id)};
    
    float raw_position = 0.0f;
    float raw_velocity = 0.0f;
    float raw_torque = 0.0f;

    if (!debug_mode_ && motor_index < static_cast<int>(motors_.size()) && motors_[motor_index]) {
        // 获取真实电机状态
        auto motor = motors_[motor_index];
        raw_position = motor->get_motor_pos();
        raw_velocity = motor->get_motor_spd();
        raw_torque = motor->get_motor_current();
    } else if (debug_mode_) {
        // 生成正弦波模拟数据
        const float PI = 3.14159265358979323846f;
        double t = this->now().seconds();
        float amp = 0.3f;      // 振幅
        float freq = 0.5f;     // 频率(Hz)
        float phase = static_cast<float>(motor_index) * PI / 4.0f;
        raw_position = amp * std::sin(2.0 * PI * freq * t + phase);
        raw_velocity = static_cast<float>(2.0 * PI * freq * amp *
                          std::cos(2.0 * PI * freq * t + phase));
        raw_torque = 0.0f;
    }
    
    // 应用方向翻转
    float position = flipped_motors_[motor_index] ? -raw_position : raw_position;
    float velocity = flipped_motors_[motor_index] ? -raw_velocity : raw_velocity;
    float torque = flipped_motors_[motor_index] ? -raw_torque : raw_torque;
    
    // 减去默认角度，得到相对位置
    float relative_position = position - joint_default_angle_[motor_index];
    // 轮电机：将编码器返回值（-12.5~12.5）卷绕到 joint_position_limits（如 -3.14~3.14）
    if (motor_index < static_cast<int>(joint_position_limits_.size())) {
        float pmin = joint_position_limits_[motor_index].first;
        float pmax = joint_position_limits_[motor_index].second;
        if (is_wheel_joint_limit(pmin, pmax)) {
            relative_position = wrap_position_to_limits(relative_position, pmin, pmax);
        }
    }

    msg.position = {relative_position};
    msg.velocity = {velocity};
    msg.effort = {torque};
    
    // 发布到对应话题
    if (motor_state_publishers_.find(motor_id) != motor_state_publishers_.end()) {
        motor_state_publishers_[motor_id]->publish(msg);
    }
}

void MotorsNode::publish_joint_states_policy() {
    // 发布整合的关节状态（供推理节点使用）
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = this->now();
    msg.header.frame_id = "base_link";
    
    // 预分配数组大小，避免动态扩容
    msg.name.reserve(motor_ids_.size());
    msg.position.reserve(motor_ids_.size());
    msg.velocity.reserve(motor_ids_.size());
    msg.effort.reserve(motor_ids_.size());
    
    // 按电机顺序收集所有状态
    for (size_t i = 0; i < motor_ids_.size(); ++i) {
        int motor_id = motor_ids_[i];
        float raw_position = 0.0f;
        float raw_velocity = 0.0f;
        float raw_torque = 0.0f;

        if (!debug_mode_ && i < motors_.size() && motors_[i]) {
            auto motor = motors_[i];
            raw_position = motor->get_motor_pos();
            raw_velocity = motor->get_motor_spd();
            raw_torque = motor->get_motor_current();
        } else if (debug_mode_) {
            const float PI = 3.14159265358979323846f;
            double t = this->now().seconds();
            float amp = 0.3f;
            float freq = 0.5f;
            float phase = static_cast<float>(i) * PI / 4.0f;
            raw_position = amp * std::sin(2.0 * PI * freq * t + phase);
            raw_velocity = static_cast<float>(2.0 * PI * freq * amp *
                               std::cos(2.0 * PI * freq * t + phase));
            raw_torque = 0.0f;
        }
        
        // 应用方向翻转
        float position = flipped_motors_[i] ? -raw_position : raw_position;
        float velocity = flipped_motors_[i] ? -raw_velocity : raw_velocity;
        float torque = flipped_motors_[i] ? -raw_torque : raw_torque;
        
        // 减去默认角度，得到相对位置
        float relative_position = position - joint_default_angle_[i];
        // 轮电机：将编码器返回值（-12.5~12.5）卷绕到 joint_position_limits（如 -3.14~3.14）
        if (i < joint_position_limits_.size()) {
            float pmin = joint_position_limits_[i].first;
            float pmax = joint_position_limits_[i].second;
            if (is_wheel_joint_limit(pmin, pmax)) {
                relative_position = wrap_position_to_limits(relative_position, pmin, pmax);
            }
        }

        msg.name.push_back("motor_" + std::to_string(motor_id));
        msg.position.push_back(relative_position);
        msg.velocity.push_back(velocity);
        msg.effort.push_back(torque);
    }
    
    // 发布到推理节点订阅的话题
    if (joint_states_policy_publisher_) {
        joint_states_policy_publisher_->publish(msg);
    }
}

void MotorsNode::policy_command_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg) {
    if(!is_init_.load()){
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Motors are not initialized; /policy/commands is ignored while DISARMED.");
        return;
    }
    if (operational_transitioning_.load()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Standby 进入斜坡尚未完成，暂不接受 /policy/commands");
        return;
    }
    if (operational_state_.load() == motors::msg::MotorRuntimeStatus::ACTIVE &&
        !direct_policy_started_.load()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "倒地自启已使能，等待 inference 明确进入 POLICY；忽略 standby 零动作");
        return;
    }

    if (zeroing_in_progress_.load()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "正在同步设置零点，忽略 /policy/commands");
        return;
    }

    if (offline_latched_.load()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "电机离线安全态已锁存，忽略 /policy/commands；请重新初始化或重新调用 /set_zeros 后再恢复控制");
        return;
    }

    // Serialize policy target updates with the 200 Hz CAN loop and the
    // synchronous soft-disarm transition. Recheck after acquiring the lock so
    // a callback that was queued just before Y cannot overwrite the frozen
    // last command.
    std::unique_lock<std::mutex> command_lock(motor_command_mutex_);
    if (!is_init_.load() || operational_transitioning_.load() ||
        zeroing_in_progress_.load() || offline_latched_.load()) {
        return;
    }

    // ================== 架构变更：50Hz 推理 -> 200Hz 控制 ==================
    // 回调函数现在只负责更新插值器目标，不直接发送 CAN 帧
    // 实际的 200Hz 控制循环在 control_loop() 定时器中执行
    
    // 再次接收到策略消息后恢复正常控制：刷新时间戳并重新激活。
    // 若配置了 policy_motor_warmup_sec，控制循环会先等待预热完成再真正下发到电机。
    const auto now = std::chrono::steady_clock::now();
    bool was_inactive = !policy_is_active_.load();
    bool warmup_already_active = policy_warmup_active_.load();
    policy_is_active_.store(true);
    has_policy_command_.store(true);
    last_policy_command_time_.store(now);
    if (was_inactive) {
        if (policy_motor_warmup_sec_ > 0.0f) {
            if (!warmup_already_active) {
                policy_warmup_start_time_.store(now);
                policy_warmup_active_.store(true);
                RCLCPP_INFO(this->get_logger(),
                            "收到策略消息，进入 %.2fs 预热；预热期间暂不向电机下发策略控制帧",
                            policy_motor_warmup_sec_);
            } else {
                RCLCPP_INFO_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "策略消息短暂中断后恢复，继续当前预热计时，不重置 %.2fs 预热窗口",
                    policy_motor_warmup_sec_);
            }
        } else {
            policy_warmup_active_.store(false);
            RCLCPP_INFO(this->get_logger(), "再次收到策略消息，恢复正常控制");
        }
    }
    
    // 计算插值步数: 200Hz 控制 / 50Hz 推理 = 4 步
    int interpolation_steps = 1;  // 默认插值步数为1（无插值）
    if (enable_interpolation_) {
        interpolation_steps = static_cast<int>(control_frequency_ / policy_frequency_);
    if (interpolation_steps < 1) {
        interpolation_steps = 1;
        RCLCPP_WARN(this->get_logger(), "策略频率 (%.1f Hz) 大于等于控制频率 (%.1f Hz)，插值步数为 1", 
                   policy_frequency_, control_frequency_);
    }
    }
    
    // 处理Policy控制命令（Float32MultiArray格式）
    // 假设data数组按电机顺序排列，每个电机一个值（位置）
    size_t num_commands = msg->data.size();
    for (size_t i = 0; i < motors_.size() && i < num_commands; ++i) {
        // 白名单检查：如果电机不在线，直接跳过
        if (!is_motor_connected_[i]) {
            continue;
        }

        JointCommand target;
        if (i < vel_only_mode_.size() && vel_only_mode_[i]) {
            // 电机 3、6 等：推理下发的 action 解读为速度目标，MIT 模式用 kp=0, kd=kd 阻尼速度控制。
            float raw_velocity = msg->data[i];
            float target_vel = flipped_motors_[i] ? -raw_velocity : raw_velocity;
            float absolute_pos = joint_default_angle_[i];
            if (flipped_motors_[i]) {
                absolute_pos = -absolute_pos;
            }
            target.position = absolute_pos;
            target.velocity = target_vel;
            target.effort = 0.0f;
            auto gains = get_motor_gains(i);
            target.kp = 0.0f;
            target.kd = gains.second;
            
        } else {
            // 其余电机：推理下发的 action 解读为位置（相对角度）
            float raw_target_pos = msg->data[i];
            float target_vel = 0.0f;
            float target_effort = 0.0f;

            // 1. 应用位置限制（在应用默认角度之前，限制相对位置）
            float original_target_pos = raw_target_pos;
            if (i < joint_position_limits_.size()) {
                float pos_min = joint_position_limits_[i].first;
                float pos_max = joint_position_limits_[i].second;
                raw_target_pos = std::max(pos_min, std::min(pos_max, raw_target_pos));
                if (raw_target_pos != original_target_pos) {
                    // 一旦发生动作范围裁切则提醒（节流：前 5 次每次都报，之后每 200 次报一次，避免刷屏）
                    static int limit_warning_count = 0;
                    limit_warning_count++;
                    if (limit_warning_count <= 5 || limit_warning_count % 200 == 0) {
                        RCLCPP_WARN(this->get_logger(),
                            "[动作范围裁切] 电机 %d 位置指令超出 joint_position_limits [%.3f, %.3f]: 原始=%.3f, 裁切后=%.3f",
                            motor_ids_[i], pos_min, pos_max, original_target_pos, raw_target_pos);
                    }
                }
            }

            // 2. 转换到绝对坐标 & 翻转处理
            float absolute_pos = raw_target_pos + joint_default_angle_[i];
            if (flipped_motors_[i]) {
                absolute_pos = -absolute_pos;
                target_vel = -target_vel;
                target_effort = -target_effort;
            }
     
            target.position = absolute_pos;
            target.velocity = target_vel;
            target.effort = target_effort;
            auto gains = get_motor_gains(i);
            target.kp = gains.first;
            target.kd = gains.second;
        }
        // 4. 更新插值器或直接设置命令
        // 注意：这里需要锁保护，防止 Timer 正在读取或修改
        {
            std::unique_lock<std::shared_mutex> lock(*motor_mutexes_[i]);
            if (enable_interpolation_) {
            // 读取当前命令作为插值起点（必须在锁内，确保数据一致性）
            JointCommand current = current_commands_[i];
            interpolators_[i].update_target(current, target, interpolation_steps);
            } else {
                // 关闭插值：直接设置当前命令为目标命令
                current_commands_[i] = target;
            }
        }
    }
}

void MotorsNode::inference_mode_callback(const std_msgs::msg::UInt8::SharedPtr msg) {
    if (msg->data == 1 &&
        operational_state_.load() == motors::msg::MotorRuntimeStatus::ACTIVE) {
        direct_policy_started_.store(true);
        RCLCPP_WARN(
            this->get_logger(),
            "倒地自启收到 inference POLICY 公告，开始接受策略控制帧");
    }
}

void MotorsNode::gazebo_joint_states_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    // 处理Gazebo中的机器人状态信息
    // 这个回调可以用于：
    // 1. 更新电机状态（在debug模式或仿真模式下）
    // 2. 验证控制框架的正确性
    // 3. 记录Gazebo状态用于对比分析
    
    if (!use_gazebo_states_) {
        return;
    }
    
    // 如果使用Gazebo状态且处于debug模式，可以用Gazebo的状态更新/policy/joint_states
    // 这样推理节点可以从Gazebo获取反馈，形成闭环
    if (debug_mode_) {
        // 可以在这里将Gazebo的状态转发到 /policy/joint_states
        // 或者保存Gazebo状态用于后续对比分析
        if (joint_states_policy_publisher_) {
            // 简单转发：将Gazebo的/joint_states转发到/policy/joint_states
            // 注意：需要确保关节名称和顺序匹配
            auto gazebo_msg = *msg;
            gazebo_msg.header.stamp = this->now();
            joint_states_policy_publisher_->publish(gazebo_msg);
        }
    }
}

void MotorsNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    if (!enable_joy_motor_buttons_) {
        last_button0_ = msg->buttons.size() > 0 ? msg->buttons[0] : 0;
        last_button5_ = msg->buttons.size() > 5 ? msg->buttons[5] : 0;
        return;
    }
    if (zeroing_in_progress_.load()) {
        last_button0_ = msg->buttons.size() > 0 ? msg->buttons[0] : 0;
        last_button5_ = msg->buttons.size() > 5 ? msg->buttons[5] : 0;
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "正在同步设置零点，忽略 motors_node 内部手柄电机快捷键");
        return;
    }

    if (msg->buttons.size() > 0 && msg->buttons[0] == 1 && msg->buttons[0] != last_button0_) {
        if (!is_init_.load()){
            // 异步执行初始化，避免阻塞回调
            std::thread([this]() {
                pthread_setname_np(pthread_self(), "motor_init");
                struct sched_param sp{}; sp.sched_priority = 60;  // 较低优先级，不干扰实时控制
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
                init_motors();
                RCLCPP_INFO(this->get_logger(), "Motors initialized.");
            }).detach();
        }else{
            // 异步执行反初始化，避免阻塞回调
            std::thread([this]() {
                pthread_setname_np(pthread_self(), "motor_deinit");
                struct sched_param sp{}; sp.sched_priority = 60;
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
                deinit_motors();
                RCLCPP_INFO(this->get_logger(), "Motors deinitialized.");
            }).detach();
        }
    }
    if (msg->buttons.size() > 5 && msg->buttons[5] == 1 && msg->buttons[5] != last_button5_) {
        // 异步执行重置，避免阻塞回调
        std::thread([this]() {
            pthread_setname_np(pthread_self(), "motor_reset");
            struct sched_param sp{}; sp.sched_priority = 60;
            pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
            reset_motors();
            RCLCPP_INFO(this->get_logger(), "Motors reset.");
        }).detach();
    }
    last_button0_ = msg->buttons.size() > 0 ? msg->buttons[0] : 0;
    last_button5_ = msg->buttons.size() > 5 ? msg->buttons[5] : 0;
}

void MotorsNode::init_motors() {
    if (debug_mode_) {
        // debug模式：不操作真实电机，直接标记已初始化
        offline_latched_.store(false);
        is_init_.store(true);
        return;
    }

    offline_latched_.store(false);
    policy_is_active_.store(false);
    has_policy_command_.store(false);
    policy_warmup_active_.store(false);

    // 初始化是低频操作，可以串行执行
    // [修改] 只有在线的才初始化
    for (size_t i = 0; i < motors_.size(); ++i) {
        if (is_motor_connected_[i]) {
            RCLCPP_INFO(this->get_logger(), "初始化电机 %d（%zu/%zu）...",
                        motor_ids_[i], i + 1, motors_.size());
            motors_[i]->MotorInit();
        }
    }
    
    // 锁外发布初始状态（减少锁持有时间）
    for (size_t i = 0; i < motors_.size(); ++i) {
        if (is_motor_connected_[i]) {
            publish_motor_state(i);
        }
    }
    
    is_init_.store(true);
}

bool MotorsNode::disable_all_motors() {
    // 失能所有电机（无论是否初始化，无论是否debug模式）
    // 这是一个安全措施，确保节点关闭时电机不会继续运行
    bool success = true;
    if (!debug_mode_ && !motors_.empty()) {
        RCLCPP_INFO(this->get_logger(), "失能所有电机...");
        for (size_t i = 0; i < motors_.size(); ++i) {
            // [修改] 只失能在线的电机
            if (motors_[i] && is_motor_connected_[i]) {
                try {
                    motors_[i]->MotorLock();  // 失能电机
                    RCLCPP_INFO(this->get_logger(), "电机 %d 已失能", motor_ids_[i]);
                } catch (const std::exception& e) {
                    success = false;
                    RCLCPP_WARN(this->get_logger(), "失能电机 %d 时出错: %s", motor_ids_[i], e.what());
                }
            }
        }
        // 短暂延迟，确保失能命令发送完成
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } else if (debug_mode_) {
        RCLCPP_INFO(this->get_logger(), "Debug模式：无需失能真实电机");
    }
    return success;
}

void MotorsNode::deinit_motors() {
    if (!debug_mode_) {
        for (size_t i = 0; i < motors_.size(); ++i) {
            // [修改] 只反初始化在线的电机
            if (is_motor_connected_[i]) {
                motors_[i]->MotorDeInit();
            }
        }
    }
    offline_latched_.store(false);
    policy_is_active_.store(false);
    has_policy_command_.store(false);
    policy_warmup_active_.store(false);
    direct_policy_started_.store(false);
    is_init_.store(false);
}

bool MotorsNode::set_zeros() {
    if(!is_init_.load()){
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized, cannot set zeros.");
        return false;
    }
    if (zeroing_in_progress_.exchange(true)) {
        RCLCPP_WARN(this->get_logger(), "Set zeros is already in progress.");
        return false;
    }
    struct ZeroingGuard {
        std::atomic<bool>& flag;
        ~ZeroingGuard() { flag.store(false); }
    } zeroing_guard{zeroing_in_progress_};
    std::lock_guard<std::mutex> command_lock(motor_command_mutex_);

    if (debug_mode_) {
        offline_latched_.store(false);
        return true;
    }

    offline_latched_.store(false);
    policy_is_active_.store(false);
    has_policy_command_.store(false);
    policy_warmup_active_.store(false);

    bool all_success = true;
    int processed_count = 0;
    
    for (size_t i = 0; i < motors_.size(); ++i) {
        // [修改] 只对在线的电机设置零点
        if (!is_motor_connected_[i]) {
            continue;
        }
        try {
            // MotorSetZero内部已经失能电机并切换到速度模式
            const bool motor_success = motors_[i]->MotorSetZero();
            processed_count++;
            if (!motor_success) {
                all_success = false;
                RCLCPP_WARN(this->get_logger(), "电机 %d 零点设置返回失败", motor_ids_[i]);
            }
            
            // 零点设置后，重新使能电机并切换回运控模式
            // 注意：顺序很重要！先切换模式，再使能
            motors_[i]->set_motor_control_mode(0);  // 切换回运控模式（0=运控模式）
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            motors_[i]->MotorUnlock();  // 使能电机
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "电机 %d 零点设置失败: %s", motor_ids_[i], e.what());
            all_success = false;
        }
    }

    if (processed_count == 0) {
        RCLCPP_ERROR(this->get_logger(), "没有在线电机可设置零点");
        return false;
    }
    if (processed_count != static_cast<int>(motor_ids_.size())) {
        RCLCPP_ERROR(this->get_logger(),
                     "零点设置只处理了 %d/%zu 个电机，未达到 6 电机全量完成要求",
                     processed_count, motor_ids_.size());
        return false;
    }

    return all_success;
}

void MotorsNode::reset_motors() {
    if(!is_init_.load()){
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized, cannot reset.");
        return;
    }
    if (zeroing_in_progress_.load()) {
        RCLCPP_WARN(this->get_logger(), "Set zeros is in progress, cannot reset motors.");
        return;
    }
    if (debug_mode_) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> command_lock(motor_command_mutex_);
        if (zeroing_in_progress_.load()) {
            RCLCPP_WARN(this->get_logger(), "Set zeros is in progress, cannot reset motors.");
            return;
        }

        // 使用当前配置增益回到默认位置，避免 /reset_motors 误触发时高 PD 硬拉。
        for (size_t i = 0; i < motors_.size(); ++i) {
            // [修改] 只重置在线的电机
            if (!is_motor_connected_[i]) {
                continue;
            }
            float target_pos = joint_default_angle_[i];
            if (flipped_motors_[i]) {
                target_pos = -target_pos;
            }
            auto gains = get_motor_gains(i);
            motors_[i]->MotorMitModeCmd(target_pos, 0.0f, gains.first, gains.second, 0.0f);
        }
    }
    
    // 锁外发布状态（减少锁持有时间）
    for (size_t i = 0; i < motors_.size(); ++i) {
        if (is_motor_connected_[i]) {
            publish_motor_state(i);
        }
    }
}

void MotorsNode::enter_offline_safe_mode(size_t motor_index, int response_count, const JointCommand& last_command) {
    offline_latched_.store(true);
    operational_state_.store(motors::msg::MotorRuntimeStatus::FAULT);
    {
        std::lock_guard<std::mutex> state_lock(runtime_state_mutex_);
        runtime_fault_code_ = "MOTOR_FEEDBACK_TIMEOUT_" + std::to_string(motor_ids_[motor_index]);
    }
    has_policy_command_.store(false);
    policy_is_active_.store(false);
    policy_warmup_active_.store(false);

    std::ostringstream response_counts;
    response_counts << "[";
    for (size_t k = 0; k < motors_.size(); ++k) {
        if (k > 0) {
            response_counts << ", ";
        }
        response_counts << "motor_" << motor_ids_[k]
                        << "("
                        << (k < can_interfaces_.size() ? can_interfaces_[k] : "unknown")
                        << ")=" << motors_[k]->get_response_count();
    }
    response_counts << "]";

    RCLCPP_ERROR(this->get_logger(),
                 "Motor ID %d is offline, 锁存 0kp0kd 安全模式并忽略后续策略消息. "
                 "response_count=%d/%d can=%s cmd(pos=%.4f vel=%.4f kp=%.3f kd=%.3f effort=%.4f) all_response_counts=%s",
                 motor_ids_[motor_index],
                 response_count,
                 offline_threshold_,
                 motor_index < can_interfaces_.size() ? can_interfaces_[motor_index].c_str() : "unknown",
                 last_command.position,
                 last_command.velocity,
                 last_command.kp,
                 last_command.kd,
                 last_command.effort,
                 response_counts.str().c_str());

    for (size_t j = 0; j < motors_.size(); ++j) {
        if (is_motor_connected_[j]) {
            try {
                motors_[j]->MotorMitModeCmd(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(),
                            "电机 %d 进入0kp0kd安全模式失败: %s",
                            motor_ids_[j], e.what());
            }
        }
    }
}

void MotorsNode::publish_runtime_status(const std::string& message) {
    if (!runtime_status_publisher_) {
        return;
    }
    motors::msg::MotorRuntimeStatus status;
    status.stamp = this->now();
    status.state = operational_state_.load();
    status.motor_ids.reserve(motor_ids_.size());
    status.online.reserve(motor_ids_.size());
    status.position.reserve(motor_ids_.size());
    status.velocity.reserve(motor_ids_.size());
    status.effort.reserve(motor_ids_.size());
    for (size_t i = 0; i < motor_ids_.size(); ++i) {
        status.motor_ids.push_back(motor_ids_[i]);
        const bool online = debug_mode_ ||
            (i < motors_.size() && motors_[i] && is_motor_connected_[i] &&
             motors_[i]->get_response_count() <= offline_threshold_);
        status.online.push_back(online);
        float raw_position = debug_mode_ ? joint_default_angle_[i] : motors_[i]->get_motor_pos();
        float raw_velocity = debug_mode_ ? 0.0f : motors_[i]->get_motor_spd();
        float raw_effort = debug_mode_ ? 0.0f : motors_[i]->get_motor_current();
        const float position = flipped_motors_[i] ? -raw_position : raw_position;
        status.position.push_back(position - joint_default_angle_[i]);
        status.velocity.push_back(flipped_motors_[i] ? -raw_velocity : raw_velocity);
        status.effort.push_back(flipped_motors_[i] ? -raw_effort : raw_effort);
    }
    {
        std::lock_guard<std::mutex> state_lock(runtime_state_mutex_);
        status.fault_code = runtime_fault_code_;
    }
    status.message = message;
    runtime_status_publisher_->publish(status);
}

void MotorsNode::disarmed_iteration() {
    const int query_interval = std::max(
        1, static_cast<int>(std::round(control_frequency_ / disarmed_query_frequency_)));
    ++disarmed_counter_;
    if (disarmed_counter_ % query_interval != 0) {
        return;
    }
    if (!debug_mode_) {
        for (auto& motor : motors_) {
            if (motor) {
                motor->refresh_motor_status();
            }
        }
    }
    publish_joint_states_policy();
    if (enable_per_motor_topics_) {
        for (size_t i = 0; i < motors_.size(); ++i) {
            publish_motor_state(static_cast<int>(i));
        }
    }
    publish_runtime_status("DISARMED: read-only feedback");
}

bool MotorsNode::enter_standby_ramp(std::string* reason) {
    const auto transition_started = std::chrono::steady_clock::now();
    operational_transitioning_.store(true);
    struct TransitionGuard {
        std::atomic<bool>& flag;
        ~TransitionGuard() { flag.store(false); }
    } transition_guard{operational_transitioning_};

    std::vector<float> start_positions(motors_.size(), 0.0f);
    for (size_t i = 0; i < motors_.size(); ++i) {
        if (!debug_mode_ &&
            (!is_motor_connected_[i] || motors_[i]->get_response_count() > offline_threshold_)) {
            if (reason) *reason = "motor " + std::to_string(motor_ids_[i]) + " is offline";
            return false;
        }
        const float raw = debug_mode_ ?
            (flipped_motors_[i] ? -joint_default_angle_[i] : joint_default_angle_[i]) :
            motors_[i]->get_motor_pos();
        if (!std::isfinite(raw)) {
            if (reason) *reason = "motor " + std::to_string(motor_ids_[i]) + " position is invalid";
            return false;
        }
        start_positions[i] = raw;
    }

    try {
        if (!is_init_.load()) {
            const auto initialization_started = std::chrono::steady_clock::now();
            RCLCPP_WARN(
                this->get_logger(),
                "开始 Standby 转换：先串行初始化 1-6 号电机，再执行 %.2f 秒斜坡",
                standby_ramp_seconds_);
            init_motors();
            const double initialization_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - initialization_started).count();
            RCLCPP_INFO(this->get_logger(), "Standby 电机初始化完成，耗时 %.2f 秒",
                        initialization_elapsed);
        }
        RCLCPP_INFO(this->get_logger(), "开始 Standby 位置/PD 斜坡，计划耗时 %.2f 秒",
                    standby_ramp_seconds_);
        const int steps = std::max(1, static_cast<int>(standby_ramp_seconds_ * control_frequency_));
        const auto sleep_duration = std::chrono::duration<double>(1.0 / control_frequency_);
        for (int step = 1; step <= steps && rclcpp::ok(); ++step) {
            const float alpha = static_cast<float>(step) / static_cast<float>(steps);
            const float gain_scale = 0.1f + 0.9f * alpha;
            for (size_t i = 0; i < motors_.size(); ++i) {
                if (!debug_mode_ && motors_[i]->get_response_count() > offline_threshold_) {
                    throw std::runtime_error(
                        "motor " + std::to_string(motor_ids_[i]) + " feedback lost during standby ramp");
                }
                if (!debug_mode_) {
                    const float feedback_raw = motors_[i]->get_motor_pos();
                    if (!std::isfinite(feedback_raw)) {
                        throw std::runtime_error(
                            "motor " + std::to_string(motor_ids_[i]) + " invalid during standby ramp");
                    }
                }
                const float final_position = flipped_motors_[i] ?
                    -joint_default_angle_[i] : joint_default_angle_[i];
                const float target = start_positions[i] + (final_position - start_positions[i]) * alpha;
                if (!std::isfinite(target)) {
                    throw std::runtime_error(
                        "motor " + std::to_string(motor_ids_[i]) + " invalid command during standby ramp");
                }
                const auto gains = get_motor_gains(i);
                const float kp = vel_only_mode_[i] ? 0.0f : gains.first * gain_scale;
                const float kd = gains.second * gain_scale;
                if (!debug_mode_) {
                    motors_[i]->MotorMitModeCmd(target, 0.0f, kp, kd, 0.0f);
                }
            }
            std::this_thread::sleep_for(sleep_duration);
        }
    } catch (const std::exception& error) {
        if (reason) *reason = error.what();
        disable_all_motors();
        is_init_.store(false);
        return false;
    }
    operational_state_.store(motors::msg::MotorRuntimeStatus::STANDBY);
    offline_latched_.store(false);
    publish_runtime_status("Standby ramp complete");
    const double transition_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - transition_started).count();
    RCLCPP_WARN(this->get_logger(), "Standby 转换完成，总耗时 %.2f 秒", transition_elapsed);
    return true;
}

bool MotorsNode::enter_direct_policy_ready(std::string* reason) {
    operational_transitioning_.store(true);
    struct TransitionGuard {
        std::atomic<bool>& flag;
        ~TransitionGuard() { flag.store(false); }
    } transition_guard{operational_transitioning_};

    for (size_t i = 0; i < motors_.size(); ++i) {
        if (!debug_mode_ &&
            (!is_motor_connected_[i] || motors_[i]->get_response_count() > offline_threshold_)) {
            if (reason) *reason = "motor " + std::to_string(motor_ids_[i]) + " is offline";
            return false;
        }
        const float raw = debug_mode_ ?
            (flipped_motors_[i] ? -joint_default_angle_[i] : joint_default_angle_[i]) :
            motors_[i]->get_motor_pos();
        if (!std::isfinite(raw)) {
            if (reason) *reason = "motor " + std::to_string(motor_ids_[i]) + " position is invalid";
            return false;
        }
    }

    try {
        direct_policy_started_.store(false);
        if (!is_init_.load()) {
            RCLCPP_WARN(
                this->get_logger(),
                "开始倒地自启转换：串行初始化 1-6 号电机，不执行默认姿态斜坡");
            init_motors();
        }
    } catch (const std::exception& error) {
        if (reason) *reason = error.what();
        disable_all_motors();
        is_init_.store(false);
        return false;
    }

    operational_state_.store(motors::msg::MotorRuntimeStatus::ACTIVE);
    offline_latched_.store(false);
    publish_runtime_status("ACTIVE: direct policy start ready; waiting for inference POLICY");
    RCLCPP_WARN(
        this->get_logger(),
        "倒地自启电机初始化完成；未执行默认姿态斜坡，等待 inference POLICY 公告");
    return true;
}

void MotorsNode::set_operational_state_srv(
    const std::shared_ptr<motors::srv::SetOperationalState::Request> request,
    std::shared_ptr<motors::srv::SetOperationalState::Response> response) {
    std::lock_guard<std::mutex> service_lock(operational_service_mutex_);
    if (request->target_state == motors::srv::SetOperationalState::Request::DISARMED) {
        disable_all_motors();
        is_init_.store(false);
        has_policy_command_.store(false);
        policy_is_active_.store(false);
        direct_policy_started_.store(false);
        offline_latched_.store(false);
        {
            std::lock_guard<std::mutex> state_lock(runtime_state_mutex_);
            runtime_fault_code_.clear();
        }
        operational_state_.store(motors::msg::MotorRuntimeStatus::DISARMED);
        response->success = true;
        response->message = "motors are DISARMED";
    } else if (request->target_state == motors::srv::SetOperationalState::Request::STANDBY) {
        if (operational_state_.load() == motors::msg::MotorRuntimeStatus::STANDBY) {
            response->success = true;
            response->message = "motors already in STANDBY";
        } else {
            std::string reason;
            response->success = enter_standby_ramp(&reason);
            response->message = response->success ? "motors entered STANDBY" : reason;
            if (!response->success) {
                {
                    std::lock_guard<std::mutex> state_lock(runtime_state_mutex_);
                    runtime_fault_code_ = "STANDBY_RAMP_FAILED";
                }
                operational_state_.store(motors::msg::MotorRuntimeStatus::FAULT);
            }
        }
    } else if (request->target_state == motors::srv::SetOperationalState::Request::ACTIVE) {
        if (operational_state_.load() == motors::msg::MotorRuntimeStatus::ACTIVE) {
            response->success = true;
            response->message = "motors already ACTIVE";
        } else {
            std::string reason;
            response->success = enter_direct_policy_ready(&reason);
            response->message = response->success ?
                "motors are ACTIVE and ready for direct policy start" : reason;
            if (!response->success) {
                {
                    std::lock_guard<std::mutex> state_lock(runtime_state_mutex_);
                    runtime_fault_code_ = "DIRECT_POLICY_ACTIVATION_FAILED";
                }
                operational_state_.store(motors::msg::MotorRuntimeStatus::FAULT);
            }
        }
    } else {
        response->success = false;
        response->message = "unsupported target state";
    }
    response->actual_state = operational_state_.load();
}

void MotorsNode::soft_disarm_srv(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::lock_guard<std::mutex> service_lock(operational_service_mutex_);

    const auto state = operational_state_.load();
    if (state == motors::msg::MotorRuntimeStatus::DISARMED) {
        response->success = true;
        response->message = "motors already DISARMED";
        return;
    }
    if (state != motors::msg::MotorRuntimeStatus::STANDBY &&
        state != motors::msg::MotorRuntimeStatus::ACTIVE) {
        response->success = false;
        response->message = "soft disarm requires STANDBY or ACTIVE motors";
        return;
    }

    operational_transitioning_.store(true);
    struct TransitionGuard {
        std::atomic<bool>& flag;
        ~TransitionGuard() { flag.store(false); }
    } transition_guard{operational_transitioning_};

    std::unique_lock<std::mutex> command_lock(motor_command_mutex_);
    std::vector<JointCommand> starting_commands(current_commands_.size());
    for (std::size_t i = 0; i < current_commands_.size(); ++i) {
        std::shared_lock<std::shared_mutex> lock(*motor_mutexes_[i]);
        starting_commands[i] = current_commands_[i];
        if (!soft_disarm_logic::command_is_finite(starting_commands[i])) {
            const bool locked = disable_all_motors();
            is_init_.store(false);
            has_policy_command_.store(false);
            policy_is_active_.store(false);
            direct_policy_started_.store(false);
            operational_state_.store(motors::msg::MotorRuntimeStatus::DISARMED);
            publish_runtime_status("DISARMED after invalid soft-disarm command");
            response->success = false;
            response->message = locked
                ? "soft disarm rejected an invalid last command; motors hard-disarmed"
                : "soft disarm rejected an invalid last command; hard-disarm reported errors";
            return;
        }
    }

    const std::size_t total_steps = std::max<std::size_t>(
        1U, static_cast<std::size_t>(
            std::ceil(soft_disarm_seconds_ * static_cast<double>(control_frequency_))));
    const auto period = std::chrono::duration<double>(1.0 / control_frequency_);
    auto next_deadline = std::chrono::steady_clock::now();
    std::string failure_reason;

    RCLCPP_WARN(
        get_logger(),
        "开始正常软失能：保持末帧目标位置，在 %.2f 秒内线性衰减速度/力矩/Kp/Kd",
        soft_disarm_seconds_);

    for (std::size_t step = 0; step <= total_steps; ++step) {
        const float remaining = soft_disarm_logic::remaining_scale(step, total_steps);
        for (std::size_t i = 0; i < motors_.size(); ++i) {
            if (!debug_mode_ &&
                (!motors_[i] || !is_motor_connected_[i] ||
                 motors_[i]->get_response_count() > offline_threshold_)) {
                failure_reason = "motor " + std::to_string(motor_ids_[i]) +
                    " went offline during soft disarm";
                break;
            }
            const JointCommand command =
                soft_disarm_logic::scaled_command(starting_commands[i], remaining);
            if (!soft_disarm_logic::command_is_finite(command)) {
                failure_reason = "non-finite command during soft disarm";
                break;
            }
            if (!debug_mode_) {
                try {
                    motors_[i]->MotorMitModeCmd(
                        command.position, command.velocity, command.kp, command.kd,
                        command.effort);
                } catch (const std::exception& error) {
                    failure_reason = "motor " + std::to_string(motor_ids_[i]) +
                        " soft-disarm command failed: " + error.what();
                    break;
                }
            }
            current_commands_[i] = command;
        }
        if (!failure_reason.empty()) {
            break;
        }
        if (step < total_steps) {
            next_deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            std::this_thread::sleep_until(next_deadline);
        }
    }

    const bool locked = disable_all_motors();
    is_init_.store(false);
    has_policy_command_.store(false);
    policy_is_active_.store(false);
    policy_warmup_active_.store(false);
    direct_policy_started_.store(false);
    offline_latched_.store(false);
    operational_state_.store(motors::msg::MotorRuntimeStatus::DISARMED);

    if (!failure_reason.empty() || !locked) {
        if (failure_reason.empty()) {
            failure_reason = "one or more MotorLock commands failed";
        }
        {
            std::lock_guard<std::mutex> state_lock(runtime_state_mutex_);
            runtime_fault_code_ = "SOFT_DISARM_FAILED";
        }
        publish_runtime_status("DISARMED after soft-disarm failure: " + failure_reason);
        response->success = false;
        response->message = failure_reason + "; immediate hard-disarm attempted";
        return;
    }

    {
        std::lock_guard<std::mutex> state_lock(runtime_state_mutex_);
        runtime_fault_code_.clear();
    }
    publish_runtime_status("DISARMED after gradual force release");
    RCLCPP_WARN(get_logger(), "三秒渐进卸力完成，1-6 号电机已硬件失能");
    response->success = true;
    response->message = "gradual force release complete; motors are DISARMED";
}

void MotorsNode::control_iteration() {
    control_counter_++;
    
    if (debug_mode_) {
        // debug模式：仅发布模拟状态
        if (control_counter_ % policy_interval_ == 0) {
            publish_joint_states_policy();
        }
        // 单电机话题（如果启用）
        if (enable_per_motor_topics_ && control_counter_ % per_motor_interval_ == 0) {
            for (size_t i = 0; i < motor_ids_.size(); ++i) {
                publish_motor_state(i);
            }
        }
        return;
    }

    if (motors_.empty()) {
        return;
    }

    if (zeroing_in_progress_.load()) {
        return;
    }
    std::unique_lock<std::mutex> command_lock(motor_command_mutex_);
    if (zeroing_in_progress_.load()) {
        return;
    }

    if (offline_latched_.load()) {
        if (enable_per_motor_topics_ && control_counter_ % per_motor_interval_ == 0) {
            for (size_t i = 0; i < motors_.size(); ++i) {
                publish_motor_state(i);
            }
        }
        if (control_counter_ % policy_interval_ == 0) {
            publish_joint_states_policy();
        }
        return;
    }

    // ================== 架构变更：200Hz 控制循环 ==================
    // 1. 从第一帧策略消息开始发控；超过 POLICY_COMMAND_TIMEOUT_MS 无新消息则对各关节发 0kp0kd
    if (has_policy_command_.load()) {
        auto now = std::chrono::steady_clock::now();
        auto last_time = last_policy_command_time_.load();
        auto time_since_policy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();

        if (time_since_policy_ms > POLICY_COMMAND_TIMEOUT_MS) {
            // 超过 200ms 未收到新策略消息，对各关节发送 0kp 0kd（松劲，避免憋劲）
            for (size_t i = 0; i < motors_.size(); ++i) {
                if (is_motor_connected_[i]) {
                    try {
                        motors_[i]->MotorMitModeCmd(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                    } catch (const std::exception& e) {
                        RCLCPP_WARN(this->get_logger(), "电机 %d 发送 0kp0kd 失败: %s", motor_ids_[i], e.what());
                    }
                }
            }
            policy_is_active_.store(false);
            if (time_since_policy_ms > POLICY_COMMAND_TIMEOUT_MS * 2) {
                has_policy_command_.store(false);
            }
        } else {
            if (policy_warmup_active_.load()) {
                auto warmup_start = policy_warmup_start_time_.load();
                auto warmup_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - warmup_start).count();
                const int warmup_duration_ms = static_cast<int>(policy_motor_warmup_sec_ * 1000.0f);

                if (warmup_elapsed_ms < warmup_duration_ms) {
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                         "策略预热中：%.2f/%.2fs，暂不向电机下发策略控制帧",
                                         warmup_elapsed_ms / 1000.0,
                                         static_cast<double>(policy_motor_warmup_sec_));
                } else {
                    policy_warmup_active_.store(false);
                    RCLCPP_INFO(this->get_logger(), "策略预热完成，开始向电机下发策略控制帧");
                }
            }

            if (policy_warmup_active_.load()) {
                // 预热期间仍发布 /policy/joint_states，让推理节点的历史观测先稳定；
                // 但不发送 MIT 控制帧，避免刚启动的策略输出直接打到电机。
            } else {
            // 未超时：正常发送策略插值/当前指令
            for (size_t i = 0; i < motors_.size(); ++i) {
                if (!is_motor_connected_[i]) {
                    continue;
                }

                JointCommand cmd_to_send;
                if (enable_interpolation_) {
                    std::unique_lock<std::shared_mutex> lock(*motor_mutexes_[i]);
                    cmd_to_send = interpolators_[i].get_next_command();
                    current_commands_[i] = cmd_to_send;
                } else {
                    std::shared_lock<std::shared_mutex> lock(*motor_mutexes_[i]);
                    cmd_to_send = current_commands_[i];
                }
                try {
                    const int response_count_before_send = motors_[i]->get_response_count();
                    if (response_count_before_send > offline_threshold_) {
                        enter_offline_safe_mode(i, response_count_before_send, cmd_to_send);
                        return;
                    }

                    motors_[i]->MotorMitModeCmd(
                        cmd_to_send.position,
                        cmd_to_send.velocity,
                        cmd_to_send.kp,
                        cmd_to_send.kd,
                        cmd_to_send.effort
                    );

                    const int response_count = motors_[i]->get_response_count();
                    if (response_count > offline_threshold_) {
                        enter_offline_safe_mode(i, response_count, cmd_to_send);
                        return;
                    }
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), "电机 %d 发送控制指令失败: %s", motor_ids_[i], e.what());
                }
            }
            }
        }
    }
    
    // ================== 控制空闲逻辑 ==================
    // 未收到第一帧策略或策略超时后，不主动刷状态/保活，避免 200Hz 空闲查询压满 CAN TX 队列。
    // response_count_ 仍只由发送成功后递增、合法反馈帧清零。
    if (!has_policy_command_.load() || !policy_is_active_.load()) {
        auto now = std::chrono::steady_clock::now();
        auto last_policy_time = last_policy_command_time_.load();
        auto time_since_policy = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_policy_time).count();

        bool policy_active = policy_is_active_.load();
        if (policy_active && time_since_policy > POLICY_COMMAND_TIMEOUT_MS) {
            policy_is_active_.store(false);
            policy_active = false;
            if (time_since_policy > POLICY_COMMAND_TIMEOUT_MS * 2) {
                has_policy_command_.store(false);
            }
        }
    }
    
    // ================== 200Hz 状态发布 (观察层) ==================
    // 由于我们在上面发送了 MotorMitModeCmd，SocketCAN 的后台线程会收到反馈并更新内存
    // 这里我们只需要从内存读取并发布给 ROS
    
    // 按配置的频率发布单电机状态
    if (enable_per_motor_topics_ && control_counter_ % per_motor_interval_ == 0) {
        for (size_t i = 0; i < motors_.size(); ++i) {
            publish_motor_state(i);
        }
    }

    // 按配置的频率发布整合的关节状态（供推理节点使用）
    // 与控制循环保持 200Hz，确保每个推理周期使用最新观测。
    if (control_counter_ % policy_interval_ == 0) {
        publish_joint_states_policy();
    }
}

void MotorsNode::control_loop() {
    // 定时器回调：周期性控制循环
    // 在debug_mode下，即使未显式初始化，也应该发布模拟状态
    if (!is_init_.load()) {
        disarmed_iteration();
        return;
    }
    control_iteration();
    if (control_counter_ % std::max(1, static_cast<int>(control_frequency_ / 20.0f)) == 0) {
        publish_runtime_status();
    }
}

void MotorsNode::clear_errors() {
    if (debug_mode_) {
        return;
    }
    std::lock_guard<std::mutex> command_lock(motor_command_mutex_);
    if (zeroing_in_progress_.load()) {
        RCLCPP_WARN(this->get_logger(), "Set zeros is in progress, cannot clear motor errors.");
        return;
    }
    for (size_t i = 0; i < motors_.size(); ++i) {
        // [修改] 只清除在线电机的错误
        if (is_motor_connected_[i]) {
            motors_[i]->clear_motor_error();
        }
    }
}

void MotorsNode::reset_motors_srv(const std::shared_ptr<motors::srv::ResetMotors::Request> request,
                              std::shared_ptr<motors::srv::ResetMotors::Response> response) {
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot reset motors.";
        return;
    }
    if (zeroing_in_progress_.load()) {
        response->success = false;
        response->message = "Set zeros is in progress, cannot reset motors.";
        return;
    }
    // 异步执行重置操作，立即返回响应
    std::thread([this]() {
        pthread_setname_np(pthread_self(), "motor_reset_srv");
        struct sched_param sp{}; sp.sched_priority = 60;
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
        try {
            reset_motors();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Reset motors failed: %s", e.what());
        }
    }).detach();
    response->success = true;
    response->message = "Motors reset command sent (executing asynchronously)";
}

void MotorsNode::read_motors_srv(const std::shared_ptr<motors::srv::ReadMotors::Request> request,
                             std::shared_ptr<motors::srv::ReadMotors::Response> response) {
    (void)request;
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot read motors.";
        return;
    }
    if (zeroing_in_progress_.load()) {
        response->success = false;
        response->message = "Set zeros is in progress, cannot read motors.";
        return;
    }
    try {
        control_iteration();
        response->success = true;
        response->message = "Motors read successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::set_zeros_srv(const std::shared_ptr<motors::srv::SetZeros::Request> request,
                           std::shared_ptr<motors::srv::SetZeros::Response> response) {
    (void)request;
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot set zeros.";
        return;
    }

    RCLCPP_INFO(this->get_logger(), "/set_zeros called; executing synchronously.");
    try {
        const bool success = set_zeros();
        response->success = success;
        response->message = success
            ? "Zero setting completed synchronously"
            : "Zero setting completed with one or more failures";
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Set zeros failed: %s", e.what());
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::control_motor_srv(const std::shared_ptr<motors::srv::ControlMotor::Request> request,
                               std::shared_ptr<motors::srv::ControlMotor::Response> response) {
    int motor_id = request->motor_id;
    float position = request->position;
    float velocity = request->velocity;
    float effort = request->effort;
    
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot control motor.";
        return;
    }
    if (zeroing_in_progress_.load()) {
        response->success = false;
        response->message = "Set zeros is in progress, cannot control motor.";
        return;
    }
    
    try {
        std::lock_guard<std::mutex> command_lock(motor_command_mutex_);
        if (zeroing_in_progress_.load()) {
            response->success = false;
            response->message = "Set zeros is in progress, cannot control motor.";
            return;
        }
        // 查找电机索引
        int motor_index = -1;
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            if (motor_ids_[i] == motor_id) {
                motor_index = i;
                break;
            }
        }
        
        if (motor_index < 0) {
            response->success = false;
            response->message = "Invalid motor ID";
            return;
        }
        
        // 应用默认角度和方向
        float absolute_pos = position + joint_default_angle_[motor_index];
        if (flipped_motors_[motor_index]) {
            absolute_pos = -absolute_pos;
            velocity = -velocity;
            effort = -effort;
        }
        
        auto gains = get_motor_gains(static_cast<size_t>(motor_index));
        {
            // 解引用智能指针以获取实际的 mutex
            std::unique_lock<std::shared_mutex> lock(*motor_mutexes_[motor_index]);
            motors_[motor_index]->MotorMitModeCmd(
                absolute_pos, velocity, gains.first, gains.second, effort);
        }
        
        response->success = true;
        response->message = "Send successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::clear_errors_srv(const std::shared_ptr<motors::srv::ClearErrors::Request> request,
                           std::shared_ptr<motors::srv::ClearErrors::Response> response) {
    (void)request;
    if (zeroing_in_progress_.load()) {
        response->success = false;
        response->message = "Set zeros is in progress, cannot clear motor errors.";
        return;
    }
    try {
        clear_errors();
        response->success = true;
        response->message = "Clear errors successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    } 
}

std::pair<float, float> MotorsNode::get_motor_gains(size_t motor_index) const {
    std::lock_guard<std::mutex> lock(gains_mutex_);
    if (motor_index >= kp_.size() || motor_index >= kd_.size()) {
        return {0.0f, 0.0f};
    }
    return {kp_[motor_index], kd_[motor_index]};
}

void MotorsNode::set_motor_gains_srv(const std::shared_ptr<motors::srv::SetMotorGains::Request> request,
                                     std::shared_ptr<motors::srv::SetMotorGains::Response> response) {
    try {
        std::vector<float> next_kp;
        std::vector<float> next_kd;

        {
            std::lock_guard<std::mutex> lock(gains_mutex_);
            next_kp = kp_;
            next_kd = kd_;

            if (request->restore_defaults) {
                next_kp = default_kp_;
                next_kd = default_kd_;
            } else {
                if (request->kp.size() != request->kd.size()) {
                    response->success = false;
                    response->message = "kp and kd arrays must have the same length";
                    return;
                }
                if (request->kp.empty()) {
                    response->success = false;
                    response->message = "kp/kd arrays are empty";
                    return;
                }
                if (!request->motor_ids.empty() && request->motor_ids.size() != request->kp.size()) {
                    response->success = false;
                    response->message = "motor_ids length must match kp/kd length";
                    return;
                }
                if (request->motor_ids.empty() && request->kp.size() != kp_.size()) {
                    response->success = false;
                    response->message = "empty motor_ids requires kp/kd for all motors in node order";
                    return;
                }

                for (size_t req_i = 0; req_i < request->kp.size(); ++req_i) {
                    if (request->kp[req_i] < 0.0f || request->kd[req_i] < 0.0f) {
                        response->success = false;
                        response->message = "kp/kd must be non-negative";
                        return;
                    }

                    size_t motor_index = req_i;
                    if (!request->motor_ids.empty()) {
                        auto it = std::find(motor_ids_.begin(), motor_ids_.end(), request->motor_ids[req_i]);
                        if (it == motor_ids_.end()) {
                            response->success = false;
                            response->message = "unknown motor_id: " + std::to_string(request->motor_ids[req_i]);
                            return;
                        }
                        motor_index = static_cast<size_t>(std::distance(motor_ids_.begin(), it));
                    }

                    next_kp[motor_index] = request->kp[req_i];
                    next_kd[motor_index] = request->kd[req_i];
                }
            }

            kp_ = next_kp;
            kd_ = next_kd;
        }

        for (size_t i = 0; i < current_commands_.size() && i < next_kp.size() && i < next_kd.size(); ++i) {
            std::unique_lock<std::shared_mutex> lock(*motor_mutexes_[i]);
            current_commands_[i].kp = (i < vel_only_mode_.size() && vel_only_mode_[i]) ? 0.0f : next_kp[i];
            current_commands_[i].kd = next_kd[i];
        }

        RCLCPP_INFO(this->get_logger(), "Updated runtime motor gains: kp=%s kd=%s",
                    format_float_vector(next_kp).c_str(), format_float_vector(next_kd).c_str());
        response->success = true;
        response->message = request->restore_defaults ? "Restored startup motor gains" : "Updated runtime motor gains";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::run_safety_check_srv(
    const std::shared_ptr<motors::srv::RunSafetyCheck::Request> /*request*/,
    std::shared_ptr<motors::srv::RunSafetyCheck::Response> response) {
    if (!is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot run safety check.";
        return;
    }
    if (zeroing_in_progress_.load()) {
        response->success = false;
        response->message = "Set zeros is in progress, cannot run safety check.";
        return;
    }

    try {
        perform_safety_check(false);
        response->success = true;
        response->message = "Safety check finished and motors moved to default position";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::identify_motor_id_srv(const std::shared_ptr<motors::srv::IdentifyMotorID::Request> request,
                                        std::shared_ptr<motors::srv::IdentifyMotorID::Response> response) {
    if (zeroing_in_progress_.load()) {
        response->success = false;
        response->message = "Set zeros is in progress, cannot identify motor ID.";
        return;
    }
    int motor_id = request->motor_id;
    
    // 查找电机索引
    int motor_index = -1;
    for (size_t i = 0; i < motor_ids_.size(); ++i) {
        if (motor_ids_[i] == motor_id) {
            motor_index = i;
            break;
        }
    }
    
    if (motor_index < 0) {
        response->success = false;
        response->message = "Invalid motor ID: " + std::to_string(motor_id);
        return;
    }
    
    try {
        // 调用电机的识别ID函数
        motors_[motor_index]->MotorResetID();
        std::string can_interface = motor_index < static_cast<int>(can_interfaces_.size())
                                        ? can_interfaces_[motor_index]
                                        : "configured CAN";
        response->success = true;
        response->message = "ID query command sent to motor " + std::to_string(motor_id) + 
                           ". Please check CAN bus (candump " + can_interface + ") for motor response. " +
                           "The motor will reply with its actual ID in the CAN frame ID (low 8 bits).";
        RCLCPP_INFO(this->get_logger(), "ID query sent for motor %d. Check candump %s for response.",
                    motor_id, can_interface.c_str());
    } catch (const std::exception& e) {
        response->success = false;
        response->message = "Error: " + std::string(e.what());
    }
}

void MotorsNode::scan_motors_srv(const std::shared_ptr<motors::srv::ScanMotors::Request> request,
                                  std::shared_ptr<motors::srv::ScanMotors::Response> response) {
    if (zeroing_in_progress_.load()) {
        response->success = false;
        response->message = "Set zeros is in progress, cannot scan motors.";
        return;
    }
    float timeout = request->timeout > 0 ? request->timeout : 1.0f;  // 默认1秒
    int timeout_ms = static_cast<int>(timeout * 1000);
    
    RCLCPP_INFO(this->get_logger(), "开始扫描电机ID，超时时间: %.1f 秒", timeout);
    
    try {
        // 发送查询命令到广播ID 0x7FFE（私有协议）
        // 通信类型 0x11 (参数读取), 参数地址 0x7001 (电机ID参数)
        uint32_t comm_type_bits = (0x11 & 0x1F) << 24;
        uint32_t data_field_bits = (0x7FFE & 0xFFFF) << 8;
        uint32_t id_bits = 0xFE & 0xFF;
        uint32_t broadcast_can_id = comm_type_bits | data_field_bits | id_bits;
        
        can_frame query_frame{};
        query_frame.can_id = broadcast_can_id | CAN_EFF_FLAG;
        query_frame.can_dlc = 8;
        query_frame.data[0] = 0x01;  // 参数地址低字节 (0x7001)
        query_frame.data[1] = 0x70;  // 参数地址高字节 (0x7001)
        std::memset(&query_frame.data[2], 0, 6);
        
        // 解析回复帧，提取电机ID
        std::set<int16_t> detected_ids;
        std::set<std::string> scanned_interfaces(can_interfaces_.begin(), can_interfaces_.end());
        for (const auto& can_interface : scanned_interfaces) {
            auto can = SocketCAN::get(can_interface);
            can->transmit(query_frame);
            RCLCPP_INFO(this->get_logger(), "已在 %s 发送查询命令到广播地址 0x7FFE", can_interface.c_str());

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            std::vector<can_frame> frames = can->receive_all_frames(timeout_ms);
            for (const auto& frame : frames) {
                // 检查是否是扩展帧（私有协议）
                if (frame.can_id & CAN_EFF_FLAG) {
                    uint32_t can_id = frame.can_id & CAN_EFF_MASK;
                    uint8_t comm_type = (can_id >> 24) & 0x1F;
                    
                    // 检查是否是参数读取回复 (0x11)
                    if (comm_type == 0x11) {
                        // 提取电机ID（低8位）
                        uint8_t motor_id = can_id & 0xFF;

                        // 检查数据内容是否是参数地址 0x7001
                        if (frame.data[0] == 0x01 && frame.data[1] == 0x70) {
                            // 这是电机ID参数的回复
                            // 如果电机ID不是广播ID (0xFE)，则记录
                            if (motor_id != 0xFE) {
                                detected_ids.insert(static_cast<int16_t>(motor_id));
                                RCLCPP_INFO(this->get_logger(), "检测到电机ID: %d (CAN: %s, CAN ID: 0x%X)",
                                            motor_id, can_interface.c_str(), can_id);
                            }
                        }
                    }
                }
            }
        }
        
        // 转换为vector
        response->detected_motor_ids.clear();
        response->detected_motor_ids.reserve(detected_ids.size());
        for (int16_t id : detected_ids) {
            response->detected_motor_ids.push_back(id);
        }
        
        if (detected_ids.empty()) {
            response->success = false;
            response->message = "未检测到任何电机。请检查：1) 电机是否已上电 2) CAN总线连接是否正常 3) 电机是否在私有协议模式";
        } else {
            response->success = true;
            std::string ids_str;
            for (size_t i = 0; i < response->detected_motor_ids.size(); ++i) {
                if (i > 0) ids_str += ", ";
                ids_str += std::to_string(response->detected_motor_ids[i]);
            }
            response->message = "扫描完成，检测到 " + std::to_string(detected_ids.size()) + " 个电机，ID: [" + ids_str + "]";
            RCLCPP_INFO(this->get_logger(), "扫描完成: %s", response->message.c_str());
        }
        
    } catch (const std::exception& e) {
        response->success = false;
        response->message = "扫描失败: " + std::string(e.what());
        RCLCPP_ERROR(this->get_logger(), "扫描电机ID时出错: %s", e.what());
    }
}




bool MotorsNode::check_can_interface_exists(const std::string& interface) {
    struct ifreq ifr;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        return false;
    }
    
    strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);
    ifr.ifr_name[IFNAMSIZ - 1] = '\0';
    
    bool exists = (ioctl(sock, SIOCGIFINDEX, &ifr) == 0);
    close(sock);
    return exists;
}

bool MotorsNode::check_can_interface_up(const std::string& interface) {
    struct ifreq ifr;
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        return false;
    }
    
    strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);
    ifr.ifr_name[IFNAMSIZ - 1] = '\0';
    
    bool ok = false;
    if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0) {
        ok = (ifr.ifr_flags & IFF_UP) != 0;
    }
    close(sock);
    return ok;
}

void MotorsNode::perform_zero_calibration(int times) {
    RCLCPP_INFO(this->get_logger(), "开始零点校准，每个电机执行 %d 次...", times);
    
    // Debug模式下模拟执行零点校准
    if (debug_mode_) {
        RCLCPP_INFO(this->get_logger(), "Debug模式：模拟执行零点校准流程...");
        
        for (int attempt = 0; attempt < times; ++attempt) {
            RCLCPP_INFO(this->get_logger(), "模拟零点校准 - 第 %d/%d 次", attempt + 1, times);
            
            for (size_t i = 0; i < motor_ids_.size(); ++i) {
                int motor_id = motor_ids_[i];
                
                // 模拟等待零点设置过程
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                
                // 模拟零点设置成功（在debug模式下总是成功）
                RCLCPP_INFO(this->get_logger(), 
                    "[模拟] 电机 %d 零点设置成功 (位置: 0.0000 rad)", motor_id);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            if (attempt < times - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "模拟零点校准完成");
        
        // 执行安全性检查（包括Enter确认功能）
        perform_safety_check();
        return;
    }
    
    {
        if (zeroing_in_progress_.exchange(true)) {
            RCLCPP_WARN(this->get_logger(), "Set zeros is already in progress, skip automatic zero calibration.");
            return;
        }
        struct ZeroingGuard {
            std::atomic<bool>& flag;
            ~ZeroingGuard() { flag.store(false); }
        } zeroing_guard{zeroing_in_progress_};

        std::lock_guard<std::mutex> command_lock(motor_command_mutex_);

        for (int attempt = 0; attempt < times; ++attempt) {
            RCLCPP_INFO(this->get_logger(), "零点校准 - 第 %d/%d 次", attempt + 1, times);

            for (size_t i = 0; i < motors_.size(); ++i) {
                // [修改] 只对在线的电机进行零点校准
                if (!is_motor_connected_[i]) {
                    continue;
                }

                int motor_id = motor_ids_[i];
                bool zero_set_success = false;

                try {
                    // MotorSetZero 内部已经失能电机、切换模式、发送归零命令并等待完成。
                    // 零点姿态由人工确认，不用读回位置“严格接近 0”作为成功条件。
                    zero_set_success = motors_[i]->MotorSetZero();
                    if (zero_set_success) {
                        RCLCPP_INFO(this->get_logger(), "电机 %d 零点设置命令完成", motor_id);
                    } else {
                        RCLCPP_WARN(this->get_logger(), "电机 %d 零点设置命令返回失败", motor_id);
                    }
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), 
                        "电机 %d 零点设置异常: %s", 
                        motor_id, e.what());
                }

                // 无论成功与否，都重新使能电机并切换回运控模式
                try {
                    // 注意：顺序很重要！先切换模式，再使能
                    motors_[i]->set_motor_control_mode(0);  // 切换回运控模式（0=运控模式）
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    motors_[i]->MotorUnlock();  // 使能电机
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    if (zero_set_success) {
                        RCLCPP_INFO(this->get_logger(), "电机 %d 已重新使能并切换回运控模式", motor_id);
                    } else {
                        RCLCPP_WARN(this->get_logger(), 
                            "电机 %d 零点设置命令未确认成功，但已重新使能并切换回运控模式", motor_id);
                    }
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "电机 %d 重新使能失败: %s", motor_id, e.what());
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 每个电机间隔100ms
            }

            if (attempt < times - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "零点校准完成");
    
    // 执行安全性检查：以极低增益向默认位置附近±0.2rad范围运动
    perform_safety_check();
}

void MotorsNode::wait_for_enter_key() {
    // 等待用户输入 Enter（阻塞读取，直到用户按 Enter）
    // 优先使用stdin，如果stdin不是终端，则尝试从/dev/tty读取
    if (isatty(STDIN_FILENO)) {
        // stdin是终端，直接从stdin读取
        std::cin.clear();
        std::cin.sync();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
    } else {
        // stdin不是终端，尝试从/dev/tty读取（控制终端）
        FILE* tty = fopen("/dev/tty", "r");
        if (tty != nullptr) {
            RCLCPP_INFO(this->get_logger(), "正在从控制终端读取输入...");
            // 读取直到遇到换行符
            int c;
            while ((c = fgetc(tty)) != EOF && c != '\n' && c != '\r') {
                // 继续读取直到换行符
            }
            fclose(tty);
        } else {
            // /dev/tty也无法打开，自动继续
            RCLCPP_WARN(this->get_logger(), "无法访问终端输入，自动继续（3秒后）...");
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
}

void MotorsNode::perform_safety_check(bool wait_for_user) {
    // if (!is_init_.load()) {
    //     RCLCPP_WARN(this->get_logger(), "电机未初始化，跳过安全性检查");
    //     return;
    // }
    // if (debug_mode_) {
    //     RCLCPP_INFO(this->get_logger(), "Debug模式：跳过安全性检查");
    //     return;
    // }
    if (wait_for_user) {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "零点校验已完成，电机即将开始安全检查");
        RCLCPP_INFO(this->get_logger(), "请检查电机位置是否正确");
        RCLCPP_INFO(this->get_logger(), "按 Enter 键继续，开始低 kp/kd 控制...");
        RCLCPP_INFO(this->get_logger(), "========================================");
        wait_for_enter_key();
    }
    
    RCLCPP_INFO(this->get_logger(), "用户已确认，开始低 kp/kd 控制");
    
    // Debug模式下模拟执行安全检查（不发送真实电机指令）
    if (debug_mode_) {
        RCLCPP_INFO(this->get_logger(), "Debug模式：模拟执行安全检查流程...");
        
        constexpr int CHECK_CYCLES = 10;  // 检查周期数
        constexpr int CHECK_DURATION_MS = 3000;  // 检查持续时间 3秒
        
        // 模拟安全检查周期
        for (int cycle = 0; cycle < CHECK_CYCLES; ++cycle) {
            std::this_thread::sleep_for(std::chrono::milliseconds(CHECK_DURATION_MS / CHECK_CYCLES));
            
            if (cycle % (CHECK_CYCLES / 2) == 0) {
                RCLCPP_INFO(this->get_logger(), "[模拟] 安全性检查进度: %d/%d 周期", cycle + 1, CHECK_CYCLES);
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "[模拟] 安全性检查完成，将电机移动到默认位置...");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        RCLCPP_INFO(this->get_logger(), "========== 安全性检查完成 ==========");
        
        if (wait_for_user) {
            RCLCPP_INFO(this->get_logger(), "========================================");
            RCLCPP_INFO(this->get_logger(), "安全性检查已完成，电机已移动到默认位置");
            RCLCPP_INFO(this->get_logger(), "请检查电机位置是否正确");
            RCLCPP_INFO(this->get_logger(), "按 Enter 键继续，开始正常的 kp/kd 控制...");
            RCLCPP_INFO(this->get_logger(), "========================================");
            wait_for_enter_key();
            RCLCPP_INFO(this->get_logger(), "用户已确认，开始正常的 kp/kd 控制模式");
        }
        return;
    }
    
    {
        std::lock_guard<std::mutex> command_lock(motor_command_mutex_);
        if (zeroing_in_progress_.load()) {
            RCLCPP_WARN(this->get_logger(), "Set zeros is in progress, skip safety check.");
            return;
        }

        // 极低的kp和kd，用于安全性检查
        constexpr float SAFETY_KP = 2.0f;  // 低位置增益
        constexpr float SAFETY_KD = 0.1f;  // 极低速度增益
        constexpr float SAFETY_RANGE = 0.2f;  // ±0.2 rad 运动范围
        constexpr int CHECK_DURATION_MS = 3000;  // 检查持续时间 3秒
        constexpr int CHECK_CYCLES = 10;  // 检查周期数（在±0.2rad范围内来回运动）

        auto start_time = std::chrono::steady_clock::now();
        int cycle = 0;

        while (cycle < CHECK_CYCLES && rclcpp::ok()) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();

            if (elapsed >= CHECK_DURATION_MS) {
                break;
            }

            // 计算目标位置：在默认位置附近±0.2rad范围内正弦运动
            float phase = 2.0f * M_PI * cycle / CHECK_CYCLES;
            float offset = SAFETY_RANGE * std::sin(phase);

            for (size_t i = 0; i < motors_.size(); ++i) {
                // 只检查在线的电机
                if (!is_motor_connected_[i]) {
                    continue;
                }

                // 计算目标位置（相对于默认位置的偏移）
                float target_pos = joint_default_angle_[i] + offset;

                // 应用方向翻转
                if (flipped_motors_[i]) {
                    target_pos = -target_pos;
                }

                try {
                    // 使用极低增益发送控制指令
                    motors_[i]->MotorMitModeCmd(target_pos, 0.0f, SAFETY_KP, SAFETY_KD, 0.0f);
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), "电机 %d 安全性检查时出错: %s", motor_ids_[i], e.what());
                }
            }

            // 短暂延迟，确保指令发送完成
            std::this_thread::sleep_for(std::chrono::milliseconds(CHECK_DURATION_MS / CHECK_CYCLES));
            cycle++;

            if (cycle % (CHECK_CYCLES / 2) == 0) {
                RCLCPP_INFO(this->get_logger(), "安全性检查进度: %d/%d 周期", cycle, CHECK_CYCLES);
            }
        }

        // 最后移动到默认位置
        RCLCPP_INFO(this->get_logger(), "安全性检查完成，将电机移动到默认位置...");
        for (size_t i = 0; i < motors_.size(); ++i) {
            if (!is_motor_connected_[i]) {
                continue;
            }

            float target_pos = joint_default_angle_[i];
            if (flipped_motors_[i]) {
                target_pos = -target_pos;
            }

            try {
                // 使用正常增益移动到默认位置
                auto gains = get_motor_gains(i);
                motors_[i]->MotorMitModeCmd(target_pos, 0.0f, gains.first, gains.second, 0.0f);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "电机 %d 移动到默认位置时出错: %s", motor_ids_[i], e.what());
            }
        }

        // 等待一段时间，确保电机移动到默认位置
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    RCLCPP_INFO(this->get_logger(), "========== 安全性检查完成 ==========");
    
    if (wait_for_user) {
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "安全性检查已完成，电机已移动到默认位置");
        RCLCPP_INFO(this->get_logger(), "请检查电机位置是否正确");
        RCLCPP_INFO(this->get_logger(), "按 Enter 键继续，开始正常的 kp/kd 控制...");
        RCLCPP_INFO(this->get_logger(), "========================================");
        wait_for_enter_key();
        RCLCPP_INFO(this->get_logger(), "用户已确认，开始正常的 kp/kd 控制模式");
    }
}

/**
 * @brief 自动初始化序列 - 修复了 Socket 竞争问题的版本
 * 
 * 修改说明：
 * 1. 移除了直接调用 receive_all_frames() 的逻辑，避免与后台接收线程竞争 socket 读取权。
 * 2. 改为利用 MotorDriver 的 response_count_ 状态来判断电机是否在线。
 * 3. 确立了后台线程为唯一 Socket 消费者的架构。
 * 
 * 工作流程：
 * [Action] -> 发送查询指令，response_count_ 自增
 * [Wait]   -> 等待后台线程处理（后台线程优先级 80，会抢占读取 CAN 帧）
 * [Process]-> 后台线程收到回复，回调函数将 response_count_ 重置为 0
 * [Observe]-> 检查 response_count_，0 = 在线，>0 = 离线
 */
void MotorsNode::auto_init_sequence() {
    RCLCPP_INFO(this->get_logger(), "========== 开始自动初始化流程  ==========");
    bool auto_zero_on_start = true;
    if (this->has_parameter("auto_zero_on_start")) {
        this->get_parameter("auto_zero_on_start", auto_zero_on_start);
    }
    
    // Debug模式下跳过CAN接口检查和电机检测，但执行模拟的初始化流程
    if (debug_mode_) {
        RCLCPP_INFO(this->get_logger(), "Debug模式：跳过CAN接口检查和电机检测，执行模拟初始化流程...");
        
        if (start_disarmed_) {
            operational_state_.store(motors::msg::MotorRuntimeStatus::DISARMED);
            publish_runtime_status("startup complete in DISARMED mode");
            RCLCPP_INFO(this->get_logger(), "========== 自动初始化流程完成（DISARMED） ==========");
            return;
        }

        // 步骤3: 初始化（在debug模式下只是标记已初始化）
        RCLCPP_INFO(this->get_logger(), "步骤3: 模拟初始化电机...");
        init_motors();

        if (!auto_zero_on_start) {
            RCLCPP_INFO(this->get_logger(), "步骤4: 跳过启动自动零点校准；等待外部调用 /set_zeros");
            RCLCPP_INFO(this->get_logger(), "========== 自动初始化流程完成 ==========");
            return;
        }
        
        // 步骤4: 零点校准 (异步执行，模拟版本)
        RCLCPP_INFO(this->get_logger(), "步骤4: 启动模拟零点校准 (异步执行)...");
        
        // 使用 detached 线程，避免阻塞 ROS executor 或主线程
        std::thread([this]() {
            pthread_setname_np(pthread_self(), "motor_zero_cal");
            
            // 执行模拟校准 (参数 1 代表校准 1 次)
            perform_zero_calibration(1);
            
            RCLCPP_INFO(this->get_logger(), "========== 自动初始化流程完成 ==========");
        }).detach();
        
        return;
    }
    
    // 步骤1: 检查 CAN 接口
    std::set<std::string> required_can_interfaces(can_interfaces_.begin(), can_interfaces_.end());
    if (required_can_interfaces.empty()) {
        RCLCPP_FATAL(this->get_logger(), "can_interfaces 为空，初始化失败");
        rclcpp::shutdown();
        return;
    }
    RCLCPP_INFO(this->get_logger(), "步骤1: 检查 SocketCAN 接口...");
    
    for (const auto& can_interface : required_can_interfaces) {
        RCLCPP_INFO(this->get_logger(), "检查CAN接口 '%s'...", can_interface.c_str());
        if (!check_can_interface_exists(can_interface)) {
            RCLCPP_FATAL(this->get_logger(),
                         "CAN接口 '%s' 不存在；请先完成 SocketCAN 配置",
                         can_interface.c_str());
            rclcpp::shutdown();
            return;
        }
        if (!check_can_interface_up(can_interface)) {
            RCLCPP_FATAL(this->get_logger(),
                         "CAN接口 '%s' 未处于UP状态；请先 ip link set %s up",
                         can_interface.c_str(), can_interface.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "CAN接口 '%s' 已就绪", can_interface.c_str());
    }
    
    // 步骤2: 检测电机在线状态 (关键修改部分 - 避免 Socket 竞争)
    RCLCPP_INFO(this->get_logger(), "步骤2: 逐个检测电机在线状态...");

    int online_count = 0;

    // [Action] 发送查询指令
    // 注意：refresh_motor_status() 会发送 CAN 帧，并使内部 response_count_ 自增。
    // 这里保持历史链路：MotorInit 之前不发送 MIT 控制帧，避免改变电机控制状态。
    for (size_t i = 0; i < motors_.size(); ++i) {
        motors_[i]->refresh_motor_status();
        // 稍微错开 10ms 发送，避免总线瞬间拥堵
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // [Wait] 等待后台线程处理
    // 此时 receiver_thread_ (Priority 80) 会抢占 CPU 读取 CAN 帧
    // 如果收到回复，回调函数会将 response_count_ 重置为 0
    RCLCPP_INFO(this->get_logger(), "已发送查询指令，等待后台线程同步状态(200ms)...");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // [Observation] 检查状态
    for (size_t i = 0; i < motors_.size(); ++i) {
        int motor_id = motor_ids_[i];
        
        // 判定逻辑: 
        // 0 = 收到反馈 (Success) -> 电机在线
        // >0 = 未收到反馈 (Timeout/Fail) -> 电机离线
        int resp_count = motors_[i]->get_response_count();

        if (resp_count == 0) {
            is_motor_connected_[i] = true;
            online_count++;
            RCLCPP_INFO(this->get_logger(), "  -> 电机 ID %d [在线] ✓", motor_id);
        } else {
            is_motor_connected_[i] = false;
            // 仅屏蔽离线电机，不抛出异常，允许调试部分在线的电机
            RCLCPP_WARN(this->get_logger(), "  -> 电机 ID %d [未响应] (无反馈计数: %d) 已屏蔽", motor_id, resp_count);
        }
    }

    if (online_count == 0) {
        RCLCPP_ERROR(this->get_logger(), "严重错误：未检测到任何电机响应！请检查：1)电源 2)CAN线连接 3)波特率");
        RCLCPP_FATAL(this->get_logger(), "未检测到在线电机，停止初始化，拒绝继续使能或设置零点");
        startup_failed_ = true;
        return;
    } else {
        RCLCPP_INFO(this->get_logger(), "检测完成：%d/%zu 个电机在线", online_count, motor_ids_.size());
    }

    if (start_disarmed_) {
        operational_state_.store(motors::msg::MotorRuntimeStatus::DISARMED);
        is_init_.store(false);
        publish_runtime_status("startup complete in DISARMED mode");
        RCLCPP_INFO(this->get_logger(), "步骤3: 按配置保持全部电机失能，仅发布只读反馈");
        RCLCPP_INFO(this->get_logger(), "========== 自动初始化流程完成（DISARMED） ==========");
        return;
    }

    // 步骤3: 初始化在线电机
    RCLCPP_INFO(this->get_logger(), "步骤3: 正常初始化在线电机（使能电机）...");
    // init_motors 内部应检查 is_motor_connected_ 标志，只操作在线电机
    init_motors(); 

    if (!auto_zero_on_start) {
        RCLCPP_INFO(this->get_logger(), "步骤4: 跳过启动自动零点校准；等待外部调用 /set_zeros");
        RCLCPP_INFO(this->get_logger(), "========== 自动初始化流程完成 ==========");
        return;
    }
    
    // 步骤4: 零点校准 (异步执行)
    RCLCPP_INFO(this->get_logger(), "步骤4: 启动零点校准 (异步执行)...");
    
    // 使用 detached 线程，避免阻塞 ROS executor 或主线程
    // 这对于 200Hz 实时系统至关重要，防止初始化卡顿影响其他回调
    std::thread([this]() {
        // 设置线程名称，方便 htop 查看资源占用
        pthread_setname_np(pthread_self(), "motor_zero_cal");
        // 设置实时优先级，但低于 CAN 通信线程，不干扰实时控制
        struct sched_param sp{}; sp.sched_priority = 60;
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
        
        // 执行校准 (参数 1 代表校准 1 次)
        perform_zero_calibration(1); 
        
        RCLCPP_INFO(this->get_logger(), "========== 自动初始化流程完成 ==========");
    }).detach();
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    // 设置当前线程（主线程，运行 Executor）为实时优先级
    // 这对于保证 200Hz 控制循环的稳定性至关重要
    struct sched_param param{};
    param.sched_priority = 70;  // 略低于 CAN RX (80)，但高于普通进程
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        RCLCPP_WARN(rclcpp::get_logger("system"), 
                   "Failed to set RT priority for main thread. Please run as root for optimal performance.");
    } else {
        RCLCPP_INFO(rclcpp::get_logger("system"), 
                   "Main thread RT priority set to %d (SCHED_FIFO)", param.sched_priority);
    }
    
    auto node = std::make_shared<MotorsNode>();
    if (node->startup_failed()) {
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    if (node->joy_motor_buttons_enabled()) {
        RCLCPP_INFO(node->get_logger(), "Press 'A' button to initialize/deinitialize motors.");
        RCLCPP_INFO(node->get_logger(), "Press 'B' button to reset motors.");
    } else {
        RCLCPP_INFO(node->get_logger(), "Joystick A/B motor shortcuts disabled; start_robot.sh owns zero/policy buttons.");
    }
    if (node->is_interpolation_enabled()) {
        RCLCPP_INFO(node->get_logger(), "架构：50Hz 推理 -> %.1fHz 控制（线性插值）",
                    node->get_control_frequency());
    } else {
        RCLCPP_INFO(node->get_logger(), "架构：50Hz 推理 -> %.1fHz 控制（无插值，直接使用策略指令）",
                    node->get_control_frequency());
    }
    try {
        executor.spin();
    } catch (const std::runtime_error& e) {
        node->deinit_motors();
        RCLCPP_FATAL(node->get_logger(), "Caught exception: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
