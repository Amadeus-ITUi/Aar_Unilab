#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// DR002 轮腿 Isaac Lab 策略部署工具。
//
// 与训练侧 (UniLab dr002_joystick_flat_we11) 对齐的关键约定:
//   - actor_history 观测: term-major 布局, 单帧 29 维, 5 帧历史 = 145 维。
//     每帧顺序: base_ang_vel(3) projected_gravity(3) joint_pos_no_wheel(4)
//               joint_vel(6) last_action(6) wing_angle(2) wing_vel(2) command(3)
//   - 动作: 腿 (idx 0,1,3,4) 位置, 轮 (idx 2,5) 速度。
//   - action term scale: 腿 0.5, 轮 10.0。
//   - 观测里的 last_action 沿用当前 Gym 部署裁剪 (腿 +/-100, 轮 +/-3.5)。
//   - 下发 motors 的 /policy/commands 布局: data[i] = raw_clipped * scale,
//     motors_node 负责加默认角 / 翻转 / 位置限位 / 速度控制 (vel_only_mode)。
namespace lab_deploy {

constexpr std::size_t kActionDim = 6;
constexpr std::size_t kWingAngleDim = 2;
constexpr std::size_t kWingVelDim = 2;
constexpr std::size_t kFrameDim = 29;
constexpr std::size_t kHistoryLength = 5;
constexpr std::size_t kActorInputDim = kFrameDim * kHistoryLength;  // 145

// 单帧内各观测项的维度, 顺序与训练侧 ActorHistoryCfg 一致。
constexpr std::size_t kNumTerms = 8;
constexpr std::array<std::size_t, kNumTerms> kTermDims{3, 3, 4, 6, 6, 2, 2, 3};

constexpr float kPi = 3.14159265358979323846f;

// A policy tick may consume a lower-state snapshot at most once. Sequence
// inequality intentionally handles uint32 wrap without introducing a clock
// timeout into the observation path.
inline bool has_new_sample_sequence(
    std::uint32_t current, std::uint32_t last, bool have_last) {
    return !have_last || current != last;
}

// 训练侧使用 angle_deg / 180，实机 JointState 使用 rad，二者等价为 angle_rad / pi。
inline float normalize_wing_angle_rad(float angle_rad) {
    return angle_rad / kPi;
}

// 训练侧 wing_vel term 使用 scale=0.1 (与 leg dof_vel 同风格)。
inline float normalize_wing_vel_rad_s(float vel_rad_s) {
    return vel_rad_s * 0.1f;
}

// 2026-09-09 实机手动辨向确认：bridge 反馈顺序为
// [motor7=左翼(0..+pi/2), motor8=右翼(-pi/2..0)]。
// 训练观测合同为 [left_wing, right_wing]，两侧从 0 向竖直翼极限均为负角，
// 因此 motor7 取反、motor8 保持符号，不再交换左右。
inline std::array<float, kWingAngleDim> motor_wing_angles_to_training_obs(
    const std::array<float, kWingAngleDim>& motor_positions_rad) {
    return {
        normalize_wing_angle_rad(-motor_positions_rad[0]),
        normalize_wing_angle_rad(motor_positions_rad[1]),
    };
}

// wing_vel 走与 wing_angle 相同的 swap+flip 映射，并乘 0.1。
inline std::array<float, kWingVelDim> motor_wing_velocities_to_training_obs(
    const std::array<float, kWingVelDim>& motor_velocities_rad_s) {
    return {
        normalize_wing_vel_rad_s(-motor_velocities_rad_s[0]),
        normalize_wing_vel_rad_s(motor_velocities_rad_s[1]),
    };
}

using ObservationFrame = std::array<float, kFrameDim>;
using ObservationHistory = std::array<ObservationFrame, kHistoryLength>;

inline std::array<float, kActorInputDim> flatten_term_major_history(
    const ObservationHistory& history) {
    std::array<float, kActorInputDim> input{};
    std::size_t output_offset = 0;
    std::size_t term_start = 0;
    for (const auto term_dim : kTermDims) {
        for (const auto& frame : history) {
            std::copy(
                frame.begin() + term_start,
                frame.begin() + term_start + term_dim,
                input.begin() + output_offset);
            output_offset += term_dim;
        }
        term_start += term_dim;
    }
    return input;
}

// 腿关节 (位置控制) 与轮关节 (速度控制) 在 6 维动作里的下标。
constexpr std::array<std::size_t, 4> kLegActionIndices{0, 1, 3, 4};
constexpr std::array<std::size_t, 2> kWheelActionIndices{2, 5};

// motors 话题顺序: [L_thigh, L_calf, L_foot, R_thigh, R_calf, R_foot]
// 训练 joint_vel 顺序: [L_thigh, L_calf, R_thigh, R_calf, L_foot, R_foot]
constexpr std::array<std::size_t, kActionDim> kTrainingJointOrder{0, 1, 3, 4, 2, 5};

// action term scale。
constexpr float kLegActionScale = 0.5f;
constexpr float kWheelActionScale = 10.0f;

// 部署侧裁剪，作用在原始 raw 上，同时用于 last_action 观测和电机命令。
constexpr float kObsClipLeg = 100.0f;
constexpr float kObsClipWheel = 3.5f;

inline bool is_wheel_index(std::size_t index) {
    return index == 2 || index == 5;
}

inline std::array<float, kActionDim> motor_to_training_joint_order(
    const std::array<float, kActionDim>& motor_order) {
    std::array<float, kActionDim> training_order{};
    for (std::size_t i = 0; i < kActionDim; ++i) {
        training_order[i] = motor_order[kTrainingJointOrder[i]];
    }
    return training_order;
}

// 对原始 raw 做部署裁剪 (腿 +/-100, 轮 +/-3.5), 用于回填 last_action 观测。
inline std::array<float, kActionDim> clip_obs_action(const std::array<float, kActionDim>& raw) {
    std::array<float, kActionDim> clipped{};
    for (std::size_t i = 0; i < kActionDim; ++i) {
        const float limit = is_wheel_index(i) ? kObsClipWheel : kObsClipLeg;
        clipped[i] = std::clamp(raw[i], -limit, limit);
    }
    return clipped;
}

// 把原始 raw 转成发布给 motors 的 /policy/commands 数据 (6 维):
//   腿 (位置控制): data[i] = clip_obs(raw) * 0.5    -> motors 再加默认角并按 joint_position_limits 限位
//   轮 (速度控制): data[i] = clip_obs(raw) * 10.0   -> motors 作为速度目标 (vel_only_mode)
// 使用 clip_obs 后的 raw 做缩放, 保证下发与观测里 last_action 的口径一致。
inline std::array<float, kActionDim> raw_to_motor_commands(const std::array<float, kActionDim>& raw_clipped) {
    std::array<float, kActionDim> cmd{};
    for (std::size_t i = 0; i < kActionDim; ++i) {
        cmd[i] = raw_clipped[i] * (is_wheel_index(i) ? kWheelActionScale : kLegActionScale);
    }
    return cmd;
}

// POLICY authority and command contents are separate decisions.  A forced-zero
// bench test must still publish a native PolicyCommand so the bridge/lower
// handoff is exercised; STANDBY must publish no native PolicyCommand at all.
inline bool should_publish_native_policy(bool policy_enabled) {
    return policy_enabled;
}

inline std::array<float, kActionDim> select_published_motor_commands(
    const std::array<float, kActionDim>& model_commands,
    bool policy_enabled,
    bool force_zero) {
    if (!policy_enabled || force_zero) {
        return {};
    }
    return model_commands;
}

}  // namespace lab_deploy
