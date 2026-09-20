#include "lab_deploy_utils.hpp"

#include <array>
#include <cassert>
#include <cmath>

namespace {

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-6f;
}

}  // namespace

int main() {
    static_assert(lab_deploy::kFrameDim == 29);
    static_assert(lab_deploy::kActorInputDim == 145);
    static_assert(lab_deploy::kNumTerms == 8);

    std::size_t summed_term_dim = 0;
    for (const auto dim : lab_deploy::kTermDims) {
        summed_term_dim += dim;
    }
    assert(summed_term_dim == lab_deploy::kFrameDim);
    assert(near(lab_deploy::normalize_wing_angle_rad(lab_deploy::kPi / 4.0f), 0.25f));
    assert(near(lab_deploy::normalize_wing_angle_rad(-lab_deploy::kPi / 4.0f), -0.25f));
    assert(near(lab_deploy::normalize_wing_vel_rad_s(1.0f), 0.1f));
    assert(near(lab_deploy::normalize_wing_vel_rad_s(-2.5f), -0.25f));

    const auto horizontal_wings = lab_deploy::motor_wing_angles_to_training_obs({0.0f, 0.0f});
    assert(near(horizontal_wings[0], 0.0f));
    assert(near(horizontal_wings[1], 0.0f));

    // 实机对称翼极限反馈 [motor7=+π/2, motor8=-π/2]，
    // 必须转为训练 [left=-0.5, right=-0.5]。
    const auto raised_wings = lab_deploy::motor_wing_angles_to_training_obs(
        {lab_deploy::kPi / 2.0f, -lab_deploy::kPi / 2.0f});
    assert(near(raised_wings[0], -0.5f));
    assert(near(raised_wings[1], -0.5f));

    // wing_vel 走同样 swap+flip 映射, 乘 0.1。
    const auto raised_wing_vels = lab_deploy::motor_wing_velocities_to_training_obs(
        {1.0f, -2.0f});
    assert(near(raised_wing_vels[0], -0.2f));  // motor8_vel * 0.1
    assert(near(raised_wing_vels[1], -0.1f));  // -motor7_vel * 0.1

    lab_deploy::ObservationHistory history{};
    for (std::size_t frame = 0; frame < lab_deploy::kHistoryLength; ++frame) {
        for (std::size_t value = 0; value < lab_deploy::kFrameDim; ++value) {
            history[frame][value] = static_cast<float>(frame * 100 + value);
        }
    }
    const auto actor_input = lab_deploy::flatten_term_major_history(history);
    assert(near(actor_input[0], history[0][0]));
    assert(near(actor_input[14], history[4][2]));
    // Wing_angle term starts after (3+3+4+6+6)*5 = 110 values.
    assert(near(actor_input[110], history[0][22]));
    assert(near(actor_input[119], history[4][23]));
    // Wing_vel term occupies indices 120..129.
    assert(near(actor_input[120], history[0][24]));
    assert(near(actor_input[129], history[4][25]));
    // Command term occupies the final 3*5 = 15 values.
    assert(near(actor_input[130], history[0][26]));
    assert(near(actor_input[144], history[4][28]));

    const std::array<float, lab_deploy::kActionDim> raw{
        0.0f, 0.0f, 4.5f, 0.0f, 0.0f, -4.5f,
    };

    const auto clipped = lab_deploy::clip_obs_action(raw);
    assert(near(clipped[2], 3.5f));
    assert(near(clipped[5], -3.5f));

    const auto commands = lab_deploy::raw_to_motor_commands(clipped);
    assert(near(commands[2], 35.0f));
    assert(near(commands[5], -35.0f));

    const std::array<float, lab_deploy::kActionDim> motor_order_velocity{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };
    const auto training_order_velocity =
        lab_deploy::motor_to_training_joint_order(motor_order_velocity);
    const std::array<float, lab_deploy::kActionDim> expected_training_order{
        1.0f, 2.0f, 4.0f, 5.0f, 3.0f, 6.0f,
    };
    assert(training_order_velocity == expected_training_order);

    return 0;
}
