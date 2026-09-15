#include "soft_disarm_logic.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

struct TestCommand {
    float position;
    float velocity;
    float kp;
    float kd;
    float effort;
};

TEST(SoftDisarmLogic, ScaleRunsLinearlyFromOneToZero) {
    EXPECT_FLOAT_EQ(soft_disarm_logic::remaining_scale(0, 600), 1.0f);
    EXPECT_FLOAT_EQ(soft_disarm_logic::remaining_scale(300, 600), 0.5f);
    EXPECT_FLOAT_EQ(soft_disarm_logic::remaining_scale(600, 600), 0.0f);
    EXPECT_FLOAT_EQ(soft_disarm_logic::remaining_scale(700, 600), 0.0f);
}

TEST(SoftDisarmLogic, PositionStaysFixedWhileAllForceTermsDecay) {
    const TestCommand source{1.25f, -2.0f, 8.0f, 0.8f, 3.0f};
    const auto halfway = soft_disarm_logic::scaled_command(source, 0.5f);
    const auto released = soft_disarm_logic::scaled_command(source, 0.0f);

    EXPECT_FLOAT_EQ(halfway.position, source.position);
    EXPECT_FLOAT_EQ(halfway.velocity, -1.0f);
    EXPECT_FLOAT_EQ(halfway.kp, 4.0f);
    EXPECT_FLOAT_EQ(halfway.kd, 0.4f);
    EXPECT_FLOAT_EQ(halfway.effort, 1.5f);

    EXPECT_FLOAT_EQ(released.position, source.position);
    EXPECT_FLOAT_EQ(released.velocity, 0.0f);
    EXPECT_FLOAT_EQ(released.kp, 0.0f);
    EXPECT_FLOAT_EQ(released.kd, 0.0f);
    EXPECT_FLOAT_EQ(released.effort, 0.0f);
    EXPECT_TRUE(soft_disarm_logic::command_is_finite(released));
}

TEST(SoftDisarmLogic, PositionAndVelocityModesDecayMonotonically) {
    const TestCommand position_mode{0.7f, 0.0f, 8.0f, 0.8f, 0.4f};
    const TestCommand velocity_mode{0.0f, 3.0f, 0.0f, 0.2f, 0.0f};
    float previous_position_kd = position_mode.kd;
    float previous_velocity = velocity_mode.velocity;
    float previous_velocity_kd = velocity_mode.kd;

    for (std::size_t step = 0; step <= 600; ++step) {
        const float scale = soft_disarm_logic::remaining_scale(step, 600);
        const auto position_command =
            soft_disarm_logic::scaled_command(position_mode, scale);
        const auto velocity_command =
            soft_disarm_logic::scaled_command(velocity_mode, scale);

        EXPECT_TRUE(soft_disarm_logic::command_is_finite(position_command));
        EXPECT_TRUE(soft_disarm_logic::command_is_finite(velocity_command));
        EXPECT_FLOAT_EQ(position_command.position, position_mode.position);
        EXPECT_LE(position_command.kd, previous_position_kd);
        EXPECT_LE(velocity_command.velocity, previous_velocity);
        EXPECT_LE(velocity_command.kd, previous_velocity_kd);
        previous_position_kd = position_command.kd;
        previous_velocity = velocity_command.velocity;
        previous_velocity_kd = velocity_command.kd;
    }
}

TEST(SoftDisarmLogic, RejectsNonFiniteLastCommand) {
    TestCommand command{0.0f, 0.0f, 2.0f, 0.1f, 0.0f};
    EXPECT_TRUE(soft_disarm_logic::command_is_finite(command));
    command.kd = std::nanf("");
    EXPECT_FALSE(soft_disarm_logic::command_is_finite(command));
}

}  // namespace
