#include "wing_control_logic.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr float kPi = 3.14159265358979323846f;

TEST(WingControlLogic, AdvancesToMirroredNinetyDegreeTargets) {
    std::array<float, 2> targets{0.0f, 0.0f};
    const std::array<float, 2> destinations{kPi / 2.0f, -kPi / 2.0f};
    const float max_step = 0.35f / 200.0f;

    bool reached = false;
    for (int tick = 0; tick < 1000 && !reached; ++tick) {
        reached = wing_control_logic::advance_targets(targets, destinations, max_step);
    }

    EXPECT_TRUE(reached);
    EXPECT_FLOAT_EQ(targets[0], kPi / 2.0f);
    EXPECT_FLOAT_EQ(targets[1], -kPi / 2.0f);
}

TEST(WingControlLogic, RejectsOutOfRangeAndNonFiniteTargets) {
    const std::array<float, 2> minimum{0.0f, -kPi / 2.0f};
    const std::array<float, 2> maximum{kPi / 2.0f, 0.0f};

    EXPECT_TRUE(wing_control_logic::positions_within_limits(
        std::array<float, 2>{kPi / 2.0f, -kPi / 2.0f}, minimum, maximum));
    EXPECT_FALSE(wing_control_logic::positions_within_limits(
        std::array<float, 2>{kPi / 2.0f + 0.01f, -kPi / 2.0f}, minimum, maximum));
    EXPECT_FALSE(wing_control_logic::positions_within_limits(
        std::array<float, 2>{0.0f, std::numeric_limits<float>::quiet_NaN()},
        minimum, maximum));
}

TEST(WingControlLogic, DisableRcHoldsCurrentFeedbackAndStopsVelocity) {
    const std::array<float, 2> actual{0.73f, -0.81f};
    std::array<float, 2> targets{1.0f, -1.0f};
    std::array<float, 2> velocity_commands{0.4f, -0.4f};

    wing_control_logic::hold_current_feedback(actual, targets, velocity_commands);

    EXPECT_EQ(targets, actual);
    EXPECT_EQ(velocity_commands, (std::array<float, 2>{0.0f, 0.0f}));
}

}  // namespace
