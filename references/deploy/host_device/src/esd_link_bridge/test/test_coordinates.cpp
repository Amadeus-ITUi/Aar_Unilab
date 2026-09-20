#include "esd_link_bridge/coordinates.hpp"

#include <gtest/gtest.h>

#include <array>

namespace coordinates = esd_link_bridge::coordinates;

TEST(Coordinates, SixLegPortsRoundTripPositionContract) {
  constexpr std::array<float, 6> signs{-1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F};
  constexpr std::array<float, 6> defaults{-0.92020F, 0.98338F, 0.0F,
                                          -0.92020F, 0.98338F, 0.0F};
  constexpr std::array<float, 6> targets{-0.4F, 0.5F, -2.0F, 0.3F, -0.7F, 1.5F};
  for (std::size_t index = 0; index < targets.size(); ++index) {
    const auto lower = coordinates::lower_position_from_policy(
        targets[index], signs[index], defaults[index]);
    EXPECT_FLOAT_EQ(coordinates::policy_position_from_lower(
                        lower, signs[index], defaults[index]),
                    targets[index]);
  }
}

TEST(Coordinates, VelocityAndEffortOnlyApplyDirection) {
  EXPECT_FLOAT_EQ(coordinates::lower_direction_from_policy(2.5F, -1.0F), -2.5F);
  EXPECT_FLOAT_EQ(coordinates::policy_direction_from_lower(-2.5F, -1.0F), 2.5F);
  EXPECT_FLOAT_EQ(coordinates::lower_direction_from_policy(2.5F, 1.0F), 2.5F);
}

TEST(Coordinates, WheelPositionWrapUsesConfiguredPolicyLimits) {
  constexpr float pi = 3.14159265358979323846F;
  EXPECT_NEAR(coordinates::wrap_to_limits(3.0F * pi, -2.0F * pi, 2.0F * pi), -pi, 1e-5F);
  EXPECT_NEAR(coordinates::wrap_to_limits(-3.0F * pi, -2.0F * pi, 2.0F * pi), pi, 1e-5F);
}

TEST(Coordinates, WingCoordinatesApplyMeasuredLowerDirection) {
  constexpr std::array<float, 2> lower_wing{-0.75F, 0.65F};
  constexpr std::array<float, 2> signs{-1.0F, -1.0F};
  for (std::size_t index = 0; index < lower_wing.size(); ++index) {
    const auto policy = coordinates::policy_direction_from_lower(
        lower_wing[index], signs[index]);
    EXPECT_FLOAT_EQ(
        coordinates::lower_direction_from_policy(policy, signs[index]),
        lower_wing[index]);
  }
}
