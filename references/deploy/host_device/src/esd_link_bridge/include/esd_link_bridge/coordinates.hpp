#pragma once

#include <cmath>

namespace esd_link_bridge::coordinates {

inline float policy_position_from_lower(float lower_position, float sign, float default_angle) {
  return sign * lower_position - default_angle;
}

inline float lower_position_from_policy(float policy_position, float sign, float default_angle) {
  return sign * (policy_position + default_angle);
}

inline float policy_direction_from_lower(float lower_value, float sign) {
  return sign * lower_value;
}

inline float lower_direction_from_policy(float policy_value, float sign) {
  return sign * policy_value;
}

inline float wrap_to_limits(float value, float lower, float upper) {
  const float period = upper - lower;
  if (period <= 0.0F) return value;
  float wrapped = std::fmod(value - lower, period);
  if (wrapped < 0.0F) wrapped += period;
  return lower + wrapped;
}

}  // namespace esd_link_bridge::coordinates
