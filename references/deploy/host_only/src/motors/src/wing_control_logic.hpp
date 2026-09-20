#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace wing_control_logic {

template <std::size_t N>
bool advance_targets(
    std::array<float, N>& targets,
    const std::array<float, N>& destinations,
    float max_step) {
    bool reached = true;
    for (std::size_t i = 0; i < N; ++i) {
        const float error = destinations[i] - targets[i];
        if (std::abs(error) <= max_step) {
            targets[i] = destinations[i];
        } else {
            targets[i] += std::copysign(max_step, error);
            reached = false;
        }
    }
    return reached;
}

template <std::size_t N>
void hold_current_feedback(
    const std::array<float, N>& actual_positions,
    std::array<float, N>& target_positions,
    std::array<float, N>& velocity_commands) {
    target_positions = actual_positions;
    velocity_commands.fill(0.0f);
}

template <std::size_t N>
bool positions_within_limits(
    const std::array<float, N>& positions,
    const std::array<float, N>& minimum,
    const std::array<float, N>& maximum) {
    for (std::size_t i = 0; i < N; ++i) {
        if (!std::isfinite(positions[i]) ||
            positions[i] < minimum[i] || positions[i] > maximum[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace wing_control_logic
