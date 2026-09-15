#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace soft_disarm_logic {

inline float remaining_scale(std::size_t step, std::size_t total_steps) {
    if (total_steps == 0U) {
        return 0.0f;
    }
    const std::size_t clamped_step = std::min(step, total_steps);
    return 1.0f - static_cast<float>(clamped_step) / static_cast<float>(total_steps);
}

template <typename Command>
Command scaled_command(const Command& source, float remaining) {
    const float scale = std::clamp(remaining, 0.0f, 1.0f);
    Command result = source;
    result.velocity = source.velocity * scale;
    result.kp = source.kp * scale;
    result.kd = source.kd * scale;
    result.effort = source.effort * scale;
    return result;
}

template <typename Command>
bool command_is_finite(const Command& command) {
    return std::isfinite(command.position) && std::isfinite(command.velocity) &&
           std::isfinite(command.kp) && std::isfinite(command.kd) &&
           std::isfinite(command.effort);
}

}  // namespace soft_disarm_logic
