#!/usr/bin/env python3
"""Confirmed bench hard limits and conservative ESD-Link software limits."""

from __future__ import annotations

import math


# Policy/base_link coordinates.  These hard endpoints were independently
# captured by two passive recorders on 2026-09-09 and confirmed by the onsite
# operator.  Never command these values directly.
CONFIRMED_HARD_LIMITS = {
    1: (-0.979066, 0.928711),
    2: (-0.971147, 0.322805),
    4: (-0.960015, 0.928018),
    5: (-0.975633, 0.303477),
    7: (-0.797862, 2.121189),
    8: (-2.075898, 0.775504),
}

# Rounded inward from the confirmed hard endpoints.  Every side retains at
# least 0.075 rad of mechanical margin; P3/P6 are velocity-controlled wheels.
POSITION_LIMITS = {
    1: (-0.89, 0.84),
    2: (-0.89, 0.24),
    4: (-0.88, 0.84),
    5: (-0.89, 0.22),
    7: (-0.70, 2.00),
    8: (-2.00, 0.70),
}

LEG_POSITION_LIMITS = {
    port: POSITION_LIMITS[port] for port in (1, 2, 4, 5)
}
WING_POSITION_LIMITS = tuple(POSITION_LIMITS[port] for port in (7, 8))


def position_limit_error(port: int, position: float) -> str | None:
    limits = POSITION_LIMITS.get(port)
    if limits is None:
        return None
    if not math.isfinite(position):
        return f"P{port} 当前策略位置不是有限数"
    if position < limits[0] or position > limits[1]:
        return (
            f"P{port} 当前策略位置 {position:+.4f} rad 超出软件限位 "
            f"[{limits[0]:+.4f}, {limits[1]:+.4f}]"
        )
    return None


def clamp_position_target(port: int, position: float, margin: float = 0.0) -> float:
    """Clamp a commanded position inward; current feedback is never rejected."""
    limits = POSITION_LIMITS.get(port)
    if limits is None:
        return position
    if not math.isfinite(position) or not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"P{port} position and margin must be finite and non-negative")
    lower = limits[0] + margin
    upper = limits[1] - margin
    if lower > upper:
        raise ValueError(f"P{port} margin {margin} leaves no command range")
    return min(max(position, lower), upper)


def pose_limit_error(positions: list[float]) -> str | None:
    if len(positions) != 8:
        return "完整姿态必须包含 P1-P8"
    for port, position in enumerate(positions, 1):
        error = position_limit_error(port, position)
        if error is not None:
            return error
    return None
