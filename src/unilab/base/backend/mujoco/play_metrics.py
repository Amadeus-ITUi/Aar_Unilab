"""MuJoCo measurements and native viewer presentation for Python playback."""

from typing import Any

import mujoco
import numpy as np


def base_motion(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> dict[str, float]:
    """Measure velocity at the body origin in body axes, and world-Z origin height."""
    velocity = np.zeros(6)
    # XBODY uses the link frame/origin; BODY uses the displaced inertial frame.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 1)
    return {
        "base_height": float(data.xpos[body_id, 2]),
        "base_velocity_x": float(velocity[3]),
        "base_velocity_y": float(velocity[4]),
        "base_yaw_rate": float(velocity[2]),
    }


def frame_base_once(viewer: Any, data: mujoco.MjData, body_id: int) -> None:
    """Set the starting view; later camera changes belong entirely to the viewer."""
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = data.xpos[body_id]


def motion_overlay(metrics: dict[str, float]) -> tuple[int, int, str, str]:
    """Native overlay columns with explicit axes and units."""
    return (
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "BASE LINK\nVx (body X)\nVy (body Y)\nYaw rate (body Z)\nHeight (world Z)",
        "Measured\n"
        f"{metrics['base_velocity_x']:+.3f} m/s\n"
        f"{metrics['base_velocity_y']:+.3f} m/s\n"
        f"{metrics['base_yaw_rate']:+.3f} rad/s\n"
        f"{metrics['base_height']:.3f} m",
    )
