"""Versioned observation layout: TRON1 PF terms, minus unsupported material channels."""

import numpy as np


def critic_layout(body_count: int, contact_length: int = 4) -> list[dict]:
    fields = [
        ("base_lin_vel", 3, "body", "actual"),
        ("base_ang_vel", 3, "body", "actual"),
        ("projected_gravity", 3, "body", "actual"),
        ("joint_pos_rel", 6, "joint", "actual"),
        ("joint_vel", 6, "joint", "actual"),
        ("last_action", 6, "policy", "raw"),
        ("phase_sin_cos", 2, "phase", "actual"),
        ("gait", 4, "command", "actual"),
        ("joint_torque", 6, "joint", "actual"),
        ("joint_acceleration", 6, "joint", "actual"),
        ("contact_force_history", contact_length * body_count * 3, "world", "actual"),
        ("body_mass", body_count, "body", "nominal"),
        ("body_inertia", body_count * 9, "body", "nominal"),
        ("joint_stiffness", 6, "joint", "nominal"),
        ("joint_damping", 6, "joint", "nominal"),
        ("root_position", 3, "world", "actual"),
        ("root_velocity", 6, "world", "actual"),
        ("root_position_duplicate", 3, "world", "actual"),
    ]
    result, offset = [], 0
    for name, size, frame, semantics in fields:
        result.append(dict(name=name, offset=offset, size=size, frame=frame, semantics=semantics))
        offset += size
    return result


def critic_size(body_count: int) -> int:
    return sum(field["size"] for field in critic_layout(body_count))


def actor_frame(clean, rng, config, *, noise: bool, noise_samples=None):
    result = clean.copy()
    if noise:
        n = config["noise"]
        amplitude = np.r_[
            np.full(3, n["ang_vel"]),
            np.full(3, n["gravity"]),
            np.full(6, n["dof_pos"]),
            np.full(6, n["dof_vel"]),
        ]
        samples = rng.normal(size=result[:, :18].shape) if noise_samples is None else noise_samples
        result[:, :18] += samples * amplitude
    s = config["normalization"]
    result[:, :18] = np.clip(result[:, :18], -s["clip_observations"], s["clip_observations"])
    result[:, :3] *= s["ang_vel"]
    result[:, 6:12] *= s["dof_pos"]
    result[:, 12:18] *= s["dof_vel"]
    return result.astype(np.float32)
