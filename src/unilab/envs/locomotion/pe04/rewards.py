"""TRON1 PF reward formulas with explicit small-robot physical adaptations."""

import numpy as np
from scipy.special import ndtr


def desired_contacts(phase, gaits, kappa):
    foot_phase = (phase[:, None] + np.column_stack((np.zeros(len(phase)), gaits[:, 1]))) % 1
    duration = gaits[:, 2:3]
    mapped = np.where(
        foot_phase < duration,
        foot_phase * 0.5 / duration,
        0.5 + (foot_phase - duration) * 0.5 / (1 - duration),
    )
    return ndtr(mapped / kappa) * (1 - ndtr((mapped - 0.5) / kappa)) + ndtr(
        (mapped - 1) / kappa
    ) * (1 - ndtr((mapped - 1.5) / kappa))


def compute_rewards(env, contact_norm=None):
    b, r = env.backend, env.cfg["reward"]
    gravity = env.gravity()
    gyro = b.qvel[:, 3:6]
    if contact_norm is None:
        contact_norm = np.linalg.norm(b.contact_history, axis=-1)
    forces = contact_norm[:, 0]
    # Illegal-contact terms use the maximum over the sensor's four samples.
    historical_forces = contact_norm.max(axis=1)
    velocity = b.foot_linear_velocity
    clearance = np.clip(b.foot_clearances(), 0, 1)
    landing = (
        (clearance < r["about_landing_threshold"])
        & (b.contact_history[:, 0, b.foot_indices, 2] <= 0.1)
        & (velocity[..., 2] < 0)
    )
    distance = np.linalg.norm(
        b.foot_link_positions[:, 0, :2] - b.foot_link_positions[:, 1, :2], axis=-1
    )
    desired = desired_contacts(env.phase, env.gaits, r["gait_kappa"])
    low, high = env.soft_limits.T
    values = dict(
        keep_balance=np.ones(env.num_envs),
        tracking_lin_vel=np.exp(
            -np.square(env.commands[:, :2] - env.base_velocity[:, :2]).sum(1) / r["tracking_sigma"]
        ),
        tracking_ang_vel=np.exp(-np.square(env.commands[:, 2] - gyro[:, 2]) / r["tracking_sigma"]),
        base_height=np.abs(b.qpos[:, 2] - env.height_target),
        lin_vel_z=env.base_velocity[:, 2] ** 2,
        ang_vel_xy=np.square(gyro[:, :2]).sum(1),
        torques=np.square(b.torque).sum(1),
        dof_acc=np.square(b.joint_acceleration).sum(1),
        action_rate=np.square(env.actions - env.previous_actions).sum(1),
        dof_pos_limits=(
            np.maximum(low - b.qpos[:, 7:], 0) + np.maximum(b.qpos[:, 7:] - high, 0)
        ).sum(1),
        collision=(historical_forces[:, env.penalized] > r["collision_force"]).sum(1),
        action_smoothness=np.square(env.actions - 2 * env.previous_actions + env.older_actions).sum(
            1
        )
        * (env.episode_steps >= 3),
        orientation=np.square(gravity[:, :2]).sum(1),
        feet_distance=np.clip(r["min_feet_distance"] - distance, 0, 1)
        + np.clip(distance - r["max_feet_distance"], 0, 1),
        feet_regulation=(
            np.exp(-clearance / env.height_target) * np.square(velocity[..., :2]).sum(2)
        ).sum(1),
        foot_landing_vel=(landing * np.square(velocity[..., 2])).sum(1),
        dof_vel=np.square(b.qvel[:, 6:]).sum(1),
        joint_powers=np.abs(b.torque * b.qvel[:, 6:]).sum(1),
        tracking_contacts_shaped_force=(
            (1 - desired)
            * (1 - np.exp(-np.square(forces[:, b.foot_indices]) / r["gait_force_sigma"]))
        ).mean(1),
        tracking_contacts_shaped_vel=(
            desired * (1 - np.exp(-np.square(velocity).sum(2) / r["gait_vel_sigma"]))
        ).mean(1),
    )
    terms = {key: values[key] * weight * env.dt for key, weight in r["scales"].items()}
    return np.sum(list(terms.values()), axis=0).astype(np.float32), terms
