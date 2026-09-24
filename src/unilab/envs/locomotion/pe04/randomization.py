"""PE04 startup dynamics and per-policy-step stochastic wrench sampling."""

import numpy as np


def configure_randomization(env):
    b, c, dr = env.backend, env.cfg["control"], env.cfg["domain_rand"]
    n, bodies = env.num_envs, len(env.cfg["env"]["body_names"])
    kp, kd = np.tile(c["kp"], (n, 1)), np.tile(c["kd"], (n, 1))
    if dr["enabled"] and not env.evaluation:
        uniform = env.rng.uniform
        kp *= uniform(*dr["kp_range"], (n, 6))
        kd *= uniform(*dr["kd_range"], (n, 6))
        buckets = uniform(*dr["friction_range"], dr["friction_buckets"])
        b.configure_physics(
            added_mass=uniform(*dr["added_mass_range"], n),
            link_scale=uniform(*dr["link_mass_range"], (n, bodies - 1)),
            mass_scale=uniform(*dr["mass_inertia_range"], (n, bodies)),
            com_offset=uniform(-np.array(dr["com_range"]), dr["com_range"], (n, bodies, 3)),
            friction=buckets[env.rng.integers(len(buckets), size=n)],
        )
    delay = round(env.cfg["play"]["delay_ms"] / 1000 / b.dt) if env.evaluation else 0
    b.configure_pd(
        kp=kp,
        kd=kd,
        torque_scale=np.ones((n, 6)),
        torque_limits=np.array(c["torque_limits"]),
        default_position=np.tile(env.home[7:], (n, 1)),
        delay_steps=np.full(n, delay),
        position_difference=env.cfg["env"]["dof_vel_use_pos_diff"],
    )


def sample_wrench(env):
    dr, b, n = env.cfg["domain_rand"], env.backend, env.num_envs
    if env.evaluation or not dr["enabled"]:
        return None
    selected = env.rng.random(n) < dr["push_probability"]
    delta_v = env.rng.uniform(-dr["push_linear_velocity"], dr["push_linear_velocity"], (n, 3))
    delta_w = env.rng.uniform(-dr["push_angular_velocity"], dr["push_angular_velocity"], (n, 3))
    delta_v[:, 2] = 0
    delta_w[:, 2] = 0
    delta_v *= selected[:, None]
    delta_w *= selected[:, None]
    torque = np.einsum("nij,nj->ni", b.root_inertia, delta_w) / env.dt
    return b.set_root_wrench(b.root_mass[:, None] * delta_v / env.dt, torque)
