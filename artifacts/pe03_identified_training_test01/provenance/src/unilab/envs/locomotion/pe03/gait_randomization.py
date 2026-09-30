"""PE03-owned robustness sampling; simulator metadata stays in the backend."""

import hashlib
import json

import numpy as np
from omegaconf import OmegaConf

from unilab.envs.locomotion.pe03.math import multiply, rotate
from unilab.envs.locomotion.pe03.robustness_curriculum import RobustnessCurriculum


def evaluation_protocol(config):
    """Bind stage acceptance to both evaluation conditions and sampled ranges."""
    conditions = config.training.get("evaluation_conditions", "nominal")
    settings = dict(
        conditions=conditions,
        noise=OmegaConf.to_container(config.noise, resolve=True),
        domain_rand=OmegaConf.to_container(config.domain_rand, resolve=True),
        reset=OmegaConf.to_container(config.env.reset_randomization, resolve=True)
        if "reset_randomization" in config.env
        else None,
    )
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return dict(schema="pe03.gait-evaluation.v2", fingerprint=digest, **settings)


def configure_randomization(env):
    n, b = env.num_envs, env.backend
    dr, c = env.cfg["domain_rand"], env.cfg["control"]

    def uniform(limits, size):
        return env.rng.uniform(*limits, size=size)

    kp, kd = np.tile(c["kp"], (n, 1)), np.tile(c["kd"], (n, 1))
    scale = np.ones((n, 6))
    default = np.tile(env.home[7:], (n, 1))
    delay = (
        np.full(n, round(env.cfg["play"]["delay_ms"] / 1000 / b.dt))
        if not env.robustness
        else np.zeros(n)
    )
    env.imu_offset = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    env.imu_angles = np.zeros((n, 2))
    env.friction = np.full(n, b.base_friction[b.robot_geoms[0], 0])
    masses = np.tile(b.base_mass, (n, 1))
    if env.robustness and dr["enabled"]:
        kp *= uniform(dr["kp_range"], (n, 6))
        kd *= uniform(dr["kd_range"], (n, 6))
        scale *= uniform(dr["torque_scale_range"], (n, 6))
        default += uniform(dr["default_joint_offset_range"], (n, 6))
        default = np.clip(default, b.joint_range[:, 0], b.joint_range[:, 1])
        delay = np.rint(uniform(dr["delay_ms_range"], n) / 1000 / b.dt)
        root = b.body_ids[0]
        masses[:, root] += uniform(dr["added_mass_range"], n)
        inertia = np.tile(b.base_inertia, (n, 1, 1))
        factors = uniform(dr["inertia_range"], (n, len(b.body_ids)))
        inertia[:, b.body_ids] *= factors[..., None]
        masses[:, b.body_ids[1:]] *= factors[:, 1:]
        com = np.tile(b.base_ipos, (n, 1, 1))
        com[:, root] += env.rng.uniform(
            -np.array(dr["base_com_range"]), dr["base_com_range"], (n, 3)
        )
        friction = np.tile(b.base_friction, (n, 1, 1))
        buckets = uniform(dr["friction_range"], dr["friction_buckets"])
        env.friction[:] = buckets[env.rng.integers(len(buckets), size=n)]
        # Equal robot/floor values prevent MuJoCo's max-combination from masking low friction.
        friction[:, np.r_[b.robot_geoms, b.ground_geoms], 0] = env.friction[:, None]
        b.randomization = dict(
            body_mass=masses, body_inertia=inertia, body_ipos=com, geom_friction=friction
        )
        angles = np.deg2rad(uniform(dr["imu_offset_deg_range"], (n, 2)))
        env.imu_angles[:] = angles
        roll = np.column_stack(
            (np.cos(angles[:, 0] / 2), np.sin(angles[:, 0] / 2), np.zeros((n, 2)))
        )
        pitch = np.column_stack(
            (np.cos(angles[:, 1] / 2), np.zeros(n), np.sin(angles[:, 1] / 2), np.zeros(n))
        )
        env.imu_offset[:] = multiply(pitch, roll)
    if np.any(masses[:, b.body_ids] <= 0):
        raise ValueError("mass randomization produced a non-positive robot mass")
    env.weight = masses.sum(axis=1) * b.gravity_acceleration
    env.push_mass = masses[:, b.body_ids[0]].copy()
    if "actuator_model" in c:
        # The measured effective delay replaces the old common 20 ms assumption.
        # play/domain_rand delay is an additional simulation-only jitter, not a
        # second copy of the communication/servo delay.
        delay = np.rint(delay[:, None] + np.array(c["actuator_model"]["delay_ms"]) / 1000 / b.dt)
    b.configure_pd(
        kp=kp,
        kd=kd,
        torque_scale=scale,
        torque_limits=np.array(c["torque_limits"]),
        default_position=default,
        delay_steps=delay,
        position_difference=env.cfg["env"]["dof_vel_use_pos_diff"],
    )
    env.noise_amplitude = np.zeros(18)
    noise, norm = env.cfg["noise"], env.cfg["normalization"]
    if env.robustness and noise["enabled"]:
        env.noise_amplitude[:] = (
            noise["level"]
            * np.r_[
                np.full(3, noise["ang_vel"] * norm["ang_vel"]),
                np.full(3, noise["gravity"]),
                np.full(6, noise["dof_pos"] * norm["dof_pos"]),
                np.full(6, noise["dof_vel"] * norm["dof_vel"]),
            ]
        )


def configure_curriculum(env, level=None):
    """Cache nominal/full-strength endpoints once; no model metadata on reset."""
    n, b, c = env.num_envs, env.backend, env.cfg["control"]
    env.robustness_curriculum = curriculum = RobustnessCurriculum(
        env.cfg["domain_rand"].get("curriculum", {}),
        enabled=env.robustness,
        evaluation=env.evaluation,
        level=level,
        num_envs=n,
    )
    env.randomization_levels = np.full(n, curriculum.level)
    env.randomization_nominal = {}
    env.randomization_targets = {}
    if not curriculum.enabled:
        return
    nominal = {
        "kp": np.tile(c["kp"], (n, 1)),
        "kd": np.tile(c["kd"], (n, 1)),
        "torque_scale": np.ones((n, 6)),
        "default_position": np.tile(env.home[7:], (n, 1)),
        "imu_angles": np.zeros((n, 2)),
        "friction": np.full(n, b.base_friction[b.robot_geoms[0], 0]),
    }
    target = {
        name: getattr(b, name).copy() for name in ("kp", "kd", "torque_scale", "default_position")
    }
    target.update(imu_angles=env.imu_angles.copy(), friction=env.friction.copy())
    for name, base in {
        "body_mass": b.base_mass,
        "body_inertia": b.base_inertia,
        "body_ipos": b.base_ipos,
        "geom_friction": b.base_friction,
    }.items():
        if name in b.randomization:
            target[name] = b.randomization[name].copy()
            nominal[name] = np.broadcast_to(base, target[name].shape).copy()
    env.randomization_nominal, env.randomization_targets = nominal, target
    env.randomization_levels[:] = -1  # Force the first application before reset.
    apply_curriculum_level(env, np.arange(n))


def apply_curriculum_level(env, ids):
    """Apply a new level only to resetting rows, preserving other episodes/FIFOs."""
    curriculum, b = env.robustness_curriculum, env.backend
    if not curriculum.enabled:
        return
    ids = ids[env.randomization_levels[ids] != curriculum.level]
    if not len(ids):
        return
    for name, nominal in env.randomization_nominal.items():
        value = nominal[ids] + curriculum.level * (
            env.randomization_targets[name][ids] - nominal[ids]
        )
        if name in b.randomization:
            b.randomization[name][ids] = value
        elif name in ("friction", "imu_angles"):
            getattr(env, name)[ids] = value
        else:
            getattr(b, name)[ids] = value
    angles = env.imu_angles[ids]
    roll = np.column_stack(
        (np.cos(angles[:, 0] / 2), np.sin(angles[:, 0] / 2), np.zeros((len(ids), 2)))
    )
    pitch = np.column_stack(
        (np.cos(angles[:, 1] / 2), np.zeros(len(ids)), np.sin(angles[:, 1] / 2), np.zeros(len(ids)))
    )
    env.imu_offset[ids] = multiply(pitch, roll)
    if "body_mass" in b.randomization:
        masses = b.randomization["body_mass"][ids]
        env.weight[ids] = masses.sum(1) * b.gravity_acceleration
        env.push_mass[ids] = masses[:, b.body_ids[0]]
    env.randomization_levels[ids] = curriculum.level


def push_forces(env):
    b, dr = env.backend, env.cfg["domain_rand"]
    env.last_push_impulse[:] = 0
    if not (env.robustness and dr["enabled"] and dr["push_enabled"]):
        return None
    interval = max(1, round(dr["push_interval_s"] / b.dt))
    trigger = (b.steps[:, None] + np.arange(1, env.substeps + 1)) % interval == 0
    rows, ticks = np.nonzero(trigger)
    active = env.randomization_levels[rows] > 0
    rows, ticks = rows[active], ticks[active]
    if not len(rows):
        return None
    forces = np.zeros((env.num_envs, env.substeps, 3))
    sampled = (
        env.rng.uniform(-1, 1, (len(rows), 3))
        * env.push_mass[rows, None]
        * dr["max_push_vel"]
        * env.randomization_levels[rows, None]
        / b.dt
    )
    sampled = rotate(b.qpos[rows, 3:7], sampled)
    sampled[:, 2] *= 0.5
    forces[rows, ticks] = sampled
    np.add.at(env.push_count, rows, 1)
    np.add.at(env.last_push_impulse, rows, sampled * b.dt)
    return forces


def reset_backend(env, ids):
    """One batch reset in the normal case; retry only initial contact outliers."""
    b, cfg = env.backend, env.cfg["env"].get("reset_randomization", {})
    apply_curriculum_level(env, ids)
    if not (env.robustness and cfg.get("enabled", False)):
        b.reset(ids, np.tile(env.home, (len(ids), 1)), np.zeros((len(ids), 12)))
        return
    qpos = np.tile(env.home, (len(ids), 1))
    qvel = np.zeros((len(ids), 12))
    level = env.randomization_levels[ids, None]
    qvel[:, :3] = env.rng.uniform(*cfg["linear_velocity_range"], (len(ids), 3)) * level
    qvel[:, 3:6] = env.rng.uniform(*cfg["angular_velocity_range"], (len(ids), 3)) * level
    low = np.maximum(env.home[7:] + cfg["joint_range"][0] * level, b.joint_range[:, 0])
    high = np.minimum(env.home[7:] + cfg["joint_range"][1] * level, b.joint_range[:, 1])
    pending = np.arange(len(ids))
    env.reset_fallback[ids] = False
    env.reset_attempts[ids] = 0
    for attempt in range(cfg["max_attempts"] + 1):
        qpos[pending, 2] = env.home[2]
        fallback = attempt == cfg["max_attempts"]
        qpos[pending, 7:] = (
            env.home[7:] if fallback else env.rng.uniform(low[pending], high[pending])
        )
        aligned = qpos[pending].copy()
        correction = b.align_reset_soles(aligned, cfg["sole_clearance"])
        qpos[pending] = aligned
        env.reset_height_correction[ids[pending]] = correction
        env.reset_attempts[ids[pending]] += 1
        b.reset(ids[pending], aligned, qvel[pending])
        if fallback:
            env.reset_fallback[ids[pending]] = True
            break
        # Soles are above the plane. Reuse contacts from the required reset forward;
        # no second collision pass is added for each sampled pose.
        contact = np.linalg.norm(b.contact_forces[ids[pending]], axis=-1).max(axis=1) > 1e-6
        pending = pending[contact]
        if not len(pending):
            break
