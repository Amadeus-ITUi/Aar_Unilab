"""Independent Hydra configuration owner for PE03."""

import math
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[5]


def load_config(overrides: Sequence[str] = (), *, name: str = "config") -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/pe03")):
        config = compose(config_name=name, overrides=list(overrides))
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    if config.training.get("matmul_precision", "highest") not in {"highest", "high"}:
        raise ValueError("training.matmul_precision must be 'highest' or 'high'")
    if not isinstance(config.training.get("cache_update_inputs", True), bool):
        raise ValueError("training.cache_update_inputs must be boolean")
    if config.observation == "pe03_v4" or config.task_id == "pe03_gait_flat":
        from unilab.envs.locomotion.pe03.gait_config import validate_gait_config

        validate_gait_config(config)
        return
    if config.robot != "pe03" or config.task_id != "pe03_flat":
        raise ValueError("PE03 configuration requires robot=pe03 and task=pe03_flat")
    expected = {
        "policy": "pe03_encoder_mlp",
        "algorithm": "pe03_custom_ppo",
        "simulator": "mujoco",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"PE03 configuration requires {key}={value}")
    if config.observation not in {"pe03_v2", "pe03_v3"}:
        raise ValueError("unsupported PE03 observation version")
    count = len(config.env.joint_order)
    if count < 1 or len(set(config.env.joint_order)) != count:
        raise ValueError("PE03 joint_order must contain unique joints")
    if config.env.frame_size < 3 * count + 6 or config.env.history_length < 1:
        raise ValueError("PE03 frame/history dimensions cannot hold the observation layout")
    keyframe = config.env.get("reset_keyframe")
    if keyframe is not None and (not isinstance(keyframe, str) or not keyframe.strip()):
        raise ValueError("PE03 reset_keyframe must be a nonempty name or null")
    height = config.env.initial_height
    if height is not None and (not math.isfinite(float(height)) or float(height) <= 0):
        raise ValueError("PE03 initial_height must be positive and finite or null")
    physics, policy = int(config.control.physics_hz), int(config.control.policy_hz)
    if physics <= 0 or policy <= 0 or physics % policy:
        raise ValueError("physics_hz must be a positive integer multiple of policy_hz")
    if config.control.action_clip <= 0 or config.control.action_scale <= 0:
        raise ValueError("action_clip and action_scale must be positive")
    if not isinstance(config.control.get("clip_joint_targets", False), bool):
        raise ValueError("clip_joint_targets must be boolean")
    if config.network.command_size != 3:
        raise ValueError("the PE03 joystick command contract requires three values")
    if config.network.latent_dim < 1 or config.network.initial_std <= 0:
        raise ValueError("PE03 latent_dim and initial_std must be positive")
    if any(
        width < 1
        for widths in (
            config.network.encoder_hidden_dims,
            config.network.actor_hidden_dims,
            config.network.critic_hidden_dims,
        )
        for width in widths
    ):
        raise ValueError("PE03 hidden layer widths must be positive")
    height_std = float(config.reward.get("base_height_std", 1.0))
    if not math.isfinite(height_std) or height_std <= 0:
        raise ValueError("reward.base_height_std must be positive and finite")
    clock_gait = config.observation == "pe03_v2"
    frame_size = 30 if clock_gait else 24
    if count != 6 or config.env.frame_size != frame_size or config.network.latent_dim != 3:
        raise ValueError(
            f"{config.observation} requires 6 joints, a {frame_size}D frame and 3D velocity encoder"
        )
    if keyframe is None:
        raise ValueError("PE03 position PD requires a named reset keyframe for its reference")
    contact_geoms = config.env.get("foot_contact_geoms")
    if contact_geoms is not None:
        if (
            len(contact_geoms) != len(config.env.foot_names)
            or len(set(contact_geoms)) != len(contact_geoms)
            or any(not isinstance(name, str) or not name.strip() for name in contact_geoms)
        ):
            raise ValueError("foot_contact_geoms requires one unique geom name per foot")
        nonfoot_bodies = set(config.env.body_names) - set(config.env.foot_names)
        if set(config.env.penalized_bodies) != nonfoot_bodies:
            raise ValueError("sole-only contact requires penalizing every nonfoot body")
    for field in ("cpu_threads", "mujoco_threads", "evaluation_episodes"):
        if int(config.training[field]) < 1:
            raise ValueError(f"training.{field} must be positive")
    for field in ("evaluation_joint_noise", "evaluation_velocity_noise"):
        noise_value = float(config.training.get(field, 0.0))
        if not math.isfinite(noise_value) or noise_value < 0:
            raise ValueError(f"training.{field} must be nonnegative and finite")
    if len(config.play.command) != 3:
        raise ValueError("PE03 play requires three command values")
    if clock_gait:
        if config.gait is None or config.play.gait is None or len(config.play.gait) != 4:
            raise ValueError("pe03_v2 requires gait configuration and four play gait values")
        if not 0 < config.play.gait[2] < 1:
            raise ValueError("play gait duration must be strictly between zero and one")
    else:
        if config.gait is not None or config.play.gait is not None:
            raise ValueError("pe03_v3 has no clock or gait command; set gait and play.gait to null")
        for name in ("tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"):
            if config.reward.scales.get(name, 0) != 0:
                raise ValueError(f"pe03_v3 cannot use phase reward {name}")
        for name in (
            "contact_force_threshold",
            "feet_air_time_cap_s",
            "moving_lin_vel_threshold",
            "moving_ang_vel_threshold",
        ):
            reward_value = float(config.reward[name])
            if not math.isfinite(reward_value) or reward_value <= 0:
                raise ValueError(f"reward.{name} must be positive and finite")
        if not 0 < config.reward.severe_tilt_deg < 90:
            raise ValueError("reward.severe_tilt_deg must be between 0 and 90 degrees")
        if config.reward.scales.get("feet_air_height", 0) != 0:
            height_range = config.reward.get("feet_air_height_range")
            if (
                height_range is None
                or len(height_range) != 2
                or not all(math.isfinite(value) for value in height_range)
                or not 0 <= height_range[0] < height_range[1]
            ):
                raise ValueError("reward.feet_air_height_range requires finite 0 <= min < max")
    if config.control.type != "P" or config.control.motor_hz != physics:
        raise ValueError("PE03 v2/v3 use position PD at each physics step")
    for field in ("kp", "kd", "torque_limits"):
        values = list(config.control[field])
        if len(values) != count or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"control.{field} requires one positive finite value per joint")
    for field in (
        "num_envs",
        "num_steps_per_env",
        "max_iterations",
        "save_interval",
        "num_learning_epochs",
        "num_mini_batches",
    ):
        if int(config.algo[field]) < 1:
            raise ValueError(f"algo.{field} must be positive")
    if config.algo.num_envs * config.algo.num_steps_per_env < config.algo.num_mini_batches:
        raise ValueError("rollout must have at least one sample per mini batch")
    if (
        not 0 < config.algo.clip_ratio < 1
        or not 0 <= config.algo.gamma <= 1
        or not 0 <= config.algo.lam <= 1
    ):
        raise ValueError("invalid PPO clip_ratio, gamma or lam")
    if config.algo.learning_rate <= 0 or config.algo.encoder_learning_rate <= 0:
        raise ValueError("PPO and encoder learning rates must be positive")
    if config.algo.schedule not in {"adaptive", "fixed"}:
        raise ValueError("unsupported learning-rate schedule")
    if not config.network.encoder_output_detach:
        raise ValueError("PE03 v2/v3 use a separately supervised, detached encoder")
    for field in ("episode_length_s", "fail_to_terminal_time_s"):
        if config.env[field] <= 0:
            raise ValueError(f"env.{field} must be positive")
    if not 0 <= config.commands.zero_probability <= 1:
        raise ValueError("zero_probability must be in [0, 1]")
    for owner, fields in (
        (
            config.domain_rand,
            (
                "friction_range",
                "added_mass_range",
                "inertia_range",
                "kp_range",
                "kd_range",
                "torque_scale_range",
                "default_joint_offset_range",
                "delay_ms_range",
                "imu_offset_deg_range",
            ),
        ),
        (
            config.gait,
            ("frequencies", "offsets", "durations", "swing_height") if clock_gait else (),
        ),
        (config.env, ("joint_reset_range", "base_velocity_reset_range")),
        (config.commands.ranges, ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading")),
    ):
        if owner is None:
            continue
        for field in fields:
            values = list(owner[field])
            if (
                len(values) != 2
                or not all(math.isfinite(v) for v in values)
                or values[0] > values[1]
            ):
                raise ValueError(f"invalid range: {field}")
    if config.gait is not None and not 0 < config.gait.durations[0] <= config.gait.durations[1] < 1:
        raise ValueError("gait durations must be strictly between zero and one")
    if config.domain_rand.delay_ms_range[0] < 0 or config.play.delay_ms < 0:
        raise ValueError("action delays cannot be negative")
    if config.domain_rand.restitution_mapping is not None:
        raise ValueError("PhysX restitution is not a MuJoCo scalar; no mapping has been calibrated")
