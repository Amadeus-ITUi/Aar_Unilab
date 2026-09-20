"""Independent Hydra configuration owner for PE01."""

import math
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[5]


def load_config(overrides: Sequence[str] = (), *, name: str = "config") -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/pe01")):
        config = compose(config_name=name, overrides=list(overrides))
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    if config.robot != "pe01" or config.task_id != "pe01_flat":
        raise ValueError("PE01 configuration requires robot=pe01 and task=pe01_flat")
    expected = {
        "policy": "pe01_encoder_mlp",
        "algorithm": "pe01_custom_ppo",
        "simulator": "mujoco",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"PE01 configuration requires {key}={value}")
    if config.observation != "pe01_v2":
        raise ValueError("unsupported PE01 observation version")
    count = len(config.env.joint_order)
    if count < 1 or len(set(config.env.joint_order)) != count:
        raise ValueError("PE01 joint_order must contain unique joints")
    if config.env.frame_size < 3 * count + 6 or config.env.history_length < 1:
        raise ValueError("PE01 frame/history dimensions cannot hold the observation layout")
    keyframe = config.env.get("reset_keyframe")
    if keyframe is not None and (not isinstance(keyframe, str) or not keyframe.strip()):
        raise ValueError("PE01 reset_keyframe must be a nonempty name or null")
    height = config.env.initial_height
    if height is not None and (not math.isfinite(float(height)) or float(height) <= 0):
        raise ValueError("PE01 initial_height must be positive and finite or null")
    physics, policy = int(config.control.physics_hz), int(config.control.policy_hz)
    if physics <= 0 or policy <= 0 or physics % policy:
        raise ValueError("physics_hz must be a positive integer multiple of policy_hz")
    if config.control.action_clip <= 0 or config.control.action_scale <= 0:
        raise ValueError("action_clip and action_scale must be positive")
    if config.network.command_size != 3:
        raise ValueError("the PE01 joystick command contract requires three values")
    if config.network.latent_dim < 1 or config.network.initial_std <= 0:
        raise ValueError("PE01 latent_dim and initial_std must be positive")
    if any(
        width < 1
        for widths in (
            config.network.encoder_hidden_dims,
            config.network.actor_hidden_dims,
            config.network.critic_hidden_dims,
        )
        for width in widths
    ):
        raise ValueError("PE01 hidden layer widths must be positive")
    if count != 6 or config.env.frame_size != 30 or config.network.latent_dim != 3:
        raise ValueError("pe01_v2 requires 6 joints, a 30D frame and 3D velocity encoder")
    if keyframe is None:
        raise ValueError("pe01_v2 requires a named reset keyframe for its PD reference")
    for field in ("cpu_threads", "mujoco_threads", "evaluation_episodes"):
        if int(config.training[field]) < 1:
            raise ValueError(f"training.{field} must be positive")
    for field in ("evaluation_joint_noise", "evaluation_velocity_noise"):
        noise_value = float(config.training.get(field, 0.0))
        if not math.isfinite(noise_value) or noise_value < 0:
            raise ValueError(f"training.{field} must be nonnegative and finite")
    if len(config.play.command) != 3 or len(config.play.gait) != 4:
        raise ValueError("pe01_v2 play requires three command values and four gait values")
    if not 0 < config.play.gait[2] < 1:
        raise ValueError("play gait duration must be strictly between zero and one")
    if config.control.type != "P" or config.control.motor_hz != physics:
        raise ValueError("pe01_v2 uses position PD at each physics step")
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
        raise ValueError("pe01_v2 uses a separately supervised, detached encoder")
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
        (config.gait, ("frequencies", "offsets", "durations", "swing_height")),
        (config.env, ("joint_reset_range", "base_velocity_reset_range")),
        (config.commands.ranges, ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading")),
    ):
        for field in fields:
            values = list(owner[field])
            if (
                len(values) != 2
                or not all(math.isfinite(v) for v in values)
                or values[0] > values[1]
            ):
                raise ValueError(f"invalid range: {field}")
    if not 0 < config.gait.durations[0] <= config.gait.durations[1] < 1:
        raise ValueError("gait durations must be strictly between zero and one")
    if config.domain_rand.delay_ms_range[0] < 0 or config.play.delay_ms < 0:
        raise ValueError("action delays cannot be negative")
    if config.domain_rand.restitution_mapping is not None:
        raise ValueError("PhysX restitution is not a MuJoCo scalar; no mapping has been calibrated")
