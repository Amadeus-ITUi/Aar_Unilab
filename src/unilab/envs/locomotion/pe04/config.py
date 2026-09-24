"""PE04 owns its configuration; no predecessor config is composed or imported."""

from pathlib import Path
from typing import Sequence

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[5]


def load_config(overrides: Sequence[str] = ()) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/pe04")):
        config = compose(config_name="config", overrides=list(overrides))
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    expected = dict(
        robot="pe04",
        task_id="pe04_flat",
        observation="pe04_tron1_v1",
        policy="pe04_encoder_mlp",
        algorithm="pe04_custom_ppo",
        simulator="mujoco",
    )
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"PE04 requires {key}={value}")
    e, c, a, n = config.env, config.control, config.algo, config.network
    if (e.frame_size, e.history_length, n.command_size, n.latent_dim) != (30, 10, 3, 3):
        raise ValueError(
            "PE04 observation contract requires 30D, 10 frames, 3 commands, 3D velocity"
        )
    if len(e.joint_order) != 6 or len(set(e.joint_order)) != 6:
        raise ValueError("PE04 requires six unique joints")
    if (c.physics_hz, c.motor_hz, c.policy_hz, e.contact_hz, e.contact_history_length) != (
        400,
        400,
        50,
        200,
        4,
    ):
        raise ValueError("PE04 timing contract is 400/400/50 Hz, 200 Hz four-sample contacts")
    if not n.encoder_output_detach or not c.clip_joint_targets:
        raise ValueError("PE04 requires detached velocity estimates and bounded targets")
    for key in ("kp", "kd", "torque_limits"):
        values = np.asarray(c[key])
        if values.shape != (6,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"invalid PE04 {key}")
    for group in (config.commands.ranges, config.gait):
        for key, value in group.items():
            if key == "resampling_time":
                continue
            if len(value) != 2 or not np.isfinite(value).all() or value[0] > value[1]:
                raise ValueError(f"invalid sampling range: {key}")
    if not 0 < config.gait.durations[0] <= config.gait.durations[1] < 1:
        raise ValueError("support fraction must be strictly between zero and one")
    if config.gait.frequencies[0] <= 0:
        raise ValueError("gait frequency must be positive")
    if (
        min(
            a.num_envs,
            a.num_steps_per_env,
            a.num_mini_batches,
            a.num_learning_epochs,
            a.save_interval,
        )
        <= 0
    ):
        raise ValueError("training dimensions and save interval must be positive")
    if a.num_envs * a.num_steps_per_env < a.num_mini_batches:
        raise ValueError("minibatches exceed rollout samples")
    if config.training.matmul_precision not in ("highest", "high"):
        raise ValueError("unsupported matmul precision")
    if config.training.get("update_precision", "float32") not in ("float32", "bfloat16"):
        raise ValueError("update_precision must be float32 or bfloat16")
    if config.play.delay_ms < 0 or len(config.play.gait) != 4:
        raise ValueError("invalid playback delay or gait")
