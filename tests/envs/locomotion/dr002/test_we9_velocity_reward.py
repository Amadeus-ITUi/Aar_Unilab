from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.dr002.joystick import DR002JoystickEnv, RewardConfig

REPO_ROOT = Path(__file__).parents[4]


def _we9_reward_config():
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(REPO_ROOT / "conf/ppo"),
    ):
        return compose(
            config_name="config",
            overrides=["task=dr002_joystick_flat_we9/mujoco"],
        ).reward


def _velocity_reward_values(abs_errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cfg = _we9_reward_config()
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(ctrl_dt=0.02)
    env._reward_cfg = RewardConfig(
        scales={
            "track_lin_vel_x": float(cfg.scales.track_lin_vel_x),
            "track_lin_vel_x_enhance": float(cfg.scales.track_lin_vel_x_enhance),
        },
        tracking_sigma=float(cfg.tracking_sigma),
        track_lin_vel_x_std=float(cfg.track_lin_vel_x_std),
        track_lin_vel_x_enhance_std=float(cfg.track_lin_vel_x_enhance_std),
        track_lin_vel_x_term_clip=float(cfg.track_lin_vel_x_term_clip),
    )
    num_envs = len(abs_errors)
    ctx = RewardContext(
        info={"commands": np.zeros((num_envs, 3), dtype=np.float64)},
        linvel=np.column_stack(
            (
                abs_errors,
                np.zeros(num_envs, dtype=np.float64),
                np.zeros(num_envs, dtype=np.float64),
            )
        ),
        gyro=np.zeros((num_envs, 3), dtype=np.float64),
        dof_pos=np.zeros((num_envs, 6), dtype=np.float64),
        num_envs=num_envs,
        tracking_sigma=env._reward_cfg.tracking_sigma,
    )
    return env._reward_track_lin_vel_x(ctx), env._reward_track_lin_vel_x_enhance(ctx)


def test_we9_velocity_parameters_exactly_preserve_real_training_curve() -> None:
    cfg = _we9_reward_config()
    assert cfg.tracking_sigma == pytest.approx(0.25)
    assert cfg.track_lin_vel_x_std == pytest.approx(math.sqrt(0.25))
    assert cfg.track_lin_vel_x_enhance_std == pytest.approx(math.sqrt(0.25 * 10.0))

    abs_errors = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)
    track, enhance = _velocity_reward_values(abs_errors)
    expected_track = np.exp(-np.square(abs_errors) / 0.25)
    expected_enhance = np.exp(-np.square(abs_errors) / (0.25 * 10.0)) - 1.0
    np.testing.assert_allclose(track, expected_track, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(enhance, expected_enhance, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(
        expected_track[1:] + expected_enhance[1:],
        [0.7541106950997375, 0.27271685920740185],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_we9_forward_velocity_width_does_not_change_yaw_tracking() -> None:
    cfg = _we9_reward_config()
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(ctrl_dt=0.02)
    env._reward_cfg = RewardConfig(
        scales={"track_ang_vel_z": 1.0},
        tracking_sigma=float(cfg.tracking_sigma),
        track_lin_vel_x_std=float(cfg.track_lin_vel_x_std),
    )
    yaw_errors = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)
    num_envs = len(yaw_errors)
    commands = np.zeros((num_envs, 3), dtype=np.float64)
    commands[:, 1] = yaw_errors
    ctx = RewardContext(
        info={"commands": commands},
        linvel=np.zeros((num_envs, 3), dtype=np.float64),
        gyro=np.zeros((num_envs, 3), dtype=np.float64),
        dof_pos=np.zeros((num_envs, 6), dtype=np.float64),
        num_envs=num_envs,
        tracking_sigma=float(cfg.tracking_sigma),
    )
    yaw_reward = env._reward_track_ang_vel_z(ctx)
    np.testing.assert_allclose(
        yaw_reward,
        np.exp(-np.square(yaw_errors) / (0.25**2)),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_we9_velocity_weights_are_not_erased_by_per_term_clipping() -> None:
    cfg = _we9_reward_config()
    assert cfg.scales.track_lin_vel_x == 1.5
    assert cfg.scales.track_lin_vel_x_enhance == 1.5
    assert cfg.track_lin_vel_x_term_clip == 1.5

    abs_errors = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)
    track, enhance = _velocity_reward_values(abs_errors)
    weighted_rate = float(cfg.scales.track_lin_vel_x) * track
    weighted_rate += float(cfg.scales.track_lin_vel_x_enhance) * enhance
    np.testing.assert_allclose(
        weighted_rate,
        1.5
        * (
            np.exp(-np.square(abs_errors) / 0.25)
            + np.exp(-np.square(abs_errors) / (0.25 * 10.0))
            - 1.0
        ),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert weighted_rate[0] == pytest.approx(1.5)


def test_force_curriculum_accumulates_x_only_at_reward_width_on_moving_steps() -> None:
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(
        commands=SimpleNamespace(
            curriculum=True,
            curriculum_moving_command_threshold=0.05,
        )
    )
    env._reward_cfg = SimpleNamespace(
        tracking_sigma=0.25,
        track_lin_vel_x_std=0.5,
    )
    env._episode_standing_mask = np.asarray([False, True, False, False])
    env._episode_track_lin_vel_x_sum = np.zeros((4,), dtype=np.float64)
    env._episode_track_ang_vel_z_sum = np.zeros((4,), dtype=np.float64)
    env._episode_force_track_lin_vel_x_sum = np.zeros((4,), dtype=np.float64)
    env._episode_track_lin_vel_x_steps = np.zeros((4,), dtype=np.int32)
    env._episode_track_total_steps = np.zeros((4,), dtype=np.int32)
    env.get_gyro = lambda: np.zeros((4, 3), dtype=np.float64)

    commands = np.asarray(
        [
            [0.25, 1.0, 0.0],  # moving on both command axes
            [0.25, 0.0, 0.0],  # standing episodes remain excluded
            [0.00, 0.0, 0.0],  # startup / near-zero commands remain excluded
            [0.00, 0.2, 0.0],  # yaw-only is moving, but force tracking remains x-only
        ],
        dtype=np.float64,
    )
    linvel = np.zeros((4, 3), dtype=np.float64)
    linvel[3, 0] = 0.25

    env._accumulate_command_curriculum({"commands": commands}, linvel)

    np.testing.assert_array_equal(env._episode_track_total_steps, [1, 1, 1, 1])
    np.testing.assert_array_equal(env._episode_track_lin_vel_x_steps, [1, 0, 0, 1])
    np.testing.assert_allclose(
        env._episode_track_lin_vel_x_sum,
        [math.exp(-1.0), 0.0, 0.0, math.exp(-1.0)],
    )
    np.testing.assert_allclose(
        env._episode_force_track_lin_vel_x_sum,
        [math.exp(-0.25), 0.0, 0.0, math.exp(-0.25)],
    )


def test_force_curriculum_uses_separate_x_score_without_changing_command_or_noise() -> None:
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(
        max_episode_steps=10,
        commands=SimpleNamespace(
            curriculum=True,
            curriculum_min_episode_fraction=0.8,
        ),
    )
    env._state = SimpleNamespace(
        terminated=np.asarray([False]),
        truncated=np.asarray([True]),
    )
    env._episode_track_lin_vel_x_sum = np.asarray([2.0])
    env._episode_track_ang_vel_z_sum = np.asarray([8.0])
    env._episode_force_track_lin_vel_x_sum = np.asarray([7.5])
    env._episode_track_lin_vel_x_steps = np.asarray([10], dtype=np.int32)
    env._episode_track_total_steps = np.asarray([10], dtype=np.int32)
    env._command_curriculum_initial_scale = 1.0
    env._command_curriculum_final_scale = 1.0
    env._command_curriculum_scale = 1.0
    env._command_curriculum_yaw_initial_scale = 1.0
    env._command_curriculum_yaw_final_scale = 1.0
    env._command_curriculum_yaw_scale = 1.0
    captured: dict[str, dict[str, np.ndarray]] = {}
    env._update_csv_force_curriculum_from_window = lambda **kwargs: captured.setdefault(
        "force", kwargs
    )
    env._noise_curriculum_follows_csv_force = lambda: False
    env._update_noise_curriculum_from_window = lambda **kwargs: captured.setdefault("noise", kwargs)

    env._update_command_curriculum(np.asarray([0], dtype=np.int32))

    np.testing.assert_allclose(captured["force"]["tracking"], [0.75])
    np.testing.assert_allclose(captured["noise"]["tracking"], [0.50])
    np.testing.assert_array_equal(captured["force"]["tracking_mask"], [True])
    assert env._last_command_curriculum_mean_tracking == pytest.approx(0.50)
