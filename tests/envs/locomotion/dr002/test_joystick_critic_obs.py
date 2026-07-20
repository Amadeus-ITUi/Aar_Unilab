from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from unilab.envs.locomotion.dr002.base import NoiseConfig
from unilab.envs.locomotion.dr002.joystick import (
    _CRITIC_DIM,
    _DR002_ASSET_ROOT,
    _HISTORY_LENGTH,
    _LEGACY_CRITIC_DIM,
    DR002DomainRandConfig,
    DR002JoystickCfg,
    DR002JoystickEnv,
    _apply_csv_force_channel_options,
    _load_force_csv,
    _sample_wrench_csv_replay,
)


class _FakeBackend:
    def __init__(self, base_pos: np.ndarray) -> None:
        self._base_pos = np.asarray(base_pos, dtype=np.float32)

    def get_base_pos(self) -> np.ndarray:
        return self._base_pos


def _bare_obs_env(num_envs: int = 2) -> DR002JoystickEnv:
    env = object.__new__(DR002JoystickEnv)
    env._np_dtype = np.dtype(np.float32)
    env._num_envs = num_envs
    env._num_action = 6
    env._actor_dim = 135
    env._critic_dim = _CRITIC_DIM
    env._use_isaaclab_critic = True
    env._wing_angle_obs_enabled = True
    env.default_angles = np.asarray([0.8, -1.6, 0.0, 0.8, -1.6, 0.0], dtype=np.float32)
    env._cfg = SimpleNamespace(
        ctrl_dt=0.02,
        noise_config=NoiseConfig(
            curriculum=True,
            curriculum_levels=[0.0, 0.25, 0.5, 0.75, 1.0],
            scale_wheel_vel=0.5,
        ),
        domain_rand=DR002DomainRandConfig(
            csv_force_enabled=True,
            csv_force_curriculum=True,
            csv_force_curriculum_hz=[0.0, 1.0, 2.0, 3.0, 4.0],
            csv_force_observation_force_normalization=50.0,
        ),
    )
    env._csv_force_curriculum_level = 0
    env._noise_curriculum_level = 4
    env._gravity_installation_bias_rp = np.zeros((num_envs, 2), dtype=np.float32)
    env._gravity_dynamic_noise_rp = np.zeros((num_envs, 2), dtype=np.float32)
    env._history_terms = [
        np.zeros((num_envs, _HISTORY_LENGTH, dim), dtype=np.float32)
        for dim in (3, 3, 4, 6, 6, 2, 3)
    ]
    env._compute_wing_angle_obs = lambda *args, **kwargs: np.full(
        (args[0], 2), 0.125, dtype=np.float32
    )
    env._backend = _FakeBackend(
        np.asarray([[0.0, 0.0, 0.7], [0.0, 0.0, 1.8]], dtype=np.float32)
    )
    env._privileged_base_mass_delta = np.asarray([[0.1], [0.2]], dtype=np.float32)
    env._privileged_base_com_offset = np.asarray(
        [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float32
    )
    env._default_joint_pos_offset = np.arange(num_envs * 6, dtype=np.float32).reshape(
        num_envs, 6
    )
    env._privileged_ground_friction_scale = np.asarray([[0.8], [0.9]], dtype=np.float32)
    env._privileged_robot_friction_scale = np.asarray([[1.1], [1.2]], dtype=np.float32)
    env._measured_csv_force_base = np.asarray(
        [[50.0, -25.0, 5.0], [100.0, 0.0, -50.0]],
        dtype=np.float32,
    )
    return env


def _obs_inputs(num_envs: int = 2) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    current = np.arange(num_envs * 6, dtype=np.float32).reshape(num_envs, 6) + 1.0
    previous = current + 20.0
    qacc = current + 40.0
    torques = current + 60.0
    commands = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)[
        :num_envs
    ]
    return {
        "info": {
            "current_actions": current,
            "last_actions": previous,
            "qacc": qacc,
            "torques": torques,
            "commands": commands,
        },
        "linvel": np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )[:num_envs],
        "gyro": np.asarray(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32
        )[:num_envs],
        "gravity": np.zeros((num_envs, 3), dtype=np.float32),
        "projected_gravity": np.asarray(
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32
        )[:num_envs],
        "dof_pos": np.asarray(
            [[1.8, -0.6, 0.3, 2.8, -0.6, 0.4], [0.8, -1.6, 0.5, 0.8, -1.6, 0.6]],
            dtype=np.float32,
        )[:num_envs],
        "dof_vel": np.asarray(
            [[1.0, 2.0, 5.0, 3.0, 4.0, 6.0], [7.0, 8.0, 11.0, 9.0, 10.0, 12.0]],
            dtype=np.float32,
        )[:num_envs],
    }


def test_critic_matches_144_force_only_contract_slice_by_slice() -> None:
    env = _bare_obs_env()
    inputs = _obs_inputs()

    obs = env._compute_obs(**inputs, reset_history=True)
    actor = obs["obs"]
    critic = obs["critic"]
    info = inputs["info"]

    assert actor.shape == (2, 135)
    assert critic.shape == (2, 144)
    np.testing.assert_array_equal(critic[:, 0:3], inputs["linvel"])
    np.testing.assert_array_equal(critic[:, 3:6], inputs["gyro"])
    np.testing.assert_array_equal(critic[:, 6:9], inputs["projected_gravity"])
    np.testing.assert_array_equal(critic[:, 9:12], info["commands"])
    np.testing.assert_allclose(
        critic[:, 12:16],
        [[1.0, 0.0, -1.0, -3.0], [-6.0, -7.0, -9.0, -10.0]],
    )
    np.testing.assert_allclose(
        critic[:, 16:22],
        np.asarray(inputs["dof_vel"])[:, [0, 1, 3, 4, 2, 5]] * 0.1,
    )
    np.testing.assert_array_equal(critic[:, 22:28], info["current_actions"])
    np.testing.assert_array_equal(critic[:, 28:34], info["current_actions"])
    np.testing.assert_array_equal(critic[:, 34:40], info["last_actions"])
    np.testing.assert_allclose(
        critic[:, 40:46],
        info["qacc"][:, [0, 1, 3, 4, 2, 5]] * 0.0025,
    )
    np.testing.assert_allclose(critic[0, 46:123], 0.2)
    np.testing.assert_allclose(critic[1, 46:123], 1.0)
    np.testing.assert_allclose(
        critic[:, 123:129],
        info["torques"][:, [0, 1, 3, 4, 2, 5]] * 0.05,
    )
    np.testing.assert_array_equal(critic[:, 129:130], env._privileged_base_mass_delta)
    np.testing.assert_array_equal(critic[:, 130:133], env._privileged_base_com_offset)
    np.testing.assert_array_equal(
        critic[:, 133:139],
        env._default_joint_pos_offset[:, [0, 1, 3, 4, 2, 5]],
    )
    np.testing.assert_allclose(critic[:, 139:141], [[1.1, 0.0], [1.2, 0.0]])
    np.testing.assert_allclose(
        critic[:, 141:144],
        [[1.0, -0.5, 0.1], [2.0, 0.0, -1.0]],
    )


def test_existing_dr002_tasks_keep_legacy_critic_contract() -> None:
    assert DR002JoystickCfg().critic_obs_mode == "legacy"
    env = _bare_obs_env()
    env._use_isaaclab_critic = False
    env._critic_dim = _LEGACY_CRITIC_DIM
    env._motor_kp = np.full((2, 6), 2.0, dtype=np.float32)
    env._motor_kd = np.full((2, 6), 0.2, dtype=np.float32)
    env._external_disturbance_current_wrench = np.arange(12, dtype=np.float32).reshape(2, 6)
    inputs = _obs_inputs()

    critic = env._compute_obs(**inputs, reset_history=True)["critic"]

    assert critic.shape == (2, _LEGACY_CRITIC_DIM)
    np.testing.assert_array_equal(critic[:, 34:40], env._motor_kp)
    np.testing.assert_array_equal(critic[:, 40:46], env._motor_kd)
    np.testing.assert_array_equal(
        critic[:, 46:52],
        env._external_disturbance_current_wrench,
    )


def test_measured_force_changes_critic_only() -> None:
    env = _bare_obs_env()
    inputs = _obs_inputs()
    first = env._compute_obs(**inputs, reset_history=True)
    env._measured_csv_force_base += 10.0
    second = env._compute_obs(**inputs, reset_history=True)

    np.testing.assert_array_equal(first["obs"], second["obs"])
    np.testing.assert_array_equal(first["critic"][:, :141], second["critic"][:, :141])
    assert not np.array_equal(first["critic"][:, 141:144], second["critic"][:, 141:144])


def test_noise_level_is_force_level_single_source_of_truth() -> None:
    env = _bare_obs_env(num_envs=1)
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]

    for force_level, noise_level in enumerate(expected):
        env._csv_force_curriculum_level = force_level
        env._noise_curriculum_level = 4 - force_level
        assert env._current_noise_level() == noise_level


def test_actor_uses_separate_leg_and_wheel_velocity_noise(monkeypatch) -> None:
    env = _bare_obs_env(num_envs=2)
    env._csv_force_curriculum_level = 4
    calls: list[tuple[tuple[int, ...], float, float]] = []

    def capture_noise(data: np.ndarray, scale: float, level: float) -> np.ndarray:
        calls.append((data.shape, float(scale), float(level)))
        return data

    monkeypatch.setattr(DR002JoystickEnv, "_obs_noise_at_level", staticmethod(capture_noise))
    env._compute_obs(**_obs_inputs(num_envs=2), reset_history=True)

    assert ((2, 4), 1.5, 1.0) in calls
    assert ((2, 2), 0.5, 1.0) in calls
    assert ((2, 3), 0.2, 1.0) in calls


def test_tilt_gravity_noise_rotates_unit_vector_and_keeps_critic_clean() -> None:
    env = _bare_obs_env(num_envs=1)
    env._cfg.noise_config = NoiseConfig(
        level=1.0,
        scale_gravity=0.0,
        gravity_noise_mode="tilt",
        gravity_installation_bias_max_deg=5.0,
        gravity_dynamic_noise_max_deg=1.0,
        gravity_dynamic_noise_time_constant_s=3.0,
    )
    env._gravity_installation_bias_rp[0] = np.deg2rad([5.0, -3.0])
    inputs = _obs_inputs(num_envs=1)
    inputs["projected_gravity"] = np.asarray([[0.0, 0.0, -2.0]], dtype=np.float32)

    obs = env._compute_obs(**inputs, reset_history=True)
    gravity_history = obs["obs"][:, 15:30].reshape(1, _HISTORY_LENGTH, 3)

    np.testing.assert_allclose(np.linalg.norm(gravity_history, axis=2), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(
        gravity_history[:, 1:],
        np.repeat(gravity_history[:, :1], _HISTORY_LENGTH - 1, axis=1),
        atol=0.0,
    )
    assert not np.allclose(gravity_history[0, 0], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(obs["critic"][:, 6:9], inputs["projected_gravity"])


def test_gravity_dynamic_noise_is_bounded_slow_and_installation_bias_is_fixed() -> None:
    env = _bare_obs_env(num_envs=4)
    env._cfg.noise_config = NoiseConfig(
        level=1.0,
        gravity_noise_mode="tilt",
        gravity_installation_bias_max_deg=5.0,
        gravity_dynamic_noise_max_deg=1.0,
        gravity_dynamic_noise_time_constant_s=3.0,
    )
    env_ids = np.asarray([1, 3], dtype=np.int32)
    np.random.seed(1234)
    env._reset_gravity_tilt_noise(env_ids)
    fixed_bias = env._gravity_installation_bias_rp.copy()

    assert np.all(np.abs(np.rad2deg(fixed_bias[env_ids])) <= 5.0)
    np.testing.assert_array_equal(env._gravity_dynamic_noise_rp, 0.0)

    max_step_rad = 0.0
    for _ in range(500):
        previous = env._gravity_dynamic_noise_rp.copy()
        env._advance_gravity_dynamic_noise(env_ids, noise_level=1.0)
        max_step_rad = max(
            max_step_rad,
            float(np.max(np.abs(env._gravity_dynamic_noise_rp - previous))),
        )

    max_noise_rad = float(np.deg2rad(1.0))
    rho = float(np.exp(-env._cfg.ctrl_dt / 3.0))
    innovation_std = (max_noise_rad / 3.0) * np.sqrt(1.0 - rho * rho)
    strict_step_bound = (1.0 - rho) * max_noise_rad + 3.0 * innovation_std
    assert np.all(np.abs(env._gravity_dynamic_noise_rp[env_ids]) <= max_noise_rad)
    assert max_step_rad <= strict_step_bound + 1.0e-7
    np.testing.assert_array_equal(env._gravity_installation_bias_rp, fixed_bias)
    np.testing.assert_array_equal(env._gravity_dynamic_noise_rp[[0, 2]], 0.0)


def test_we6_config_uses_episode_bias_and_slow_gravity_tilt_noise() -> None:
    config_path = (
        Path(__file__).parents[4]
        / "conf/ppo/task/dr002_joystick_flat_we6/mujoco.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    noise = config["env"]["noise_config"]

    assert noise["scale_gyro"] == 0.1
    assert noise["scale_joint_angle"] == 0.001
    assert noise["scale_joint_vel"] == 0.2
    assert noise["scale_wheel_vel"] == 0.3
    assert noise["scale_gravity"] == 0.0
    assert noise["gravity_noise_mode"] == "tilt"
    assert noise["gravity_installation_bias_max_deg"] == 5.0
    assert noise["gravity_dynamic_noise_max_deg"] == 1.0
    assert noise["gravity_dynamic_noise_time_constant_s"] == 3.0


def test_noise_curriculum_rejects_level_count_mismatch() -> None:
    env = _bare_obs_env(num_envs=1)
    env._cfg.noise_config.curriculum_levels = [0.0, 1.0]

    with pytest.raises(ValueError, match="one level per CSV force curriculum"):
        env._validate_noise_curriculum_cfg()


def test_we6_base_config_uses_only_latest_1_to_4_hz_assets() -> None:
    config_path = (
        Path(__file__).parents[4]
        / "conf/ppo/task/dr002_joystick_flat_we6/base.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    env_cfg = config["env"]
    domain_rand = env_cfg["domain_rand"]

    assert domain_rand["csv_force_curriculum_hz"] == [0, 1, 2, 3, 4]
    assert env_cfg["noise_config"]["level"] == 1.0
    assert env_cfg["noise_config"]["curriculum"] is False
    assert env_cfg["noise_config"]["curriculum_levels"] == [1.0] * 5
    assert env_cfg["control_config"]["action_delay_min_steps"] == 4
    assert env_cfg["control_config"]["action_delay_max_steps"] == 16
    assert env_cfg["control_config"]["resample_action_delay"] is False
    assert domain_rand["max_force"] == [10.0, 10.0, 0.0]
    assert domain_rand["csv_force_zero_fy"] is True
    assert config["reward"]["scales"]["joint_acc_wheel_l2"] == pytest.approx(-2.5e-6)
    assert len(domain_rand["csv_force_curriculum_paths"]) == 4
    assert len(env_cfg["wing_angle_obs"]["curriculum_paths"]) == 4
    assert all(
        "sweep_20260713_230639" in path
        for path in domain_rand["csv_force_curriculum_paths"]
    )
    assert all(
        "sweep_20260713_230639" in path
        for path in env_cfg["wing_angle_obs"]["curriculum_paths"]
    )
    assert config["algo"]["actor"]["history_term_dims"] == [3, 3, 4, 6, 6, 2, 3]


@pytest.mark.parametrize("hz", [1, 2, 3, 4])
def test_latest_measured_wrench_assets_preserve_all_six_axes(hz: int) -> None:
    path = _DR002_ASSET_ROOT / "measured_wrench/sweep_20260713_230639" / f"{hz}hz.csv"
    samples = _load_force_csv(str(path))

    assert samples.shape[1] == 7
    assert samples[0, 0] == pytest.approx(0.0)
    assert samples[-1, 0] == pytest.approx(10.0)
    assert np.all(np.isfinite(samples))
    assert np.max(np.abs(samples[:, 4:7])) > 0.1
    np.testing.assert_allclose(samples[0, 1:7], samples[-1, 1:7], atol=1.0e-9)


def test_csv_force_fy_zeroing_preserves_source_and_other_axes() -> None:
    source = np.asarray([[[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]]], dtype=np.float64)
    result = _apply_csv_force_channel_options(source, zero_fy=True)

    np.testing.assert_array_equal(source, [[[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]]])
    np.testing.assert_array_equal(result, [[[1.0, 0.0, 3.0], [-4.0, 0.0, -6.0]]])


def test_wrench_replay_interpolates_force_and_moment_together() -> None:
    samples = np.asarray(
        [
            [0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [1.0, 2.0, 12.0, 22.0, 32.0, 42.0, 52.0],
        ],
        dtype=np.float64,
    )

    wrench = _sample_wrench_csv_replay(
        samples,
        replay_t=np.asarray([0.5]),
        elapsed_s=np.asarray([1.0]),
        period=0.0,
        transition_seconds=0.0,
    )

    np.testing.assert_allclose(wrench, [[1.0, 11.0, 21.0, 31.0, 41.0, 51.0]])
