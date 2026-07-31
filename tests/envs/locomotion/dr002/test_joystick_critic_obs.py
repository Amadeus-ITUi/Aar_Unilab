from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from unilab.envs.common.rotation import np_yaw_to_quat
from unilab.envs.locomotion.dr002.base import NoiseConfig
from unilab.envs.locomotion.dr002.joystick import (
    _CRITIC_DIM,
    _CRITIC_WITH_MEASURED_MOMENT_DIM,
    _DR002_ASSET_ROOT,
    _HISTORY_LENGTH,
    _LEGACY_CRITIC_DIM,
    DR002DomainRandConfig,
    DR002JoystickCfg,
    DR002JoystickDomainRandomizationProvider,
    DR002JoystickEnv,
    WingAngleObservationConfig,
    _apply_csv_force_channel_options,
    _csv_replay_channels_have_closed_seam,
    _csv_replay_has_closed_seam,
    _csv_wrench_effective_seam_columns,
    _load_force_csv,
    _sample_wrench_csv_replay,
    _transform_csv_wrench_to_base_com,
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
    env._critic_includes_measured_moment = False
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
        wing_angle_obs=WingAngleObservationConfig(
            enabled=True,
            normalization_deg=180.0,
            noise_half_range_deg=0.1,
        ),
        domain_rand=DR002DomainRandConfig(
            csv_force_enabled=True,
            csv_force_curriculum=True,
            csv_force_curriculum_hz=[0.0, 1.0, 2.0, 3.0, 4.0],
            csv_force_observation_force_normalization=50.0,
            csv_force_observation_include_measured_moment=False,
            csv_force_observation_moment_normalization=15.0,
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
    env._backend = _FakeBackend(np.asarray([[0.0, 0.0, 0.7], [0.0, 0.0, 1.8]], dtype=np.float32))
    env._privileged_base_mass_delta = np.asarray([[0.1], [0.2]], dtype=np.float32)
    env._privileged_base_com_offset = np.asarray(
        [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float32
    )
    env._default_joint_pos_offset = np.arange(num_envs * 6, dtype=np.float32).reshape(num_envs, 6)
    env._privileged_ground_friction_scale = np.asarray([[0.8], [0.9]], dtype=np.float32)
    env._privileged_robot_friction_scale = np.asarray([[1.1], [1.2]], dtype=np.float32)
    env._measured_csv_force_base = np.asarray(
        [[50.0, -25.0, 5.0], [100.0, 0.0, -50.0]],
        dtype=np.float32,
    )
    env._measured_csv_moment_base = np.asarray(
        [[15.0, -7.5, 1.5], [30.0, 0.0, -15.0]],
        dtype=np.float32,
    )
    return env


def _obs_inputs(num_envs: int = 2) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    current = np.arange(num_envs * 6, dtype=np.float32).reshape(num_envs, 6) + 1.0
    previous = current + 20.0
    qacc = current + 40.0
    torques = current + 60.0
    commands = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)[:num_envs]
    return {
        "info": {
            "current_actions": current,
            "last_actions": previous,
            "qacc": qacc,
            "torques": torques,
            "commands": commands,
        },
        "linvel": np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)[:num_envs],
        "gyro": np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)[:num_envs],
        "gravity": np.zeros((num_envs, 3), dtype=np.float32),
        "projected_gravity": np.asarray([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)[
            :num_envs
        ],
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


def test_measured_moment_extends_critic_only_to_147_dimensions() -> None:
    env = _bare_obs_env()
    env._critic_includes_measured_moment = True
    env._critic_dim = _CRITIC_WITH_MEASURED_MOMENT_DIM
    env._cfg.domain_rand.csv_force_observation_include_measured_moment = True
    inputs = _obs_inputs()

    first = env._compute_obs(**inputs, reset_history=True)
    env._measured_csv_moment_base += 3.0
    second = env._compute_obs(**inputs, reset_history=True)

    assert first["obs"].shape == (2, 135)
    assert first["critic"].shape == (2, 147)
    np.testing.assert_allclose(
        first["critic"][:, 144:147],
        [[1.0, -0.5, 0.1], [2.0, 0.0, -1.0]],
    )
    np.testing.assert_array_equal(first["obs"], second["obs"])
    np.testing.assert_array_equal(first["critic"][:, :144], second["critic"][:, :144])
    assert not np.array_equal(first["critic"][:, 144:147], second["critic"][:, 144:147])


def test_wing_gaussian_noise_changes_only_actor_wing_history(monkeypatch) -> None:
    env = _bare_obs_env()
    env._critic_includes_measured_moment = True
    env._critic_dim = _CRITIC_WITH_MEASURED_MOMENT_DIM
    env._csv_force_curriculum_level = 4
    env._cfg.wing_angle_obs.noise_half_range_deg = 0.0
    env._cfg.wing_angle_obs.gaussian_noise_relative_std = 0.0
    monkeypatch.setattr(env, "_obs_noise_at_level", lambda data, scale, level: data)
    monkeypatch.setattr(
        np.random,
        "normal",
        lambda loc, scale, size: np.ones(size, dtype=np.float32),
    )
    inputs = _obs_inputs()

    clean = env._compute_obs(**inputs, reset_history=True)
    env._cfg.wing_angle_obs.gaussian_noise_relative_std = 0.05
    noisy = env._compute_obs(**inputs, reset_history=True)

    assert clean["obs"].shape == (2, 135)
    assert clean["critic"].shape == (2, 147)
    np.testing.assert_array_equal(clean["obs"][:, :110], noisy["obs"][:, :110])
    np.testing.assert_allclose(
        noisy["obs"][:, 110:120],
        clean["obs"][:, 110:120] * 1.05,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(clean["obs"][:, 120:], noisy["obs"][:, 120:])
    np.testing.assert_array_equal(clean["critic"], noisy["critic"])
    np.testing.assert_array_equal(clean["privileged_target"], noisy["privileged_target"])


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
    assert ((2, 2), pytest.approx(0.1 / 180.0), 1.0) in calls


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


def test_we11_config_uses_episode_bias_and_slow_gravity_tilt_noise() -> None:
    config_path = Path(__file__).parents[4] / "conf/ppo/task/dr002_joystick_flat_we11/base.yaml"
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


def test_we11_base_config_owns_latest_network_observation_and_force_contract() -> None:
    config_path = Path(__file__).parents[4] / "conf/ppo/task/dr002_joystick_flat_we11/base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    env_cfg = config["env"]
    domain_rand = env_cfg["domain_rand"]

    assert domain_rand["csv_force_curriculum_hz"] == [0, 1, 2, 3]
    assert env_cfg["noise_config"]["level"] == 1.0
    assert env_cfg["noise_config"]["curriculum"] is False
    assert env_cfg["noise_config"]["curriculum_levels"] == [1.0] * 4
    assert env_cfg["control_config"]["action_delay_min_steps"] == 2
    assert env_cfg["control_config"]["action_delay_max_steps"] == 8
    assert env_cfg["control_config"]["resample_action_delay"] is True
    assert domain_rand["push_force_limit"] == [10.0, 10.0, 0.0]
    assert domain_rand["csv_force_zero_fy"] is True
    assert config["reward"]["scales"]["joint_acc_wheel_l2"] == pytest.approx(-2.5e-7)
    assert len(domain_rand["csv_force_curriculum_paths"]) == 3
    assert len(env_cfg["wing_angle_obs"]["curriculum_paths"]) == 3
    assert all(
        "we11/training_data/measured_wrench_20260728_skin" in path
        for path in domain_rand["csv_force_curriculum_paths"]
    )
    assert all(
        "we11/training_data/wing_angle_20260713" in path
        for path in env_cfg["wing_angle_obs"]["curriculum_paths"]
    )
    assert config["algo"]["actor"]["history_term_dims"] == [3, 3, 4, 6, 6, 2, 3]


@pytest.mark.parametrize("hz", [1, 2, 3])
def test_latest_measured_wrench_assets_preserve_all_six_axes(hz: int) -> None:
    path = _DR002_ASSET_ROOT / "we11/training_data/measured_wrench_20260728_skin" / f"{hz}hz.csv"
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


def test_closed_wrench_seam_preserves_raw_tail_without_period_warp() -> None:
    samples = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
            [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    replay_t = np.asarray([0.95], dtype=np.float64)

    assert _csv_replay_has_closed_seam(samples, period=1.0, num_channels=6)
    wrench = _sample_wrench_csv_replay(
        samples,
        replay_t=replay_t,
        elapsed_s=np.asarray([1.95]),
        period=1.0,
        transition_seconds=0.1,
    )
    raw = np.asarray(
        [np.interp(replay_t[0], samples[:, 0], samples[:, channel]) for channel in range(1, 7)]
    )

    np.testing.assert_allclose(wrench[0], raw, rtol=0.0, atol=1.0e-12)


def test_open_wrench_seam_retains_compatibility_tail_blend() -> None:
    samples = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        ],
        dtype=np.float64,
    )

    assert not _csv_replay_has_closed_seam(samples, period=1.0, num_channels=6)
    wrench = _sample_wrench_csv_replay(
        samples,
        replay_t=np.asarray([0.95]),
        elapsed_s=np.asarray([1.95]),
        period=1.0,
        transition_seconds=0.1,
    )
    raw = np.asarray([1.9, 3.8, 5.7, 7.6, 9.5, 11.4])

    # At the middle of the 0.1 s tail transition, smoothstep alpha is 0.5.
    np.testing.assert_allclose(wrench[0], 0.5 * raw, rtol=0.0, atol=1.0e-12)


def test_closed_wrench_seam_keeps_only_episode_start_smoothstep() -> None:
    samples = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    replay_t = np.asarray([0.05, 0.05])

    wrench = _sample_wrench_csv_replay(
        samples,
        replay_t=replay_t,
        elapsed_s=np.asarray([0.05, 1.05]),
        period=1.0,
        transition_seconds=0.1,
    )
    raw = np.asarray(
        [np.interp(replay_t[0], samples[:, 0], samples[:, channel]) for channel in range(1, 7)]
    )

    np.testing.assert_allclose(wrench[0], 0.5 * raw, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(wrench[1], raw, rtol=0.0, atol=1.0e-12)


def test_closed_seam_detection_uses_exact_activity_coverage_boundaries() -> None:
    exact = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    late_start = exact.copy()
    late_start[0, 0] = 5.0e-10
    short_end = exact.copy()
    short_end[-1, 0] = 1.0 - 5.0e-10

    assert _csv_replay_has_closed_seam(exact, period=1.0, num_channels=6)
    assert not _csv_replay_has_closed_seam(late_start, period=1.0, num_channels=6)
    assert not _csv_replay_has_closed_seam(short_end, period=1.0, num_channels=6)


def test_closed_seam_detection_can_ignore_physically_unused_moment_channels() -> None:
    samples = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            [1.0, 1.0, 2.0, 3.0, 11.0, 22.0, 33.0],
        ],
        dtype=np.float64,
    )

    assert _csv_replay_channels_have_closed_seam(
        samples,
        period=1.0,
        channel_columns=(1, 2, 3),
    )
    assert not _csv_replay_has_closed_seam(samples, period=1.0, num_channels=6)


def test_effective_seam_channels_include_critic_only_measured_moment() -> None:
    force_only = DR002DomainRandConfig(
        csv_force_zero_fy=True,
        csv_force_apply_measured_moment=False,
        csv_force_observation_include_measured_moment=False,
    )
    critic_moment = DR002DomainRandConfig(
        csv_force_zero_fy=True,
        csv_force_apply_measured_moment=False,
        csv_force_observation_include_measured_moment=True,
    )

    assert _csv_wrench_effective_seam_columns(force_only) == (1, 3)
    assert _csv_wrench_effective_seam_columns(critic_moment) == (1, 3, 4, 5, 6)


def test_csv_wrench_amplitude_scale_resamples_only_reset_environments(monkeypatch) -> None:
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(
        domain_rand=DR002DomainRandConfig(
            csv_force_amplitude_scale_range=[0.5, 1.5],
        )
    )
    env._num_envs = 4
    env._csv_force_amplitude_scales = np.ones((4,), dtype=np.float32)
    draws = iter(
        (
            np.asarray([0.55, 0.75, 1.25, 1.45], dtype=np.float64),
            np.asarray([0.65, 1.35], dtype=np.float64),
        )
    )

    def _uniform(low: float, high: float, size: tuple[int, ...]) -> np.ndarray:
        assert low == pytest.approx(0.5)
        assert high == pytest.approx(1.5)
        value = next(draws)
        assert value.shape == size
        return value

    monkeypatch.setattr(np.random, "uniform", _uniform)
    initial = env.sample_reset_csv_force_amplitude_scales(4)
    env.set_csv_force_amplitude_scales(np.arange(4, dtype=np.int32), initial)
    before_partial_reset = env.csv_force_amplitude_scales().copy()

    partial = env.sample_reset_csv_force_amplitude_scales(2)
    env.set_csv_force_amplitude_scales(np.asarray([1, 3], dtype=np.int32), partial)

    np.testing.assert_allclose(
        before_partial_reset,
        [0.55, 0.75, 1.25, 1.45],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        env.csv_force_amplitude_scales(),
        [0.55, 0.65, 1.25, 1.35],
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    "scale_range",
    (
        [0.5],
        [1.5, 0.5],
        [-0.1, 1.0],
        [np.nan, 1.0],
    ),
)
def test_csv_wrench_amplitude_scale_rejects_invalid_ranges(
    scale_range: list[float],
) -> None:
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(
        domain_rand=DR002DomainRandConfig(
            csv_force_amplitude_scale_range=scale_range,
        )
    )

    with pytest.raises(ValueError, match="csv_force_amplitude_scale_range"):
        env._csv_force_amplitude_scale_bounds()


def test_csv_wrench_transform_can_select_measured_and_transport_moments() -> None:
    wrench_sensor = np.asarray([[1.0, 2.0, 3.0, 40.0, 50.0, 60.0]], dtype=np.float64)
    moment_arm_base = np.asarray([0.5, -0.25, 0.75], dtype=np.float64)

    transport_only = _transform_csv_wrench_to_base_com(
        wrench_sensor,
        sensor_to_base_rotation=np.eye(3),
        moment_arm_base=moment_arm_base,
        zero_fy=False,
        apply_measured_moment=False,
        apply_point_torque=True,
    )
    measured_only = _transform_csv_wrench_to_base_com(
        wrench_sensor,
        sensor_to_base_rotation=np.eye(3),
        moment_arm_base=moment_arm_base,
        zero_fy=False,
        apply_measured_moment=True,
        apply_point_torque=False,
    )

    np.testing.assert_allclose(transport_only, [[1.0, 2.0, 3.0, -2.25, -0.75, 1.25]])
    np.testing.assert_allclose(measured_only, [[1.0, 2.0, 3.0, 40.0, 50.0, 60.0]])


@pytest.mark.parametrize("amplitude_scale", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("apply_to_standing", [False, True])
@pytest.mark.parametrize("resume_suppressed", [False, True])
def test_csv_full_wrench_aligns_sensor_base_and_world_frames(
    tmp_path: Path,
    amplitude_scale: float,
    apply_to_standing: bool,
    resume_suppressed: bool,
) -> None:
    csv_path = tmp_path / "measured_wrench.csv"
    csv_path.write_text(
        "time,Fx,Fy,Fz,Mx,My,Mz\n0.0,1.0,2.0,3.0,4.0,5.0,6.0\n1.0,1.0,2.0,3.0,4.0,5.0,6.0\n",
        encoding="utf-8",
    )
    sensor_to_base = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    domain_rand = DR002DomainRandConfig(
        csv_force_enabled=True,
        csv_force_path=str(csv_path),
        csv_force_curriculum_hz=[0.0, 1.0],
        csv_force_curriculum=False,
        csv_force_period=0.0,
        csv_force_transition_seconds=0.0,
        csv_force_apply_to_standing=apply_to_standing,
        csv_force_zero_fy=True,
        csv_force_rotation=sensor_to_base.reshape(-1).tolist(),
        csv_force_push_point=[0.5, -0.1, 0.75],
        csv_force_apply_measured_moment=True,
        csv_force_apply_point_torque=True,
    )
    base_quat = np_yaw_to_quat(np.asarray([np.pi / 2.0], dtype=np.float64))
    backend = SimpleNamespace(
        get_base_quat=lambda: base_quat,
        get_body_id=lambda name: 0,
        get_body_ipos=lambda: np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            domain_rand=domain_rand,
            sim_substeps=1,
            sim_dt=0.01,
            ctrl_dt=0.01,
            asset=SimpleNamespace(base_name="base_link"),
        ),
        _backend=backend,
        _num_envs=1,
        _standing_envs_episode_persistent=True,
        _csv_force_curriculum_level=1,
        _startup_stand_steps=0,
        _privileged_base_com_offset=np.asarray([[0.1, -0.1, 0.2]], dtype=np.float64),
        _external_disturbance_current_wrench=np.zeros((1, 6), dtype=np.float64),
        _measured_csv_force_base=np.zeros((1, 3), dtype=np.float64),
        _measured_csv_moment_base=np.zeros((1, 3), dtype=np.float64),
        episode_steps=lambda: np.asarray([0], dtype=np.int64),
        csv_force_start_delay_steps=lambda: np.asarray([0], dtype=np.int64),
        csv_force_amplitude_scales=lambda: np.asarray([amplitude_scale], dtype=np.float64),
        csv_force_resume_suppressed=lambda: np.asarray([resume_suppressed]),
        episode_standing_mask=lambda: np.asarray([True]),
    )

    trajectory = DR002JoystickDomainRandomizationProvider()._build_csv_force_trajectory(
        env, step_counter=0
    )

    assert trajectory is not None
    assert trajectory.shape == (1, 1, 1, 6)
    # zero_fy acts in sensor axes: F_s=[1,0,3]. Site rotation gives
    # F_b=[0,1,3], while all measured moment axes remain active.
    effective_scale = amplitude_scale if apply_to_standing and not resume_suppressed else 0.0
    np.testing.assert_allclose(
        env._measured_csv_force_base,
        np.asarray([[0.0, 1.0, 3.0]]) * effective_scale,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        env._measured_csv_moment_base,
        np.asarray([[-5.0, 4.0, 6.0]]) * effective_scale,
        atol=1.0e-12,
    )
    # M_b=[-5,4,6] and r=[0.3,-0.2,0.25], so the COM moment is
    # M_b+r x F_b=[-5.85,3.1,6.3]. A +90 deg base yaw then maps both
    # force and moment into MuJoCo's world-frame xfrc_applied contract.
    expected_world = np.asarray([-1.0, 0.0, 3.0, -3.1, -5.85, 6.3]) * effective_scale
    np.testing.assert_allclose(trajectory[0, 0, 0], expected_world, atol=1.0e-12)
    np.testing.assert_allclose(
        env._external_disturbance_current_wrench[0],
        expected_world,
        atol=1.0e-12,
    )


def _bare_curriculum_checkpoint_env(
    *,
    window_episodes: int = 100,
) -> DR002JoystickEnv:
    env = object.__new__(DR002JoystickEnv)
    env._cfg = SimpleNamespace(
        commands=SimpleNamespace(
            curriculum=True,
            curriculum_min_episode_fraction=0.5,
            curriculum_moving_command_threshold=0.05,
        ),
        domain_rand=DR002DomainRandConfig(
            csv_force_enabled=True,
            csv_force_curriculum=True,
            csv_force_curriculum_hz=[0.0, 1.0, 2.0, 3.0],
            csv_force_curriculum_paths=[
                "measured_wrench/test/1hz.csv",
                "measured_wrench/test/2hz.csv",
                "measured_wrench/test/3hz.csv",
            ],
            csv_force_curriculum_window_episodes=window_episodes,
        ),
        noise_config=NoiseConfig(curriculum=False),
        max_episode_steps=1000,
    )
    env._reward_cfg = SimpleNamespace(
        tracking_sigma=0.25,
        track_lin_vel_x_std=0.5,
    )
    env._command_curriculum_initial_scale = 1.0
    env._command_curriculum_final_scale = 2.0
    env._command_curriculum_scale = 2.0
    env._command_curriculum_yaw_initial_scale = 0.5
    env._command_curriculum_yaw_final_scale = 1.0
    env._command_curriculum_yaw_scale = 1.0
    env._csv_force_curriculum_level = 3
    env._csv_force_curriculum_num_promoted = 3
    env._csv_force_curriculum_num_demoted = 1
    env._csv_force_curriculum_last_direction = 1
    env._csv_force_window_episodes = 50
    env._csv_force_window_tracking_episodes = 40
    env._csv_force_window_tracking_sum = 32.0
    env._csv_force_window_episode_length_fraction_sum = 45.0
    env._csv_force_window_mature_count = 44
    env._csv_force_window_fail_count = 2
    env._csv_force_active_levels = np.asarray([3, 2, 1, 0], dtype=np.int32)
    env._csv_force_start_delay_steps = np.asarray([150, 160, 170, 180], dtype=np.int32)
    env._csv_force_resume_pending_reset = np.zeros((4,), dtype=np.bool_)
    env._external_disturbance_current_wrench = np.ones((4, 6), dtype=np.float32)
    env._measured_csv_force_base = np.ones((4, 3), dtype=np.float32)
    env._measured_csv_moment_base = np.ones((4, 3), dtype=np.float32)
    env._last_csv_force_window_tracking = 0.8
    env._last_csv_force_window_fail_rate = 0.1
    env._last_csv_force_window_mean_episode_length_fraction = 0.9
    env._last_csv_force_window_mature_fraction = 0.88
    env._noise_curriculum_level = 0
    env._noise_curriculum_num_promoted = 0
    env._noise_curriculum_num_demoted = 0
    env._noise_curriculum_last_direction = 0
    env._noise_window_episodes = 0
    env._noise_window_tracking_episodes = 0
    env._noise_window_tracking_sum = 0.0
    env._noise_window_episode_length_fraction_sum = 0.0
    env._noise_window_mature_count = 0
    env._noise_window_fail_count = 0
    env._episode_standing_mask = np.asarray([True, False, True, False])
    env._state = SimpleNamespace(info={"steps": np.asarray([1149, 700, 250, 0], dtype=np.uint32)})
    return env


def test_curriculum_training_state_round_trip_preserves_only_aggregate_progress() -> None:
    source = _bare_curriculum_checkpoint_env()
    saved = source.training_state_dict()
    target = _bare_curriculum_checkpoint_env()
    target._command_curriculum_scale = 1.0
    target._command_curriculum_yaw_scale = 0.5
    target._csv_force_curriculum_level = 0
    target._csv_force_curriculum_num_promoted = 0
    target._csv_force_curriculum_num_demoted = 0
    target._csv_force_curriculum_last_direction = 0
    target._reset_csv_force_curriculum_window()
    episode_steps_before = target._state.info["steps"].copy()
    standing_before = target._episode_standing_mask.copy()

    target.load_training_state_dict(saved)

    assert target._command_curriculum_scale == pytest.approx(2.0)
    assert target._command_curriculum_yaw_scale == pytest.approx(1.0)
    assert target._csv_force_curriculum_level == 3
    assert target._csv_force_curriculum_num_promoted == 3
    assert target._csv_force_curriculum_num_demoted == 1
    assert target._csv_force_curriculum_last_direction == 1
    assert "window" not in saved["csv_force_curriculum"]
    assert target._csv_force_window_episodes == 0
    assert target._csv_force_window_tracking_episodes == 0
    assert target._csv_force_window_tracking_sum == pytest.approx(0.0)
    assert target._csv_force_window_episode_length_fraction_sum == pytest.approx(0.0)
    assert target._csv_force_window_mature_count == 0
    assert target._csv_force_window_fail_count == 0
    np.testing.assert_array_equal(target._csv_force_active_levels, 0)
    np.testing.assert_array_equal(target._csv_force_start_delay_steps, 0)
    np.testing.assert_array_equal(target._csv_force_resume_pending_reset, True)
    np.testing.assert_array_equal(target._external_disturbance_current_wrench, 0.0)
    np.testing.assert_array_equal(target._measured_csv_force_base, 0.0)
    np.testing.assert_array_equal(target._measured_csv_moment_base, 0.0)
    np.testing.assert_array_equal(target._state.info["steps"], episode_steps_before)
    np.testing.assert_array_equal(target._episode_standing_mask, standing_before)

    target._update_command_curriculum(np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(target._csv_force_resume_pending_reset, False)
    assert target._csv_force_window_episodes == 0
    assert target._csv_force_window_tracking_sum == pytest.approx(0.0)


def test_curriculum_restore_resets_partial_window_when_force_config_changes() -> None:
    source = _bare_curriculum_checkpoint_env(window_episodes=100)
    target = _bare_curriculum_checkpoint_env(window_episodes=200)

    target.load_training_state_dict(source.training_state_dict())

    assert target._csv_force_curriculum_level == 3
    assert target._csv_force_curriculum_num_promoted == 3
    assert target._csv_force_window_episodes == 0
    assert target._csv_force_window_tracking_episodes == 0
    assert target._csv_force_window_tracking_sum == pytest.approx(0.0)


def test_curriculum_restore_keeps_force_zero_until_command_is_full() -> None:
    source = _bare_curriculum_checkpoint_env()
    saved = source.training_state_dict()
    saved["command_curriculum"]["scale"] = 1.5
    target = _bare_curriculum_checkpoint_env()

    target.load_training_state_dict(saved)

    assert target._command_curriculum_scale == pytest.approx(1.5)
    assert not target._command_curriculum_is_full()
    assert target._csv_force_curriculum_level == 0
    assert target._csv_force_window_episodes == 0
    np.testing.assert_array_equal(target._csv_force_active_levels, 0)


def test_curriculum_restore_preserves_duplicate_hz_level_for_exact_mapping() -> None:
    source = _bare_curriculum_checkpoint_env()
    target = _bare_curriculum_checkpoint_env()
    for env in (source, target):
        env._cfg.domain_rand.csv_force_curriculum_hz = [0.0, 1.0, 1.0, 2.0]
        env._cfg.domain_rand.csv_force_curriculum_paths = [
            "measured_wrench/test/1hz_a.csv",
            "measured_wrench/test/1hz_b.csv",
            "measured_wrench/test/2hz.csv",
        ]
    source._csv_force_curriculum_level = 2

    saved = source.training_state_dict()
    assert all(
        path.startswith("asset://") for path in saved["csv_force_curriculum"]["config"]["paths"]
    )
    target.load_training_state_dict(saved)

    assert target._csv_force_curriculum_level == 2

    migrated_target = _bare_curriculum_checkpoint_env()
    migrated_target._cfg.domain_rand.csv_force_curriculum_hz = [0.0, 1.0, 1.0, 2.0]
    migrated_target._cfg.domain_rand.csv_force_curriculum_paths = [
        "measured_wrench/migrated/1hz_a.csv",
        "measured_wrench/migrated/1hz_b.csv",
        "measured_wrench/migrated/2hz.csv",
    ]
    migrated_target.load_training_state_dict(saved)
    assert migrated_target._csv_force_curriculum_level == 2


def test_curriculum_restore_never_maps_removed_hz_to_a_harder_level() -> None:
    source = _bare_curriculum_checkpoint_env()
    source._csv_force_curriculum_level = 2
    saved = source.training_state_dict()
    target = _bare_curriculum_checkpoint_env()
    target._cfg.domain_rand.csv_force_curriculum_hz = [0.0, 1.0, 3.0]
    target._cfg.domain_rand.csv_force_curriculum_paths = [
        "measured_wrench/test/1hz.csv",
        "measured_wrench/test/3hz.csv",
    ]

    target.load_training_state_dict(saved)

    assert target._csv_force_curriculum_level == 1


def test_curriculum_restore_preserves_path_only_level_for_exact_mapping() -> None:
    source = _bare_curriculum_checkpoint_env()
    target = _bare_curriculum_checkpoint_env()
    for env in (source, target):
        env._cfg.domain_rand.csv_force_curriculum_hz = []
        env._cfg.domain_rand.csv_force_curriculum_paths = [
            "measured_wrench/test/easy.csv",
            "measured_wrench/test/medium.csv",
            "measured_wrench/test/hard.csv",
        ]
    source._csv_force_curriculum_level = 2

    target.load_training_state_dict(source.training_state_dict())

    assert target._csv_force_curriculum_level == 2


def test_curriculum_restore_round_trips_independent_noise_progress() -> None:
    source = _bare_curriculum_checkpoint_env()
    target = _bare_curriculum_checkpoint_env()
    for env in (source, target):
        env._cfg.domain_rand.csv_force_curriculum = False
        env._cfg.noise_config = NoiseConfig(
            curriculum=True,
            curriculum_levels=[0.0, 0.5, 1.0],
        )
    source._noise_curriculum_level = 2
    source._noise_curriculum_num_promoted = 4
    source._noise_curriculum_num_demoted = 1
    source._noise_curriculum_last_direction = 1
    source._noise_window_episodes = 20
    source._noise_window_tracking_episodes = 18
    source._noise_window_tracking_sum = 12.0
    source._noise_window_episode_length_fraction_sum = 17.0
    source._noise_window_mature_count = 16
    source._noise_window_fail_count = 2
    target._noise_curriculum_level = 0
    target._reset_noise_curriculum_window()

    target.load_training_state_dict(source.training_state_dict())

    assert target._noise_curriculum_level == 2
    assert target._noise_curriculum_num_promoted == 4
    assert target._noise_curriculum_num_demoted == 1
    assert target._noise_curriculum_last_direction == 1
    assert target._noise_window_episodes == 0
    assert target._noise_window_tracking_episodes == 0
    assert target._noise_window_tracking_sum == pytest.approx(0.0)


def test_disabled_empty_noise_levels_remain_checkpoint_safe() -> None:
    source = _bare_curriculum_checkpoint_env()
    target = _bare_curriculum_checkpoint_env()
    source._cfg.noise_config = NoiseConfig(curriculum=False, curriculum_levels=[])
    target._cfg.noise_config = NoiseConfig(curriculum=False, curriculum_levels=[])

    saved = source.training_state_dict()
    target.load_training_state_dict(saved)

    assert saved["noise_curriculum"]["config"]["levels"] == []
    assert target._noise_curriculum_level == 0
