from types import SimpleNamespace

import numpy as np
import pytest

import unilab.envs.locomotion.dr002.joystick as joystick_module
from unilab.envs.locomotion.dr002.joystick import (
    DR002DomainRandConfig,
    DR002JoystickDomainRandomizationProvider,
    DR002JoystickEnv,
    WingAngleObservationConfig,
    _csv_force_zero_hz_mask,
)


def _bare_wing_env() -> DR002JoystickEnv:
    env = object.__new__(DR002JoystickEnv)
    env._np_dtype = np.dtype(np.float32)
    env._wing_angle_obs_enabled = True
    env._cfg = SimpleNamespace(
        wing_angle_obs=WingAngleObservationConfig(
            enabled=True,
            standing_normalized_range=[-0.05, 0.05],
        )
    )
    env._wing_angle_obs_amplitude_scales = np.ones((4,), dtype=np.float32)
    return env


def test_zero_hz_mask_uses_frequency_labels_instead_of_level_index() -> None:
    domain_rand = DR002DomainRandConfig(csv_force_curriculum_hz=[2.0, 0.0, 1.0])

    mask = _csv_force_zero_hz_mask(domain_rand, np.asarray([0, 1, 2, 1]))

    np.testing.assert_array_equal(mask, [False, True, False, True])


def test_sample_zero_hz_wing_obs_only_fills_masked_rows(monkeypatch) -> None:
    env = _bare_wing_env()
    env._zero_hz_wing_angle_obs = np.full((4, 2), -0.04, dtype=np.float32)
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda low, high, size: np.full(size, 0.025, dtype=np.float64),
    )

    sampled = env.sample_reset_zero_hz_wing_angle_obs(np.asarray([True, False, True, False]))

    assert sampled.shape == (4, 2)
    assert sampled.dtype == np.float32
    np.testing.assert_allclose(sampled[[0, 2]], 0.025)
    np.testing.assert_array_equal(sampled[[1, 3]], 0.0)

    env.set_zero_hz_wing_angle_obs(np.arange(4, dtype=np.int32), sampled)
    np.testing.assert_array_equal(env._zero_hz_wing_angle_obs[[1, 3]], 0.0)


def test_wing_obs_amplitude_scale_resamples_only_reset_environments(monkeypatch) -> None:
    env = _bare_wing_env()
    env._num_envs = 4
    env._cfg.wing_angle_obs.csv_amplitude_scale_range = [0.8, 1.2]
    draws = iter(
        (
            np.asarray([0.82, 0.94, 1.06, 1.18], dtype=np.float64),
            np.asarray([0.88, 1.12], dtype=np.float64),
        )
    )

    def _uniform(low: float, high: float, size: tuple[int, ...]) -> np.ndarray:
        assert low == pytest.approx(0.8)
        assert high == pytest.approx(1.2)
        value = next(draws)
        assert value.shape == size
        return value

    monkeypatch.setattr(np.random, "uniform", _uniform)
    initial = env.sample_reset_wing_angle_obs_amplitude_scales(4)
    env.set_wing_angle_obs_amplitude_scales(np.arange(4, dtype=np.int32), initial)
    before_partial_reset = env.wing_angle_obs_amplitude_scales().copy()

    partial = env.sample_reset_wing_angle_obs_amplitude_scales(2)
    env.set_wing_angle_obs_amplitude_scales(np.asarray([1, 3], dtype=np.int32), partial)

    np.testing.assert_allclose(
        before_partial_reset,
        [0.82, 0.94, 1.06, 1.18],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        env.wing_angle_obs_amplitude_scales(),
        [0.82, 0.88, 1.06, 1.12],
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    "scale_range",
    (
        [0.8],
        [1.2, 0.8],
        [-0.1, 1.0],
        [np.nan, 1.0],
    ),
)
def test_wing_obs_amplitude_scale_rejects_invalid_ranges(
    scale_range: list[float],
) -> None:
    cfg = WingAngleObservationConfig(
        enabled=True,
        csv_amplitude_scale_range=scale_range,
    )

    with pytest.raises(ValueError, match="csv_amplitude_scale_range"):
        joystick_module._wing_angle_obs_csv_amplitude_scale_bounds(cfg)


def test_wing_obs_multiplicative_gaussian_noise_uses_five_percent_per_channel(
    monkeypatch,
) -> None:
    clean = np.asarray([[0.4, -0.2], [0.5, -0.25]], dtype=np.float32)
    standard_normal = np.asarray([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)

    def _normal(
        loc: float,
        scale: float,
        size: tuple[int, ...],
    ) -> np.ndarray:
        assert loc == pytest.approx(0.0)
        assert scale == pytest.approx(1.0)
        assert size == clean.shape
        return standard_normal.copy()

    monkeypatch.setattr(np.random, "normal", _normal)
    noisy = DR002JoystickEnv._obs_multiplicative_gaussian_noise_at_level(
        clean,
        relative_std=0.05,
        level=1.0,
    )

    np.testing.assert_allclose(
        noisy,
        clean * (1.0 + 0.05 * standard_normal),
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        DR002JoystickEnv._obs_multiplicative_gaussian_noise_at_level(
            clean,
            relative_std=0.05,
            level=0.0,
        ),
        clean,
    )
    np.testing.assert_allclose(
        clean,
        [[0.4, -0.2], [0.5, -0.25]],
        atol=1.0e-7,
    )


def test_compute_wing_obs_reuses_episode_value_for_all_zero_hz_rows() -> None:
    env = _bare_wing_env()
    env._num_envs = 3
    env._startup_stand_steps = 0
    env._cfg = SimpleNamespace(
        domain_rand=DR002DomainRandConfig(
            csv_force_curriculum_hz=[0.0],
            csv_force_curriculum_paths=[],
            csv_force_period=10.0,
        ),
        wing_angle_obs=env._cfg.wing_angle_obs,
        ctrl_dt=0.02,
    )
    env._csv_force_active_levels = np.zeros((3,), dtype=np.int32)
    env._csv_force_start_delay_steps = np.zeros((3,), dtype=np.int32)
    env._zero_hz_wing_angle_obs = np.asarray(
        [[0.01, -0.02], [0.03, 0.04], [-0.05, 0.02]],
        dtype=np.float32,
    )
    env._wing_angle_obs_amplitude_scales[:3] = [0.8, 1.0, 1.2]
    env._episode_standing_mask = np.asarray([True, False, False])
    env._state = SimpleNamespace(info={"steps": np.asarray([5, 7, 9], dtype=np.int64)})

    first = env._compute_wing_angle_obs(3, None, preview_next_step=False)
    second = env._compute_wing_angle_obs(3, None, preview_next_step=True)
    subset = env._compute_wing_angle_obs(
        2,
        np.asarray([2, 1], dtype=np.int32),
        preview_next_step=False,
    )

    np.testing.assert_array_equal(first, env._zero_hz_wing_angle_obs)
    np.testing.assert_array_equal(second, env._zero_hz_wing_angle_obs)
    np.testing.assert_array_equal(subset, env._zero_hz_wing_angle_obs[[2, 1]])


def test_positive_hz_csv_motor_position_scale_is_obs_only_and_force_independent(
    tmp_path,
) -> None:
    force_path = tmp_path / "force.csv"
    force_path.write_text(
        "time,Fx,Fy,Fz,Mx,My,Mz\n0.0,1.0,2.0,3.0,4.0,5.0,6.0\n1.0,1.0,2.0,3.0,4.0,5.0,6.0\n",
        encoding="utf-8",
    )
    angle_path = tmp_path / "motor_position.csv"
    angle_path.write_text(
        "time,left,right\n0.0,0.0,0.0\n0.5,180.0,-90.0\n1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    env = _bare_wing_env()
    env._num_envs = 3
    env._startup_stand_steps = 0
    env._cfg = SimpleNamespace(
        domain_rand=DR002DomainRandConfig(
            csv_force_enabled=True,
            csv_force_path=str(force_path),
            csv_force_curriculum_paths=[str(force_path)],
            csv_force_curriculum_hz=[1.0],
            csv_force_period=1.0,
            csv_force_transition_seconds=0.0,
        ),
        wing_angle_obs=WingAngleObservationConfig(
            enabled=True,
            curriculum_paths=[str(angle_path)],
            normalization_deg=180.0,
            zero_offsets_deg=[0.0, 0.0],
            csv_amplitude_scale_range=[0.8, 1.2],
        ),
        ctrl_dt=0.01,
    )
    env._csv_force_active_levels = np.zeros((3,), dtype=np.int32)
    env._csv_force_start_delay_steps = np.zeros((3,), dtype=np.int32)
    env._csv_force_amplitude_scales = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    env._wing_angle_obs_amplitude_scales = np.asarray([0.8, 1.0, 1.2], dtype=np.float32)
    env._zero_hz_wing_angle_obs = np.zeros((3, 2), dtype=np.float32)
    # Standing is a command contract only; positive-Hz rows 0 and 2 must use
    # the same paired motor-position replay as the moving row.
    env._episode_standing_mask = np.asarray([True, False, True])
    env._wing_angle_force_samples_by_path = {}
    env._wing_angle_samples_by_path = {}
    env._state = SimpleNamespace(info={"steps": np.full((3,), 25, dtype=np.int64)})

    before_scale_state = env.wing_angle_obs_amplitude_scales().copy()
    first = env._compute_wing_angle_obs(3, None, preview_next_step=False)
    env._csv_force_amplitude_scales[:] = [1.5, 0.5, 0.9]
    second = env._compute_wing_angle_obs(3, None, preview_next_step=False)

    expected = np.asarray(
        [[0.4, -0.2], [0.5, -0.25], [0.6, -0.3]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(first, expected, atol=1.0e-6)
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(
        env.wing_angle_obs_amplitude_scales(),
        before_scale_state,
    )

    env._csv_force_resume_pending_reset = np.asarray([False, True, False])
    suppressed = env._compute_wing_angle_obs(3, None, preview_next_step=False)
    np.testing.assert_allclose(
        suppressed,
        np.asarray([[0.4, -0.2], [0.0, 0.0], [0.6, -0.3]], dtype=np.float32),
        atol=1.0e-6,
    )


def test_compute_wing_obs_first_reset_does_not_require_env_state() -> None:
    env = _bare_wing_env()
    env._num_envs = 1
    env._startup_stand_steps = 150
    env._cfg = SimpleNamespace(
        domain_rand=DR002DomainRandConfig(
            csv_force_curriculum_hz=[0.0],
            csv_force_curriculum_paths=[],
            csv_force_period=10.0,
        ),
        wing_angle_obs=env._cfg.wing_angle_obs,
        ctrl_dt=0.02,
    )
    env._csv_force_active_levels = np.zeros((1,), dtype=np.int32)
    env._csv_force_start_delay_steps = np.zeros((1,), dtype=np.int32)
    env._zero_hz_wing_angle_obs = np.asarray([[0.02, -0.03]], dtype=np.float32)
    env._episode_standing_mask = np.asarray([False])
    env._state = None

    obs = env._compute_wing_angle_obs(
        1,
        np.asarray([0], dtype=np.int32),
        preview_next_step=False,
        episode_steps_override=np.zeros((1,), dtype=np.int64),
    )

    np.testing.assert_array_equal(obs, env._zero_hz_wing_angle_obs)


@pytest.mark.parametrize(
    (
        "apply_to_standing",
        "expected_zero_hz_mask",
        "expected_active_levels",
        "expected_start_delays",
    ),
    [
        (False, [True, True, False], [0, 0, 1], [0, 6, 7]),
        (True, [False, True, False], [2, 0, 1], [5, 6, 7]),
    ],
)
def test_reset_plan_configures_standing_force_replay(
    monkeypatch,
    apply_to_standing: bool,
    expected_zero_hz_mask: list[bool],
    expected_active_levels: list[int],
    expected_start_delays: list[int],
) -> None:
    class _Spawn:
        @staticmethod
        def origins_for(env_ids: np.ndarray) -> np.ndarray:
            return np.zeros((env_ids.size, 3), dtype=np.float32)

    class _FakeEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                domain_rand=DR002DomainRandConfig(
                    csv_force_curriculum_hz=[0.0, 1.0, 2.0],
                    csv_force_apply_to_standing=apply_to_standing,
                    init_xy_range=[0.0, 0.0],
                    randomize_init_yaw=False,
                )
            )
            self._init_qpos = np.zeros((13,), dtype=np.float32)
            self._init_qvel = np.zeros((12,), dtype=np.float32)
            self._spawn = _Spawn()
            self._startup_stand_steps = 1
            self._standing_envs_episode_persistent = True
            self._num_action = 6
            self.sampled_zero_hz_mask = None
            self.active_levels = None
            self.start_delays = None
            self.amplitude_scales = None
            self.wing_obs_amplitude_scales = None

        @staticmethod
        def sample_reset_motor_gains(num_reset: int) -> tuple[np.ndarray, np.ndarray]:
            values = np.zeros((num_reset, 6), dtype=np.float32)
            return values, values

        @staticmethod
        def set_motor_gains(env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
            return None

        @staticmethod
        def sample_reset_motor_runtime_randomization(
            num_reset: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.ones((num_reset, 6), dtype=np.float32),
                np.zeros((num_reset, 6), dtype=np.float32),
            )

        @staticmethod
        def set_motor_runtime_randomization(
            env_ids: np.ndarray,
            torque_scale: np.ndarray,
            default_joint_pos_offset: np.ndarray,
        ) -> None:
            return None

        @staticmethod
        def sample_reset_standing_mask(env_ids: np.ndarray) -> np.ndarray:
            return np.asarray([True, False, False])

        @staticmethod
        def set_episode_standing_mask(
            env_ids: np.ndarray,
            standing_mask: np.ndarray,
        ) -> None:
            return None

        @staticmethod
        def sample_reset_csv_force_levels(num_reset: int) -> np.ndarray:
            return np.asarray([2, 0, 1], dtype=np.int32)

        @staticmethod
        def sample_reset_csv_force_start_delay_steps(num_reset: int) -> np.ndarray:
            return np.asarray([5, 6, 7], dtype=np.int32)

        @staticmethod
        def sample_reset_csv_force_amplitude_scales(num_reset: int) -> np.ndarray:
            return np.asarray([0.6, 0.9, 1.4], dtype=np.float32)

        @staticmethod
        def sample_reset_wing_angle_obs_amplitude_scales(num_reset: int) -> np.ndarray:
            return np.asarray([0.85, 1.0, 1.15], dtype=np.float32)

        def sample_reset_zero_hz_wing_angle_obs(self, mask: np.ndarray) -> np.ndarray:
            self.sampled_zero_hz_mask = np.array(mask, copy=True)
            return np.zeros((mask.size, 2), dtype=np.float32)

        @staticmethod
        def set_zero_hz_wing_angle_obs(
            env_ids: np.ndarray,
            wing_angle_obs: np.ndarray,
        ) -> None:
            return None

        def set_csv_force_active_levels(
            self,
            env_ids: np.ndarray,
            levels: np.ndarray,
        ) -> None:
            self.active_levels = np.array(levels, copy=True)

        def set_csv_force_start_delay_steps(
            self,
            env_ids: np.ndarray,
            delay_steps: np.ndarray,
        ) -> None:
            self.start_delays = np.array(delay_steps, copy=True)

        def set_csv_force_amplitude_scales(
            self,
            env_ids: np.ndarray,
            amplitude_scales: np.ndarray,
        ) -> None:
            self.amplitude_scales = np.array(amplitude_scales, copy=True)

        def set_wing_angle_obs_amplitude_scales(
            self,
            env_ids: np.ndarray,
            amplitude_scales: np.ndarray,
        ) -> None:
            self.wing_obs_amplitude_scales = np.array(amplitude_scales, copy=True)

        @staticmethod
        def set_privileged_reset_randomization(env_ids, reset_randomization) -> None:
            return None

        @staticmethod
        def startup_commands(num_reset: int) -> np.ndarray:
            return np.zeros((num_reset, 3), dtype=np.float32)

    monkeypatch.setattr(
        joystick_module,
        "build_dr002_backend_reset_randomization",
        lambda *args, **kwargs: None,
    )
    env = _FakeEnv()
    provider = DR002JoystickDomainRandomizationProvider()

    provider.build_reset_plan(env, np.asarray([0, 1, 2], dtype=np.int32))

    np.testing.assert_array_equal(env.sampled_zero_hz_mask, expected_zero_hz_mask)
    np.testing.assert_array_equal(env.active_levels, expected_active_levels)
    np.testing.assert_array_equal(env.start_delays, expected_start_delays)
    np.testing.assert_allclose(env.amplitude_scales, [0.6, 0.9, 1.4])
    np.testing.assert_allclose(env.wing_obs_amplitude_scales, [0.85, 1.0, 1.15])
