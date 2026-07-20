from types import SimpleNamespace

import numpy as np

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

    sampled = env.sample_reset_zero_hz_wing_angle_obs(
        np.asarray([True, False, True, False])
    )

    assert sampled.shape == (4, 2)
    assert sampled.dtype == np.float32
    np.testing.assert_allclose(sampled[[0, 2]], 0.025)
    np.testing.assert_array_equal(sampled[[1, 3]], 0.0)

    env.set_zero_hz_wing_angle_obs(np.arange(4, dtype=np.int32), sampled)
    np.testing.assert_array_equal(env._zero_hz_wing_angle_obs[[1, 3]], 0.0)


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


def test_reset_plan_samples_after_standing_force_level_override(monkeypatch) -> None:
    class _Spawn:
        @staticmethod
        def origins_for(env_ids: np.ndarray) -> np.ndarray:
            return np.zeros((env_ids.size, 3), dtype=np.float32)

    class _FakeEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                domain_rand=DR002DomainRandConfig(
                    csv_force_curriculum_hz=[0.0, 1.0, 2.0],
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

    np.testing.assert_array_equal(env.sampled_zero_hz_mask, [True, True, False])
    np.testing.assert_array_equal(env.active_levels, [0, 0, 1])
    np.testing.assert_array_equal(env.start_delays, [0, 6, 7])
