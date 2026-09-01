from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.dr002.joystick import DR002JoystickEnv, RewardConfig


def _bare_termination_env(num_envs: int = 1) -> DR002JoystickEnv:
    env = object.__new__(DR002JoystickEnv)
    env._num_envs = num_envs
    env._cfg = SimpleNamespace(ctrl_dt=0.02)
    env._reward_cfg = RewardConfig(
        scales={},
        termination_contact_threshold=0.1,
        termination_contact_fail_steps=25,
        termination_gravity_z_threshold=0.7,
        termination_fail_time_s=0.5,
    )
    env._lingzu_fail_steps = np.zeros(num_envs, dtype=np.int32)
    env._lingzu_contact_fail_accum_steps = np.zeros(num_envs, dtype=np.int32)
    env._episode_getup_mask = np.zeros(num_envs, dtype=np.bool_)
    env._getup_success_hold_steps = np.zeros(num_envs, dtype=np.int32)
    env._getup_succeeded = np.zeros(num_envs, dtype=np.bool_)
    env._getup_timeout_recorded = np.zeros(num_envs, dtype=np.bool_)
    env._getup_grace_steps = 150
    env._getup_timeout_count = 0
    env._state = SimpleNamespace(info={"steps": np.zeros(num_envs, dtype=np.uint32)})
    env._update_getup_success = lambda _gravity: None
    env._contact_failed_now = np.zeros(num_envs, dtype=np.bool_)
    env._has_termination_contact = lambda *_args, **_kwargs: env._contact_failed_now.copy()
    return env


def _upright(num_envs: int = 1) -> np.ndarray:
    gravity = np.zeros((num_envs, 3), dtype=np.float32)
    gravity[:, 2] = 1.0
    return gravity


def _tilted(num_envs: int = 1) -> np.ndarray:
    gravity = _upright(num_envs)
    gravity[:, 2] = 0.7
    return gravity


def test_contact_requires_25_consecutive_control_steps() -> None:
    env = _bare_termination_env()
    env._contact_failed_now[:] = True

    for _ in range(24):
        assert not env._compute_terminated(_upright())[0]
    assert env._lingzu_contact_fail_accum_steps[0] == 24
    assert env._compute_terminated(_upright())[0]


def test_contact_counter_resets_after_contact_free_step() -> None:
    env = _bare_termination_env()
    env._contact_failed_now[:] = True

    for _ in range(24):
        assert not env._compute_terminated(_upright())[0]
    env._contact_failed_now[:] = False
    assert not env._compute_terminated(_upright())[0]
    assert env._lingzu_contact_fail_accum_steps[0] == 0

    env._contact_failed_now[:] = True
    for _ in range(24):
        assert not env._compute_terminated(_upright())[0]


def test_tilt_requires_25_consecutive_control_steps_and_resets() -> None:
    env = _bare_termination_env()

    for _ in range(24):
        assert not env._compute_terminated(_tilted())[0]
    assert env._lingzu_fail_steps[0] == 24

    assert not env._compute_terminated(_upright())[0]
    assert env._lingzu_fail_steps[0] == 0

    for _ in range(24):
        assert not env._compute_terminated(_tilted())[0]
    assert env._compute_terminated(_tilted())[0]


def test_contact_and_tilt_counters_remain_independent() -> None:
    env = _bare_termination_env()
    env._contact_failed_now[:] = True

    for _ in range(10):
        assert not env._compute_terminated(_upright())[0]
    for _ in range(14):
        assert not env._compute_terminated(_tilted())[0]

    assert env._lingzu_contact_fail_accum_steps[0] == 24
    assert env._lingzu_fail_steps[0] == 14
    assert env._compute_terminated(_tilted())[0]
    assert env._lingzu_contact_fail_accum_steps[0] == 25
    assert env._lingzu_fail_steps[0] == 15


def test_getup_contact_is_suppressed_until_three_second_grace_expires() -> None:
    env = _bare_termination_env()
    env._episode_getup_mask[:] = True
    env._contact_failed_now[:] = True

    for step in range(150):
        env._state.info["steps"][:] = step
        assert not env._compute_terminated(_upright())[0]

    env._state.info["steps"][:] = 150
    assert env._compute_terminated(_upright())[0]
    assert env._getup_timeout_count == 1


def test_getup_tilt_requires_150_consecutive_control_steps() -> None:
    env = _bare_termination_env()
    env._episode_getup_mask[:] = True

    for step in range(149):
        env._state.info["steps"][:] = step
        assert not env._compute_terminated(_tilted())[0]

    env._state.info["steps"][:] = 149
    assert env._compute_terminated(_tilted())[0]


def test_getup_success_requires_pose_height_and_all_nonwheel_contacts_clear() -> None:
    env = _bare_termination_env()
    env._episode_getup_mask[:] = True
    env._update_getup_success = DR002JoystickEnv._update_getup_success.__get__(env)
    env._backend = SimpleNamespace(
        get_base_pos=lambda: np.asarray([[0.0, 0.0, 0.24]], dtype=np.float32)
    )
    env.get_local_linvel = lambda: np.zeros((1, 3), dtype=np.float32)
    env.get_gyro = lambda: np.zeros((1, 3), dtype=np.float32)
    contact_values = np.zeros((1, 9), dtype=np.float32)
    env._undesired_contact_values = lambda _num_envs: contact_values
    env._getup_success_count = 0
    env._getup_success_time_sum_s = 0.0
    env._state.info["steps"][:] = 25

    contact_values[0, -1] = 0.2
    for _ in range(30):
        env._update_getup_success(_upright())
        assert not env._getup_succeeded[0]
    contact_values.fill(0.0)
    for _ in range(24):
        env._update_getup_success(_upright())
        assert not env._getup_succeeded[0]
    env._update_getup_success(_upright())

    assert env._getup_succeeded[0]
    assert env._getup_success_count == 1
    assert env._getup_success_time_sum_s == 0.5


def test_getup_success_requires_velocity_but_not_reward_workspace_and_one_second_hold() -> None:
    env = _bare_termination_env()
    env._episode_getup_mask[:] = True
    env._update_getup_success = DR002JoystickEnv._update_getup_success.__get__(env)
    env._reward_cfg.getup_success_gravity_z_threshold = 0.95
    env._reward_cfg.getup_success_base_height_range = (0.20, 0.30)
    env._reward_cfg.getup_success_max_abs_lin_vel_xy = 0.10
    env._reward_cfg.getup_success_max_abs_ang_vel_xyz = 0.20
    env._reward_cfg.getup_workspace_half_extent = 0.10
    env._reward_cfg.getup_success_workspace_half_extent = None
    env._reward_cfg.getup_success_hold_time_s = 1.0
    base_pos = np.asarray([[0.0, 0.0, 0.24]], dtype=np.float32)
    linvel = np.zeros((1, 3), dtype=np.float32)
    gyro = np.zeros((1, 3), dtype=np.float32)
    env._backend = SimpleNamespace(get_base_pos=lambda: base_pos)
    env.get_local_linvel = lambda: linvel
    env.get_gyro = lambda: gyro
    env._episode_reset_xy = np.zeros((1, 2), dtype=np.float32)
    env._undesired_contact_values = lambda _num_envs: np.zeros((1, 9), dtype=np.float32)
    env._getup_success_count = 0
    env._getup_success_time_sum_s = 0.0

    linvel[0, 0] = 0.11
    env._update_getup_success(_upright())
    assert env._getup_success_hold_steps[0] == 0
    linvel.fill(0.0)
    base_pos[0, 0] = 0.11
    for _ in range(49):
        env._update_getup_success(_upright())
    assert not env._getup_succeeded[0]
    env._update_getup_success(_upright())
    assert env._getup_succeeded[0]
