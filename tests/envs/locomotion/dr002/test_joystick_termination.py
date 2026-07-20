from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.dr002.joystick import DR002JoystickEnv, RewardConfig


def _bare_termination_env(num_envs: int = 1) -> DR002JoystickEnv:
    env = object.__new__(DR002JoystickEnv)
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
    env._contact_failed_now = np.zeros(num_envs, dtype=np.bool_)
    env._has_undesired_contact = lambda *_args, **_kwargs: env._contact_failed_now.copy()
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
