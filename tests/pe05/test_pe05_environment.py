"""PE03 physical adaptations and original PE01 behavior at the environment boundary."""

import numpy as np
import pytest

from unilab.envs.locomotion.pe05.config import load_config
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


@pytest.fixture
def env():
    cfg = load_config(["task=pe05_legacy", "algo.num_envs=4", "training.mujoco_threads=1"])
    instance = PE05VectorEnv(cfg)
    try:
        yield instance
    finally:
        instance.close()


def test_zero_command_probability_and_subset_reset(env):
    ids = np.array([1, 3])
    untouched = env.commands[[0, 2]].copy()
    zeros = 0
    for _ in range(2000):
        env._resample_commands(ids)
        zeros += int((env.commands[ids] == 0).all(1).sum())
    assert 0.88 < zeros / 4000 < 0.92
    np.testing.assert_array_equal(env.commands[[0, 2]], untouched)
    env.step(np.full((4, 6), 0.1))
    state, history, fifo = (
        env.backend.state.copy(),
        env.history.copy(),
        env.backend.delay_buffer.copy(),
    )
    env._reset_ids(ids)
    np.testing.assert_array_equal(env.backend.state[[0, 2]], state[[0, 2]])
    np.testing.assert_array_equal(env.history[[0, 2]], history[[0, 2]])
    np.testing.assert_array_equal(env.backend.delay_buffer[[0, 2]], fifo[[0, 2]])
    np.testing.assert_array_equal(env.backend.delay_buffer[ids], 0)
    np.testing.assert_array_equal(env.history[ids, 0], env.history[ids, -1])


def test_body_home_and_target_clipping_include_random_zero_offsets(env):
    assert env.height_target == pytest.approx(0.2913829166803791)
    assert env.home[7] == pytest.approx(0.02315985984147312)
    assert env.home[12] == pytest.approx(1.254112635692029)
    for action in (1000, -1000):
        env.step(np.full((4, 6), action))
        target = env.backend.default_position + env.actions * 0.25
        assert np.all(target >= env.backend.joint_range[:, 0] - 1e-12)
        assert np.all(target <= env.backend.joint_range[:, 1] + 1e-12)
        np.testing.assert_allclose(
            env.backend.delay_buffer[:, 0] + env.backend.default_position, target
        )
        assert np.all(np.abs(env.backend.torque) <= np.array(env.cfg["control"]["torque_limits"]))


def test_friction_range_is_not_masked_by_the_nominal_floor(env):
    b = env.backend
    friction = b.randomization["geom_friction"]
    robot = friction[:, b.robot_geoms, 0]
    floor = friction[:, b.ground_geoms, 0]
    np.testing.assert_allclose(robot, np.broadcast_to(floor[:, :1], robot.shape))
    assert np.all((robot >= 0.1) & (robot <= 2.0))


def test_current_frame_history_and_clean_critic_are_synchronized(env):
    for _ in range(3):
        previous = env.history.copy()
        state = env.step(np.zeros((4, 6)))
        np.testing.assert_array_equal(state.obs["frame"], state.obs["actor"][:, -30:])
        np.testing.assert_array_equal(env.history[:, :-1], previous[:, 1:])
        # Command changes never enter history; critic uses the current clean frame.
        np.testing.assert_allclose(
            state.obs["critic"][:, 9:15],
            env.backend.qpos[:, 7:] - env.backend.default_position,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            state.obs["critic"][:, :3], env.base_velocity.astype(np.float32)
        )


def test_landing_reward_uses_vertical_force_and_height_error_is_squared(env, monkeypatch):
    monkeypatch.setattr(env.backend, "foot_clearances", lambda: np.full((4, 2), 0.01))
    env.foot_velocity[:] = [0, 0, -0.2]
    env.backend.contact_history[:] = 0
    # Tangential force alone does not count as a landed foot in the original reward.
    env.backend.contact_history[:, 0, env.backend.foot_indices, 0] = 10
    _, terms = env._rewards()
    np.testing.assert_allclose(terms["foot_landing_vel"], 2 * 0.2**2 * -0.15 * 0.02)
    np.testing.assert_allclose(
        terms["base_height"], (env.backend.qpos[:, 2] - env.height_target) ** 2 * -3 * 0.02
    )
    env.backend.contact_history[:, 0, env.backend.foot_indices, 2] = 1
    _, terms = env._rewards()
    np.testing.assert_array_equal(terms["foot_landing_vel"], 0)


def test_failure_ticks_accumulate_and_timeout_does_not_mask_failure(env, monkeypatch):
    # Isolate the termination rule from physics: failures need not be consecutive.
    monkeypatch.setattr(env.backend, "step", lambda *args, **kwargs: None)
    env.auto_reset = False
    env.backend.contact_history[:] = 0
    root = env.termination_bodies[0]
    for step in range(51):
        env.backend.contact_history[:, 0, root, 2] = 6 if step % 2 == 0 else 0
        state = env.step(np.zeros((4, 6)))
        assert state.terminated.all() == (step == 50)
    env.episode_steps[:] = env.max_episode_steps
    state = env.step(np.zeros((4, 6)))
    assert state.terminated.all() and not state.truncated.any()
