import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.envs.locomotion.pe04.observations import critic_layout
from unilab.envs.locomotion.pe04.rewards import compute_rewards, desired_contacts
from unilab.envs.locomotion.pe04.vector_env import PE04VectorEnv


def test_observation_layout_history_and_sparse_reset(config):
    env = PE04VectorEnv(config, evaluation=True)
    try:
        initial = env.state.obs
        assert {k: v.shape for k, v in initial.items()} == {
            "actor": (2, 300),
            "frame": (2, 30),
            "command": (2, 3),
            "critic": (2, 267),
        }
        np.testing.assert_array_equal(initial["frame"][:, 24:26], [[0, 1], [0, 1]])
        np.testing.assert_allclose(
            initial["actor"].reshape(2, 10, 30), np.repeat(initial["frame"][:, None], 10, axis=1)
        )
        env.set_command([0.2, -0.1, 0.4])
        obs = env._observe()
        np.testing.assert_allclose(obs["command"], [[0.2, -0.1, 0.4]] * 2)
        np.testing.assert_array_equal(initial["actor"], obs["actor"])
        env.episode_steps[0] = env.max_episode_steps - 1
        action = np.full((2, 6), 0.1)
        result = env.step(action)
        assert result.truncated.tolist() == [True, False]
        np.testing.assert_array_equal(result.obs["frame"][0, 18:24], 0)
        np.testing.assert_allclose(result.final_observation["frame"][0, 18:24], 0.1)
        np.testing.assert_array_equal(result.obs["actor"][1, :-30], initial["actor"][1, 30:])
        np.testing.assert_allclose(
            result.final_observation["critic"][0, :3], result.final_observation["critic"][0, :3]
        )
        env.gaits[1, 0] = 2.5
        env.step(np.zeros((2, 6)))
        assert env.phase[1] == pytest.approx(env.episode_steps[1] * 0.02 * 2.5 % 1)
    finally:
        env.close()


def test_native_pd_matches_python_and_real_joint_acceleration(config):
    cfg = OmegaConf.merge(config, {"training": {"native_pd": False}})
    first, second = PE04VectorEnv(config, evaluation=True), PE04VectorEnv(cfg, evaluation=True)
    try:
        for step in range(8):
            action = np.full((2, 6), np.sin(step) * 0.2)
            a, b = first.step(action), second.step(action)
            np.testing.assert_allclose(a.obs["actor"], b.obs["actor"], atol=1e-6)
            np.testing.assert_allclose(a.obs["critic"], b.obs["critic"], atol=1e-5)
        backend = first.backend
        data = backend.create_visual_data()
        mujoco.mj_forward(backend.model, data)
        np.testing.assert_allclose(
            backend.joint_acceleration[0], data.qacc[6:], rtol=1e-6, atol=1e-5
        )
        assert backend.contact_history.shape == (2, 4, 9, 3)
        np.testing.assert_allclose(
            backend.contact_history[:, 0], -backend.contact_forces, atol=1e-9
        )
    finally:
        first.close()
        second.close()


def test_reward_formula_physical_inputs_and_dt(config):
    env = PE04VectorEnv(config, evaluation=True)
    try:
        env.height_target += 0.01
        reward, terms = compute_rewards(env)
        np.testing.assert_allclose(terms["base_height"], -0.01 * 20 * 0.02, atol=1e-12)
        np.testing.assert_allclose(terms["keep_balance"], 0.02)
        np.testing.assert_allclose(reward, np.sum(list(terms.values()), axis=0), atol=1e-7)
        target = desired_contacts(np.array([0.25, 0.75]), np.array([[2, 0.5, 0.5, 0.06]] * 2), 0.05)
        assert target[0, 0] > 0.99 and target[0, 1] < 0.01
        assert target[1, 1] > 0.99 and target[1, 0] < 0.01
        force = np.zeros_like(env.backend.contact_history)
        force[:, 2, 0, 2] = 20
        env.backend.contact_history[:] = force
        _, terms = compute_rewards(env)
        np.testing.assert_allclose(terms["collision"], -0.5 * 0.02)
    finally:
        env.close()


def test_delay_randomization_nominal_critic_and_resume_snapshot(config):
    cfg = OmegaConf.merge(config, {"play": {"delay_ms": 25}})
    env = PE04VectorEnv(cfg, evaluation=True)
    try:
        action = np.array([[1] * 6, [0] * 6])
        env.step(action)
        np.testing.assert_array_equal(env.backend.qpos[0], env.backend.qpos[1])
        env.step(action)
        assert np.max(np.abs(env.backend.qpos[0] - env.backend.qpos[1])) > 1e-6
        assert np.all(np.abs(env.backend.torque) <= config.control.torque_limits)
    finally:
        env.close()
    env = PE04VectorEnv(config)
    try:
        mass = env.backend.randomization["body_mass"].copy()
        assert not np.array_equal(mass[0], mass[1])
        field = next(f for f in critic_layout(9) if f["name"] == "body_mass")
        np.testing.assert_array_equal(
            env.state.obs["critic"][:, field["offset"] : field["offset"] + 9],
            np.tile(env.backend.nominal_mass.astype(np.float32), (2, 1)),
        )
        env._reset_ids(np.array([0]))
        np.testing.assert_array_equal(env.backend.randomization["body_mass"], mass)
        snapshot = env.snapshot()
        expected = env.step(np.full((2, 6), 0.1))
        env.restore(snapshot)
        actual = env.step(np.full((2, 6), 0.1))
        for name in expected.obs:
            np.testing.assert_array_equal(expected.obs[name], actual.obs[name])
        np.testing.assert_array_equal(expected.reward, actual.reward)
    finally:
        env.close()
