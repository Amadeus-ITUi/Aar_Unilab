"""Success-driven velocity expansion without enabling physical randomization."""

import copy

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.envs.locomotion.pe05.command_curriculum import VelocityCommandCurriculum
from unilab.envs.locomotion.pe05.config import load_config
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


def make_course(count=4, interval=10):
    cfg = OmegaConf.to_container(load_config().commands, resolve=True)
    return VelocityCommandCurriculum(cfg, count, interval)


def test_initial_ranges_zero_quota_and_neighbor_expansion():
    c = make_course(count=10000, interval=2)
    commands = c.sample(np.random.default_rng(42), np.arange(10000))
    assert 0.18 < (commands == 0).all(1).mean() < 0.22
    assert (np.abs(commands) <= [0.3 + 1e-12, 0.1 + 1e-12, 0.5 + 1e-12]).all()
    before = c.weights.copy()
    c.observe(np.ones((10000, 4)), np.zeros(10000, bool), np.zeros(10000, bool))
    np.testing.assert_array_equal(c.weights, before)
    c.observe(np.ones((10000, 4)), np.zeros(10000, bool), np.zeros(10000, bool))
    assert (c.weights > 0).sum() > (before > 0).sum()
    np.testing.assert_allclose(c.weights[before == 0][c.weights[before == 0] > 0], 0.2)
    sampled = c.sample(np.random.default_rng(1), np.arange(10000))
    assert np.max(np.abs(sampled[:, 0])) > 0.3
    assert (sampled >= c.low).all() and (sampled <= c.high).all()


@pytest.mark.parametrize("case", ["zero", "failed", "partial", "poor_tracking", "poor_contact"])
def test_ineligible_windows_do_not_expand(case):
    c = make_course(count=1)
    c.sample(np.random.default_rng(1), np.array([0]))
    c.zero[:] = case == "zero"
    before = c.weights.copy()
    score = np.ones((1, 4))
    if case == "poor_tracking":
        score[:, 0] = 0.79
    if case == "poor_contact":
        score[:, 2] = 0.89
    steps = 2 if case == "partial" else 10
    for i in range(steps):
        c.observe(
            score,
            np.array([case == "failed" and i == steps - 1]),
            np.array([case == "partial" and i == steps - 1]),
        )
    np.testing.assert_array_equal(c.weights, before)
    assert c.successful_windows == 0


def test_scores_use_tracking_and_phase_contact_quality():
    c = make_course(count=1)
    phase = np.array([0.25])
    gait = np.array([[2, 0.5, 0.5, 0.03]])
    command = np.array([[0.2, 0, 0.3]])
    feet = {
        "ground_force": np.array([[[0.0, 0.0, 32.0], [0.0, 0.0, 0.0]]]),
        "contact_velocity": np.zeros((1, 2, 3)),
    }
    good = c.tracking_scores(
        command, np.array([[0.2, 0, 0]]), np.array([0.3]), phase, gait, feet, np.array([32.0])
    )
    assert (good >= c.config["thresholds"]).all()
    feet["ground_force"][:] = feet["ground_force"][:, ::-1].copy()
    feet["contact_velocity"][:, 0, 0] = 10
    bad = c.tracking_scores(
        command, np.array([[1.0, 0, 0]]), np.array([2.0]), phase, gait, feet, np.array([32.0])
    )
    assert (bad < c.config["thresholds"]).all()


def test_command_boundary_scores_old_command_and_restores_partial_window(monkeypatch):
    cfg = load_config(
        [
            "algo.num_envs=4",
            "training.mujoco_threads=1",
            "commands.resampling_time=0.04",
            "commands.zero_probability=0",
        ]
    )
    env = PE05VectorEnv(cfg)
    try:
        c = env.adaptation.command_course
        observed = []

        def scores(commands, *args):
            observed.append(commands.copy())
            return np.ones((4, 4))

        monkeypatch.setattr(c, "tracking_scores", scores)
        action = np.zeros((4, 6))
        old = env.commands.copy()
        env.step(action)
        saved = env.snapshot()
        assert (saved["adaptation"]["command_course"]["ticks"] == 1).all()
        state = env.step(action)
        np.testing.assert_array_equal(observed[-1], old)
        assert not np.array_equal(old, env.commands)
        np.testing.assert_allclose(state.obs["command"], env.commands)
        expected = copy.deepcopy(env.snapshot())
        env.restore(saved)
        env.step(action)
        actual = env.snapshot()
        for key in ("weights", "bins", "scores", "ticks", "zero"):
            np.testing.assert_array_equal(
                actual["adaptation"]["command_course"][key],
                expected["adaptation"]["command_course"][key],
            )
        np.testing.assert_array_equal(env.commands, expected["arrays"]["commands"])
        np.testing.assert_array_equal(actual["backend"]["state"], expected["backend"]["state"])
        np.testing.assert_array_equal(env.adaptation.level, 0)
        np.testing.assert_array_equal(env.backend.delay_steps, 0)
    finally:
        env.close()


def test_timeout_keeps_terminal_command_and_evaluation_does_not_learn():
    cfg = load_config(
        [
            "algo.num_envs=2",
            "training.mujoco_threads=1",
            "commands.resampling_time=0.04",
            "env.episode_length_s=0.04",
        ]
    )
    env = PE05VectorEnv(cfg)
    try:
        commands = env.commands.copy()
        env.step(np.zeros((2, 6)))
        state = env.step(np.zeros((2, 6)))
        assert state.truncated.all()
        np.testing.assert_allclose(state.final_observation["command"], commands)
        np.testing.assert_allclose(state.obs["command"], env.commands)
        np.testing.assert_array_equal(env.adaptation.command_course.ticks, 0)
    finally:
        env.close()
    env = PE05VectorEnv(cfg, evaluation=True)
    try:
        assert env.adaptation.command_course is None
        env.step(np.zeros((2, 6)))
        np.testing.assert_allclose(env.commands, np.tile(cfg.play.command, (2, 1)))
    finally:
        env.close()


@pytest.mark.parametrize(
    "override",
    [
        "commands.heading_command=true",
        "commands.bin_width=[0,0.1,0.1]",
        "commands.initial_low=[-2,-0.1,-0.5]",
        "commands.thresholds=[0.8,0.7,0.9]",
        "commands.weight_increment=2",
    ],
)
def test_invalid_course_config_rejected(override):
    with pytest.raises(ValueError):
        load_config([override])
