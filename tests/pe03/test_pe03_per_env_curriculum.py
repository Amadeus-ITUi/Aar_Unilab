"""Equal environment coverage, synchronized promotion cuts, and PPO continuation."""

import copy

import numpy as np
import pytest
import torch

from unilab.algos.torch.pe03.ppo import generalized_advantage
from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.envs.locomotion.pe03.config import load_config
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.robustness_curriculum import RobustnessCurriculum


def config(*overrides, experiment="gait_fixed"):
    return load_config(
        [
            f"+experiment={experiment}",
            "algo.num_envs=4",
            "algo.num_steps_per_env=3",
            "algo.num_learning_epochs=1",
            "algo.num_mini_batches=1",
            "training.device=cpu",
            "training.mujoco_threads=1",
            "training.logger=none",
            "training.export=false",
            "training.evaluation_interval=0",
            *overrides,
        ]
    )


def course(**overrides):
    return RobustnessCurriculum(
        dict(enabled=True, strategy="per_env_mean", mean_episode_length=950, **overrides),
        enabled=True,
        num_envs=4,
    )


def assert_tree_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    else:
        np.testing.assert_array_equal(a, b)


def test_first_result_per_environment_includes_failure_and_rounds_are_disjoint():
    c = course()
    assert not c.observe([100], [0], [0])
    # This fast-reset environment must not acquire extra weight or replace its failure.
    for _ in range(100):
        assert not c.observe([1000], [0], [0])
    assert c.count == 1 and c.length_sum == 100
    assert not c.observe([1000, 1000], [0, 0], [1, 2])
    assert c.completed_windows == 0 and c.metrics()["randomization/round_coverage"] == 0.75
    assert not c.observe([1000], [0], [3])
    assert c.last_window_mean == 775 and c.completed_windows == 1
    assert c.level == 0 and c.count == 0
    assert c.observe([950] * 4, [0] * 4, np.arange(4))
    assert c.last_window_mean == 950 and c.level == 0.1
    assert c.count == 0 and (c.env_lengths == -1).all()
    # The whole previous batch is ineligible for the new level.
    assert not c.observe([1000] * 4, [0] * 4, np.arange(4))
    assert c.count == 0


@pytest.mark.parametrize("last, promoted", [(799, False), (800, True)])
def test_mean_950_boundary_instead_of_individual_success_fraction(last, promoted):
    c = course()
    assert c.observe([1000, 1000, 1000, last], [0] * 4, np.arange(4)) == promoted
    assert c.last_window_mean == (3000 + last) / 4
    assert c.level == (0.1 if promoted else 0)


def test_partial_round_resume_and_monotone_cap():
    c = course()
    for expected in range(1, 11):
        level = c.level
        c.observe([949] * 4, [level] * 4, np.arange(4))
        assert c.level == level
        c.observe([1000, 900], [level, level], [2, 0])
        restored = course()
        restored.restore(c.snapshot())
        for current in (c, restored):
            # Duplicate environment 2 cannot overwrite the saved first result.
            assert current.observe([1, 950, 950], [level] * 3, [2, 1, 3])
        assert_tree_equal(c.snapshot(), restored.snapshot())
        assert c.level == expected / 10
    assert not c.observe([1000] * 4, [1] * 4, np.arange(4))
    assert c.level == 1 and c.promotions == 10
    with pytest.raises(ValueError, match="saved results"):
        course().restore({"level": 0})


@pytest.mark.parametrize("stage", ["gait_fixed", "gait_variable"])
def test_default_preset_and_evaluation_freeze(stage):
    cfg = config(experiment=stage)
    assert cfg.domain_rand.curriculum.strategy == "per_env_mean"
    assert cfg.domain_rand.curriculum.mean_episode_length == 950
    assert "recent_episodes" not in cfg.domain_rand.curriculum
    env = PE03GaitEnv(cfg, evaluation=True, robustness=True, randomization_level=0.2)
    try:
        env.episode_steps[:] = 999
        state = env.step(np.zeros((4, 6)))
        assert state.truncated.all() and not state.info["promotion_truncated"].any()
        assert env.robustness_curriculum.level == 0.2
        assert env.robustness_curriculum.count == 0
    finally:
        env.close()


@pytest.mark.parametrize("value", [0, -1, "nan"])
def test_invalid_mean_threshold(value):
    with pytest.raises(ValueError):
        config(f"domain_rand.curriculum.mean_episode_length={value}")


def test_promotion_preserves_old_reward_terminal_observation_and_true_failure(monkeypatch):
    env, control = PE03GaitEnv(config()), PE03GaitEnv(config())
    try:
        env.step(np.full((4, 6), 0.2))
        c = env.robustness_curriculum
        c.observe([1000] * 3, [0] * 3, [0, 1, 2])
        env.episode_steps[:] = [50, 100, 400, 999]
        env.failure_steps[0] = env.failure_limit - 1
        control.restore(env.snapshot())
        control.robustness_curriculum.evaluation = True
        old_commands = env._scaled_commands().copy()

        def force_failure(original):
            def rewards(feet):
                reward, terms, info = original(feet)
                info["base_tilt_deg"][0] = 180
                return reward, terms, info

            return rewards

        for instance in (env, control):
            monkeypatch.setattr(instance, "_rewards", force_failure(instance._rewards))
        action = np.full((4, 6), 0.3)
        result, baseline = env.step(action), control.step(action)
        np.testing.assert_array_equal(result.reward, baseline.reward)
        assert_tree_equal(result.final_observation, baseline.final_observation)
        np.testing.assert_allclose(result.final_observation["command"], old_commands)
        assert result.terminated.tolist() == [True, False, False, False]
        assert result.truncated.tolist() == [False, True, True, True]
        assert result.info["promotion_truncated"].tolist() == [False, True, True, False]
        assert result.info["episode_lengths"].tolist() == [51, 1000]
        assert len(result.info["episode_returns"]) == 2
        assert c.level == 0.1 and c.count == 0 and c.promotion_truncations == 2
        np.testing.assert_array_equal(env.randomization_levels, 0.1)
        for name in ("episode_steps", "failure_steps", "actions", "last_actions", "old_actions"):
            np.testing.assert_array_equal(getattr(env, name), 0)
        np.testing.assert_array_equal(env.backend.qvel, 0)
        np.testing.assert_array_equal(env.backend.delay_buffer, 0)
        np.testing.assert_array_equal(env.backend.steps, 0)
        np.testing.assert_array_equal(
            result.obs["actor"].reshape(4, 30, 38),
            np.repeat(result.obs["frame"][:, None], 30, axis=1),
        )
        assert not np.array_equal(result.obs["actor"], result.final_observation["actor"])
        # The two partial command windows did not count as command curriculum results.
        np.testing.assert_array_equal(
            env.velocity_curriculum.weights, control.velocity_curriculum.weights
        )
        env.step(np.zeros((4, 6)))
        assert c.count == 0
    finally:
        env.close()
        control.close()


def test_runner_bootstraps_cuts_and_does_not_log_them_as_completed_episodes(tmp_path):
    runner = PE03Runner(config(), tmp_path)
    try:
        env = runner.env
        env.robustness_curriculum.observe([1000] * 3, [0] * 3, [0, 1, 2])
        env.episode_steps[:] = [100, 200, 300, 999]
        original = env.step
        states = []

        def record(action):
            state = original(action)
            states.append(state)
            return state

        env.step = record
        rollout, metrics = runner.collect()
        assert rollout["truncated"][0].all() and not rollout["terminated"].any()
        assert list(runner.lengths) == [1000]
        assert metrics["randomization/level"] == 0.1
        assert metrics["randomization/promotion_truncations"] == 3
        assert metrics["randomization/episode_count"] == 0
        with torch.no_grad():
            expected = runner.algorithm.value(runner._tensors(states[0].final_observation))
            reset_value = runner.algorithm.value(runner._tensors(states[0].obs))
        torch.testing.assert_close(rollout["next_value"][0], expected)
        assert not torch.equal(expected, reset_value)
        advantages, _ = generalized_advantage(
            rollout["reward"],
            rollout["value"],
            rollout["next_value"],
            rollout["terminated"],
            rollout["truncated"],
            0.99,
            0.95,
        )
        # The new level's later rewards must not leak across the old episode boundary.
        torch.testing.assert_close(
            advantages[0], rollout["reward"][0] + 0.99 * expected - rollout["value"][0]
        )
    finally:
        runner.close()


def test_exact_checkpoint_resume_across_partial_round_promotion_and_ppo_update(tmp_path):
    cfg = config()
    runner = PE03Runner(cfg, tmp_path / "first")
    checkpoint = tmp_path / "partial.pt"
    try:
        # Include a real PPO update and nonempty optimizer state before saving.
        rollout, _ = runner.collect()
        runner.algorithm.update(rollout)
        runner.env.robustness_curriculum.observe([1000, 900], [0, 0], [0, 1])
        runner.env.episode_steps[:] = [50, 100, 998, 999]
        runner.save(checkpoint)
        expected_rollout, expected_metrics = runner.collect()
        runner.algorithm.update(expected_rollout)
        expected_policy = copy.deepcopy(runner.policy.state_dict())
        expected_env = runner.env.snapshot()
        expected_optimizer = copy.deepcopy(runner.algorithm.optimizer.state_dict())
        expected_encoder = copy.deepcopy(runner.algorithm.encoder_optimizer.state_dict())
        assert runner.env.robustness_curriculum.level == 0.1
    finally:
        runner.close()
    cfg.training.resume = str(checkpoint)
    resumed = PE03Runner(cfg, tmp_path / "resumed")
    try:
        assert resumed.env.robustness_curriculum.count == 2
        actual_rollout, actual_metrics = resumed.collect()
        assert_tree_equal(actual_rollout, expected_rollout)
        assert actual_metrics == expected_metrics
        resumed.algorithm.update(actual_rollout)
        assert_tree_equal(resumed.policy.state_dict(), expected_policy)
        assert_tree_equal(resumed.env.snapshot(), expected_env)
        assert_tree_equal(resumed.algorithm.optimizer.state_dict(), expected_optimizer)
        assert_tree_equal(resumed.algorithm.encoder_optimizer.state_dict(), expected_encoder)
    finally:
        resumed.close()
    cfg.domain_rand.curriculum.mean_episode_length = 900
    with pytest.raises(ValueError, match="resume changes"):
        PE03Runner(cfg, tmp_path / "changed")
