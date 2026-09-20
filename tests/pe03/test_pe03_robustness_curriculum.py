"""Fresh-episode promotion, gradual physical effects, and exact course resume."""

import copy

import numpy as np
import pytest
import torch

from unilab.algos.torch.pe03.gait_evaluation import evaluate_gait
from unilab.algos.torch.pe03.policy import PE03EncoderPolicy
from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.algos.torch.pe03.tensorboard import tensorboard_metrics
from unilab.base.backend.mujoco.native_batch import native_joint_position_pd_available
from unilab.envs.locomotion.pe03.config import load_config
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.gait_randomization import push_forces
from unilab.envs.locomotion.pe03.robustness_curriculum import RobustnessCurriculum


def config(*overrides):
    # Retain coverage of checkpoints using the previous rolling-fraction contract.
    return load_config(
        [
            "+experiment=gait_fixed",
            "domain_rand.curriculum.strategy=episode_fraction",
            "+domain_rand.curriculum.recent_episodes=100",
            "+domain_rand.curriculum.min_episode_length=950",
            "+domain_rand.curriculum.success_fraction=0.5",
            "algo.num_envs=4",
            "algo.num_steps_per_env=3",
            "algo.max_iterations=1",
            "algo.num_learning_epochs=1",
            "algo.num_mini_batches=1",
            "training.device=cpu",
            "training.mujoco_threads=1",
            "training.logger=none",
            "training.evaluation_interval=0",
            *overrides,
        ]
    )


def course(**overrides):
    return RobustnessCurriculum(
        dict(enabled=True, window_episodes=4096, consecutive_windows=2, **overrides),
        enabled=True,
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


def fraction_course():
    return RobustnessCurriculum(dict(config().domain_rand.curriculum), enabled=True)


def test_fraction_uses_individual_950_step_threshold_not_mean():
    c = fraction_course()
    c.observe([949] * 51 + [1000] * 49, np.zeros(100))
    assert c.last_window_mean > 950
    assert c.last_success_fraction == 0.49 and c.level == 0
    c.observe([950], [0])  # Evicts a 949-step episode; exactly 50% now pass.
    assert c.level == 0.1 and c.last_success_fraction == 0.5
    assert c.count == 0 and len(c.recent_lengths) == 0
    c = fraction_course()
    c.observe([0] * 50 + [950] * 50, np.zeros(100))
    assert c.level == 0.1 and c.last_window_mean == 475


def test_fraction_waits_for_100_fresh_episodes_and_never_reuses_old_level():
    c = fraction_course()
    c.observe(np.full(99, 950), np.zeros(99))
    assert c.count == 99 and c.level == 0
    c.observe([950], [0])
    assert c.level == 0.1
    c.observe(np.full(10000, 1000), np.zeros(10000))
    assert c.level == 0.1 and c.count == 0
    c.observe(np.full(10000, 950), np.full(10000, 0.1))
    assert c.level == 0.2 and c.promotions == 2
    c.observe([], [])
    assert c.level == 0.2


def test_fraction_keeps_latest_100_and_preserves_rolling_buffer_on_resume():
    c = fraction_course()
    c.observe([1000] * 100 + [949] * 100, np.zeros(200))
    assert c.level == 0 and c.count == 100
    assert list(c.recent_lengths) == [949] * 100
    c.observe([950] * 49, np.zeros(49))
    assert c.level == 0 and c.last_success_fraction == 0.49
    restored = fraction_course()
    restored.restore(c.snapshot())
    for current in (c, restored):
        current.observe([950], [0])
        assert current.level == 0.1
    assert c.snapshot() == restored.snapshot()
    assert c.metrics()["randomization/last_check_success_fraction"] == 0.5


def test_fraction_never_demotes_and_caps_at_one():
    c = fraction_course()
    for expected in range(1, 11):
        level = c.level
        c.observe(np.full(100, 949), np.full(100, level))
        assert c.level == level
        c.observe(np.full(100, 950), np.full(100, level))
        assert c.level == expected / 10
    c.observe(np.full(100, 950), np.ones(100))
    assert c.level == 1 and c.promotions == 10


def test_two_full_fresh_windows_and_no_reuse_or_multi_promotion():
    c = course()
    c.observe(np.full(4095, 1000), np.zeros(4095))
    assert c.count == 4095 and c.successful_windows == 0 and c.level == 0
    c.observe([1000], [0])
    assert c.count == 0 and c.successful_windows == 1
    c.observe(np.full(20000, 1000), np.zeros(20000))
    assert c.level == 0.1 and c.promotions == 1 and c.count == 0
    c.observe(np.full(8192, 1000), np.zeros(8192))
    assert c.level == 0.1 and c.count == 0
    c.observe([], [])
    assert c.promotions == 1


def test_short_failure_resets_streak_but_never_lowers_level():
    c = course(initial_level=0.3)
    c.observe(np.full(4096, 1000), np.full(4096, 0.3))
    lengths = np.full(4096, 1000)
    lengths[0] = 999  # No rounding of the console's 1000.00 display.
    c.observe(lengths, np.full(4096, 0.3))
    assert c.successful_windows == 0 and c.last_window_mean < 1000
    assert c.level == 0.3
    c.observe(np.full(4096, 1000), np.full(4096, 0.3))
    assert c.level == 0.3
    c.observe(np.full(4096, 1000), np.full(4096, 0.3))
    assert c.level == 0.4


def test_mixed_levels_partial_window_resume_and_maximum():
    c = course()
    for expected in range(1, 11):
        level = c.level
        c.observe(np.full(2000, 1000), np.full(2000, level))
        restored = course()
        restored.restore(c.snapshot())
        lengths = np.full(2 * (8192 - 2000), 1000)
        levels = np.tile([level, level - 0.1], 8192 - 2000)
        c.observe(lengths, levels)
        restored.observe(lengths, levels)
        assert c.snapshot() == restored.snapshot()
        assert c.level == expected / 10
    c.observe(np.full(8192, 1000), np.ones(8192))
    assert c.level == 1 and c.promotions == 10


@pytest.mark.parametrize(
    "override",
    [
        "initial_level=-0.1",
        "initial_level=1.1",
        "increment=0",
        "increment=nan",
        "max_level=2",
        "recent_episodes=0",
        "recent_episodes=1.5",
        "success_fraction=0",
        "success_fraction=1.1",
        "success_fraction=nan",
        "strategy=unknown",
        "min_episode_length=0",
    ],
)
def test_invalid_course_is_rejected(override):
    with pytest.raises(ValueError):
        config("domain_rand.curriculum." + override)


@pytest.mark.parametrize("level", [0.0, 0.1, 1.0])
def test_scaled_parameters_noise_reset_and_fixed_delay(level):
    env = PE03GaitEnv(config(f"domain_rand.curriculum.initial_level={level}"))
    try:
        b = env.backend
        np.testing.assert_array_equal(b.delay_steps, 8)
        np.testing.assert_array_equal(b.qvel, 0)
        np.testing.assert_array_equal(env.randomization_levels, level)
        np.testing.assert_allclose(b.foot_clearances().min(1), 0.001, atol=1e-14)
        assert np.max(np.abs(b.qpos[:, 7:] - env.home[7:])) <= level * 0.1
        assert np.all(np.abs(env.measurement_noise) <= env.noise_amplitude * level + 1e-15)
        for name, nominal in env.randomization_nominal.items():
            expected = nominal + level * (env.randomization_targets[name] - nominal)
            actual = (
                b.randomization[name]
                if name in b.randomization
                else getattr(env if name in ("friction", "imu_angles") else b, name)
            )
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(
            env.weight, b.randomization["body_mass"].sum(1) * b.gravity_acceleration
        )
        b.steps[:] = round(5 / b.dt) - 1
        rng = copy.deepcopy(env.rng.bit_generator.state)
        force = push_forces(env)
        if level == 0:
            assert force is None and rng == env.rng.bit_generator.state
            np.testing.assert_array_equal(env.imu_offset, [[1, 0, 0, 0]] * 4)
            np.testing.assert_array_equal(env.measurement_noise, 0)
        else:
            assert force is not None and np.any(force)
            np.testing.assert_allclose(env.last_push_impulse, force[:, 0] * b.dt)
            assert np.all(
                np.linalg.norm(env.last_push_impulse, axis=1) <= env.push_mass * 0.5 * level * 1.5
            )
        tags = tensorboard_metrics(env.robustness_curriculum.metrics(), policy_dt=env.dt)
        assert tags["randomization/level"] == level
    finally:
        env.close()


def test_new_parameters_only_apply_on_reset_and_preserve_other_episodes():
    env = PE03GaitEnv(config())
    try:
        b = env.backend
        env.step(np.full((4, 6), 0.2))
        before = env.snapshot()
        env.robustness_curriculum.observe(np.full(8192, 1000), np.zeros(8192))
        np.testing.assert_array_equal(b.kp, before["backend"]["kp"])
        env._reset_ids(np.array([1]))
        np.testing.assert_array_equal(env.randomization_levels, [0, 0.1, 0, 0])
        others = np.array([0, 2, 3])
        for key in ("state", "kp", "kd", "torque_scale", "default_position", "delay_buffer"):
            np.testing.assert_array_equal(getattr(b, key)[others], before["backend"][key][others])
        for key in b.randomization:
            np.testing.assert_array_equal(
                b.randomization[key][others], before["backend"]["randomization"][key][others]
            )
        np.testing.assert_array_equal(
            env.measurement_noise[others], before["arrays"]["measurement_noise"][others]
        )
        np.testing.assert_array_equal(b.qvel[1], 0)
        assert np.any(b.kp[1] != b.kp[0])
        env.robustness_curriculum.observe([1000, 1000], env.randomization_levels[[0, 1]])
        assert env.robustness_curriculum.count == 1
        # Restore mixed levels and the partial window, then cross another reset.
        snapshot = env.snapshot()
        env._reset_ids(np.array([0, 2, 3]))
        expected = env.snapshot()
        env.restore(snapshot)
        env._reset_ids(np.array([0, 2, 3]))
        assert_tree_equal(expected, env.snapshot())
    finally:
        env.close()


def test_actual_timeouts_promote_and_replay_exactly_across_level_changes():
    env = PE03GaitEnv(
        config(
            "env.episode_length_s=0.04",
            "domain_rand.curriculum.recent_episodes=8",
            "domain_rand.curriculum.min_episode_length=2",
        )
    )
    try:
        actions = np.zeros((4, 6))
        for _ in range(3):
            env.step(actions)
        assert env.robustness_curriculum.count == 4
        snapshot = env.snapshot()
        expected = [env.step(actions) for _ in range(8)]
        end = env.snapshot()
        assert env.robustness_curriculum.level == 0.2
        env.restore(snapshot)
        for target in expected:
            actual = env.step(actions)
            np.testing.assert_array_equal(actual.reward, target.reward)
            assert_tree_equal(actual.obs, target.obs)
            assert_tree_equal(actual.info["diagnostics"], target.info["diagnostics"])
        assert_tree_equal(env.snapshot(), end)
    finally:
        env.close()


def test_checkpoint_resume_preserves_partial_window_and_subsequent_promotions(tmp_path):
    cfg = config(
        "env.episode_length_s=0.04",
        "domain_rand.curriculum.recent_episodes=8",
        "domain_rand.curriculum.min_episode_length=2",
    )

    def train(settings, name):
        runner = PE03Runner(settings, tmp_path / name)
        try:
            checkpoint = runner.learn()
            return checkpoint, copy.deepcopy(runner.policy.state_dict()), runner.env.snapshot()
        finally:
            runner.close()

    first, _, partial = train(cfg, "first")
    assert partial["robustness_curriculum"]["count"] == 4
    cfg.training.resume = str(first)
    _, actual, actual_env = train(cfg, "resume")
    cfg.training.resume = None
    cfg.algo.max_iterations = 2
    _, expected, expected_env = train(cfg, "whole")
    assert expected_env["robustness_curriculum"]["level"] == 0.1
    assert_tree_equal(actual, expected)
    assert_tree_equal(actual_env, expected_env)
    cfg.training.resume = str(first)
    cfg.domain_rand.curriculum.increment = 0.2
    with pytest.raises(ValueError, match="resume changes"):
        PE03Runner(cfg, tmp_path / "incompatible")


@pytest.mark.skipif(not native_joint_position_pd_available(), reason="native PD not built")
def test_python_native_agree_through_parameter_promotions():
    envs = [
        PE03GaitEnv(
            config(
                f"training.native_pd={str(native).lower()}",
                "env.episode_length_s=0.04",
                "domain_rand.curriculum.recent_episodes=8",
                "domain_rand.curriculum.min_episode_length=2",
            )
        )
        for native in (False, True)
    ]
    try:
        for action in np.random.default_rng(5).normal(0, 0.1, (12, 4, 6)):
            a, b = [env.step(action) for env in envs]
            np.testing.assert_allclose(a.reward, b.reward, atol=1e-9, rtol=1e-9)
            for key in a.obs:
                np.testing.assert_allclose(a.obs[key], b.obs[key], atol=1e-7, rtol=1e-7)
            assert (
                envs[0].robustness_curriculum.snapshot() == envs[1].robustness_curriculum.snapshot()
            )
        assert envs[0].robustness_curriculum.level == 0.3
    finally:
        for env in envs:
            env.close()


def test_evaluation_freezes_requested_level_and_requires_maximum_for_acceptance():
    cfg = config("training.evaluation_episodes=4", "env.episode_length_s=0.04")
    policy = PE03EncoderPolicy(cfg)
    low, report = evaluate_gait(policy, cfg, "cpu", randomization_level=0.1)
    assert low["evaluation/randomization_level"] == 0.1
    assert report["randomization"]["level"] == 0.1
    assert not report["acceptance"]["checks"]["randomization_level"]
    full, report = evaluate_gait(policy, cfg, "cpu")
    assert full["evaluation/randomization_level"] == 1
    assert report["acceptance"]["checks"]["randomization_level"]
    c = RobustnessCurriculum(
        dict(cfg.domain_rand.curriculum), enabled=True, evaluation=True, level=0.1
    )
    c.observe(np.full(8192, 1000), np.full(8192, 0.1))
    assert c.level == 0.1 and c.count == 0
