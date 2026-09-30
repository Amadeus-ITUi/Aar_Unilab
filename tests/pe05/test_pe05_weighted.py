"""Physical reward/task regressions without enlarging the deployable observation."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.envs.locomotion.pe05.config import ROOT, load_config
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


@pytest.fixture
def env(request):
    overrides = getattr(request, "param", [])
    instance = PE05VectorEnv(
        load_config(["algo.num_envs=4", "training.mujoco_threads=1", *overrides])
    )
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def old_reward(env):
    # Exercise the actual 17-term contract saved before the PE01-shaped migration.
    saved = OmegaConf.load(ROOT / "docs/assets/pe05_pe01_gait/before_task.yaml")
    env.cfg["reward"] = OmegaConf.to_container(saved.reward, resolve=True)
    return env


def test_signed_normalized_sum_and_quadratic_tracking(env):
    env.backend.qpos[:, 2] = env.height_target - 0.03
    env.base_velocity[:] = 0
    env.commands[:] = [0.5, 0, 0.5]
    total, terms = env._rewards()
    np.testing.assert_allclose(terms["base_height"], -0.0027)
    np.testing.assert_allclose(terms["tracking_lin_vel"], 0)
    np.testing.assert_allclose(terms["tracking_ang_vel"], 0)
    np.testing.assert_allclose(total, sum(terms.values()), rtol=1e-6)
    assert (total < 0).all()
    assert "keep_balance" not in terms
    # Quadratic growth is preserved after calibrating the physical coefficient.
    env.backend.qpos[:, 2] = env.height_target - 0.08
    np.testing.assert_allclose(env._rewards()[1]["base_height"], -0.0192)


def test_short_calf_contact_is_not_lost_between_policy_steps(env):
    b = env.backend
    calf = env.cfg["env"]["body_names"].index("L_calf_Link")
    b.contact_history[:] = b.ground_contact_history[:] = 0
    b.contact_history[:, 4, calf, 2] = 10
    b.ground_contact_history[:, 4, calf, 2] = 10
    _, terms = env._rewards()
    np.testing.assert_allclose(terms["collision"], -0.1)
    assert env.adaptation.latest["ground_contact"][:, env.penalized.index(calf)].all()
    assert not env.adaptation.latest["self_contact"].any()
    assert not env.adaptation.termination().any()
    assert not env.adaptation.ground_ticks.any()  # Final samples have separated again.


def test_persistent_ground_is_consecutive_per_body_and_self_is_not_ground(env):
    b, a = env.backend, env.adaptation
    calf = env.cfg["env"]["body_names"].index("L_calf_Link")
    b.contact_history[:] = b.ground_contact_history[:] = 0
    b.contact_history[:, :, calf, 2] = 10
    for _ in range(6):
        env._rewards()
        assert not a.termination().any()  # Self-only contact is penalized, not ground dwell.
    b.ground_contact_history[:] = b.contact_history
    for i in range(5):
        env._rewards()
        np.testing.assert_array_equal(a.termination(), i == 4)
    b.contact_history[0] = b.ground_contact_history[0] = 0
    env._rewards()
    np.testing.assert_array_equal(a.termination(), [False, True, True, True])
    a.episode_sums[:] = 7
    ticks = a.ground_ticks.copy()
    env._reset_ids(np.array([1, 3]))
    np.testing.assert_array_equal(a.ground_ticks[[0, 2]], ticks[[0, 2]])
    np.testing.assert_array_equal(a.ground_ticks[[1, 3]], 0)
    np.testing.assert_array_equal(a.episode_sums[[0, 2]], 7)
    np.testing.assert_array_equal(b.ground_contact_history[[1, 3]], 0)


def test_low_height_failure_resets_after_recovery(env):
    env.backend.qpos[:, 2] = 0.14
    for _ in range(4):
        env._rewards()
        assert not env.adaptation.termination().any()
    env.backend.qpos[:, 2] = env.height_target
    env._rewards()
    assert not env.adaptation.termination().any()
    np.testing.assert_array_equal(env.failure_steps, 0)
    env.backend.qpos[:, 2] = 0.14
    for i in range(5):
        env._rewards()
        np.testing.assert_array_equal(env.adaptation.termination(), i == 4)


@pytest.mark.parametrize("command", [[0, 0, 0], [0.05, 0, 0], [0.3, 0, 0]])
@pytest.mark.parametrize("phase,swing_foot", [(0.25, 1), (0.75, 0)])
def test_slip_only_at_stance_and_clearance_only_at_swing(
    old_reward, monkeypatch, command, phase, swing_foot
):
    env = old_reward
    feet = env.backend.gait_foot_state(np.array(env.cfg["env"]["reference_points"]))
    feet["contact_velocity"][:] = [0.2, 0, 0]
    feet["ground_force"][:] = [0, 0, 10]
    feet["ground_force"][:, swing_foot] = 0
    feet["clearance"][:] = 0
    monkeypatch.setattr(env.backend, "gait_foot_state", lambda points: feet)
    env.commands[:] = command
    env.phase[:] = phase
    _, terms = env._rewards()
    np.testing.assert_allclose(terms["feet_slip"], -0.000048)
    np.testing.assert_allclose(terms["contact_schedule"], 0)
    # Swing apex, including zero command: 3 cm / 2 cm = 1.5; calibrated weight .018.
    np.testing.assert_allclose(terms["foot_clearance"], -0.00081)
    feet["clearance"][:, swing_foot] = 0.03
    np.testing.assert_allclose(env._rewards()[1]["foot_clearance"], 0, atol=1e-12)
    feet["ground_force"][:, swing_foot, 2] = 10
    np.testing.assert_allclose(env._rewards()[1]["contact_schedule"], -0.005)


def test_old_weighted_config_preserves_zero_command_stance(old_reward, monkeypatch):
    env = old_reward
    del env.cfg["reward"]["zero_command_stance"]
    feet = env.backend.gait_foot_state(np.array(env.cfg["env"]["reference_points"]))
    feet["ground_force"][:] = [0, 0, 10]
    feet["clearance"][:] = 0
    monkeypatch.setattr(env.backend, "gait_foot_state", lambda points: feet)
    env.commands[:] = 0
    env.phase[:] = 0.25
    _, terms = env._rewards()
    np.testing.assert_allclose(terms["foot_clearance"], 0)
    np.testing.assert_allclose(terms["contact_schedule"], 0)


@pytest.mark.parametrize(
    "env", [["domain_rand.enabled=true", "domain_rand.curriculum.enabled=true"]], indirect=True
)
def test_safe_full_randomization_reset_and_nominal_start(env):
    np.testing.assert_allclose(env.backend.kp, np.tile(env.cfg["control"]["kp"], (4, 1)))
    env.adaptation.level[:] = 1
    for _ in range(15):
        env._reset_ids(np.arange(4))
        q = env.backend.qpos[:, 7:]
        assert (q >= env.backend.joint_range[:, 0]).all()
        assert (q <= env.backend.joint_range[:, 1]).all()
        assert np.max(np.linalg.norm(env.backend.contact_forces, axis=-1)) < 1e-6
        np.testing.assert_allclose(env.backend.foot_clearances().min(1), 0.001, atol=1e-6)
    np.testing.assert_allclose(env.backend.kp, env.adaptation.targets["kp"])
    assert ((env.backend.delay_steps >= 10) & (env.backend.delay_steps <= 20)).all()


@pytest.mark.parametrize(
    "env", [["domain_rand.enabled=true", "domain_rand.curriculum.enabled=true"]], indirect=True
)
def test_curriculum_requires_quality_and_changes_on_reset(env):
    a = env.adaptation
    env.episode_steps[:] = env.max_episode_steps
    a.episode_sums[:] = 0
    a.episode_sums[0, 0] = env.max_episode_steps * 0.1
    a.episode_sums[1, 1] = env.max_episode_steps * 0.08
    previous = env.backend.kp.copy()
    a.finish_episodes(np.arange(4), np.array([False, False, False, True]))
    np.testing.assert_allclose(a.level, [0, 0, 0.1, 0])
    np.testing.assert_array_equal(env.backend.kp, previous)
    env._reset_ids(np.array([2]))
    assert not np.array_equal(env.backend.kp[2], previous[2])
    np.testing.assert_array_equal(env.backend.kp[[0, 1, 3]], previous[[0, 1, 3]])


def test_command_probability_and_reset_isolation(env):
    a = env.adaptation
    for level in (0, 1):
        a.level[:] = level
        count = 0
        for _ in range(1000):
            env._resample_commands(np.array([1, 3]))
            count += int((env.commands[[1, 3]] == 0).all(1).sum())
        assert 0.17 < count / 2000 < 0.23
    env.step(np.zeros((4, 6)))
    history, state, fifo = (
        env.history.copy(),
        env.backend.state.copy(),
        env.backend.delay_buffer.copy(),
    )
    env._reset_ids(np.array([1, 3]))
    np.testing.assert_array_equal(env.history[[0, 2]], history[[0, 2]])
    np.testing.assert_array_equal(env.backend.state[[0, 2]], state[[0, 2]])
    np.testing.assert_array_equal(env.backend.delay_buffer[[0, 2]], fifo[[0, 2]])
    assert env.state.obs["actor"].shape == (4, 300)


def test_nominal_defaults_keep_resets_noise_delay_and_curriculum_disabled(env):
    b, a = env.backend, env.adaptation
    initial = b.qpos.copy()
    np.testing.assert_array_equal(b.delay_steps, 0)
    assert not b.randomization
    np.testing.assert_array_equal(env.imu_offset, np.tile([1, 0, 0, 0], (4, 1)))
    for _ in range(3):
        env.step(np.ones((4, 6)) * 0.1)
        env._reset_ids(np.arange(4))
        np.testing.assert_array_equal(b.qpos, initial)
        np.testing.assert_array_equal(b.qvel, 0)
        frame, clean = env._frame(np.arange(4))
        np.testing.assert_array_equal(frame, clean)
    env.episode_steps[:] = env.max_episode_steps
    a.episode_sums[:] = 0
    a.finish_episodes(np.arange(4), np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(a.level, 0)
    # The independent velocity course starts small even with physical randomization off.
    samples = []
    for _ in range(100):
        env._resample_commands(np.arange(4))
        samples.append(env.commands.copy())
    values = np.concatenate(samples)
    assert 0.25 < np.max(np.abs(values[:, 0])) <= 0.3 + 1e-12
    assert 0.08 < np.max(np.abs(values[:, 1])) <= 0.1 + 1e-12
    assert np.max(np.abs(values[:, 2])) <= 0.5 + 1e-12


def test_ground_telemetry_matches_unfused_backend(env):
    snapshot = env.snapshot()
    actions = np.full((4, 6), 0.2)
    for _ in range(20):
        env.step(actions)
    expected = env.backend.snapshot()
    assert expected["ground_contact_history"].max() > 1
    env.restore(snapshot)
    env.backend.fused_telemetry = False
    for _ in range(20):
        env.step(actions)
    for name in ("contact_history", "ground_contact_history", "state"):
        np.testing.assert_allclose(
            env.backend.snapshot()[name], expected[name], atol=1e-8, rtol=1e-8
        )


@pytest.mark.parametrize(
    "override",
    [
        "reward.base_height_std=0",
        "reward.zero_command_stance=invalid",
        "reward.scales.collision=1",
        "env.contact_hz=50",
        "domain_rand.curriculum.initial_level=2",
    ],
)
def test_invalid_weighted_contract_rejected(override):
    with pytest.raises(ValueError):
        load_config([override])


def test_failure_cost_is_signed_and_timeout_is_not_failure(env, monkeypatch):
    monkeypatch.setattr(env.backend, "step", lambda *args, **kwargs: None)
    env.backend.qpos[0, 2] = 0.14
    env.failure_steps[0] = 4
    env.episode_steps[1] = env.max_episode_steps - 1
    state = env.step(np.zeros((4, 6)))
    assert state.terminated.tolist() == [True, False, False, False]
    assert state.truncated.tolist() == [False, True, False, False]
    assert state.info["reward_terms"]["termination"] == pytest.approx(-0.5)
    assert state.info["diagnostics"]["reward/total"] == pytest.approx(state.reward.mean())
    assert state.reward.mean() == pytest.approx(sum(state.info["reward_terms"].values()))
