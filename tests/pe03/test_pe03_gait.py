"""Behavioral contracts for the independently designed WTW-inspired PE03 task."""

import copy
import hashlib
import json

import mujoco
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from unilab.algos.torch.pe03.policy import PE03EncoderPolicy
from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.catalog.launch import DEFAULTS, build_launch
from unilab.envs.locomotion.pe03.config import load_config
from unilab.envs.locomotion.pe03.gait import CommandCurriculum, phase_targets, placement_targets
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.gait_randomization import evaluation_protocol


def small(*overrides):
    return load_config(
        [
            "+experiment=gait_fixed",
            "algo.num_envs=2",
            "algo.num_steps_per_env=4",
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


@pytest.mark.parametrize("stage", ["fixed", "variable"])
def test_gait_launch_and_control_contract(stage):
    command, _, _ = build_launch(DEFAULTS, "pe03_gait_" + stage, [], environment={})
    config = load_config(command[2:])
    previous = load_config(["+experiment=walking"])
    control = OmegaConf.to_container(config.control)
    assert control["clip_joint_targets"] is True
    assert control == OmegaConf.to_container(previous.control)
    assert config.task_id == "pe03_gait_flat" and config.observation == "pe03_v4"
    assert config.gait.stage == stage
    assert config.algo.num_envs == 4096 and config.training.evaluation_episodes == 64
    assert "feet_air_time" not in config.reward.scales


@pytest.mark.parametrize(
    "override",
    [
        "gait.high=[3,1,0.1]",
        "gait.low=[0,0.5,0.02]",
        "commands.initial_low=[-2,-0.1,-0.5]",
        "network.latent_dim=2",
    ],
)
def test_invalid_gait_contracts_are_rejected_before_allocating_simulators(override):
    with pytest.raises(ValueError):
        small(override)


@pytest.mark.parametrize("beta", [0.5, 0.6, 0.7])
def test_phase_contact_height_and_placement_share_support_fraction(beta):
    phase = np.array([0.0, beta / 2, beta, beta + (1 - beta) / 2, 1 - 1e-9])
    gait = np.tile([2.0, beta, 0.03], (len(phase), 1))
    contact, height, travel, stance = phase_targets(phase, gait, 0.07)
    np.testing.assert_allclose(height[:, 0], [0, 0, 0, 0.03, 0], atol=1e-8)
    assert contact[1, 0] > 0.99 and contact[3, 0] < 0.01
    assert stance[1, 0] and not stance[3, 0]
    nominal = np.array([[-0.04834, 0.07], [-0.04834, -0.07]])
    targets = placement_targets(np.tile([0.2, 0.1, 0], (len(phase), 1)), gait, travel, nominal)
    np.testing.assert_allclose(targets[0, 0] - nominal[0], np.array([0.2, 0.1]) * beta / 4)
    np.testing.assert_allclose(targets[2, 0] - nominal[0], -np.array([0.2, 0.1]) * beta / 4)


def test_actor_has_direct_history_path_and_only_velocity_supervision():
    policy = PE03EncoderPolicy(small())
    assert policy.actor[0].in_features == 1143
    assert policy.critic[0].in_features == 1154
    assert policy.encoder[-1].out_features == 3
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        for layer in policy.actor:
            if isinstance(layer, nn.Linear):
                layer.weight[0, 0] = 1
    history = torch.zeros(2, 1140, requires_grad=True)
    with torch.no_grad():
        history[1, 0] = 1
    action = policy.action_mean(history, torch.zeros(2, 38), torch.zeros(2, 6))
    torch.testing.assert_close(policy.encode(history), torch.zeros(2, 3))
    assert action[0, 0] == 0 and action[1, 0] == 1
    action.sum().backward()
    assert history.grad[1, 0] == 1
    assert all(parameter.grad is None for parameter in policy.encoder.parameters())
    (policy.encode(history.detach()) - 1).square().mean().backward()
    assert policy.encoder[-1].bias.grad.abs().sum() > 0


def test_commands_replace_current_history_and_reach_velocity_head():
    policy = PE03EncoderPolicy(small())
    history, frame, command = torch.zeros(1, 1140), torch.zeros(1, 38), torch.ones(1, 6)
    seen = []
    hook = policy.encoder.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone())
    )
    try:
        policy.action_mean(history, frame, command)
        np.testing.assert_array_equal(seen[0][0, -8:-2], 1)
        assert seen[0][0, :-38].count_nonzero() == 0
    finally:
        hook.remove()


def test_reset_observations_soft_limits_and_timeout_contract():
    env = PE03GaitEnv(small("env.episode_length_s=0.04"), evaluation=True)
    try:
        obs = env.state.obs
        assert {k: v.shape for k, v in obs.items()} == dict(
            actor=(2, 1140), frame=(2, 38), command=(2, 6), critic=(2, 14)
        )
        np.testing.assert_array_equal(
            obs["actor"].reshape(2, 30, 38), np.repeat(obs["frame"][:, None], 30, axis=1)
        )
        assert (
            (env.home[7:] >= env.soft_limits[:, 0]) & (env.home[7:] <= env.soft_limits[:, 1])
        ).all()
        state = env.step(np.ones((2, 6)) * 0.2)
        assert not state.truncated.any()
        np.testing.assert_allclose(state.obs["frame"][:, 18:24], env.actions)
        np.testing.assert_array_equal(state.obs["frame"][:, 24:30], 0)
        env.episode_steps[1] = 0
        state = env.step(np.ones((2, 6)) * 0.1)
        assert state.truncated.tolist() == [True, False]
        assert state.final_observation is not None
        np.testing.assert_array_equal(
            state.obs["actor"][0].reshape(30, 38),
            np.repeat(state.obs["frame"][0, None], 30, axis=0),
        )
    finally:
        env.close()


def test_failure_ticks_are_consecutive_and_collision_is_added_after_aggregation(monkeypatch):
    env = PE03GaitEnv(small(), evaluation=True, auto_reset=False)
    try:
        monkeypatch.setattr(env.backend, "step", lambda *args, **kwargs: None)
        failure_height = env.cfg["env"]["failure_height"]
        assert env.height_target > failure_height
        for low in [True, True, False, True, True, True, True, True]:
            env.backend.qpos[:, 2] = failure_height - 0.01 if low else env.height_target
            state = env.step(np.zeros((2, 6)))
            np.testing.assert_allclose(
                state.reward,
                state.info["positive_reward"]
                * np.exp(state.info["exponential_negative_reward"] / 0.02)
                + state.info["additive_reward"],
                rtol=1e-6,
            )
        assert state.terminated.all() and (env.failure_steps == 5).all()
    finally:
        env.close()


@pytest.mark.parametrize("large_motion_cost", [False, True])
def test_each_colliding_body_keeps_its_full_penalty_even_when_exponential_saturates(
    large_motion_cost,
):
    env = PE03GaitEnv(small("algo.num_envs=4"), evaluation=True)
    try:
        b = env.backend
        b.sensors[:, b.contact_adr[:, None] + np.arange(3)] = 0
        feet = b.gait_foot_state(env.reference_points)
        if large_motion_cost:
            env.actions[:] = 100
        for row, count in enumerate(range(4)):
            for body in env.penalized[:count]:
                b.sensors[row, b.contact_adr[body]] = 2.0
        reward, terms, info = env._rewards(feet)
        np.testing.assert_array_equal(info["raw_rewards"]["collision"], [0, 1, 2, 3])
        np.testing.assert_allclose(terms["collision"], [0, -0.1, -0.2, -0.3])
        np.testing.assert_allclose(np.diff(reward), -0.1, atol=2e-8)
        assert reward[-1] < 0
        np.testing.assert_allclose(info["attenuation"], info["attenuation"][0])
        if large_motion_cost:
            assert info["attenuation"].max() < 1e-20
        logs = env._diagnostics(
            {
                **info,
                "cycle_event": np.zeros((4, 2), bool),
                "touch_event": np.zeros((4, 2), bool),
                "actual_frequency": np.zeros((4, 2)),
                "peak_error": np.zeros((4, 2)),
                "duty_error": np.zeros((4, 2)),
                "actual_duty": np.zeros((4, 2)),
                "failure_reasons": np.zeros((4, 3), bool),
            }
        )
        weighted = sum(v for k, v in logs.items() if k.endswith("/weighted_penalty"))
        assert weighted == pytest.approx(terms["collision"].mean())
    finally:
        env.close()


def test_collision_threshold_and_body_multiplier_are_independent_of_termination():
    env = PE03GaitEnv(
        small(
            "reward.collision.force_threshold=2", "reward.collision.body_multipliers.R_thigh_Link=2"
        ),
        evaluation=True,
    )
    try:
        b = env.backend
        b.sensors[:, b.contact_adr[:, None] + np.arange(3)] = 0
        foot = env.config.env.body_names.index("R_foot_Link")
        thigh = env.config.env.body_names.index("R_thigh_Link")
        b.sensors[:, b.contact_adr[foot]] = 100
        b.sensors[0, b.contact_adr[thigh]] = 2.0
        b.sensors[1, b.contact_adr[thigh]] = 2.01
        _, terms, info = env._rewards(b.gait_foot_state(env.reference_points))
        np.testing.assert_array_equal(info["raw_rewards"]["collision"], [0, 1])
        np.testing.assert_allclose(terms["collision"], [0, -0.2])
        assert env.config.env.failure_contact_force == 1.0
    finally:
        env.close()


def test_checkpoint_without_collision_block_keeps_original_exponential_reward():
    cfg = small("~reward.collision")
    env = PE03GaitEnv(cfg, evaluation=True)
    try:
        b = env.backend
        b.sensors[:, b.contact_adr[:, None] + np.arange(3)] = 0
        for body in env.penalized[:3]:
            b.sensors[:, b.contact_adr[body]] = 2
        reward, terms, info = env._rewards(b.gait_foot_state(env.reference_points))
        assert env.collision_aggregation == "exponential"
        expected_negative = sum(np.minimum(v, 0) for v in terms.values())
        np.testing.assert_array_equal(terms["collision"], np.full(2, 3 * -5.0 * env.dt))
        np.testing.assert_array_equal(
            reward, (info["positive_reward"] * np.exp(expected_negative / 0.02)).astype(np.float32)
        )
        np.testing.assert_array_equal(info["additive_reward"], 0)
    finally:
        env.close()


@pytest.mark.parametrize(
    "override",
    [
        "reward.collision.aggregation=bad",
        "reward.collision.force_threshold=-1",
        "reward.collision.body_multipliers.R_thigh_Link=-1",
        "+reward.collision.body_multipliers.R_foot_Link=1",
        "reward.scales.collision=5",
    ],
)
def test_invalid_collision_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        small(override)


def test_parameter_transition_is_continuous_and_phase_is_not_reset():
    env = PE03GaitEnv(small(), evaluation=True, auto_reset=False)
    try:
        env.set_command([0, 0, 0], [3, 0.7, 0.1])
        expected_phase = env.phase.copy()
        previous = env.gaits.copy()
        for _ in range(25):
            expected_phase = (expected_phase + env.gaits[:, 0] * env.dt) % 1
            state = env.step(np.zeros((2, 6)))
            np.testing.assert_allclose(env.phase, expected_phase, atol=1e-12)
            assert (env.gaits >= previous - 1e-12).all()
            previous = env.gaits.copy()
            np.testing.assert_allclose(
                state.obs["frame"][:, 33:36], env.gaits * [0.5, 1, 10], rtol=1e-6
            )
        np.testing.assert_allclose(env.gaits, [[3, 0.7, 0.1]] * 2)
    finally:
        env.close()


def test_curriculum_only_expands_successful_cells_and_respects_limits():
    c = CommandCurriculum(
        [-1, -0.6, -1], [1, 0.6, 1], [0.1, 0.1, 0.1], [-0.3, -0.1, -0.5], [0.3, 0.1, 0.5]
    )
    before = c.weights.copy()
    _, ids = c.sample(np.random.default_rng(1), 100)
    c.update(ids, np.zeros(100, bool))
    np.testing.assert_array_equal(c.weights, before)
    c.update(ids, np.ones(100, bool))
    assert (c.weights > before).any() and (c.weights <= 1).all()
    samples, _ = c.sample(np.random.default_rng(2), 1000)
    assert (samples >= c.low).all() and (samples <= c.high).all()


def test_zero_commands_and_early_failure_cannot_expand_velocity_course():
    env = PE03GaitEnv(small())
    try:
        old = env.velocity_curriculum.snapshot()
        env.window_scores[:] = env.command_interval
        env.standing_command[:] = True
        env._update_curriculum(np.array([0, 1]), np.zeros(2, bool))
        np.testing.assert_array_equal(old, env.velocity_curriculum.weights)
        env.standing_command[:] = False
        env.window_scores[:] = env.command_interval / 2
        env._update_curriculum(np.array([0, 1]), np.zeros(2, bool))
        np.testing.assert_array_equal(old, env.velocity_curriculum.weights)
    finally:
        env.close()


def test_backend_contact_point_velocity_matches_mujoco_jacobian():
    env = PE03GaitEnv(small(), evaluation=True)
    try:
        b = env.backend
        qvel = np.zeros((2, 12))
        qvel[:, :6] = [0.1, -0.2, 0.03, 0.3, -0.1, 0.2]
        qvel[:, 6:] = [0.1, 0.3, -0.2, -0.1, -0.3, 0.2]
        b.reset(np.arange(2), np.tile(env.home, (2, 1)), qvel)
        state = b.gait_foot_state(env.reference_points)
        data = mujoco.MjData(b.model)
        data.qpos[:] = b.qpos[0]
        data.qvel[:] = b.qvel[0]
        mujoco.mj_forward(b.model, data)
        for foot, name in enumerate(env.config.env.foot_names):
            adr = b.gait_sensor_adr["contact_point"][foot]
            point = data.sensordata[adr + 4 : adr + 7]
            jac = np.zeros((3, b.model.nv))
            mujoco.mj_jac(b.model, data, jac, None, point, b.model.body(name).id)
            np.testing.assert_allclose(
                state["contact_velocity"][0, foot], jac @ data.qvel, atol=1e-10
            )
            np.testing.assert_allclose(
                state["ground_force"][0, foot], -data.sensordata[adr + 1 : adr + 4]
            )
    finally:
        env.close()


def test_exact_training_resume_and_stage_gate(tmp_path):
    def run(cfg, path):
        runner = PE03Runner(cfg, path)
        try:
            checkpoint = runner.learn()
            return checkpoint, {k: v.clone() for k, v in runner.policy.state_dict().items()}
        finally:
            runner.close()

    cfg = small("algo.max_iterations=2")
    _, expected = run(cfg, tmp_path / "whole")
    cfg.algo.max_iterations = 1
    first, _ = run(cfg, tmp_path / "first")
    changed = copy.deepcopy(cfg)
    changed.training.resume = str(first)
    changed.reward.collision.body_multipliers.R_thigh_Link = 2.0
    with pytest.raises(ValueError, match="resume changes"):
        PE03Runner(changed, tmp_path / "changed_reward_rejected")
    cfg.training.resume = str(first)
    _, actual = run(cfg, tmp_path / "resume")
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
    variable = small("gait.stage=variable", f"training.stage_from={first}")
    with pytest.raises(ValueError, match="three consecutive"):
        PE03Runner(variable, tmp_path / "rejected")
    # Synthetic acceptance is confined to this test fixture, never a real run.
    payload = torch.load(first, weights_only=True)
    payload["iteration"] = 300
    payload["training_config"]["training"]["evaluation_episodes"] = 64
    payload["training_config"]["training"]["evaluation_interval"] = 100
    payload["acceptance_records"] = [dict(iteration=i, passed=True) for i in (100, 200, 300)]
    stale = tmp_path / "stale_clean_acceptance.pt"
    torch.save(payload, stale)
    variable.training.stage_from = str(stale)
    with pytest.raises(ValueError, match="randomized acceptance"):
        PE03Runner(variable, tmp_path / "stale_rejected")
    payload["acceptance_records"] = [
        dict(iteration=i, passed=True, protocol=evaluation_protocol(cfg)["fingerprint"])
        for i in (100, 200, 300)
    ]
    accepted = tmp_path / "synthetic.pt"
    torch.save(payload, accepted)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            dict(
                checkpoint_sha256=hashlib.sha256(accepted.read_bytes()).hexdigest(),
                reviewer="test fixture",
                no_dragging=True,
                no_crossing=True,
                no_nonfoot_support=True,
                no_frequent_limits=True,
            )
        )
    )
    variable.training.stage_from = str(accepted)
    variable.training.visual_review = str(review)
    runner = PE03Runner(variable, tmp_path / "stage")
    try:
        assert runner.iteration == 0 and runner.algorithm.updates == 0
        assert not runner.algorithm.optimizer.state
        for key, value in payload["actor_state_dict"].items():
            torch.testing.assert_close(runner.policy.state_dict()[key], value)
        checkpoint = runner.learn()
        assert checkpoint.is_file() and runner.iteration == 1
        assert runner.algorithm.optimizer.state
        assert all(torch.isfinite(p).all() for p in runner.policy.parameters())
    finally:
        runner.close()
