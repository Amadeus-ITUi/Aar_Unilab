"""Behavioral checks for the formal PE02 migration, separate from v1 smoke tests."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from unilab.algos.torch.pe02.policy import PE02EncoderPolicy
from unilab.algos.torch.pe02.ppo import generalized_advantage
from unilab.algos.torch.pe02.runner import PE02Runner
from unilab.envs.locomotion.pe02.config import ROOT, load_config
from unilab.envs.locomotion.pe02.vector_env import PE02VectorEnv


def small_config(*overrides):
    return load_config(
        [
            "algo.num_envs=4",
            "algo.num_steps_per_env=4",
            "algo.max_iterations=2",
            "algo.num_learning_epochs=2",
            "algo.num_mini_batches=2",
            "training.device=cpu",
            "training.logger=none",
            "training.export=false",
            "training.evaluation_interval=0",
            *overrides,
        ]
    )


def source_constant(*names):
    # Read literal configuration only: references is never imported or executed.
    path = (
        ROOT / "references/pe01/legacy_training/legged_gym/envs/dragon_flat/dragon_flat_config.py"
    )
    node = ast.parse(path.read_text())
    for name in names[:-1]:
        node = next(
            child for child in node.body if isinstance(child, ast.ClassDef) and child.name == name
        )
    assignment = next(
        child
        for child in node.body
        if isinstance(child, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == names[-1] for target in child.targets)
    )
    return ast.literal_eval(assignment.value)


def test_formal_defaults_match_original_source():
    cfg = load_config()
    assert cfg.observation == "pe02_v2"
    assert cfg.algo.num_envs == source_constant("DragonCfgFlat", "env", "num_envs") == 8192
    for name in ("num_steps_per_env", "max_iterations", "save_interval"):
        assert cfg.algo[name] == source_constant("DragonCfgPPOD", "runner", name)
    for name in (
        "num_learning_epochs",
        "num_mini_batches",
        "gamma",
        "lam",
        "learning_rate",
        "max_grad_norm",
    ):
        assert cfg.algo[name] == source_constant("DragonCfgPPOD", "algorithm", name)
    assert list(cfg.network.actor_hidden_dims) == source_constant(
        "DragonCfgPPOD", "policy", "actor_hidden_dims"
    )
    assert list(cfg.network.critic_hidden_dims) == source_constant(
        "DragonCfgPPOD", "policy", "critic_hidden_dims"
    )
    assert list(cfg.network.encoder_hidden_dims) == source_constant(
        "DragonCfgPPOD", "MLP_Encoder", "hidden_dims"
    )
    assert cfg.network.latent_dim == source_constant(
        "DragonCfgPPOD", "MLP_Encoder", "num_output_dim"
    )
    for name, value in cfg.reward.scales.items():
        assert value == source_constant("DragonCfgFlat", "rewards", "scales", name)
    assert cfg.control.physics_hz / cfg.control.policy_hz == source_constant(
        "DragonCfgFlat", "control", "decimation"
    )
    assert cfg.control.action_scale == source_constant("DragonCfgFlat", "control", "action_scale")


def test_timeout_bootstrap_and_episode_boundaries():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    next_values = torch.tensor([[0.6], [10.0], [100.0]])
    terminated = torch.tensor([[False], [False], [True]])
    truncated = torch.tensor([[False], [True], [False]])
    advantages, returns = generalized_advantage(
        rewards, values, next_values, terminated, truncated, 0.9, 0.8
    )
    torch.testing.assert_close(advantages[:, 0], torch.tensor([8.528, 10.4, 2.3]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([9.028, 11.0, 3.0]))


def test_encoder_is_supervised_separately_and_actor_is_unbounded():
    policy = PE02EncoderPolicy(small_config())
    assert [
        layer.out_features for layer in policy.encoder if isinstance(layer, torch.nn.Linear)
    ] == [256, 128, 3]
    with torch.no_grad():
        policy.actor[-1].weight.zero_()
        policy.actor[-1].bias.fill_(2)
    history, frame, command = torch.zeros(2, 300), torch.zeros(2, 30), torch.zeros(2, 3)
    action = policy.action_mean(history, frame, command)
    torch.testing.assert_close(action, torch.full((2, 6), 2.0))
    action.sum().backward()
    assert all(parameter.grad is None for parameter in policy.encoder.parameters())
    policy.encode(history).square().mean().backward()
    assert any(parameter.grad is not None for parameter in policy.encoder.parameters())


def test_observation_layout_and_sparse_timeout_reset():
    env = PE02VectorEnv(small_config("env.episode_length_s=0.02"), evaluation=True)
    try:
        obs = env.reset().obs
        assert isinstance(obs, dict)
        np.testing.assert_allclose(obs["frame"][:, 3:6], [[0, 0, -1]] * 4, atol=1e-7)
        np.testing.assert_array_equal(obs["frame"][:, 6:24], 0)
        np.testing.assert_array_equal(obs["frame"][:, 24:26], [[0, 1]] * 4)
        np.testing.assert_array_equal(
            obs["actor"].reshape(4, 10, 30), np.repeat(obs["frame"][:, None], 10, axis=1)
        )
        env.episode_steps[0] = 1
        state = env.step(np.zeros((4, 6)))
        assert state.truncated.tolist() == [True, False, False, False]
        assert state.final_observation is not None
        assert env.episode_steps.tolist() == [0, 1, 1, 1]
        assert not np.array_equal(state.obs["actor"][0], state.final_observation["actor"][0])
        np.testing.assert_array_equal(state.obs["frame"][0, 24:26], [0, 1])
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_delay_pd_limits_and_randomization_isolation():
    cfg = small_config("play.delay_ms=25")
    env = PE02VectorEnv(cfg, evaluation=True, num_envs=2)
    try:
        actions = np.array([[1] * 6, [0] * 6])
        env.step(actions)
        np.testing.assert_array_equal(env.backend.qpos[0], env.backend.qpos[1])
        env.step(actions)
        assert np.max(np.abs(env.backend.qpos[0] - env.backend.qpos[1])) > 1e-6
        assert np.all(np.abs(env.backend.torque) <= np.array(cfg.control.torque_limits))
    finally:
        env.close()
    env = PE02VectorEnv(cfg)
    try:
        sampled = env.backend.randomization["body_mass"].copy()
        assert not np.array_equal(sampled[0], sampled[1])
        env._reset_ids(np.array([0]))
        np.testing.assert_array_equal(sampled, env.backend.randomization["body_mass"])
        assert np.all((env.backend.delay_steps >= 10) & (env.backend.delay_steps <= 20))
    finally:
        env.close()


def test_standing_rewards_are_targeted_and_keep_original_weights():
    env = PE02VectorEnv(small_config(), evaluation=True)
    try:
        total, terms = env._rewards()
        np.testing.assert_allclose(terms["base_height"], 0, atol=1e-9)
        np.testing.assert_allclose(terms["orientation"], 0, atol=1e-9)
        np.testing.assert_allclose(terms["keep_balance"], 0.02)
        np.testing.assert_allclose(terms["tracking_lin_vel"], 0.02)
        assert len(terms) == 18
        assert np.isfinite(total).all()
    finally:
        env.close()


def run_training(config, path):
    runner = PE02Runner(config, path)
    try:
        initial = {name: value.clone() for name, value in runner.policy.state_dict().items()}
        checkpoint = runner.learn()
        final = {name: value.clone() for name, value in runner.policy.state_dict().items()}
        return checkpoint, initial, final
    finally:
        runner.close()


def test_multiple_updates_and_exact_resume(tmp_path):
    cfg = small_config()
    checkpoint, initial, final = run_training(cfg, tmp_path / "whole")
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["iteration"] == 2 and payload["total_samples"] == 32
    assert payload["optimizer_updates"] == 8 and payload["encoder_updates"] == 8
    assert not torch.equal(initial["actor.0.weight"], final["actor.0.weight"])
    assert not torch.equal(initial["encoder.0.weight"], final["encoder.0.weight"])
    for line in (tmp_path / "whole/metrics.jsonl").read_text().splitlines():
        assert all(np.isfinite(v) for v in json.loads(line).values())
    cfg.algo.max_iterations = 1
    split, _, _ = run_training(cfg, tmp_path / "split")
    cfg.training.resume = str(split)
    resumed, _, resumed_final = run_training(cfg, tmp_path / "resume")
    assert torch.load(resumed, weights_only=True)["iteration"] == 2
    for name in final:
        torch.testing.assert_close(final[name], resumed_final[name], rtol=0, atol=0)


def test_resume_rejects_contract_changes(tmp_path):
    cfg = small_config("algo.max_iterations=1")
    checkpoint, _, _ = run_training(cfg, tmp_path / "original")
    cfg.training.resume = str(checkpoint)
    cfg.control.action_scale = 0.5
    with pytest.raises(ValueError, match="contract"):
        PE02Runner(cfg, tmp_path / "wrong")


def test_finite_evaluation_metrics(tmp_path):
    runner = PE02Runner(
        small_config("env.episode_length_s=0.06", "training.evaluation_episodes=2"), tmp_path
    )
    try:
        metrics = runner.evaluate()
        assert 0 < metrics["evaluation/episode_seconds"] <= 0.08
        assert 0 <= metrics["evaluation/failure_rate"] <= 1
        assert 0 <= metrics["evaluation/standing_fraction"] <= 1
        assert 0 <= metrics["evaluation/nonfoot_contact"] <= 1
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        runner.close()


def test_evaluation_perturbations_are_reproducible_and_play_stays_at_home():
    cfg = small_config()
    env = PE02VectorEnv(cfg, evaluation=True, evaluation_reset_noise=True)
    try:
        perturbed = env.backend.qpos.copy()
        assert not np.array_equal(perturbed[0], perturbed[1])
        assert np.max(np.abs(perturbed[:, 7:] - env.home[7:])) <= 0.01
    finally:
        env.close()
    replay = PE02VectorEnv(cfg, evaluation=True, evaluation_reset_noise=True)
    try:
        np.testing.assert_array_equal(perturbed, replay.backend.qpos)
    finally:
        replay.close()
    play = PE02VectorEnv(cfg, evaluation=True)
    try:
        np.testing.assert_array_equal(play.backend.qpos, np.tile(play.home, (4, 1)))
    finally:
        play.close()


def test_formal_training_export_and_relocated_play_are_independent(tmp_path):
    # Run the ONNX legacy tracer outside pytest (Torch/Python native tracing isolation).
    script = tmp_path / "formal_release.py"
    script.write_text(
        """
import importlib.abc
import shutil
import sys
from pathlib import Path

class RejectReferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if 'pe01' in fullname.lower() or fullname.startswith('references'):
            raise AssertionError(fullname)

sys.meta_path.insert(0, RejectReferenceImports())
sys.path.insert(0, sys.argv[1])
import numpy as np
import onnxruntime as ort
import torch
from unilab.adapters.pe02_ppo import train, load_policy
from unilab.envs.locomotion.pe02.config import load_config
from unilab.envs.locomotion.pe02.play_env import make_play_env
from unilab.release_contract import load_manifest
from scripts.train_pe02 import export_release

cfg = load_config([
    'algo.num_envs=2', 'algo.num_steps_per_env=2', 'algo.max_iterations=1',
    'training.device=cpu', 'training.logger=none', 'training.evaluation_interval=0',
    'play.delay_ms=25', 'play.command=[0.2,-0.1,0.3]',
])
result = train(Path('run'), config=cfg)
release = export_release(result.checkpoint, 'formal')
relocated = Path('relocated')
shutil.move(release, relocated)
manifest = load_manifest(relocated / 'deployment_manifest.json')
assert manifest['policy']['observation_builder'] == 'pe02_v2'
assert manifest['control']['type'] == 'position_pd'
assert manifest['control']['command_delay_steps'] == 10
policy = load_policy(result.checkpoint)
session = ort.InferenceSession(str(relocated / 'policy.onnx'), providers=['CPUExecutionProvider'])
env = make_play_env(policy.config, relocated / manifest['artifacts']['scene_path'])
try:
    observation = env.reset()
    command = np.load(relocated / 'golden_inputs/command.npy')
    for _ in range(12):
        inputs = {
            'observation_history': observation.actor[None],
            'observation': observation.actor[None, -30:],
            'command': command,
        }
        action = session.run(['action'], inputs)[0]
        with torch.no_grad():
            expected = policy.action_mean(*[torch.from_numpy(value) for value in inputs.values()])
        np.testing.assert_allclose(action, expected.numpy(), atol=1e-6, rtol=1e-5)
        observation, reward, _, _ = env.step(action[0])
        assert np.isfinite(observation.actor).all() and np.isfinite(reward)
finally:
    env.close()
"""
    )
    subprocess.run([sys.executable, str(script), str(ROOT)], cwd=tmp_path, check=True, timeout=60)
