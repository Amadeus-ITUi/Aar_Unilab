"""Behavioral checks for the formal PE03 migration, separate from v1 smoke tests."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from unilab.algos.torch.pe03.policy import PE03EncoderPolicy
from unilab.algos.torch.pe03.ppo import generalized_advantage
from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.envs.locomotion.pe03.config import ROOT, load_config
from unilab.envs.locomotion.pe03.vector_env import PE03VectorEnv


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
    policy = PE03EncoderPolicy(small_config())
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
    env = PE03VectorEnv(small_config("env.episode_length_s=0.02"), evaluation=True)
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
    env = PE03VectorEnv(cfg, evaluation=True, num_envs=2)
    try:
        actions = np.array([[1] * 6, [0] * 6])
        env.step(actions)
        np.testing.assert_array_equal(env.backend.qpos[0], env.backend.qpos[1])
        env.step(actions)
        assert np.max(np.abs(env.backend.qpos[0] - env.backend.qpos[1])) > 1e-6
        assert np.all(np.abs(env.backend.torque) <= np.array(cfg.control.torque_limits))
    finally:
        env.close()
    env = PE03VectorEnv(cfg)
    try:
        sampled = env.backend.randomization["body_mass"].copy()
        assert not np.array_equal(sampled[0], sampled[1])
        env._reset_ids(np.array([0]))
        np.testing.assert_array_equal(sampled, env.backend.randomization["body_mass"])
        assert np.all((env.backend.delay_steps >= 10) & (env.backend.delay_steps <= 20))
    finally:
        env.close()


@pytest.mark.parametrize("old_config", [False, True])
def test_standing_rewards_are_targeted_and_keep_original_weights(old_config):
    cfg = small_config(*([] if old_config else ["+reward.base_height_std=1.0"]))
    env = PE03VectorEnv(cfg, evaluation=True)
    try:
        total, terms = env._rewards()
        np.testing.assert_allclose(terms["base_height"], 0, atol=1e-9)
        np.testing.assert_allclose(terms["orientation"], 0, atol=1e-9)
        np.testing.assert_allclose(terms["keep_balance"], 0.02)
        np.testing.assert_allclose(terms["tracking_lin_vel"], 0.02)
        assert len(terms) == 18
        assert np.isfinite(total).all()
        env.backend.qpos[:, 2] += 0.03
        np.testing.assert_allclose(env._rewards()[1]["base_height"], -3 * 0.03**2 * 0.02)
    finally:
        env.close()


def run_training(config, path):
    runner = PE03Runner(config, path)
    try:
        initial = {name: value.clone() for name, value in runner.policy.state_dict().items()}
        checkpoint = runner.learn()
        final = {name: value.clone() for name, value in runner.policy.state_dict().items()}
        return checkpoint, initial, final
    finally:
        runner.close()


@pytest.mark.parametrize("experiment", ["standing", "walking"])
def test_multiple_updates_and_exact_resume(tmp_path, experiment):
    cfg = small_config(*([f"+experiment={experiment}"] if experiment else []))
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
        PE03Runner(cfg, tmp_path / "wrong")


def test_finite_evaluation_metrics(tmp_path):
    runner = PE03Runner(
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
    env = PE03VectorEnv(cfg, evaluation=True, evaluation_reset_noise=True)
    try:
        perturbed = env.backend.qpos.copy()
        assert not np.array_equal(perturbed[0], perturbed[1])
        assert np.max(np.abs(perturbed[:, 7:] - env.home[7:])) <= 0.01
    finally:
        env.close()
    replay = PE03VectorEnv(cfg, evaluation=True, evaluation_reset_noise=True)
    try:
        np.testing.assert_array_equal(perturbed, replay.backend.qpos)
    finally:
        replay.close()
    play = PE03VectorEnv(cfg, evaluation=True)
    try:
        np.testing.assert_array_equal(play.backend.qpos, np.tile(play.home, (4, 1)))
    finally:
        play.close()


@pytest.mark.parametrize(
    "experiment,version,frame_size", [("standing", "pe03_v2", 30), ("walking", "pe03_v3", 24)]
)
def test_formal_training_export_and_relocated_play_are_independent(
    tmp_path, experiment, version, frame_size
):
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
        if any(x in fullname.lower() for x in ('pe01', 'pe02')) or fullname.startswith('references'):
            raise AssertionError(fullname)

sys.meta_path.insert(0, RejectReferenceImports())
sys.path.insert(0, sys.argv[1])
import numpy as np
import onnxruntime as ort
import torch
from unilab.adapters.pe03_ppo import train, load_policy
from unilab.envs.locomotion.pe03.config import load_config
from unilab.envs.locomotion.pe03.play_env import make_play_env
from unilab.release_contract import load_manifest
from scripts.train_pe03 import export_release

cfg = load_config([
    'algo.num_envs=2', 'algo.num_steps_per_env=2', 'algo.max_iterations=1',
    'training.device=cpu', 'training.logger=none', 'training.evaluation_interval=0',
    'play.delay_ms=25', 'play.command=[0.2,-0.1,0.3]',
] + ([f'+experiment={sys.argv[2]}'] if sys.argv[2] != 'none' else []))
result = train(Path('run'), config=cfg)
release = export_release(result.checkpoint, 'formal')
relocated = Path('relocated')
shutil.move(release, relocated)
manifest = load_manifest(relocated / 'deployment_manifest.json')
assert manifest['policy']['observation_builder'] == sys.argv[3]
frame_size = int(sys.argv[4])
assert manifest['policy']['history']['frame_size'] == frame_size
assert manifest['control']['type'] == 'position_pd'
assert manifest['control']['command_delay_steps'] == 10
import json
runtime=json.loads((relocated/'robot/pe03_runtime.json').read_text())
assert runtime['schema']==sys.argv[3].replace('pe03_v','pe03.runtime.v')+'.joint-limits.v1'
np.testing.assert_allclose(runtime['joint_target_limits'],
 [[-.20,1.50],[-1.44,0],[-2.02,0],[-1.50,.20],[0,1.44],[0,2.02]])
stale=torch.load(result.checkpoint,weights_only=True)
stale['asset_sha256']='pre-measurement-asset-fingerprint'
torch.save(stale,Path('stale.pt'))
try: export_release(Path('stale.pt'),'should-not-export')
except ValueError as error: assert 'assets changed' in str(error)
else: raise AssertionError('stale checkpoint exported against current assets')
policy = load_policy(result.checkpoint)
session = ort.InferenceSession(str(relocated / 'policy.onnx'), providers=['CPUExecutionProvider'])
env = make_play_env(policy.config, relocated / manifest['artifacts']['scene_path'])
try:
    observation = env.reset()
    command = np.load(relocated / 'golden_inputs/command.npy')
    for _ in range(12):
        inputs = {
            'observation_history': observation.actor[None],
            'observation': observation.actor[None, -frame_size:],
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

from tools.compare_pe03_sim2sim import python_trajectory, cpp_trajectory
import onnx
from onnx import helper, numpy_helper
release_path=Path('relocated')
binary=Path(sys.argv[1])/'sim2sim/build/aar_sim2sim'
for label,constant in [('policy',None),('upper',1000.),('lower',-1000.)]:
 if constant is not None:
  model=onnx.load(release_path/'policy.onnx')
  del model.graph.node[:]
  del model.graph.initializer[:]
  model.graph.node.append(helper.make_node('Constant',[],['action'],
   value=numpy_helper.from_array(np.full((1,6),constant,dtype=np.float32))))
  onnx.save(model,release_path/'policy.onnx')
  np.save(release_path/'golden_outputs/action.npy',np.full((1,6),constant,dtype=np.float32))
 for steps in (1,10,100):
  expected=python_trajectory(release_path,steps)
  actual=cpp_trajectory(binary,release_path,steps,Path(f'telemetry-{label}-{steps}'))
  for a,b in zip(expected,actual,strict=True):
   assert a.keys()==b.keys()
   np.testing.assert_allclose([a[k] for k in a],[b[k] for k in a],rtol=0,atol=2e-5)
# Old self-contained releases still select the original, unclipped protocol.
from omegaconf import OmegaConf
import json
runtime_path=release_path/'robot/pe03_runtime.json'
runtime=json.loads(runtime_path.read_text())
runtime['schema']=runtime['schema'].removesuffix('.joint-limits.v1')
del runtime['joint_target_limits']
runtime_path.write_text(json.dumps(runtime))
old_config=OmegaConf.load(release_path/'runtime_config.yaml')
del old_config.control.clip_joint_targets
OmegaConf.save(old_config,release_path/'runtime_config.yaml')
expected=python_trajectory(release_path,10)
actual=cpp_trajectory(binary,release_path,10,Path('telemetry-legacy'))
for a,b in zip(expected,actual,strict=True):
 np.testing.assert_allclose([a[k] for k in a],[b[k] for k in a],rtol=0,atol=2e-5)
"""
    )
    subprocess.run(
        [sys.executable, str(script), str(ROOT), experiment or "none", version, str(frame_size)],
        cwd=tmp_path,
        check=True,
        timeout=180,
    )
