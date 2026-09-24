"""Performance paths preserve contacts, random draws, delays and PPO updates."""

import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.pe04.ppo import PE04PPO
from unilab.algos.torch.pe04.runner import PE04Runner
from unilab.envs.locomotion.pe04.math import rotate
from unilab.envs.locomotion.pe04.vector_env import PE04VectorEnv


@pytest.mark.parametrize("position_difference", [True, False])
def test_fused_telemetry_matches_legacy_with_randomization_and_delays(config, position_difference):
    cfg = OmegaConf.merge(
        config,
        {
            "algo": {"num_envs": 8},
            "env": {"episode_length_s": 0.14, "dof_vel_use_pos_diff": position_difference},
            "training": {"mujoco_threads": 2},
        },
    )
    first, second = PE04VectorEnv(cfg), PE04VectorEnv(cfg)
    try:
        assert first.backend.fused_telemetry
        second.backend.fused_telemetry = False
        # Exercise the FIFO as well as nonzero forces, randomized models and resets.
        for env in (first, second):
            b = env.backend
            b.delay_steps[:] = np.arange(8) % 4
            b.delay_buffer = np.zeros((8, 4, 6))
        rng = np.random.default_rng(21)
        for _ in range(40):
            action = rng.normal(size=(8, 6))
            a, b = first.step(action), second.step(action)
            for key in a.obs:
                np.testing.assert_array_equal(a.obs[key], b.obs[key])
            np.testing.assert_array_equal(a.reward, b.reward)
            np.testing.assert_array_equal(a.terminated, b.terminated)
            np.testing.assert_array_equal(a.truncated, b.truncated)
            np.testing.assert_array_equal(first.backend.state, second.backend.state)
            np.testing.assert_array_equal(
                first.backend.contact_history, second.backend.contact_history
            )
    finally:
        first.close()
        second.close()


def test_sparse_observations_preserve_full_batch_values_and_rng(config):
    env = PE04VectorEnv(config)
    try:
        state = copy.deepcopy(env.rng.bit_generator.state)
        full = env._observe()
        expected_rng = copy.deepcopy(env.rng.bit_generator.state)
        env.rng.bit_generator.state = state
        partial = env._observe(ids=np.array([1]))
        for key in full:
            np.testing.assert_array_equal(partial[key], full[key][[1]])
        assert env.rng.bit_generator.state == expected_rng
    finally:
        env.close()


def test_three_component_rotation_matches_numpy_cross():
    rng = np.random.default_rng(12)
    q, v = rng.normal(size=(128, 1, 4)), rng.normal(size=(128, 9, 3))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    t = 2 * np.cross(q[..., 1:], v)
    expected = v + q[..., :1] * t + np.cross(q[..., 1:], t)
    np.testing.assert_array_equal(rotate(q, v), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_ppo_targets_preserve_parameters_and_metrics(config, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = OmegaConf.merge(
        config,
        {"training": {"device": device}, "algo": {"num_learning_epochs": 2, "num_mini_batches": 2}},
    )
    runner = PE04Runner(cfg, tmp_path)
    try:
        reference_cfg = OmegaConf.merge(cfg, {"training": {"cache_update_inputs": False}})
        reference = PE04PPO(copy.deepcopy(runner.policy), reference_cfg)
        for _ in range(2):
            rollout, _ = runner.collect()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
            actual = runner.algorithm.update(rollout)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            expected = reference.update(rollout)
            assert actual == expected
            for key, value in runner.policy.state_dict().items():
                torch.testing.assert_close(
                    value, reference.policy.state_dict()[key], rtol=0, atol=0
                )
    finally:
        runner.close()


def test_bfloat16_updates_keep_fp32_checkpoints_and_inference(config, tmp_path):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    from unilab.adapters.pe04_ppo import load_policy

    cfg = OmegaConf.merge(config, {"training": {"device": "cuda", "update_precision": "bfloat16"}})
    runner = PE04Runner(cfg, tmp_path)
    try:
        before = {key: value.clone() for key, value in runner.policy.state_dict().items()}
        checkpoint = runner.learn()
        for prefix in ("actor.", "critic.", "encoder."):
            assert any(
                not torch.equal(value, runner.policy.state_dict()[key])
                for key, value in before.items()
                if key.startswith(prefix)
            )
        assert all(
            p.dtype == torch.float32 and torch.isfinite(p).all() for p in runner.policy.parameters()
        )
        assert not torch.is_autocast_enabled("cuda")
        loaded = load_policy(checkpoint)
        assert all(p.dtype == torch.float32 for p in loaded.parameters())
        obs = runner.env.state.obs
        args = [torch.from_numpy(obs[key]) for key in ("actor", "frame", "command")]
        with torch.no_grad():
            actual = loaded.action_mean(*args)
            expected = runner.policy.action_mean(*[x.cuda() for x in args]).cpu()
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    finally:
        runner.close()
