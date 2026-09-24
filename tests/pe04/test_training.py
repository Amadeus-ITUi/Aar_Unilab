import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.adapters.pe04_ppo import load_policy
from unilab.algos.torch.pe04.policy import PE04EncoderPolicy
from unilab.algos.torch.pe04.ppo import generalized_advantage
from unilab.algos.torch.pe04.runner import PE04Runner


def test_actor_velocity_bottleneck_and_detached_gradients(config):
    policy = PE04EncoderPolicy(config)
    history = torch.randn(2, 300)
    frame, command = torch.randn(2, 30), torch.randn(2, 3)
    assert policy.actor[0].in_features == 36 and policy.critic[0].in_features == 270
    with torch.no_grad():
        for parameter in policy.encoder.parameters():
            parameter.zero_()
    torch.testing.assert_close(
        policy.action_mean(history, frame, command),
        policy.action_mean(history + 10, frame, command),
    )
    policy.action_mean(history, frame, command).sum().backward()
    assert all(p.grad is None for p in policy.encoder.parameters())
    (policy.encode(history) - 1).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoder.parameters())


def test_timeout_bootstrap_does_not_cross_reset():
    reward = torch.tensor([[1.0], [2.0], [100.0]])
    value = torch.zeros_like(reward)
    terminated = torch.tensor([[False], [False], [True]])
    truncated = torch.tensor([[False], [True], [False]])
    adv, _ = generalized_advantage(
        reward, value, torch.ones_like(value) * 3, terminated, truncated, 0.9, 1
    )
    torch.testing.assert_close(adv, torch.tensor([[7.93], [4.7], [100.0]]))


def test_training_checkpoint_resume_and_foreign_checkpoint_rejection(config, tmp_path):
    config.env.episode_length_s = 0.04
    runner = PE04Runner(config, tmp_path / "first")
    try:
        before = {k: v.clone() for k, v in runner.policy.state_dict().items()}
        checkpoint = runner.learn()
        for prefix in ("encoder.", "actor.", "critic."):
            assert any(
                not torch.equal(value, runner.policy.state_dict()[key])
                for key, value in before.items()
                if key.startswith(prefix)
            )
        expected, _ = runner.collect()
    finally:
        runner.close()
    resumed = PE04Runner(
        OmegaConf.merge(config, {"training": {"resume": str(checkpoint)}}), tmp_path / "resumed"
    )
    try:
        actual, _ = resumed.collect()
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        assert resumed.iteration == 1
    finally:
        resumed.close()
    payload = torch.load(checkpoint, weights_only=True)
    assert len(payload["observation_layout"]) == 18
    for key, invalid, error in (
        ("observation_layout", [], "layout"),
        ("asset_sha256", "changed", "assets"),
    ):
        original = payload[key]
        payload[key] = invalid
        torch.save(payload, tmp_path / "incompatible.pt")
        with pytest.raises(ValueError, match=error):
            load_policy(tmp_path / "incompatible.pt")
        payload[key] = original
    payload["robot_id"] = "pe03"
    torch.save(payload, tmp_path / "foreign.pt")
    with pytest.raises(ValueError, match="robot_id"):
        load_policy(tmp_path / "foreign.pt")
