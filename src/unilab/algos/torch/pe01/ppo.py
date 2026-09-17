# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# See LICENSE.pe01 for terms; PE01 migration changes are maintained independently.
"""PE01 PPO and separately supervised velocity encoder.

Algorithm provenance: original PE01 BSD-3-Clause implementation, retained under
references/pe01/legacy_training. See LICENSE.pe01 and the migration audit.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributions import Normal

from unilab.algos.torch.pe01.policy import PE01EncoderPolicy


def generalized_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap timeouts from terminal observations; never recurse across resets."""
    advantage = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]) - values[step]
        running = delta + gamma * lam * (~(terminated[step] | truncated[step])) * running
        advantage[step] = running
    return advantage, advantage + values


class PE01PPO:
    def __init__(self, policy: PE01EncoderPolicy, config: DictConfig) -> None:
        self.policy = policy
        self.cfg = config.algo
        self.parameters = [policy.log_std, *policy.actor.parameters(), *policy.critic.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=float(self.cfg.learning_rate))
        self.encoder_optimizer = torch.optim.Adam(
            policy.encoder.parameters(), lr=float(self.cfg.encoder_learning_rate)
        )
        self.learning_rate = float(self.cfg.learning_rate)
        self.updates = 0
        self.encoder_updates = 0

    def distribution(self, obs: dict[str, torch.Tensor]) -> Normal:
        mean = self.policy.action_mean(obs["actor"], obs["frame"], obs["command"])
        return Normal(mean, self.policy.log_std.exp().expand_as(mean))

    def value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy.value(obs["actor"], obs["critic"], obs["command"])

    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        advantage, returns = generalized_advantage(
            rollout["reward"],
            rollout["value"],
            rollout["next_value"],
            rollout["terminated"],
            rollout["truncated"],
            float(self.cfg.gamma),
            float(self.cfg.lam),
        )
        data = {key: value.flatten(0, 1) for key, value in rollout.items()}
        advantage = advantage.flatten()
        data["advantage"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        data["returns"] = returns.flatten()
        count = len(data["reward"])
        indices = torch.randperm(count, device=advantage.device)
        metrics: dict[str, list[float]] = {
            key: []
            for key in (
                "policy_loss",
                "value_loss",
                "entropy",
                "kl",
                "clip_fraction",
                "encoder_loss",
            )
        }
        clip = float(self.cfg.clip_ratio)
        for _ in range(int(self.cfg.num_learning_epochs)):
            for batch in torch.tensor_split(indices, int(self.cfg.num_mini_batches)):
                obs = {key: data[key][batch] for key in ("actor", "frame", "critic", "command")}
                distribution = self.distribution(obs)
                log_prob = distribution.log_prob(data["action"][batch]).sum(-1)
                value = self.value(obs)
                with torch.no_grad():
                    old_sigma = data["sigma"][batch]
                    sigma = distribution.stddev
                    kl = (
                        (
                            torch.log(sigma / old_sigma + 1e-5)
                            + (
                                old_sigma.square()
                                + (data["mean"][batch] - distribution.mean).square()
                            )
                            / (2 * sigma.square())
                            - 0.5
                        )
                        .sum(-1)
                        .mean()
                    )
                    desired_kl = float(self.cfg.desired_kl)
                    if self.cfg.schedule == "adaptive":
                        if kl > 2 * desired_kl:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0 < kl < desired_kl / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for group in self.optimizer.param_groups:
                            group["lr"] = self.learning_rate
                ratio = (log_prob - data["log_prob"][batch]).exp()
                adv = data["advantage"][batch]
                policy_loss = torch.maximum(
                    -adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip)
                ).mean()
                value_error = (value - data["returns"][batch]).square()
                if self.cfg.use_clipped_value_loss:
                    clipped = data["value"][batch] + (value - data["value"][batch]).clamp(
                        -clip, clip
                    )
                    value_error = torch.maximum(
                        value_error, (clipped - data["returns"][batch]).square()
                    )
                value_loss = value_error.mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + float(self.cfg.value_loss_coefficient) * value_loss
                    - float(self.cfg.entropy_coefficient) * entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PE01 PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters, float(self.cfg.max_grad_norm), error_if_nonfinite=True
                )
                self.optimizer.step()
                self.updates += 1
                for key, val in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                    ("kl", kl),
                    ("clip_fraction", ((ratio - 1).abs() > clip).float().mean()),
                ):
                    metrics[key].append(float(val.detach()))
        # Encoder is frozen during all policy updates, then supervised by body velocity.
        indices = torch.randperm(count, device=advantage.device)
        for _ in range(int(self.cfg.num_learning_epochs)):
            for batch in torch.tensor_split(indices, int(self.cfg.num_mini_batches)):
                prediction = self.policy.encode(data["actor"][batch])
                loss = (prediction - data["critic"][batch, :3]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PE01 encoder loss")
                self.encoder_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.encoder_optimizer.step()
                self.encoder_updates += 1
                metrics["encoder_loss"].append(float(loss.detach()))
        return {
            **{key: float(np.mean(values)) for key, values in metrics.items()},
            "learning_rate": self.learning_rate,
            "action_std": float(self.policy.log_std.exp().mean().detach()),
            "optimizer_updates": self.updates,
            "encoder_updates": self.encoder_updates,
        }
