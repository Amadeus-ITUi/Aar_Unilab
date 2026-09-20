# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# See LICENSE.pe01 for terms; PE03 migration changes are maintained independently.
"""PE03 PPO and separately supervised velocity encoder.

Algorithm provenance: original PE01 BSD-3-Clause implementation, retained under
references/pe01/legacy_training. See LICENSE.pe01 and the migration audit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from omegaconf import DictConfig
from torch.distributions import Normal

from unilab.algos.torch.pe03.policy import PE03EncoderPolicy


@contextmanager
def matmul_precision(precision: str) -> Iterator[None]:
    """Scope Tensor Core opt-in to updates; preserve inference/export settings."""
    previous = torch.get_float32_matmul_precision()
    if previous != precision:
        torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        if previous != precision:
            torch.set_float32_matmul_precision(previous)


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


class PE03PPO:
    def __init__(self, policy: PE03EncoderPolicy, config: DictConfig) -> None:
        self.policy = policy
        self.cfg = config.algo
        self.matmul_precision = str(config.training.get("matmul_precision", "highest"))
        self.cache_update_inputs = bool(config.training.get("cache_update_inputs", True))
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
        return self._distribution(mean)

    def _distribution(self, mean: torch.Tensor) -> Normal:
        # Loss/gradient checks and the environment's finite-action check guard
        # training. Distribution validation otherwise synchronizes CUDA several
        # times for every inference and minibatch.
        return Normal(mean, self.policy.log_std.exp().expand_as(mean), validate_args=False)

    def value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy.value(obs["actor"], obs["critic"], obs["command"])

    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        with matmul_precision(self.matmul_precision):
            return self._update(rollout)

    def _update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
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
        batches = torch.tensor_split(indices, int(self.cfg.num_mini_batches))

        # The encoder is frozen throughout PPO, and the permutation is already
        # shared by all epochs. Prepare each batch once, in exactly the same
        # batch shape, rather than repeating gathers, encoding and concatenation.
        def prepare_inputs(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                history, commands = data["actor"][batch], data["command"][batch]
                return (
                    self.policy.actor_input(history, data["frame"][batch], commands),
                    self.policy.critic_input(history, data["critic"][batch], commands),
                )

        inputs = [prepare_inputs(batch) for batch in batches] if self.cache_update_inputs else None
        metrics: dict[str, list[torch.Tensor]] = {
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
            for index, batch in enumerate(batches):
                actor_input, critic_input = (
                    inputs[index] if inputs is not None else prepare_inputs(batch)
                )
                distribution = self._distribution(self.policy.actor(actor_input))
                log_prob = distribution.log_prob(data["action"][batch]).sum(-1)
                value = self.policy.critic(critic_input).squeeze(-1)
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
                        kl_value = float(kl)
                        if kl_value > 2 * desired_kl:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0 < kl_value < desired_kl / 2:
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
                    raise FloatingPointError("non-finite PE03 PPO loss")
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
                    metrics[key].append(val.detach())
        # Drop the larger PPO inputs before preparing the supervised phase.
        del inputs, actor_input, critic_input
        # Encoder is frozen during all policy updates, then supervised by body velocity.
        indices = torch.randperm(count, device=advantage.device)
        batches = torch.tensor_split(indices, int(self.cfg.num_mini_batches))

        def prepare_encoder(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return data["actor"][batch], data["critic"][batch, :3]

        encoder_batches = (
            [prepare_encoder(batch) for batch in batches] if self.cache_update_inputs else None
        )
        for _ in range(int(self.cfg.num_learning_epochs)):
            for index, batch in enumerate(batches):
                history, velocity = (
                    encoder_batches[index]
                    if encoder_batches is not None
                    else prepare_encoder(batch)
                )
                prediction = self.policy.encode(history)
                loss = (prediction - velocity).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PE03 encoder loss")
                self.encoder_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.encoder_optimizer.step()
                self.encoder_updates += 1
                metrics["encoder_loss"].append(loss.detach())
        del encoder_batches, history, velocity
        if self.policy.gait_policy:
            with torch.no_grad():
                error = self.policy.encode(data["actor"]) - data["critic"][:, :3]
                for index, axis in enumerate(("x", "y", "z")):
                    metrics[f"velocity_estimator/rmse_{axis}"] = [
                        error[:, index].square().mean().sqrt()
                    ]
        # Transfer reporting scalars together, after all optimization work.
        reduced = torch.stack([torch.stack(values).double().mean() for values in metrics.values()])
        reported = dict(zip(metrics, reduced.cpu().tolist(), strict=True))
        return {
            **reported,
            "learning_rate": self.learning_rate,
            "action_std": float(self.policy.log_std.exp().mean().detach()),
            "optimizer_updates": self.updates,
            "encoder_updates": self.encoder_updates,
        }
