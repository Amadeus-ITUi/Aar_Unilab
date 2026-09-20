# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# See LICENSE.pe01 for terms; PE03 migration changes are maintained independently.
"""PE03-owned encoder, actor and critic networks."""

import math
from collections.abc import Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from unilab.envs.locomotion.pe03.config import load_config, validate_config


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class PE03EncoderPolicy(nn.Module):
    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = OmegaConf.merge(config if config is not None else load_config())
        validate_config(self.config)
        env, network = self.config.env, self.config.network
        self.gait_policy = self.config.observation == "pe03_v4"
        self.frame_size = int(env.frame_size)
        self.history_dim = self.frame_size * int(env.history_length)
        self.command_dim = int(network.command_size)
        self.critic_dim = 14 if self.gait_policy else self.frame_size + 3
        self.action_dim = len(env.joint_order)
        latent = int(network.latent_dim)
        self.encoder = _mlp(self.history_dim, list(network.encoder_hidden_dims), latent)
        self.actor = _mlp(
            latent + self.history_dim
            if self.gait_policy
            else latent + self.frame_size + self.command_dim,
            list(network.actor_hidden_dims),
            self.action_dim,
        )
        self.critic = _mlp(
            self.critic_dim + self.history_dim
            if self.gait_policy
            else self.critic_dim + self.command_dim + latent,
            list(network.critic_hidden_dims),
            1,
        )
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), math.log(float(network.initial_std)))
        )

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)

    def actor_input(
        self, history: torch.Tensor, observation: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:
        if self.gait_policy:
            frame = torch.cat((observation[:, :30], commands, observation[:, 36:]), dim=-1)
            history = torch.cat((history[:, : -self.frame_size], frame), dim=-1)
            with torch.no_grad():
                latent = self.encode(history)
            return torch.cat((history, latent), dim=-1)
        with torch.no_grad():
            latent = self.encode(history)
        return torch.cat((latent, observation, commands), dim=-1)

    def critic_input(
        self, history: torch.Tensor, critic_observation: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:
        if self.gait_policy:
            return torch.cat((history, critic_observation), dim=-1)
        with torch.no_grad():
            latent = self.encode(history)
        return torch.cat((critic_observation, commands, latent), dim=-1)

    def action_mean(
        self, history: torch.Tensor, observation: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:
        return self.actor(self.actor_input(history, observation, commands))

    def value(
        self, history: torch.Tensor, critic_observation: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:
        return self.critic(self.critic_input(history, critic_observation, commands)).squeeze(-1)
