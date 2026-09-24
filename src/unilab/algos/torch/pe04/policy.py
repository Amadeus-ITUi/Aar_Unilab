# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
"""Independent TRON1-style velocity encoder and asymmetric PE04 actor/critic."""

import math
from collections.abc import Sequence

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from unilab.envs.locomotion.pe04.config import load_config, validate_config
from unilab.envs.locomotion.pe04.observations import critic_size


def _mlp(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class PE04EncoderPolicy(nn.Module):
    def __init__(self, config: DictConfig | None = None):
        super().__init__()
        self.config = OmegaConf.merge(config if config is not None else load_config())
        validate_config(self.config)
        env, net = self.config.env, self.config.network
        self.frame_size, self.history_dim, self.command_dim = 30, 300, 3
        self.action_dim = len(env.joint_order)
        self.critic_dim = critic_size(len(env.body_names))
        self.encoder = _mlp(self.history_dim, net.encoder_hidden_dims, 3)
        self.actor = _mlp(36, net.actor_hidden_dims, self.action_dim)
        self.critic = _mlp(self.critic_dim + 3, net.critic_hidden_dims, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(net.initial_std)))

    def encode(self, history):
        return self.encoder(history)

    def actor_input(self, history, observation, commands):
        with torch.no_grad():
            velocity = self.encode(history)
        return torch.cat((velocity, observation, commands), dim=-1)

    def critic_input(self, history, critic_observation, commands):
        return torch.cat((critic_observation, commands), dim=-1)

    def action_mean(self, history, observation, commands):
        return self.actor(self.actor_input(history, observation, commands))

    def value(self, history, critic_observation, commands):
        return self.critic(self.critic_input(history, critic_observation, commands)).squeeze(-1)
