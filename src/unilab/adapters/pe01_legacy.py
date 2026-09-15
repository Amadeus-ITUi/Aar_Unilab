"""PE01 encoder-policy and minimal custom PPO-compatible training adapter.

This is a clean MuJoCo implementation of the predecessor interface, not an
Isaac Gym compatibility layer. The source algorithm remains attributed under
``references/pe01``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from unilab.envs.locomotion.pe01 import PE01Env


class PE01EncoderPolicy(nn.Module):
    def __init__(self, history_dim: int = 300, frame_dim: int = 30, command_dim: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(history_dim, 256), nn.ELU(), nn.Linear(256, 16))
        self.actor = nn.Sequential(
            nn.Linear(16 + frame_dim + command_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 6),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(33 + command_dim + 16, 256), nn.ELU(), nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(6))

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)

    def action_mean(
        self, history: torch.Tensor, observation: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:
        return self.actor(torch.cat((self.encode(history), observation, commands), dim=-1))

    def value(
        self, history: torch.Tensor, critic_observation: torch.Tensor, commands: torch.Tensor
    ) -> torch.Tensor:
        return self.critic(
            torch.cat((critic_observation, commands, self.encode(history)), dim=-1)
        ).squeeze(-1)


@dataclass(frozen=True)
class PE01TrainResult:
    checkpoint: Path
    mean_reward: float
    steps: int


def train_minimal(
    output_dir: Path, *, steps: int = 128, seed: int = 1, device: str = "cpu"
) -> PE01TrainResult:
    """Run a bounded clipped-policy update to verify the PE01 training path."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PE01Env()
    policy = PE01EncoderPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    current = env.reset()
    commands = torch.zeros((1, 3), device=device)
    losses: list[torch.Tensor] = []
    rewards: list[float] = []
    for _ in range(steps):
        history = torch.as_tensor(current.actor, device=device).unsqueeze(0)
        frame = history[:, -30:]
        critic = torch.as_tensor(current.critic, device=device).unsqueeze(0)
        mean = policy.action_mean(history, frame, commands)
        distribution = Normal(mean, policy.log_std.exp().expand_as(mean))
        action = distribution.sample()
        following, reward, done, _ = env.step(action.detach().cpu().numpy()[0])
        advantage = torch.tensor([reward], dtype=torch.float32, device=device) - policy.value(
            history, critic, commands
        )
        old_log_probability = distribution.log_prob(action).sum(-1).detach()
        ratio = torch.exp(distribution.log_prob(action).sum(-1) - old_log_probability)
        clipped = torch.clamp(ratio, 0.8, 1.2)
        losses.append(
            -torch.minimum(ratio * advantage.detach(), clipped * advantage.detach()).mean()
            + 0.5 * advantage.square().mean()
        )
        rewards.append(reward)
        current = env.reset() if done else following
    optimizer.zero_grad()
    torch.stack(losses).mean().backward()
    optimizer.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model_1.pt"
    torch.save({"actor_state_dict": policy.state_dict(), "steps": steps, "seed": seed}, checkpoint)
    return PE01TrainResult(checkpoint, float(np.mean(rewards)), steps)


def load_policy(checkpoint: Path, device: str = "cpu") -> PE01EncoderPolicy:
    policy = PE01EncoderPolicy().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state["actor_state_dict"])
    policy.eval()
    return policy
