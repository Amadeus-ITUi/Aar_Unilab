"""PE02-owned PPO training and checkpoint adapter."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal

from unilab.algos.torch.pe02 import PE02EncoderPolicy
from unilab.envs.locomotion.pe02 import PE02Env
from unilab.envs.locomotion.pe02.config import load_legacy_config, validate_config


@dataclass(frozen=True)
class PE02TrainResult:
    checkpoint: Path
    mean_reward: float
    steps: int


@dataclass(frozen=True)
class PE02TrainingResult:
    checkpoint: Path
    iterations: int
    samples: int


def train(output_dir: Path, *, config: DictConfig) -> PE02TrainingResult:
    """Run the PE02-owned full PPO/encoder training loop."""
    from unilab.algos.torch.pe02.runner import PE02Runner

    runner = PE02Runner(config, output_dir)
    try:
        checkpoint = runner.learn()
        return PE02TrainingResult(checkpoint, runner.iteration, runner.total_samples)
    finally:
        runner.close()


def train_minimal(output_dir: Path, *, config: DictConfig | None = None) -> PE02TrainResult:
    config = OmegaConf.merge(config if config is not None else load_legacy_config())
    validate_config(config)
    if config.observation != "pe02_v1":
        raise ValueError("train_minimal only supports pe02_v1; use train for pe02_v2")
    steps, seed, device = (
        int(config.training.steps),
        int(config.training.seed),
        str(config.training.device),
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PE02Env(config=config)
    policy = PE02EncoderPolicy(config)
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.training.learning_rate))
    current = env.reset()
    commands = torch.zeros((1, policy.command_dim), device=device)
    losses: list[torch.Tensor] = []
    rewards: list[float] = []
    for _ in range(steps):
        history = torch.as_tensor(current.actor, device=device).unsqueeze(0)
        frame = history[:, -policy.frame_size :]
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
        clip = float(config.training.clip_ratio)
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
        losses.append(
            -torch.minimum(ratio * advantage.detach(), clipped * advantage.detach()).mean()
            + float(config.training.value_loss_coefficient) * advantage.square().mean()
        )
        rewards.append(reward)
        current = env.reset() if done else following
    optimizer.zero_grad()
    torch.stack(losses).mean().backward()
    optimizer.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model_1.pt"
    torch.save(
        {
            "actor_state_dict": policy.state_dict(),
            "steps": steps,
            "seed": seed,
            "robot_id": "pe02",
            "training_config": OmegaConf.to_container(config, resolve=True),
        },
        checkpoint,
    )
    OmegaConf.save(config, output_dir / "training_config.yaml", resolve=True)
    return PE02TrainResult(checkpoint, float(np.mean(rewards)), steps)


def resolve_play_checkpoint(checkpoint: str | int | Path, *, log_root: Path) -> Path:
    """Resolve -1 by run timestamp, then by numeric model iteration within that run."""
    if str(checkpoint) != "-1":
        return Path(str(checkpoint)).expanduser()
    log_root = log_root.expanduser()
    if not log_root.is_dir():
        raise FileNotFoundError(f"PE02 training log directory does not exist: {log_root}")
    runs = []
    for path in log_root.iterdir():
        if not path.is_dir():
            continue
        for fmt in ("%Y-%m-%d_%H-%M-%S_%f_mujoco", "%Y-%m-%d_%H-%M-%S_mujoco"):
            try:
                timestamp = datetime.strptime(path.name, fmt)
            except ValueError:
                continue
            runs.append((timestamp, path.name, path))
            break
    if not runs:
        raise FileNotFoundError(f"No timestamped PE02 training runs found in {log_root}")
    latest_run = max(runs)[2]
    models = []
    for path in latest_run.iterdir():
        match = re.fullmatch(r"model_([0-9]+)\.pt", path.name)
        if match is not None and path.is_file():
            models.append((int(match[1]), path.name, path))
    if not models:
        raise FileNotFoundError(
            f"Latest PE02 run has no saved model_<iteration>.pt yet: {latest_run}; "
            "wait for a checkpoint or specify an explicit checkpoint path"
        )
    return max(models)[2]


def load_policy(checkpoint: Path, device: str = "cpu") -> PE02EncoderPolicy:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("robot_id") != "pe02":
        raise ValueError("PE02 requires a checkpoint with robot_id='pe02'")
    if "training_config" in state:
        config = OmegaConf.create(state["training_config"])
    else:
        # Early PE02 checkpoints predate standing-pose/config snapshots.
        config = load_legacy_config(["env.reset_keyframe=null", "env.initial_height=0.38"])
    validate_config(config)
    policy = PE02EncoderPolicy(config)
    policy.to(device)
    policy.load_state_dict(state["actor_state_dict"])
    policy.eval()
    return policy
