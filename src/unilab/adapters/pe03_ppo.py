"""PE03-owned PPO training and checkpoint adapter."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.pe03 import PE03EncoderPolicy
from unilab.envs.locomotion.pe03.config import validate_config


@dataclass(frozen=True)
class PE03TrainingResult:
    checkpoint: Path
    iterations: int
    samples: int


def train(output_dir: Path, *, config: DictConfig) -> PE03TrainingResult:
    """Run the PE03-owned full PPO/encoder training loop."""
    from unilab.algos.torch.pe03.runner import PE03Runner

    runner = PE03Runner(config, output_dir)
    try:
        checkpoint = runner.learn()
        return PE03TrainingResult(checkpoint, runner.iteration, runner.total_samples)
    finally:
        runner.close()


def resolve_play_checkpoint(checkpoint: str | int | Path, *, log_root: Path) -> Path:
    """Resolve -1 by run timestamp, then by numeric model iteration within that run."""
    if str(checkpoint) != "-1":
        return Path(str(checkpoint)).expanduser()
    log_root = log_root.expanduser()
    if not log_root.is_dir():
        raise FileNotFoundError(f"PE03 training log directory does not exist: {log_root}")
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
        raise FileNotFoundError(f"No timestamped PE03 training runs found in {log_root}")
    latest_run = max(runs)[2]
    models = []
    for path in latest_run.iterdir():
        match = re.fullmatch(r"model_([0-9]+)\.pt", path.name)
        if match is not None and path.is_file():
            models.append((int(match[1]), path.name, path))
    if not models:
        raise FileNotFoundError(
            f"Latest PE03 run has no saved model_<iteration>.pt yet: {latest_run}; "
            "wait for a checkpoint or specify an explicit checkpoint path"
        )
    return max(models)[2]


def load_policy(checkpoint: Path, device: str = "cpu") -> PE03EncoderPolicy:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("robot_id") != "pe03":
        raise ValueError("PE03 requires a checkpoint with robot_id='pe03'")
    if "training_config" in state:
        config = OmegaConf.create(state["training_config"])
    else:
        raise ValueError("PE03 checkpoints must contain training_config")
    validate_config(config)
    policy = PE03EncoderPolicy(config)
    policy.to(device)
    policy.load_state_dict(state["actor_state_dict"])
    policy.eval()
    return policy
