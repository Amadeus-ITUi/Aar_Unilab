"""Independent formal PE01 PPO training and checkpoint boundary."""

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.pe01.policy import PE01EncoderPolicy
from unilab.envs.locomotion.pe01.config import validate_config


@dataclass(frozen=True)
class PE01TrainingResult:
    checkpoint: Path
    iterations: int
    samples: int


def train(output_dir: Path, *, config: DictConfig) -> PE01TrainingResult:
    from unilab.algos.torch.pe01.runner import PE01Runner

    runner = PE01Runner(config, output_dir)
    try:
        checkpoint = runner.learn()
        return PE01TrainingResult(checkpoint, runner.iteration, runner.total_samples)
    finally:
        runner.close()


def load_policy(checkpoint: Path, device: str = "cpu") -> PE01EncoderPolicy:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("schema") != "pe01.training.v2" or state.get("robot_id") != "pe01":
        raise ValueError(
            "formal PE01 requires a pe01.training.v2 checkpoint; use legacy playback for v1"
        )
    config = OmegaConf.create(state["training_config"])
    validate_config(config)
    policy = PE01EncoderPolicy(config)
    policy.to(device)
    policy.load_state_dict(state["actor_state_dict"])
    policy.eval()
    return policy
