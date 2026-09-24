"""Independent formal PE05 PPO training and checkpoint boundary."""

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig

from unilab.algos.torch.pe05.policy import PE05EncoderPolicy
from unilab.envs.locomotion.pe05.contracts import validate_checkpoint


@dataclass(frozen=True)
class PE05TrainingResult:
    checkpoint: Path
    iterations: int
    samples: int


def train(output_dir: Path, *, config: DictConfig) -> PE05TrainingResult:
    from unilab.algos.torch.pe05.runner import PE05Runner

    runner = PE05Runner(config, output_dir)
    try:
        checkpoint = runner.learn()
        return PE05TrainingResult(checkpoint, runner.iteration, runner.total_samples)
    finally:
        runner.close()


def load_policy(checkpoint: Path, device: str = "cpu") -> PE05EncoderPolicy:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    config = validate_checkpoint(state)
    policy = PE05EncoderPolicy(config)
    policy.to(device)
    policy.load_state_dict(state["actor_state_dict"])
    policy.eval()
    return policy
