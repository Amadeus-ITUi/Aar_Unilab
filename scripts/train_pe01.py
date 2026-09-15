#!/usr/bin/env python3
"""PE01 custom PPO adapter entrypoint used by generic train/play selectors."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from unilab.adapters.pe01_legacy import load_policy, train_minimal
from unilab.envs.locomotion.pe01 import PE01Env
from unilab.playback import InteractiveSession
from unilab.release_contract import create_release


class ExportedPE01Actor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation_history, observation, command):
        return self.policy.action_mean(observation_history, observation, command)


def export_release(checkpoint: Path, run_id: str, device: str) -> Path:
    policy = load_policy(checkpoint, device=device)
    actor = ExportedPE01Actor(policy)
    run_dir = checkpoint.parent
    onnx_path = run_dir / "policy.onnx"
    torch.onnx.export(
        actor,
        (torch.zeros(1, 300, device=device), torch.zeros(1, 30, device=device), torch.zeros(1, 3, device=device)),
        onnx_path,
        input_names=["observation_history", "observation", "command"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    runtime_config = run_dir / "runtime_config.yaml"
    runtime_config.write_text(
        "robot: pe01\ntask: pe01_flat\nsimulator: mujoco\nphysics_hz: 400\npolicy_hz: 50\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "aar-unilab.actor.v1",
        "robot": {
            "id": "pe01",
            "asset_version": "dragon-3-final-20260915",
            "joint_order": ["left_hip_joint", "left_thigh_joint", "left_calf_joint", "right_hip_joint", "right_thigh_joint", "right_calf_joint"],
        },
        "task": {"id": "pe01_flat"},
        "policy": {
            "inputs": [
                {"name": "observation_history", "dtype": "float32", "shape": [1, 300]},
                {"name": "observation", "dtype": "float32", "shape": [1, 30]},
                {"name": "command", "dtype": "float32", "shape": [1, 3]},
            ],
            "outputs": [{"name": "action", "dtype": "float32", "shape": [1, 6]}],
            "history": {"length": 10, "frame_size": 30, "layout": "frame-major"},
            "hidden_state": [],
        },
        "control": {"physics_hz": 400, "motor_hz": 400, "policy_hz": 50, "action_scale": 0.25, "action_clip": 1.0, "command_delay_steps": 0, "gains": {"hip": [4.3, 0.34], "thigh": [4.3, 0.34], "calf": [4.9, 0.24]}},
        "artifacts": {"scene_path": "robot/scene.xml"},
    }
    release_dir = Path("releases/pe01/pe01_flat") / run_id
    return create_release(
        release_dir,
        onnx=onnx_path,
        runtime_config=runtime_config,
        manifest=manifest,
        robot_files=["src/unilab/assets/robots/pe01/pe01.xml", "src/unilab/assets/robots/pe01/scene.xml"],
    )


def values(argv: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in argv if "=" in item)


def main() -> int:
    args = values(sys.argv[1:])
    mode = args.get("mode", "train")
    device = args.get("training.device", "cpu")
    if mode == "train":
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_mujoco")
        destination = Path("logs/pe01_custom_ppo/pe01/pe01_flat") / run_id
        result = train_minimal(
            destination, steps=int(args.get("training.steps", "128")),
            seed=int(args.get("training.seed", "1")), device=device
        )
        release = export_release(result.checkpoint, run_id, device)
        print(f"PE01 checkpoint={result.checkpoint} release={release} mean_reward={result.mean_reward:.6f}")
        return 0
    checkpoint_value = args.get("checkpoint")
    if not checkpoint_value:
        print("PE01 play requires checkpoint=<path>", file=sys.stderr)
        return 2
    policy = load_policy(Path(checkpoint_value), device=device)
    env = PE01Env()
    def action(observation, command):
        history = torch.as_tensor(observation.actor, device=device).unsqueeze(0)
        commands = torch.as_tensor(command, device=device).unsqueeze(0)
        with torch.inference_mode():
            selected = policy.action_mean(history, history[:, -30:], commands)
        return selected.cpu().numpy()[0]

    telemetry_dir = Path(args.get("play.telemetry", "logs/play/pe01/pe01_flat/latest"))
    with InteractiveSession(
        env.model,
        env.data,
        telemetry_dir=telemetry_dir,
        render=args.get("play.render", "interactive") == "interactive",
        plot=args.get("play.plot", "true").lower() == "true",
    ) as session:
        session.run(int(args.get("play.steps", "1000")), action, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
