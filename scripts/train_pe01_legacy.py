#!/usr/bin/env python3
"""PE01 custom PPO entrypoint."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from unilab.adapters.pe01_legacy import load_policy, train_minimal
from unilab.catalog import catalog
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe01 import PE01Env
from unilab.playback import InteractiveSession
from unilab.release_contract import create_release


class ExportedPE01Actor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation_history, observation, command):
        return self.policy.action_mean(observation_history, observation, command)


def export_release(
    checkpoint: Path,
    run_id: str,
    device: str,
    *,
    robot_id: str = "pe01",
    task_id: str = "pe01_flat",
) -> Path:
    robot = catalog.robots[robot_id]
    root = Path(__file__).resolve().parents[1]
    robot_root = repository_path(robot.scene, root).parent
    policy = load_policy(checkpoint, device=device, robot_id=robot_id)
    actor = ExportedPE01Actor(policy)
    run_dir = checkpoint.parent
    onnx_path = run_dir / "policy.onnx"
    torch.onnx.export(
        actor,
        (
            torch.zeros(1, 300, device=device),
            torch.zeros(1, 30, device=device),
            torch.zeros(1, 3, device=device),
        ),
        onnx_path,
        input_names=["observation_history", "observation", "command"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    golden_inputs = {
        "observation_history": np.zeros((1, 300), dtype=np.float32),
        "observation": np.zeros((1, 30), dtype=np.float32),
        "command": np.zeros((1, 3), dtype=np.float32),
    }
    with torch.inference_mode():
        torch_action = (
            actor(
                *(
                    torch.as_tensor(golden_inputs[name], device=device)
                    for name in (
                        "observation_history",
                        "observation",
                        "command",
                    )
                )
            )
            .cpu()
            .numpy()
        )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    golden_action = session.run(["action"], golden_inputs)[0]
    np.testing.assert_allclose(torch_action, golden_action, rtol=1.0e-5, atol=1.0e-6)
    runtime_config = run_dir / "runtime_config.yaml"
    runtime_config.write_text(
        f"robot: {robot_id}\ntask: {task_id}\nsimulator: mujoco\nphysics_hz: 400\npolicy_hz: 50\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "aar-unilab.actor.v1",
        "robot": {
            "id": robot.id,
            "asset_version": robot.asset_version,
            "joint_order": list(robot.joints),
        },
        "task": {"id": task_id},
        "policy": {
            "observation_builder": "pe01_v1",
            "inputs": [
                {"name": "observation_history", "dtype": "float32", "shape": [1, 300]},
                {"name": "observation", "dtype": "float32", "shape": [1, 30]},
                {"name": "command", "dtype": "float32", "shape": [1, 3]},
            ],
            "outputs": [{"name": "action", "dtype": "float32", "shape": [1, 6]}],
            "history": {"length": 10, "frame_size": 30, "layout": "frame-major"},
            "hidden_state": [],
        },
        "control": {
            "physics_hz": 400,
            "motor_hz": 400,
            "policy_hz": 50,
            "action_scale": 1.0,
            "action_clip": 1.0,
            "command_delay_steps": 0,
            "gains": {"hip": [4.3, 0.34], "thigh": [4.3, 0.34], "calf": [4.9, 0.24]},
        },
        "artifacts": {"scene_path": "robot/scene.xml"},
    }
    release_dir = Path("releases") / robot_id / task_id / run_id
    robot_files = []
    for relative in robot.runtime_assets:
        source = robot_root / relative
        robot_files.extend(sorted(source.rglob("*")) if source.is_dir() else [source])
    return create_release(
        release_dir,
        onnx=onnx_path,
        runtime_config=runtime_config,
        manifest=manifest,
        robot_files=[path for path in robot_files if path.is_file()],
        robot_root=robot_root,
        golden_inputs=golden_inputs,
        golden_outputs={"action": golden_action},
    )


def values(argv: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in argv if "=" in item)


def main() -> int:
    args = values(sys.argv[1:])
    robot_id = args.get("robot", "pe01")
    task_id = args.get("task", f"{robot_id}_flat")
    selection = catalog.resolve(
        {
            "robot": robot_id,
            "task": task_id,
            "observation": "pe01_legacy",
            "policy": "pe01_encoder_mlp",
            "algorithm": "pe01_custom_ppo",
            "simulator": "mujoco",
        }
    )
    model_path = repository_path(selection.robot.scene, Path(__file__).resolve().parents[1])
    mode = args.get("mode", "train")
    device = args.get("training.device", "cpu")
    if mode == "train":
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_mujoco")
        destination = Path("logs/pe01_custom_ppo") / robot_id / task_id / run_id
        result = train_minimal(
            destination,
            steps=int(args.get("training.steps", "128")),
            seed=int(args.get("training.seed", "1")),
            device=device,
            model_path=model_path,
            robot_id=robot_id,
        )
        release = export_release(
            result.checkpoint, run_id, device, robot_id=robot_id, task_id=task_id
        )
        print(
            f"{robot_id.upper()} checkpoint={result.checkpoint} release={release} mean_reward={result.mean_reward:.6f}"
        )
        return 0
    checkpoint_value = args.get("checkpoint")
    if not checkpoint_value:
        print(f"{robot_id.upper()} play requires checkpoint=<path>", file=sys.stderr)
        return 2
    policy = load_policy(Path(checkpoint_value), device=device, robot_id=robot_id)
    env = PE01Env(model_path)

    def action(observation, command):
        history = torch.as_tensor(observation.actor, device=device).unsqueeze(0)
        commands = torch.as_tensor(command, device=device).unsqueeze(0)
        with torch.inference_mode():
            selected = policy.action_mean(history, history[:, -30:], commands)
        return selected.cpu().numpy()[0]

    telemetry_dir = Path(args.get("play.telemetry", f"logs/play/{robot_id}/{task_id}/latest"))
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
