#!/usr/bin/env python3
"""Independent PE02 training, playback and ONNX export entrypoint."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch import nn

from unilab.adapters.pe02_ppo import load_policy, resolve_play_checkpoint, train, train_minimal
from unilab.algos.torch.pe02 import PE02EncoderPolicy
from unilab.base.backend.mujoco.single_robot import configure_release_model
from unilab.catalog import catalog
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe02.config import ROOT, load_config, load_legacy_config
from unilab.envs.locomotion.pe02.play_env import make_play_env
from unilab.playback import InteractiveSession
from unilab.release_contract import create_release


class ExportedPE02Actor(nn.Module):
    def __init__(self, policy: PE02EncoderPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation_history, observation, command):
        return self.policy.action_mean(observation_history, observation, command)


def export_release(checkpoint: Path, run_id: str, device: str = "cpu") -> Path:
    import onnxruntime as ort

    policy = load_policy(checkpoint, device=device)
    config = policy.config
    robot = catalog.robots["pe02"]
    if tuple(config.env.joint_order) != robot.joints:
        raise ValueError("PE02 config joint_order differs from its catalog contract")
    actor = ExportedPE02Actor(policy)
    onnx_path = checkpoint.parent / "policy.onnx"
    golden_inputs = {
        "observation_history": np.zeros((1, policy.history_dim), dtype=np.float32),
        "observation": np.zeros((1, policy.frame_size), dtype=np.float32),
        "command": np.zeros((1, policy.command_dim), dtype=np.float32),
    }
    tensors = tuple(torch.as_tensor(value, device=device) for value in golden_inputs.values())
    torch.onnx.export(
        actor,
        tensors,
        onnx_path,
        input_names=list(golden_inputs),
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    with torch.inference_mode():
        torch_action = actor(*tensors).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    golden_action = session.run(["action"], golden_inputs)[0]
    np.testing.assert_allclose(torch_action, golden_action, rtol=1.0e-5, atol=1.0e-6)
    runtime_config = checkpoint.parent / "runtime_config.yaml"
    OmegaConf.save(config, runtime_config, resolve=True)
    model_path = repository_path(str(config.env.model_path), ROOT)
    manifest = {
        "schema": "aar-unilab.actor.v1",
        "robot": {
            "id": "pe02",
            "asset_version": robot.asset_version,
            "joint_order": list(config.env.joint_order),
        },
        "task": {"id": str(config.task_id)},
        "policy": {
            "observation_builder": str(config.observation),
            "inputs": [
                {"name": name, "dtype": "float32", "shape": list(value.shape)}
                for name, value in golden_inputs.items()
            ],
            "outputs": [{"name": "action", "dtype": "float32", "shape": [1, policy.action_dim]}],
            "history": {
                "length": int(config.env.history_length),
                "frame_size": policy.frame_size,
                "layout": "frame-major",
            },
            "hidden_state": [],
        },
        "control": {
            "physics_hz": int(config.control.physics_hz),
            "motor_hz": int(config.control.get("motor_hz", config.control.physics_hz)),
            "policy_hz": int(config.control.policy_hz),
            "action_scale": float(config.control.action_scale),
            "action_clip": float(config.control.action_clip),
            "command_delay_steps": 0,
        },
        "artifacts": {"scene_path": f"robot/{model_path.name}"},
    }
    reset_keyframe = config.env.get("reset_keyframe")
    if reset_keyframe is not None:
        manifest["task"]["reset_keyframe"] = str(reset_keyframe)
    # Bake timing/reset settings into the release without altering the source assets.
    with tempfile.TemporaryDirectory(prefix="pe02-release-") as temporary:
        staging = Path(temporary)
        for relative in robot.runtime_assets:
            source = model_path.parent / relative
            if source.is_dir():
                shutil.copytree(source, staging / relative)
            else:
                shutil.copy2(source, staging / relative)
        configure_release_model(
            staging / "pe02.xml",
            scene_path=staging / model_path.name,
            physics_hz=int(config.control.physics_hz),
            base_height=(
                float(config.env.initial_height) if config.env.initial_height is not None else None
            ),
            reset_keyframe=reset_keyframe,
        )
        if config.observation in {"pe02_v2", "pe02_v3"}:
            from unilab.base.backend.mujoco.batched_robot import compile_robot_scene

            model = compile_robot_scene(
                staging / model_path.name, tuple(config.env.body_names), visual=False
            )
            runtime = {
                "schema": f"pe02.runtime.{str(config.observation).removeprefix('pe02_')}",
                "default_joint_position": model.key(str(reset_keyframe)).qpos[7:].tolist(),
                "kp": list(config.control.kp),
                "kd": list(config.control.kd),
                "torque_limits": list(config.control.torque_limits),
                "user_torque_limit": float(config.control.user_torque_limit),
                "position_difference": bool(config.env.dof_vel_use_pos_diff),
                "delay_steps": round(
                    float(config.play.delay_ms) * int(config.control.physics_hz) / 1000
                ),
                "normalization": OmegaConf.to_container(config.normalization, resolve=True),
            }
            if config.observation == "pe02_v2":
                runtime["gait"] = list(config.play.gait)
            (staging / "pe02_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
            manifest["artifacts"]["pe02_runtime_path"] = "robot/pe02_runtime.json"
            manifest["control"]["command_delay_steps"] = runtime["delay_steps"]
            manifest["control"]["type"] = "position_pd"
            golden_inputs["command"][:] = np.array(config.play.command) * [
                config.normalization.lin_vel,
                config.normalization.lin_vel,
                config.normalization.ang_vel,
            ]
            # Recompute golden output if a nonzero default play command was configured.
            golden_action = session.run(["action"], golden_inputs)[0]
        return create_release(
            Path("releases/pe02") / str(config.task_id) / run_id,
            onnx=onnx_path,
            runtime_config=runtime_config,
            manifest=manifest,
            robot_root=staging,
            robot_files=sorted(path for path in staging.rglob("*") if path.is_file()),
            golden_inputs=golden_inputs,
            golden_outputs={"action": golden_action},
        )


def main() -> int:
    loader = load_legacy_config if "observation=pe02_v1" in sys.argv[1:] else load_config
    config = loader(sys.argv[1:])
    catalog.resolve(
        {
            "robot": str(config.robot),
            "task": str(config.task_id),
            "observation": str(config.observation),
            "policy": str(config.policy),
            "algorithm": str(config.algorithm),
            "simulator": str(config.simulator),
        }
    )
    device = str(config.training.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.mode == "train":
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f_mujoco")
        if config.observation == "pe02_v1":
            destination = Path("logs/pe02_custom_ppo/pe02") / str(config.task_id) / run_id
            minimal = train_minimal(destination, config=config)
            release = export_release(minimal.checkpoint, run_id, device)
            print(f"PE02 legacy smoke checkpoint={minimal.checkpoint} release={release}")
        else:
            destination = Path(str(config.training.log_root)) / run_id
            result = train(destination, config=config)
            release = (
                export_release(result.checkpoint, run_id, device)
                if config.training.export
                else None
            )
            print(
                f"PE02 checkpoint={result.checkpoint} iterations={result.iterations} "
                f"samples={result.samples} release={release}"
            )
        return 0
    if config.mode != "play":
        raise ValueError(f"unsupported PE02 mode={config.mode!r}")
    if not config.checkpoint:
        print("PE02 play requires checkpoint=<path> or checkpoint=-1", file=sys.stderr)
        return 2
    checkpoint = resolve_play_checkpoint(
        config.checkpoint,
        log_root=Path(
            str(config.training.get("log_root", f"logs/pe02_custom_ppo/pe02/{config.task_id}"))
        ),
    )
    print(f"PE02 play checkpoint={checkpoint.resolve()}", flush=True)
    policy = load_policy(checkpoint, device=device)
    play_config = OmegaConf.merge(policy.config)
    if play_config.observation in {"pe02_v2", "pe02_v3"}:
        overrides = OmegaConf.to_container(config.play, resolve=True)
        # The checkpoint owns the observation layout. A v3 launcher can play old v2 models.
        if play_config.observation == "pe02_v3":
            overrides["gait"] = None
        elif overrides.get("gait") is None:
            overrides.pop("gait")
        # Playback options can be newer than the checkpoint's structured config.
        with open_dict(play_config.play):
            play_config.play.merge_with(overrides)
    env = make_play_env(play_config)

    def action(observation, command):
        if hasattr(env, "set_command"):
            command = env.set_command(command)
        history = torch.as_tensor(observation.actor, device=device).unsqueeze(0)
        commands = torch.as_tensor(command, device=device).unsqueeze(0)
        with torch.inference_mode():
            selected = policy.action_mean(history, history[:, -policy.frame_size :], commands)
        return selected.cpu().numpy()[0]

    try:
        with InteractiveSession(
            env.model,
            env.data,
            telemetry_dir=Path(str(config.play.telemetry)),
            render=config.play.render == "interactive",
            plot=bool(config.play.plot),
            paused=bool(config.play.paused),
            command_source=str(config.play.command_source),
            fixed_command=config.play.command,
            gamepad_index=int(config.play.gamepad.index),
            gamepad_deadzone=float(config.play.gamepad.deadzone),
            gamepad_scale=config.play.gamepad.scale,
        ) as session:
            session.run(int(config.play.steps), action, env)
    finally:
        if hasattr(env, "close"):
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
