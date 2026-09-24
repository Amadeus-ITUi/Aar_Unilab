#!/usr/bin/env python3
"""Independent PE04 train/play and relocatable ONNX release entrypoint."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from unilab.adapters.pe04_ppo import load_policy, resolve_play_checkpoint, train
from unilab.base.backend.mujoco.batched_robot import compile_robot_scene
from unilab.catalog import catalog
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe04.config import ROOT, load_config
from unilab.envs.locomotion.pe04.play_env import make_play_env
from unilab.playback import InteractiveSession
from unilab.release_contract import create_release


class ExportedPE04Actor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation_history, observation, command):
        return self.policy.action_mean(observation_history, observation, command)


def export_release(checkpoint: Path, run_id: str, device="cpu") -> Path:
    import onnxruntime as ort

    policy = load_policy(checkpoint, device=device)
    cfg, robot = policy.config, catalog.robots["pe04"]
    if tuple(cfg.env.joint_order) != robot.joints:
        raise ValueError("PE04 joint order differs from catalog")
    actor = ExportedPE04Actor(policy).eval()
    rng = np.random.default_rng(42)
    inputs = dict(
        observation_history=rng.normal(0, 0.1, (1, 300)).astype(np.float32),
        observation=rng.normal(0, 0.1, (1, 30)).astype(np.float32),
        command=np.array([[0.2, -0.1, 0.3]], np.float32),
    )
    tensors = tuple(torch.as_tensor(v, device=device) for v in inputs.values())
    onnx = checkpoint.parent / "policy.onnx"
    torch.onnx.export(
        actor,
        tensors,
        onnx,
        input_names=list(inputs),
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    output = session.run(["action"], inputs)[0]
    with torch.no_grad():
        np.testing.assert_allclose(output, actor(*tensors).cpu().numpy(), rtol=1e-5, atol=1e-6)
    runtime_config = checkpoint.parent / "runtime_config.yaml"
    OmegaConf.save(cfg, runtime_config, resolve=True)
    source = repository_path(str(cfg.env.model_path), ROOT)
    model = compile_robot_scene(source, tuple(cfg.env.body_names), visual=False)
    runtime = dict(
        schema="pe04.runtime.tron1.v1",
        default_joint_position=model.key(str(cfg.env.reset_keyframe)).qpos[7:].tolist(),
        kp=list(cfg.control.kp),
        kd=list(cfg.control.kd),
        torque_limits=list(cfg.control.torque_limits),
        user_torque_limit=float(cfg.control.user_torque_limit),
        position_difference=bool(cfg.env.dof_vel_use_pos_diff),
        delay_steps=round(float(cfg.play.delay_ms) * int(cfg.control.physics_hz) / 1000),
        normalization=OmegaConf.to_container(cfg.normalization, resolve=True),
        gait=list(cfg.play.gait),
        joint_target_limits=model.jnt_range[1:].tolist(),
    )
    manifest = {
        "schema": "aar-unilab.actor.v1",
        "robot": {
            "id": "pe04",
            "asset_version": str(cfg.env.asset_version),
            "joint_order": list(cfg.env.joint_order),
        },
        "task": {"id": str(cfg.task_id), "reset_keyframe": str(cfg.env.reset_keyframe)},
        "policy": {
            "observation_builder": str(cfg.observation),
            "inputs": [
                {"name": k, "dtype": "float32", "shape": list(v.shape)} for k, v in inputs.items()
            ],
            "outputs": [{"name": "action", "dtype": "float32", "shape": [1, 6]}],
            "history": {"length": 10, "frame_size": 30, "layout": "frame-major"},
            "hidden_state": [],
        },
        "control": {
            "physics_hz": int(cfg.control.physics_hz),
            "motor_hz": int(cfg.control.motor_hz),
            "policy_hz": int(cfg.control.policy_hz),
            "action_scale": float(cfg.control.action_scale),
            "action_clip": float(cfg.control.action_clip),
            "type": "position_pd",
            "command_delay_steps": runtime["delay_steps"],
        },
        "artifacts": {
            "scene_path": "robot/scene.xml",
            "pe04_runtime_path": "robot/pe04_runtime.json",
        },
    }
    with tempfile.TemporaryDirectory(prefix="pe04-release-") as folder:
        staging = Path(folder)
        for name in robot.runtime_assets:
            path = source.parent / name
            if path.is_dir():
                shutil.copytree(path, staging / name)
            else:
                shutil.copy2(path, staging / name)
        (staging / "pe04_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        return create_release(
            Path("releases/pe04") / str(cfg.task_id) / run_id,
            onnx=onnx,
            runtime_config=runtime_config,
            manifest=manifest,
            robot_root=staging,
            robot_files=sorted(p for p in staging.rglob("*") if p.is_file()),
            golden_inputs=inputs,
            golden_outputs={"action": output},
        )


def main() -> int:
    cfg = load_config(sys.argv[1:])
    catalog.resolve(
        {
            "robot": cfg.robot,
            "task": cfg.task_id,
            "observation": cfg.observation,
            "policy": cfg.policy,
            "algorithm": cfg.algorithm,
            "simulator": cfg.simulator,
        }
    )
    if cfg.mode == "train":
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f_mujoco")
        result = train(Path(str(cfg.training.log_root)) / run_id, config=cfg)
        release = export_release(result.checkpoint, run_id) if cfg.training.export else None
        print(
            f"PE04 checkpoint={result.checkpoint} iterations={result.iterations} release={release}"
        )
        return 0
    if cfg.mode != "play":
        raise ValueError(f"unsupported PE04 mode: {cfg.mode}")
    checkpoint = resolve_play_checkpoint(cfg.checkpoint, log_root=Path(str(cfg.training.log_root)))
    policy = load_policy(checkpoint)
    play_cfg = OmegaConf.merge(
        policy.config, {"play": OmegaConf.to_container(cfg.play, resolve=True)}
    )
    env = make_play_env(play_cfg)

    def action(observation, command):
        commands = torch.from_numpy(env.set_command(command)[None])
        history = torch.from_numpy(observation.actor[None])
        with torch.no_grad():
            return policy.action_mean(history, history[:, -30:], commands).numpy()[0]

    try:
        with InteractiveSession(
            env.model,
            env.data,
            telemetry_dir=Path(str(cfg.play.telemetry)),
            render=cfg.play.render == "interactive",
            plot=cfg.play.plot,
            paused=cfg.play.paused,
            command_source=cfg.play.command_source,
            fixed_command=cfg.play.command,
            gamepad_index=cfg.play.gamepad.index,
            gamepad_deadzone=cfg.play.gamepad.deadzone,
            gamepad_scale=cfg.play.gamepad.scale,
        ) as session:
            session.run(int(cfg.play.steps), action, env)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
