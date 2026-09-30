"""Evaluate a PE03 candidate without resuming training or contacting hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from unilab.adapters.pe03_ppo import load_policy
from unilab.algos.torch.pe03.gait_evaluation import evaluate_gait
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.math import inverse_rotate

COMMANDS = {
    "standing": [0, 0, 0],
    "forward": [0.3, 0, 0],
    "backward": [-0.3, 0, 0],
    "left": [0, 0.1, 0],
    "right": [0, -0.1, 0],
    "turn_left": [0, 0, 0.4],
    "turn_right": [0, 0, -0.4],
}


def action(policy, state):
    obs = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in state.obs.items()}
    with torch.inference_mode():
        return policy.action_mean(obs["actor"], obs["frame"], obs["command"]).numpy()


def nominal_commands(policy):
    cfg = policy.config
    env = PE03GaitEnv(cfg, evaluation=True, num_envs=len(COMMANDS), auto_reset=False)
    commands = np.array(list(COMMANDS.values()), dtype=float)
    try:
        env.set_command(commands, np.tile(cfg.gait.fixed, (len(COMMANDS), 1)))
        env.history[:] = env._frame()[:, None]
        env.state.obs = env._observe()
        live = np.ones(len(COMMANDS), bool)
        failed = np.zeros(len(COMMANDS), bool)
        lengths = np.zeros(len(COMMANDS))
        sums = np.zeros((len(COMMANDS), 5))
        for _ in range(env.max_episode_steps):
            state = env.step(action(policy, env.state))
            b = env.backend
            velocity = inverse_rotate(b.qpos[:, 3:7], b.qvel[:, :3])
            sums[live] += np.column_stack(
                (
                    np.linalg.norm(velocity[:, :2] - commands[:, :2], axis=1),
                    np.abs(b.qvel[:, 5] - commands[:, 2]),
                    state.info["base_height_error"],
                    state.info["nonfoot_contact"],
                    state.info["joint_limit"],
                )
            )[live]
            lengths[live] += 1
            failed |= live & state.terminated
            live &= ~(state.terminated | state.truncated)
            if not live.any():
                break
        return {
            name: dict(
                failed=bool(failed[i]),
                episode_seconds=float(lengths[i] * env.dt),
                **dict(
                    zip(
                        (
                            "planar_error_m_s",
                            "yaw_error_rad_s",
                            "height_error_m",
                            "nonfoot_fraction",
                            "joint_limit_fraction",
                        ),
                        (sums[i] / max(1, lengths[i])).tolist(),
                        strict=True,
                    )
                ),
            )
            for i, name in enumerate(COMMANDS)
        }
    finally:
        env.close()


def render(policy, output):
    import mujoco

    for name in ("standing", "forward", "turn_left"):
        env = PE03GaitEnv(policy.config, evaluation=True, num_envs=1, visual=True, auto_reset=False)
        renderer = mujoco.Renderer(env.backend.model, height=360, width=480)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 135, -18, 0.9
        data = env.backend.create_visual_data()
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "480x360",
                "-r",
                "25",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(output / f"{name}.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
        try:
            env.set_command(np.array(COMMANDS[name]), policy.config.gait.fixed)
            env.history[:] = env._frame()[:, None]
            env.state.obs = env._observe()
            for tick in range(500):
                env.step(action(policy, env.state))
                if tick % 2 == 0:
                    env.backend.sync_visual_data(data)
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data, camera=camera)
                    process.stdin.write(renderer.render().tobytes())
        finally:
            process.stdin.close()
            code = process.wait(timeout=30)
            renderer.close()
            env.close()
        if code:
            raise RuntimeError(f"ffmpeg failed: {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint, device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, report = evaluate_gait(
        policy,
        policy.config,
        torch.device("cpu"),
        randomization_level=float(policy.config.domain_rand.curriculum.max_level),
    )
    result = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        full_range=report,
        nominal_commands=nominal_commands(policy),
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(dict(acceptance=report["acceptance"], nominal=result["nominal_commands"])),
        flush=True,
    )
    if args.render:
        render(policy, args.output)


if __name__ == "__main__":
    main()
