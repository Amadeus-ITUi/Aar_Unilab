"""Evaluate a saved PE05 checkpoint against basic nominal acceptance and render it.

Does not allocate a training runner or change checkpoint configuration on disk.
Use MUJOCO_GL=egl for --render on headless hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

COMMANDS = {
    "standing": [0, 0, 0],
    "forward": [0.3, 0, 0],
    "backward": [-0.3, 0, 0],
    "left": [0, 0.1, 0],
    "right": [0, -0.1, 0],
    "turn_left": [0, 0, 0.4],
    "turn_right": [0, 0, -0.4],
}
LIMITS = {
    "failure_rate": 0.05,
    "tracking_error_mps": 0.15,
    "yaw_error_radps": 0.2,
    "height_error_m": 0.02,
    "nonfoot_contact": 0.01,
}


def acceptance(metrics):
    groups = {}
    for command in COMMANDS:
        prefix = f"evaluation/{command}/"
        values = {key: metrics.get(prefix + key) for key in LIMITS}
        failures = [
            key
            for key, limit in LIMITS.items()
            if values[key] is None or not np.isfinite(values[key]) or values[key] > limit
        ]
        duration = metrics.get(prefix + "episode_seconds")
        if duration is None or not np.isfinite(duration) or duration < 19:
            failures.append("episode_seconds")
        groups[command] = dict(
            passed=not failures, failed_metrics=failures, episode_seconds=duration, **values
        )
    return dict(passed=all(g["passed"] for g in groups.values()), groups=groups)


def summarize_run(path):
    records = []
    previous_pass = False
    for line in path.read_text().splitlines():
        metric = json.loads(line)
        if "evaluation/standing/failure_rate" not in metric:
            continue
        result = acceptance(metric)
        result.update(
            iteration=metric["iteration"], consecutive_pass=previous_pass and result["passed"]
        )
        previous_pass = result["passed"]
        records.append(result)
    return records


def plot_history(path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    evaluations = [row for row in rows if "evaluation/standing/failure_rate" in row]
    specs = [
        ("episode_seconds", 19, "Survival (s)"),
        ("failure_rate", 0.05, "Failure fraction"),
        ("tracking_error_mps", 0.15, "Planar error (m/s)"),
        ("yaw_error_radps", 0.2, "Yaw error (rad/s)"),
        ("height_error_m", 0.02, "Height error (m)"),
        ("nonfoot_contact", 0.01, "Nonfoot contact fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, threshold, title) in zip(axes.flat, specs):
        for command in COMMANDS:
            ax.plot(
                [row["iteration"] for row in evaluations],
                [row[f"evaluation/{command}/{metric}"] for row in evaluations],
                marker=".",
                label=command,
            )
        ax.axhline(threshold, color="black", linestyle="--", label="acceptance")
        ax.set(title=title, xlabel="Training iteration")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "training_evaluation.png", dpi=150)
    plt.close(fig)
    records = summarize_run(path)
    (output / "evaluation_history.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def render_replays(config, policy, output):
    import mujoco
    import torch
    from omegaconf import OmegaConf

    from unilab.algos.torch.pe05.gait_metrics import GaitMetrics
    from unilab.envs.locomotion.pe05.math import inverse_rotate
    from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv

    output.mkdir(parents=True, exist_ok=True)
    frames_by_command = []
    diagnostics = {}
    for name, command in COMMANDS.items():
        cfg = OmegaConf.merge(config, {"play": {"command": command, "delay_ms": 0}})
        env = PE05VectorEnv(
            cfg,
            evaluation=True,
            num_envs=1,
            visual=True,
            auto_reset=False,
            # Zero configured noise, but identical sole alignment to evaluate_commands.
            evaluation_reset_noise=True,
        )
        renderer = mujoco.Renderer(env.backend.model, height=360, width=480)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 135, -18, 0.9
        data = env.backend.create_visual_data()
        destination = output / f"{name}.mp4"
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
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
                str(destination),
            ],
            stdin=subprocess.PIPE,
        )
        samples, crossing, swing_drag, contact_steps, body_contacts = [], [], [], [], []
        measured_commands = []
        gait_metrics = GaitMetrics(env.dt)
        reward_trace = []
        terms = {}
        state = env.state
        try:
            for tick in range(env.max_episode_steps):
                obs = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in state.obs.items()}
                with torch.inference_mode():
                    action = policy.action_mean(obs["actor"], obs["frame"], obs["command"]).numpy()
                state = env.step(action)
                measured_commands.append([*env.base_velocity[0, :2], env.backend.qvel[0, 5]])
                feet = env.backend.gait_foot_state(np.array(env.cfg["env"]["reference_points"]))
                relative = feet["reference_position"] - env.backend.qpos[:, None, :3]
                body_feet = inverse_rotate(env.backend.qpos[:, None, 3:7], relative)
                crossing.append(bool(body_feet[0, 0, 1] <= body_feet[0, 1, 1]))
                behavior = state.info["behavior"]
                gait_metrics.update(behavior, np.ones(1, dtype=bool))
                reward_trace.append(
                    dict(
                        time_s=(tick + 1) * env.dt,
                        phase=float(env.phase[0]),
                        swing=behavior["swing"][0].tolist(),
                        desired_contact=behavior["desired_contact"][0].tolist(),
                        foot_contact=behavior["foot_contact"][0].tolist(),
                        clearance_m=behavior["foot_clearance"][0].tolist(),
                        target_clearance_m=behavior["target_clearance"][0].tolist(),
                        reference_velocity_mps=behavior["foot_reference_velocity"][0].tolist(),
                        terms={k: float(v[0]) for k, v in behavior["reward_terms"].items()},
                        foot_terms={
                            k: v[0].tolist() for k, v in behavior["foot_reward_terms"].items()
                        },
                    )
                )
                swing = behavior["swing"]
                # Mid-swing near the apex is a more useful drag check than phase boundaries.
                high_target = behavior["target_clearance"] > 0.02
                swing_drag.extend(
                    (behavior["foot_clearance"][swing & high_target] < 0.005).tolist()
                )
                contact_steps.append(bool(state.info["nonfoot_contact"][0]))
                body_contacts.append(behavior["ground_contact"][0].copy())
                for key, value in state.info["reward_terms"].items():
                    terms.setdefault(key, []).append(value)
                if tick % 2 == 0:
                    env.backend.sync_visual_data(data)
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data, camera=camera)
                    frame = renderer.render()
                    picture = Image.fromarray(frame)
                    ImageDraw.Draw(picture).text(
                        (8, 8),
                        f"{name}  {(tick + 1) * env.dt:.2f}s",
                        fill="white",
                        stroke_width=1,
                        stroke_fill="black",
                    )
                    if tick in (0, 98, 248, 498, 748, 998):
                        samples.append(picture.copy())
                    ffmpeg.stdin.write(np.asarray(picture).tobytes())
                if state.terminated[0] or state.truncated[0]:
                    if len(samples) < 6:
                        samples.append(picture.copy())
                    break
            diagnostics[name] = dict(
                seconds=(tick + 1) * env.dt,
                commanded_velocity=command,
                mean_measured_velocity=np.mean(measured_commands, axis=0).tolist(),
                failed=bool(state.terminated[0]),
                crossing_fraction=float(np.mean(crossing)),
                mid_swing_drag_fraction=float(np.mean(swing_drag)) if swing_drag else None,
                nonfoot_contact_fraction=float(np.mean(contact_steps)),
                ground_contact_by_body=dict(
                    zip(env.cfg["env"]["penalized_bodies"], np.mean(body_contacts, axis=0).tolist())
                ),
                last_failure_diagnostics={
                    k: v for k, v in env.adaptation.diagnostics.items() if k.startswith("failure/")
                },
                mean_reward_terms={key: float(np.mean(value)) for key, value in terms.items()},
            )
            diagnostics[name]["gait_metrics"] = gait_metrics.result()
            (output / f"{name}_reward_trace.json").write_text(json.dumps(reward_trace) + "\n")
            frames_by_command.append(samples)
        finally:
            ffmpeg.stdin.close()
            code = ffmpeg.wait(timeout=30)
            renderer.close()
            env.close()
            if code:
                raise RuntimeError(f"ffmpeg failed with {code}")
    columns = max(len(samples) for samples in frames_by_command)
    sheet = Image.new("RGB", (480 * columns, 360 * 7), "white")
    for row, samples in enumerate(frames_by_command):
        for column, picture in enumerate(samples[:6]):
            sheet.paste(picture, (column * 480, row * 360))
    sheet.save(output / "contact_sheet.jpg")
    (output / "replay_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return diagnostics


def evaluate(checkpoint, output, render):
    import torch

    from unilab.algos.torch.pe05.evaluation import evaluate_commands
    from unilab.algos.torch.pe05.policy import PE05EncoderPolicy
    from unilab.envs.locomotion.pe05.contracts import validate_checkpoint

    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = validate_checkpoint(payload)
    cfg.training.mujoco_threads = 1
    cfg.evaluation.commands = COMMANDS
    cfg.evaluation.max_height_error = 0.02
    cfg.evaluation.max_tracking_error = 0.15
    cfg.evaluation.max_yaw_error = 0.2
    cfg.evaluation.max_nonfoot_fraction = 0.01
    cfg.evaluation.delay_ms = 0
    policy = PE05EncoderPolicy(cfg).eval()
    policy.load_state_dict(payload["actor_state_dict"])
    runner = SimpleNamespace(
        config=cfg,
        policy=policy,
        _tensors=lambda obs: {k: torch.as_tensor(v, dtype=torch.float32) for k, v in obs.items()},
    )
    metrics = evaluate_commands(runner)
    report = acceptance(metrics)
    report.update(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        iteration=payload["iteration"],
        metrics=metrics,
    )
    output.mkdir(parents=True, exist_ok=True)
    if render:
        report["replays"] = render_replays(cfg, policy, output / "replays")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("iteration", "passed", "groups")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/reports/pe05_scale_calibration/final")
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.metrics:
        print(json.dumps(plot_history(args.metrics, args.output), indent=2))
    elif args.checkpoint:
        evaluate(args.checkpoint, args.output, args.render)
    else:
        parser.error("provide --checkpoint or --metrics")
