"""Renderer, gamepad, camera, chart and telemetry services for Python Play."""

from __future__ import annotations

import csv
import json
import time
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from typing import Callable

import numpy as np

from unilab.base.backend.mujoco.play_metrics import base_motion, frame_base_once, motion_overlay


def joystick_command(axes: np.ndarray | None, deadzone: float = 0.12) -> np.ndarray:
    """Map standardized GLFW Xbox axes to WE11 ``[vx, yaw, height_rate]``."""
    command = np.zeros(3, dtype=np.float32)
    if axes is None or len(axes) < 4:
        return command
    values = np.asarray((-axes[1], -axes[2], axes[3]), dtype=np.float32)
    values[np.abs(values) < deadzone] = 0.0
    command[:] = np.clip(values, -1.0, 1.0)
    return command


class InteractiveSession:
    """Optional services around a MuJoCo step loop.

    MuJoCo's native passive viewer owns camera rotate/pan/zoom and Ctrl-drag
    perturbation. This class adds a velocity/height HUD, GLFW gamepad polling,
    live Matplotlib plots and matching CSV/JSONL diagnostic streams. The camera
    is framed once on startup; Free and Tracking remain native viewer choices.
    """

    def __init__(
        self,
        model,
        data,
        *,
        telemetry_dir: Path,
        render: bool,
        plot: bool,
        follow_body: int = 1,
    ) -> None:
        self.model = model
        self.data = data
        self.render = render
        self.plot = plot
        self.follow_body = follow_body
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._csv_stream = (telemetry_dir / "telemetry.csv").open("w", newline="", encoding="utf-8")
        self._jsonl_stream = (telemetry_dir / "telemetry.jsonl").open("w", encoding="utf-8")
        self._csv = csv.DictWriter(
            self._csv_stream,
            fieldnames=(
                "time_seconds",
                "base_height",
                "command_x",
                "command_y",
                "command_yaw",
                "base_velocity_x",
                "base_velocity_y",
                "base_yaw_rate",
            ),
        )
        self._csv.writeheader()
        self._viewer_context = nullcontext(None)
        if render:
            import mujoco.viewer

            self._viewer_context = mujoco.viewer.launch_passive(model, data)
        self.viewer = None
        self._chart = None
        self._samples: list[tuple[float, float]] = []

    def __enter__(self):
        self.viewer = self._viewer_context.__enter__()
        if self.plot:
            import matplotlib.pyplot as plt

            plt.ion()
            figure, axis = plt.subplots()
            (line,) = axis.plot([], [])
            axis.set(xlabel="time [s]", ylabel="base height [m]", title="Play telemetry")
            self._chart = (plt, figure, axis, line)
        return self

    def __exit__(self, *error):
        self._csv_stream.close()
        self._jsonl_stream.close()
        return self._viewer_context.__exit__(*error)

    def gamepad(self) -> np.ndarray:
        if not self.render:
            return joystick_command(None)
        try:
            import glfw

            axes = glfw.get_joystick_axes(glfw.JOYSTICK_1)
        except (ImportError, TypeError):
            axes = None
        return joystick_command(axes)

    def tick(self, command: np.ndarray) -> bool:
        row = {
            "time_seconds": float(self.data.time),
            **base_motion(self.model, self.data, self.follow_body),
            "command_x": float(command[0]),
            "command_y": float(command[1]),
            "command_yaw": float(command[2]),
        }
        self._csv.writerow(row)
        self._jsonl_stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.set_texts(motion_overlay(row))
            self.viewer.sync()
        if self._chart is not None:
            plt, figure, axis, line = self._chart
            self._samples.append((row["time_seconds"], row["base_height"]))
            line.set_data(*zip(*self._samples[-500:]))
            axis.relim()
            axis.autoscale_view()
            figure.canvas.draw_idle()
            plt.pause(0.001)
        return True

    def run(self, steps: int, action: Callable[[object, np.ndarray], np.ndarray], env) -> None:
        """Run policy steps; -1 continues until viewer closure or interruption."""
        if steps < -1:
            raise ValueError("play.steps must be nonnegative or -1 for unlimited playback")
        observation = env.reset()
        if self.viewer is not None:
            frame_base_once(self.viewer, self.data, self.follow_body)
        # env.step advances one policy period, including all physics substeps.
        policy_hz = getattr(env, "policy_hz", None)
        timestep = (
            1.0 / float(policy_hz) if policy_hz is not None else float(env.model.opt.timestep)
        )
        iterations = count() if steps == -1 else range(steps)
        for _ in iterations:
            started = time.monotonic()
            command = self.gamepad()
            selected_action = action(observation, command)
            observation, _, done, _ = env.step(selected_action)
            if not self.tick(command):
                break
            if done:
                observation = env.reset()
            remaining = timestep - (time.monotonic() - started)
            if self.render and remaining > 0:
                time.sleep(remaining)
