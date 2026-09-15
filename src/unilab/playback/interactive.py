"""Renderer, gamepad, camera, chart and telemetry services for Python Play."""

from __future__ import annotations

import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np


def joystick_command(axes: np.ndarray | None, deadzone: float = 0.12) -> np.ndarray:
    command = np.zeros(3, dtype=np.float32)
    if axes is None or len(axes) < 4:
        return command
    values = np.asarray((axes[1], axes[0], axes[3]), dtype=np.float32)
    values[np.abs(values) < deadzone] = 0.0
    command[:] = np.clip(values, -1.0, 1.0)
    command[0] *= -1.0
    command[2] *= -1.0
    return command


class InteractiveSession:
    """Optional services around a MuJoCo step loop.

    MuJoCo's native passive viewer owns camera rotate/pan/zoom and Ctrl-drag
    perturbation. This class adds follow mode, GLFW gamepad polling, live
    Matplotlib plots and matching CSV/JSONL diagnostic streams.
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
            fieldnames=("time_seconds", "base_height", "command_x", "command_y", "command_yaw"),
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
        try:
            import glfw

            axes = glfw.get_joystick_axes(glfw.JOYSTICK_1)
        except (ImportError, TypeError):
            axes = None
        return joystick_command(axes)

    def tick(self, command: np.ndarray) -> bool:
        row = {
            "time_seconds": float(self.data.time),
            "base_height": float(self.data.qpos[2]),
            "command_x": float(command[0]),
            "command_y": float(command[1]),
            "command_yaw": float(command[2]),
        }
        self._csv.writerow(row)
        self._jsonl_stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.cam.lookat[:] = self.data.xpos[self.follow_body]
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
        observation = env.reset()
        timestep = float(env.model.opt.timestep)
        for _ in range(steps):
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
