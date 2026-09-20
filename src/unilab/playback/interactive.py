"""Renderer, gamepad, camera, chart and telemetry services for Python Play."""

from __future__ import annotations

import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Callable, Sequence

import numpy as np

from unilab.base.backend.mujoco.play_metrics import (
    base_motion,
    clear_viewer_perturbation,
    frame_base_once,
    motion_overlay,
)


def joystick_command(axes: np.ndarray | None, deadzone: float = 0.12) -> np.ndarray:
    """Map GLFW gamepad axes to body-frame ``[vx, vy, vyaw]``.

    Left stick up/left commands forward/left; right stick left commands
    positive yaw. Right-stick vertical motion and triggers are unused.
    """
    command = np.zeros(3, dtype=np.float32)
    if axes is None or len(axes) < 4:
        return command
    values = np.asarray((-axes[1], -axes[0], -axes[2]), dtype=np.float32)
    if not np.isfinite(values).all():
        return command
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
        command_source: str = "gamepad",
        fixed_command: Sequence[float] = (0.0, 0.0, 0.0),
        gamepad_index: int = 0,
        gamepad_deadzone: float = 0.12,
        gamepad_scale: Sequence[float] = (1.0, 1.0, 1.0),
        paused: bool = False,
    ) -> None:
        if paused and not render:
            raise ValueError("play.paused=true requires play.render=interactive")
        if command_source not in {"gamepad", "fixed"}:
            raise ValueError("play.command_source must be 'gamepad' or 'fixed'")
        if not 0 <= gamepad_index <= 15:
            raise ValueError("play.gamepad.index must be between 0 and 15")
        if not 0 <= gamepad_deadzone < 1:
            raise ValueError("play.gamepad.deadzone must be in [0, 1)")
        self.command_source = command_source
        self.fixed_command = np.array(fixed_command, dtype=np.float32)
        self.gamepad_index = gamepad_index
        self.gamepad_deadzone = gamepad_deadzone
        self.gamepad_scale = np.array(gamepad_scale, dtype=np.float32)
        for name, values in (
            ("play.command", self.fixed_command),
            ("play.gamepad.scale", self.gamepad_scale),
        ):
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite values")
        if (self.gamepad_scale < 0).any():
            raise ValueError("play.gamepad.scale must be nonnegative")
        self.model = model
        self.data = data
        self.render = render
        self.plot = plot
        self.follow_body = follow_body
        self.paused = paused
        # The viewer calls back on its UI thread. Only the simulation loop may
        # reset the environment or change playback state.
        self._events: SimpleQueue[str] = SimpleQueue()
        self._gamepad_pressed: set[str] = set()
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

            self._viewer_context = mujoco.viewer.launch_passive(
                model, data, key_callback=self.key_callback
            )
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
        if self._chart is not None:
            plt, figure, _, _ = self._chart
            plt.close(figure)
        return self._viewer_context.__exit__(*error)

    def key_callback(self, keycode: int) -> None:
        """Queue native viewer keys without accessing simulation data."""
        event = {
            32: "pause",  # Space
            ord("P"): "pause",
            ord("R"): "reset",
            259: "reset",  # GLFW Backspace
            ord("N"): "step",
            ord("C"): "camera",
            ord("Q"): "quit",
        }.get(keycode)
        if event is not None:
            self._events.put(event)

    def gamepad(self) -> np.ndarray:
        if not self.render:
            return joystick_command(None)
        try:
            import glfw

            # Raw joystick axes have device-specific ordering (and this binding
            # returns a pointer/count pair). GLFW's gamepad mapping normalizes it.
            state = glfw.get_gamepad_state(glfw.JOYSTICK_1 + self.gamepad_index)
            axes = None if state is None else np.asarray(state.axes, dtype=np.float32)
            buttons = getattr(state, "buttons", ())
            pressed = set()
            if len(buttons) > glfw.GAMEPAD_BUTTON_START:
                if buttons[glfw.GAMEPAD_BUTTON_START]:
                    pressed.add("pause")
                if buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER] and buttons[glfw.GAMEPAD_BUTTON_Y]:
                    pressed.add("reset")
            for event in sorted(pressed - self._gamepad_pressed):
                self._events.put(event)
            self._gamepad_pressed = pressed
        except (ImportError, TypeError):
            axes = None
            self._gamepad_pressed.clear()
        return joystick_command(axes, self.gamepad_deadzone) * self.gamepad_scale

    def tick(self, command: np.ndarray, *, record: bool = True) -> bool:
        row = {
            "time_seconds": float(self.data.time),
            **base_motion(self.model, self.data, self.follow_body),
            "command_x": float(command[0]),
            "command_y": float(command[1]),
            "command_yaw": float(command[2]),
        }
        if record:
            self._csv.writerow(row)
            self._jsonl_stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.set_texts(
                motion_overlay(row, paused=self.paused, command_source=self.command_source)
            )
            self.viewer.sync()
        if self._chart is not None:
            plt, figure, axis, line = self._chart
            if record:
                self._samples.append((row["time_seconds"], row["base_height"]))
            if self._samples:
                line.set_data(*zip(*self._samples[-500:]))
            else:
                line.set_data([], [])
            axis.relim()
            axis.autoscale_view()
            figure.canvas.draw_idle()
            plt.pause(0.001)
        return True

    def run(self, steps: int, action: Callable[[object, np.ndarray], np.ndarray], env) -> None:
        """Run policy steps; -1 continues until viewer closure or interruption."""
        if steps < -1:
            raise ValueError("play.steps must be nonnegative or -1 for unlimited playback")
        if self.command_source == "gamepad":
            print(
                "Play command: gamepad; left stick up/left = +vx/+vy, "
                "right stick left = +vyaw; centered/disconnected = zero. "
                f"Scale [m/s, m/s, rad/s]: {self.gamepad_scale.tolist()}"
            )
        else:
            print(f"Play command: fixed {self.fixed_command.tolist()}")
        print(
            "Play controls: Space/P pause/resume; R/Backspace reset; "
            "N single step while paused; C recenter camera; Q quit. "
            "In gamepad mode: Start pause/resume; RB+Y reset."
        )

        def reset():
            command = (
                self.fixed_command.copy()
                if self.command_source == "fixed"
                else np.zeros(3, dtype=np.float32)
            )
            if hasattr(env, "set_command"):
                env.set_command(command)
            observation = env.reset()
            if self.viewer is not None:
                clear_viewer_perturbation(self.viewer)
            self._samples.clear()
            return observation, command

        observation, command = reset()
        if self.viewer is not None:
            frame_base_once(self.viewer, self.data, self.follow_body)
        # env.step advances one policy period, including all physics substeps.
        policy_hz = getattr(env, "policy_hz", None)
        timestep = (
            1.0 / float(policy_hz) if policy_hz is not None else float(env.model.opt.timestep)
        )
        completed_steps = 0
        single_steps = 0
        while steps == -1 or completed_steps < steps:
            if self.viewer is not None and not self.viewer.is_running():
                break
            started = time.monotonic()
            command = (
                self.gamepad() if self.command_source == "gamepad" else self.fixed_command.copy()
            )
            reset_requested = False
            while True:
                try:
                    event = self._events.get_nowait()
                except Empty:
                    break
                if event == "quit":
                    return
                if event == "pause":
                    self.paused = not self.paused
                    single_steps = 0
                elif event == "reset":
                    reset_requested = True
                    single_steps = 0
                elif event == "step" and self.paused:
                    single_steps += 1
                elif event == "camera" and self.viewer is not None:
                    frame_base_once(self.viewer, self.data, self.follow_body)
            advanced = False
            done = False
            if reset_requested:
                observation, command = reset()
            elif not self.paused or single_steps:
                selected_action = action(observation, command)
                observation, _, done, _ = env.step(selected_action)
                completed_steps += 1
                single_steps = max(0, single_steps - 1)
                advanced = True
            if not self.tick(command, record=advanced):
                break
            if done:
                observation, command = reset()
            remaining = timestep - (time.monotonic() - started)
            if (self.render or self.paused) and remaining > 0:
                time.sleep(remaining)
