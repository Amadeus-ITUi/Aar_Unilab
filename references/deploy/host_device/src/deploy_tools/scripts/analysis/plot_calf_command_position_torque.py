#!/usr/bin/env python3
"""Plot calf commands, policy-side positions, and measured torque together."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    values = {column: [] for column in columns}
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            parsed = [float(row[column]) for column in columns]
            if not all(math.isfinite(value) for value in parsed):
                continue
            for column, value in zip(columns, parsed):
                values[column].append(value)
    return {column: np.asarray(data) for column, data in values.items()}


def mse(left: np.ndarray, right: np.ndarray) -> float:
    difference = left - right
    return float(np.mean(difference * difference))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    commands = load(args.record_dir / "policy_commands.csv", ("t_rel", "cmd_1", "cmd_4"))
    joints = load(
        args.record_dir / "policy_joint_states.csv",
        ("t_rel", "motor_2_pos", "motor_5_pos", "motor_2_effort", "motor_5_effort"),
    )
    command_time = commands["t_rel"]
    joint_time = joints["t_rel"]
    if command_time.size == 0 or joint_time.size == 0:
        raise SystemExit("No finite command/joint samples found")

    valid = (command_time >= joint_time[0]) & (command_time <= joint_time[-1])
    time_values = command_time[valid]
    command_left = commands["cmd_1"][valid]
    command_right = commands["cmd_4"][valid]
    position_left = np.interp(time_values, joint_time, joints["motor_2_pos"])
    position_right = np.interp(time_values, joint_time, joints["motor_5_pos"])
    torque_left = np.interp(time_values, joint_time, joints["motor_2_effort"])
    torque_right = np.interp(time_values, joint_time, joints["motor_5_effort"])

    command_mse = mse(command_left, command_right)
    position_mse = mse(position_left, position_right)
    torque_mse = mse(torque_left, torque_right)

    output = args.output or args.record_dir / "motor_2_5_command_position_torque_comparison.png"
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    figure.suptitle("Left/Right Calf Comparison: Command, Position, and Torque", fontsize=15)

    axes[0].plot(time_values, command_left, label="cmd_1 -> motor_2 left calf", linewidth=0.8)
    axes[0].plot(time_values, command_right, label="cmd_4 -> motor_5 right calf", linewidth=0.8)
    axes[0].set_ylabel("Command (rad)")
    axes[0].set_title(f"Policy command; left-right MSE = {command_mse:.8f} rad²")

    axes[1].plot(time_values, position_left, label="motor_2 policy position", linewidth=0.8)
    axes[1].plot(time_values, position_right, label="motor_5 policy position", linewidth=0.8)
    axes[1].set_ylabel("Position (rad)")
    axes[1].set_title(f"Measured policy-side position; left-right MSE = {position_mse:.8f} rad²")

    axes[2].plot(time_values, torque_left, label="motor_2 effort", linewidth=0.8)
    axes[2].plot(time_values, torque_right, label="motor_5 effort", linewidth=0.8)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_title(f"Measured torque; left-right MSE = {torque_mse:.8f} Nm²")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)
    print(f"command MSE={command_mse:.9f} rad^2")
    print(f"position MSE={position_mse:.9f} rad^2")
    print(f"torque MSE={torque_mse:.9f} Nm^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
