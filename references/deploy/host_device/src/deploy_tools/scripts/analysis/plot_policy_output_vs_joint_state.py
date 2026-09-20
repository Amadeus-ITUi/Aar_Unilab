#!/usr/bin/env python3
"""Plot calf policy outputs against measured policy-side joint positions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_columns(path: Path, columns: tuple[str, ...]) -> dict[str, list[float]]:
    result = {column: [] for column in columns}
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            values = [float(row[column]) for column in columns]
            if not all(math.isfinite(value) for value in values):
                continue
            for column, value in zip(columns, values):
                result[column].append(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    command_path = args.record_dir / "policy_commands.csv"
    joint_path = args.record_dir / "policy_joint_states.csv"
    output = args.output or args.record_dir / "policy_output_vs_joint_state_motor_2_5.png"

    commands = load_columns(command_path, ("t_rel", "cmd_1", "cmd_4"))
    joints = load_columns(joint_path, ("t_rel", "motor_2_pos", "motor_5_pos"))
    command_time = np.asarray(commands["t_rel"])
    if command_time.size == 0 or not joints["t_rel"]:
        raise SystemExit("No finite command/joint samples found")

    joint_time = np.asarray(joints["t_rel"])
    valid = (command_time >= joint_time[0]) & (command_time <= joint_time[-1])
    command_time = command_time[valid]
    command_left = np.asarray(commands["cmd_1"])[valid]
    command_right = np.asarray(commands["cmd_4"])[valid]
    measured_left = np.interp(command_time, joint_time, joints["motor_2_pos"])
    measured_right = np.interp(command_time, joint_time, joints["motor_5_pos"])
    error_left = command_left - measured_left
    error_right = command_right - measured_right

    mse_left = float(np.mean(error_left * error_left))
    mse_right = float(np.mean(error_right * error_right))

    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    figure.suptitle("Policy Output vs Measured Calf Joint State", fontsize=15)

    axes[0].plot(command_time, command_left, label="policy cmd_1 (motor_2)", linewidth=0.8)
    axes[0].plot(command_time, measured_left, label="measured motor_2 position", linewidth=0.8)
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title("Left calf / motor 2")

    axes[1].plot(command_time, command_right, label="policy cmd_4 (motor_5)", linewidth=0.8)
    axes[1].plot(command_time, measured_right, label="measured motor_5 position", linewidth=0.8)
    axes[1].set_ylabel("Position (rad)")
    axes[1].set_title("Right calf / motor 5")

    axes[2].plot(command_time, error_left, label="motor_2 cmd - feedback", linewidth=0.8)
    axes[2].plot(command_time, error_right, label="motor_5 cmd - feedback", linewidth=0.8)
    axes[2].axhline(0.0, color="black", linewidth=0.6)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Tracking error (rad)")
    axes[2].set_title(f"Tracking error; MSE motor_2={mse_left:.6f}, motor_5={mse_right:.6f} rad^2")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)
    print(f"motor_2 tracking MSE={mse_left:.9f} rad^2")
    print(f"motor_5 tracking MSE={mse_right:.9f} rad^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
