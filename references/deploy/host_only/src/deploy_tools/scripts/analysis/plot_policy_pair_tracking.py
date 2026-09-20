#!/usr/bin/env python3
"""Plot policy output and measured policy-side position for a motor pair."""

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
    parser.add_argument("--left-motor", type=int, required=True)
    parser.add_argument("--right-motor", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    left = args.left_motor
    right = args.right_motor
    command_columns = ("t_rel", f"cmd_{left - 1}", f"cmd_{right - 1}")
    joint_columns = ("t_rel", f"motor_{left}_pos", f"motor_{right}_pos")
    commands = load_columns(args.record_dir / "policy_commands.csv", command_columns)
    joints = load_columns(args.record_dir / "policy_joint_states.csv", joint_columns)
    command_time = np.asarray(commands["t_rel"])
    joint_time = np.asarray(joints["t_rel"])
    if command_time.size == 0 or joint_time.size == 0:
        raise SystemExit("No finite command/joint samples found")

    valid = (command_time >= joint_time[0]) & (command_time <= joint_time[-1])
    command_time = command_time[valid]
    command_left = np.asarray(commands[command_columns[1]])[valid]
    command_right = np.asarray(commands[command_columns[2]])[valid]
    measured_left = np.interp(joint_time, joint_time, joints[joint_columns[1]])
    measured_right = np.interp(joint_time, joint_time, joints[joint_columns[2]])
    measured_left = np.interp(command_time, joint_time, measured_left)
    measured_right = np.interp(command_time, joint_time, measured_right)
    error_left = command_left - measured_left
    error_right = command_right - measured_right
    output_difference = command_left - command_right
    feedback_difference = measured_left - measured_right

    output_mse = float(np.mean(output_difference * output_difference))
    feedback_mse = float(np.mean(feedback_difference * feedback_difference))
    tracking_left_mse = float(np.mean(error_left * error_left))
    tracking_right_mse = float(np.mean(error_right * error_right))

    output = args.output or args.record_dir / f"policy_output_vs_joint_state_motor_{left}_{right}.png"
    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    figure.suptitle(f"Motor {left} vs Motor {right}: Policy Output and Feedback", fontsize=15)

    axes[0].plot(command_time, command_left, label=f"policy cmd_{left - 1} / motor_{left}", linewidth=0.8)
    axes[0].plot(command_time, measured_left, label=f"feedback motor_{left}", linewidth=0.8)
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title(f"Left motor {left}")

    axes[1].plot(command_time, command_right, label=f"policy cmd_{right - 1} / motor_{right}", linewidth=0.8)
    axes[1].plot(command_time, measured_right, label=f"feedback motor_{right}", linewidth=0.8)
    axes[1].set_ylabel("Position (rad)")
    axes[1].set_title(f"Right motor {right}")

    axes[2].plot(command_time, error_left, label=f"motor_{left} cmd - feedback", linewidth=0.8)
    axes[2].plot(command_time, error_right, label=f"motor_{right} cmd - feedback", linewidth=0.8)
    axes[2].axhline(0.0, color="black", linewidth=0.6)
    axes[2].set_ylabel("Tracking error (rad)")
    axes[2].set_title(
        f"Tracking MSE: motor_{left}={tracking_left_mse:.6f}, "
        f"motor_{right}={tracking_right_mse:.6f} rad^2"
    )

    axes[3].plot(command_time, output_difference, label="policy output difference", linewidth=0.8)
    axes[3].plot(command_time, feedback_difference, label="feedback difference", linewidth=0.8)
    axes[3].axhline(0.0, color="black", linewidth=0.6)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Left - right (rad)")
    axes[3].set_title(f"Output MSE={output_mse:.6f} rad^2; feedback MSE={feedback_mse:.6f} rad^2")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)
    print(f"policy output MSE={output_mse:.9f} rad^2")
    print(f"feedback MSE={feedback_mse:.9f} rad^2")
    print(f"motor_{left} tracking MSE={tracking_left_mse:.9f} rad^2")
    print(f"motor_{right} tracking MSE={tracking_right_mse:.9f} rad^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
