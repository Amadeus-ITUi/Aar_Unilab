#!/usr/bin/env python3
"""Plot frame-aligned motor_2/motor_5 joint-state comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def finite(value: str) -> float:
    number = float(value)
    return number if math.isfinite(number) else math.nan


def load_csv(path: Path):
    time_values = []
    position_2 = []
    position_5 = []
    velocity_2 = []
    velocity_5 = []
    effort_2 = []
    effort_5 = []

    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            values = [
                finite(row[key])
                for key in (
                    "t_rel",
                    "motor_2_pos",
                    "motor_5_pos",
                    "motor_2_vel",
                    "motor_5_vel",
                    "motor_2_effort",
                    "motor_5_effort",
                )
            ]
            if not all(math.isfinite(value) for value in values):
                continue
            t, p2, p5, v2, v5, e2, e5 = values
            time_values.append(t)
            position_2.append(p2)
            position_5.append(p5)
            velocity_2.append(v2)
            velocity_5.append(v5)
            effort_2.append(e2)
            effort_5.append(e5)

    return (
        time_values,
        position_2,
        position_5,
        velocity_2,
        velocity_5,
        effort_2,
        effort_5,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.csv_path.with_name("motor_2_5_joint_state_comparison.png")
    (
        t,
        p2,
        p5,
        v2,
        v5,
        e2,
        e5,
    ) = load_csv(args.csv_path)
    if not t:
        raise SystemExit("No finite motor_2/motor_5 samples found")

    position_difference = [left - right for left, right in zip(p2, p5)]
    position_squared_error = [difference * difference for difference in position_difference]
    position_mse = sum(position_squared_error) / len(position_squared_error)

    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    figure.suptitle("Motor 2 vs Motor 5: Frame-Aligned Joint-State Comparison", fontsize=15)

    axes[0].plot(t, p2, label="motor_2 left calf", linewidth=0.8)
    axes[0].plot(t, p5, label="motor_5 right calf", linewidth=0.8)
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title("Policy-side position")

    axes[1].plot(t, v2, label="motor_2", linewidth=0.7)
    axes[1].plot(t, v5, label="motor_5", linewidth=0.7)
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_title("Policy-side velocity")

    axes[2].plot(t, e2, label="motor_2", linewidth=0.7)
    axes[2].plot(t, e5, label="motor_5", linewidth=0.7)
    axes[2].set_ylabel("Effort (Nm)")
    axes[2].set_title("Policy-side effort")

    axes[3].plot(t, position_difference, color="tab:red", linewidth=0.8, label="motor_2 - motor_5")
    axes[3].plot(t, position_squared_error, color="tab:purple", linewidth=0.6, alpha=0.65, label="squared position error")
    axes[3].axhline(0.0, color="black", linewidth=0.6)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Difference / error")
    axes[3].set_title(f"Position difference and squared error; MSE = {position_mse:.8f} rad^2")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
