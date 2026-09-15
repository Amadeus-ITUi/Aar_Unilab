#!/usr/bin/env python3
"""Summarize standing deploy observations from record_deploy_runtime output.

This script is read-only. It reconstructs the single-frame actor observation
terms from recorded /IMU_data, /policy/joint_states, /cmd_vel and
/policy/commands CSV files.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path


MOTOR_NAMES = [f"motor_{i}" for i in range(1, 7)]
LEG_INDICES = [0, 1, 3, 4]
TRAINING_JOINT_VEL_ORDER = [0, 1, 3, 4, 2, 5]


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="") as fp:
        return list(csv.DictReader(fp))


def f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def crop_last_seconds(rows, seconds):
    if not rows or seconds <= 0.0:
        return rows
    last_t = f(rows[-1], "t_rel")
    if not math.isfinite(last_t):
        return rows
    start = last_t - seconds
    return [row for row in rows if f(row, "t_rel", -1e9) >= start]


def mean_std(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def column_mean_std(rows, key):
    return mean_std([f(row, key) for row in rows])


def fmt(values, digits=6):
    return "[" + ", ".join(f"{v:+.{digits}f}" if math.isfinite(v) else "nan" for v in values) + "]"


def latest_or_default(rows, default=None):
    return rows[-1] if rows else default


def command_from_cmd_vel(cmd_rows, default_height):
    row = latest_or_default(cmd_rows)
    if row is None:
        return [0.0, 0.0, default_height]
    height = f(row, "lin_z", default_height)
    if not math.isfinite(height) or height <= 1e-3:
        height = default_height
    return [f(row, "lin_x", 0.0), f(row, "ang_z", 0.0), height]


def last_action_from_policy_commands(command_rows):
    row = latest_or_default(command_rows)
    if row is None:
        return [0.0] * 6, [0.0] * 6
    motor_cmd = [f(row, f"cmd_{i}", 0.0) for i in range(6)]
    raw = []
    for i, value in enumerate(motor_cmd):
        raw.append(value / 10.0 if i in (2, 5) else value / 0.5)
    return motor_cmd, raw


def main():
    parser = argparse.ArgumentParser(
        description="Print standing actor observation terms from a deploy_record_* directory."
    )
    parser.add_argument("record_dir", help="logs/deploy_record_YYYYmmdd_HHMMSS directory")
    parser.add_argument("--last-sec", type=float, default=5.0, help="average over last N seconds")
    parser.add_argument("--height", type=float, default=0.24, help="default height command")
    args = parser.parse_args()

    record_dir = Path(args.record_dir).expanduser().resolve()
    imu_rows = crop_last_seconds(read_csv(record_dir / "imu_data.csv"), args.last_sec)
    joint_rows = crop_last_seconds(read_csv(record_dir / "policy_joint_states.csv"), args.last_sec)
    cmd_vel_rows = crop_last_seconds(read_csv(record_dir / "cmd_vel.csv"), args.last_sec)
    policy_cmd_rows = crop_last_seconds(read_csv(record_dir / "policy_commands.csv"), args.last_sec)

    if not imu_rows:
        raise SystemExit(f"[ERROR] no imu_data.csv samples in {record_dir}")
    if not joint_rows:
        raise SystemExit(f"[ERROR] no policy_joint_states.csv samples in {record_dir}")

    gyro = [column_mean_std(imu_rows, key)[0] for key in ("wx", "wy", "wz")]
    gyro_std = [column_mean_std(imu_rows, key)[1] for key in ("wx", "wy", "wz")]
    gravity = [column_mean_std(imu_rows, key)[0] for key in ("gravity_x", "gravity_y", "gravity_z")]
    gravity_std = [column_mean_std(imu_rows, key)[1] for key in ("gravity_x", "gravity_y", "gravity_z")]

    pos = [column_mean_std(joint_rows, f"{name}_pos")[0] for name in MOTOR_NAMES]
    pos_std = [column_mean_std(joint_rows, f"{name}_pos")[1] for name in MOTOR_NAMES]
    vel = [column_mean_std(joint_rows, f"{name}_vel")[0] for name in MOTOR_NAMES]
    vel_std = [column_mean_std(joint_rows, f"{name}_vel")[1] for name in MOTOR_NAMES]

    joint_pos_no_wheel = [pos[i] for i in LEG_INDICES]
    joint_vel_training_raw = [vel[i] for i in TRAINING_JOINT_VEL_ORDER]
    joint_vel_training_scaled = [v * 0.1 for v in joint_vel_training_raw]
    motor_cmd, inferred_last_action = last_action_from_policy_commands(policy_cmd_rows)
    command = command_from_cmd_vel(cmd_vel_rows, args.height)

    print("=== Standing Observation Summary ===")
    print(f"record_dir: {record_dir}")
    print(f"window: last {args.last_sec:.2f}s")
    print(f"samples: imu={len(imu_rows)} joint={len(joint_rows)} cmd_vel={len(cmd_vel_rows)} policy_cmd={len(policy_cmd_rows)}")
    print()
    print("actor single-frame terms:")
    print(f"  base_ang_vel / gyro [wx, wy, wz]:             {fmt(gyro)}")
    print(f"  gyro std:                                    {fmt(gyro_std)}")
    print(f"  projected_gravity [gx, gy, gz]:              {fmt(gravity)}")
    print(f"  projected_gravity std:                       {fmt(gravity_std)}")
    print(f"  joint_pos_no_wheel [0,1,3,4]:                {fmt(joint_pos_no_wheel)}")
    print(f"  joint_vel motor order [0,1,2,3,4,5]:         {fmt(vel)}")
    print(f"  joint_vel motor std:                         {fmt(vel_std)}")
    print(f"  joint_vel actor raw [0,1,3,4,2,5]:           {fmt(joint_vel_training_raw)}")
    print(f"  joint_vel actor scaled (*0.1):               {fmt(joint_vel_training_scaled)}")
    print(f"  last_action inferred from /policy/commands:  {fmt(inferred_last_action)}")
    print(f"  published /policy/commands:                  {fmt(motor_cmd)}")
    print(f"  command [lin_x, yaw, height]:                {fmt(command)}")
    print()
    print("full motor relative joint_pos [1..6]:")
    print(f"  mean: {fmt(pos)}")
    print(f"  std:  {fmt(pos_std)}")

    g_norm = math.sqrt(sum(v * v for v in gravity if math.isfinite(v)))
    print()
    print(f"gravity norm: {g_norm:.6f}")
    if abs(gravity[0]) > 0.1 or abs(gravity[1]) > 0.1 or gravity[2] > -0.95:
        print("[WARN] projected_gravity is not close to [0, 0, -1].")


if __name__ == "__main__":
    main()
