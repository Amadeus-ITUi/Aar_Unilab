#!/usr/bin/env python3
import argparse
import csv
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


DEFAULT_SIGNS = (-1.0, -1.0, 1.0)


def deployment_root():
    configured_root = os.environ.get("WALKING_EAGLE_DEPLOY_ROOT")
    if configured_root:
        return os.path.abspath(os.path.expanduser(configured_root))

    source_path = os.path.abspath(__file__)
    candidate = os.path.dirname(source_path)
    while candidate != os.path.dirname(candidate):
        if os.path.isfile(os.path.join(candidate, "start_robot.sh")) and os.path.isdir(
            os.path.join(candidate, "src")
        ):
            return candidate
        candidate = os.path.dirname(candidate)
    return os.getcwd()


def norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def default_output_path():
    root = deployment_root()
    out_dir = os.path.join(root, "logs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"policy_imu_sequence_{stamp}.csv")


def parse_signs(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("signs must have 3 comma-separated values, e.g. -1,-1,1")
    signs = []
    for part in parts:
        value = float(part)
        signs.append(1.0 if value >= 0.0 else -1.0)
    return tuple(signs)


def main():
    parser = argparse.ArgumentParser(
        description="Record /IMU_data with the same IMU->base_link sign transform used by policy."
    )
    parser.add_argument("--topic", default="/IMU_data")
    parser.add_argument("--output", default=default_output_path())
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means record until Ctrl+C")
    parser.add_argument("--signs", type=parse_signs, default=DEFAULT_SIGNS, help="IMU->base signs, default -1,-1,1")
    parser.add_argument("--no-wait", action="store_true", help="start recording immediately")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("=== policy IMU sequence recorder ===")
    print(f"topic:  {args.topic}")
    print(f"output: {args.output}")
    print(f"signs:  [{args.signs[0]:+.0f}, {args.signs[1]:+.0f}, {args.signs[2]:+.0f}]")
    if args.duration > 0.0:
        print(f"duration: {args.duration:.2f}s")
    else:
        print("duration: until Ctrl+C")

    if not args.no_wait:
        input("Press Enter to start recording, then move from lean-back to lean-forward...")

    samples = 0
    first_t = None
    last_t = None

    rclpy.init()
    node = rclpy.create_node("record_policy_imu_sequence")

    fields = [
        "sample",
        "t_rel",
        "t_wall",
        "stamp_sec",
        "stamp_nanosec",
        "qx",
        "qy",
        "qz",
        "qw",
        "q_norm",
        "acc_x",
        "acc_y",
        "acc_z",
        "raw_wx",
        "raw_wy",
        "raw_wz",
        "policy_wx",
        "policy_wy",
        "policy_wz",
        "raw_gx",
        "raw_gy",
        "raw_gz",
        "policy_gx",
        "policy_gy",
        "policy_gz",
        "policy_g_norm",
    ]

    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    def callback(msg: Imu):
        nonlocal samples, first_t, last_t
        now = time.monotonic()
        if first_t is None:
            first_t = now
        last_t = now
        t_rel = now - first_t

        q = msg.orientation
        a = msg.linear_acceleration
        w = msg.angular_velocity
        g = msg.angular_velocity_covariance

        policy_wx = args.signs[0] * w.x
        policy_wy = args.signs[1] * w.y
        policy_wz = args.signs[2] * w.z
        policy_gx = args.signs[0] * g[0]
        policy_gy = args.signs[1] * g[1]
        policy_gz = args.signs[2] * g[2]

        writer.writerow(
            {
                "sample": samples,
                "t_rel": f"{t_rel:.9f}",
                "t_wall": f"{time.time():.9f}",
                "stamp_sec": msg.header.stamp.sec,
                "stamp_nanosec": msg.header.stamp.nanosec,
                "qx": f"{q.x:.9f}",
                "qy": f"{q.y:.9f}",
                "qz": f"{q.z:.9f}",
                "qw": f"{q.w:.9f}",
                "q_norm": f"{math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w):.9f}",
                "acc_x": f"{a.x:.9f}",
                "acc_y": f"{a.y:.9f}",
                "acc_z": f"{a.z:.9f}",
                "raw_wx": f"{w.x:.9f}",
                "raw_wy": f"{w.y:.9f}",
                "raw_wz": f"{w.z:.9f}",
                "policy_wx": f"{policy_wx:.9f}",
                "policy_wy": f"{policy_wy:.9f}",
                "policy_wz": f"{policy_wz:.9f}",
                "raw_gx": f"{g[0]:.9f}",
                "raw_gy": f"{g[1]:.9f}",
                "raw_gz": f"{g[2]:.9f}",
                "policy_gx": f"{policy_gx:.9f}",
                "policy_gy": f"{policy_gy:.9f}",
                "policy_gz": f"{policy_gz:.9f}",
                "policy_g_norm": f"{norm3(policy_gx, policy_gy, policy_gz):.9f}",
            }
        )
        samples += 1

        if samples % 200 == 0:
            print(
                f"t={t_rel:.2f}s samples={samples} "
                f"policy_gyro=[{policy_wx:+.3f}, {policy_wy:+.3f}, {policy_wz:+.3f}] "
                f"policy_gravity=[{policy_gx:+.3f}, {policy_gy:+.3f}, {policy_gz:+.3f}]"
            )

    node.create_subscription(Imu, args.topic, callback, qos_profile_sensor_data)

    deadline = None if args.duration <= 0.0 else time.monotonic() + args.duration
    print("Recording...")
    try:
        while rclpy.ok():
            if deadline is not None and time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\nInterrupted, saving CSV...")
    finally:
        csv_file.flush()
        csv_file.close()
        node.destroy_node()
        rclpy.shutdown()

    elapsed = (last_t - first_t) if first_t is not None and last_t is not None else 0.0
    rate = (samples - 1) / elapsed if samples > 1 and elapsed > 1e-9 else 0.0
    print("=== done ===")
    print(f"samples: {samples}")
    print(f"rate:    {rate:.2f} Hz")
    print(f"csv:     {args.output}")
    return 0 if samples > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
