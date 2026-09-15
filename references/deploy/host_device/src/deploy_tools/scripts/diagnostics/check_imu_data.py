#!/usr/bin/env python3
import argparse
import math
import statistics
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


IMU_TO_BASE_SIGNS = (-1.0, -1.0, 1.0)


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Check /IMU_data after the same IMU->base_link sign transform used by lab_inference_node."
    )
    parser.add_argument("--topic", default="/IMU_data")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--expect-gravity-z", type=float, default=-1.0)
    parser.add_argument("--max-static-gyro", type=float, default=0.15, help="rad/s warning threshold while robot is static")
    args = parser.parse_args()

    samples = []
    rclpy.init()
    node = rclpy.create_node("check_imu_data")

    def callback(msg: Imu):
        now = time.monotonic()
        q = msg.orientation
        w = msg.angular_velocity
        g = msg.angular_velocity_covariance
        samples.append(
            {
                "t": now,
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "qw": q.w,
                "wx": IMU_TO_BASE_SIGNS[0] * w.x,
                "wy": IMU_TO_BASE_SIGNS[1] * w.y,
                "wz": IMU_TO_BASE_SIGNS[2] * w.z,
                "gx": IMU_TO_BASE_SIGNS[0] * g[0],
                "gy": IMU_TO_BASE_SIGNS[1] * g[1],
                "gz": IMU_TO_BASE_SIGNS[2] * g[2],
            }
        )

    node.create_subscription(Imu, args.topic, callback, qos_profile_sensor_data)

    print(f"Listening {args.topic} for {args.duration:.1f}s ...")
    print(
        "Applying IMU->base_link signs used by policy: "
        f"[{IMU_TO_BASE_SIGNS[0]:+.0f}, {IMU_TO_BASE_SIGNS[1]:+.0f}, {IMU_TO_BASE_SIGNS[2]:+.0f}]"
    )
    deadline = time.monotonic() + args.duration
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if not samples:
        print("[FAIL] No IMU messages received.")
        return 1

    elapsed = samples[-1]["t"] - samples[0]["t"] if len(samples) > 1 else args.duration
    rate = (len(samples) - 1) / elapsed if elapsed > 1e-6 and len(samples) > 1 else 0.0

    fields = {key: [s[key] for s in samples] for key in samples[0] if key != "t"}
    q_norm = [
        math.sqrt(s["qx"] ** 2 + s["qy"] ** 2 + s["qz"] ** 2 + s["qw"] ** 2)
        for s in samples
    ]
    g_norm = [
        math.sqrt(s["gx"] ** 2 + s["gy"] ** 2 + s["gz"] ** 2)
        for s in samples
    ]
    gyro_abs = [
        math.sqrt(s["wx"] ** 2 + s["wy"] ** 2 + s["wz"] ** 2)
        for s in samples
    ]

    print("\n=== IMU summary ===")
    print(f"samples: {len(samples)}")
    print(f"rate:    {rate:.2f} Hz")
    print(f"quat norm mean/std: {mean(q_norm):+.6f} / {std(q_norm):.6f}")
    print(
        "policy gyro mean rad/s: "
        f"wx={mean(fields['wx']):+.6f}, wy={mean(fields['wy']):+.6f}, wz={mean(fields['wz']):+.6f}"
    )
    print(
        "policy gyro std  rad/s: "
        f"wx={std(fields['wx']):.6f}, wy={std(fields['wy']):.6f}, wz={std(fields['wz']):.6f}"
    )
    print(f"policy gyro |w| mean/max: {mean(gyro_abs):.6f} / {max(gyro_abs):.6f} rad/s")
    print(
        "policy projected_gravity mean: "
        f"gx={mean(fields['gx']):+.6f}, gy={mean(fields['gy']):+.6f}, gz={mean(fields['gz']):+.6f}"
    )
    print(
        "policy projected_gravity std:  "
        f"gx={std(fields['gx']):.6f}, gy={std(fields['gy']):.6f}, gz={std(fields['gz']):.6f}"
    )
    print(f"policy projected_gravity norm mean/std: {mean(g_norm):.6f} / {std(g_norm):.6f}")

    warnings = []
    if rate < 50.0:
        warnings.append(f"rate too low: {rate:.2f} Hz")
    if abs(mean(q_norm) - 1.0) > 0.05:
        warnings.append(f"quaternion norm far from 1: {mean(q_norm):.3f}")
    if abs(mean(g_norm) - 1.0) > 0.10:
        warnings.append(f"projected_gravity norm far from 1: {mean(g_norm):.3f}")
    if abs(mean(fields["gz"]) - args.expect_gravity_z) > 0.30:
        warnings.append(
            f"projected_gravity z unexpected: {mean(fields['gz']):+.3f}, expected about {args.expect_gravity_z:+.1f}"
        )
    if mean(gyro_abs) > args.max_static_gyro:
        warnings.append(f"gyro too large for static robot: mean |w|={mean(gyro_abs):.3f} rad/s")

    if warnings:
        print("\n[WARN]")
        for item in warnings:
            print(f"- {item}")
        return 2

    print("\n[OK] IMU data looks reasonable for static inference input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
