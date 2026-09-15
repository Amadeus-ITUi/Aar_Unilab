#!/usr/bin/env python3
"""Runtime PD tuner for foot motors through /set_motor_gains.

This script only calls motors_node's SetMotorGains service. It does not publish
/policy/commands and does not enable, zero, or move motors by itself.
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node

try:
    from motors.srv import SetMotorGains
except ImportError as exc:  # pragma: no cover - depends on sourced ROS workspace
    SetMotorGains = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


DEFAULT_FOOT_IDS = [3, 6]


def parse_motor_ids(text: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid motor id list: {text}") from exc
    if not ids:
        raise argparse.ArgumentTypeError("motor id list cannot be empty")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamically set foot motor PD gains with /set_motor_gains."
    )
    parser.add_argument(
        "--foot-ids",
        type=parse_motor_ids,
        default=DEFAULT_FOOT_IDS,
        help="Comma-separated physical foot motor ids. Default: 3,6.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=0.0,
        help="Kp to set for foot motors. Wheels normally use 0.0.",
    )
    parser.add_argument(
        "--kd",
        "--foot-kd",
        dest="kd",
        type=float,
        default=0.1,
        help="Kd to set for foot motors. Default: 0.1.",
    )
    parser.add_argument(
        "--restore-defaults",
        action="store_true",
        help="Restore kp/kd loaded from motors.yaml at motors_node startup.",
    )
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for /set_motor_gains and its response.",
    )
    return parser.parse_args()


class FootPDTuner(Node):
    def __init__(self) -> None:
        super().__init__("set_foot_pd")
        self.client = self.create_client(SetMotorGains, "/set_motor_gains")

    def call(self, args: argparse.Namespace) -> int:
        if not self.client.wait_for_service(timeout_sec=args.service_timeout):
            self.get_logger().error("/set_motor_gains service is not available. Is motors_node running?")
            return 1

        req = SetMotorGains.Request()
        req.restore_defaults = bool(args.restore_defaults)
        if not req.restore_defaults:
            if args.kp < 0.0 or args.kd < 0.0:
                self.get_logger().error("kp/kd must be non-negative")
                return 1
            req.motor_ids = [int(motor_id) for motor_id in args.foot_ids]
            req.kp = [float(args.kp)] * len(req.motor_ids)
            req.kd = [float(args.kd)] * len(req.motor_ids)

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=args.service_timeout)
        if not future.done():
            self.get_logger().error("/set_motor_gains request timed out")
            return 1

        result = future.result()
        if result is None:
            self.get_logger().error("/set_motor_gains returned no response")
            return 1
        if not result.success:
            self.get_logger().error(result.message)
            return 1

        if req.restore_defaults:
            self.get_logger().info(f"Restored startup motor gains: {result.message}")
        else:
            self.get_logger().info(
                f"Updated foot gains: motor_ids={list(req.motor_ids)} kp={list(req.kp)} kd={list(req.kd)}"
            )
            self.get_logger().info(result.message)
        return 0


def main() -> int:
    args = parse_args()
    if SetMotorGains is None:
        print(
            f"[ERROR] Cannot import motors.srv.SetMotorGains: {IMPORT_ERROR}\n"
            "Please run: source install/setup.bash",
            file=sys.stderr,
        )
        return 1

    rclpy.init()
    node = FootPDTuner()
    try:
        return node.call(args)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
