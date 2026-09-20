#!/usr/bin/env python3
"""LEGACY SocketCAN-era two-motor publisher.

No ESD-Link bridge subscribes to /policy/commands in maintenance mode. Use
``ros2 run deploy_tools esd_selected_motor_sweep --motor-ids ID1,ID2`` instead.

Publishes a full 6-element /policy/commands vector. Selected motors receive a
sine command; all other entries stay at zero.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


MOTOR_COUNT = 6


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip(), 0) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep two motors together through /policy/commands.")
    parser.add_argument("--motor-ids", default="2,5", help="Physical motor ids, comma-separated. Default: 2,5")
    parser.add_argument("--frequency", type=float, default=1.0, help="Fixed sine frequency in Hz.")
    parser.add_argument("--amplitude", type=float, default=0.2, help="Sine amplitude in command units/rad.")
    parser.add_argument("--duration", type=float, default=10.0, help="Formal sweep duration in seconds.")
    parser.add_argument("--publish-rate", type=float, default=200.0)
    parser.add_argument("--record-rate", type=float, default=200.0)
    parser.add_argument("--pre-hold-sec", type=float, default=5.0, help="Publish zero command before sweep.")
    parser.add_argument("--wait-for-enter-after-hold", action="store_true", help="Hold zero until Enter before sweep.")
    parser.add_argument(
        "--signs",
        default="auto",
        help="Comma-separated signs for selected motors, or auto. Auto: motors 1-3 negative, 4-6 positive.",
    )
    parser.add_argument("--output-file", default="", help="CSV output path. Empty means auto-generate under plots/.")
    return parser.parse_args()


def default_output_file(motor_ids: list[int]) -> Path:
    root = os.environ.get("DEPLOY_CPP_ROOT")
    plots_dir = Path(root) / "plots" if root else Path.cwd() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1.0) * 1000)
    ids = "_".join(str(motor_id) for motor_id in motor_ids)
    return plots_dir / f"sweep_motors_{ids}_fixed1hz_{stamp}_{ms:03d}.csv"


class DualMotorSweepNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sweep_motors_2_5_fixed_1hz")

        if args.frequency <= 0.0 or args.amplitude <= 0.0 or args.duration <= 0.0:
            raise ValueError("--frequency, --amplitude and --duration must be positive")
        if args.publish_rate <= 0.0 or args.record_rate <= 0.0:
            raise ValueError("--publish-rate and --record-rate must be positive")
        if args.publish_rate > 250.0 or args.record_rate > 250.0:
            raise ValueError("--publish-rate and --record-rate must not exceed the 250 Hz ESD-Link limit")
        if args.pre_hold_sec < 0.0:
            raise ValueError("--pre-hold-sec must be non-negative")

        self.args = args
        self.motor_ids = parse_csv_ints(args.motor_ids)
        if len(self.motor_ids) != 2:
            raise ValueError("--motor-ids must contain exactly two motor ids")
        for motor_id in self.motor_ids:
            if motor_id < 1 or motor_id > MOTOR_COUNT:
                raise ValueError(f"motor id must be in [1, {MOTOR_COUNT}], got {motor_id}")

        if args.signs.strip().lower() == "auto":
            self.signs = [-1.0 if motor_id <= 3 else 1.0 for motor_id in self.motor_ids]
        else:
            self.signs = parse_csv_floats(args.signs)
            if len(self.signs) != len(self.motor_ids):
                raise ValueError("--signs length must match --motor-ids")
            self.signs = [1.0 if sign >= 0.0 else -1.0 for sign in self.signs]

        self.output_file = Path(args.output_file) if args.output_file else default_output_file(self.motor_ids)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.commands: List[float] = [0.0] * MOTOR_COUNT
        self.positions: List[float] = [0.0] * MOTOR_COUNT
        self.velocities: List[float] = [0.0] * MOTOR_COUNT
        self.torques: List[float] = [0.0] * MOTOR_COUNT
        self.records: list[list[float]] = []
        self.phase = "pre_hold" if (args.pre_hold_sec > 0.0 or args.wait_for_enter_after_hold) else "sweep"
        self.start_time = self.get_clock().now()
        self.phase_start_time = self.start_time
        self.sweep_started = self.phase == "sweep"
        self.finished = False
        self.enter_event = threading.Event()
        self.enter_thread: threading.Thread | None = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState, "/policy/joint_states", self.joint_state_callback, sensor_qos)
        self.pub = self.create_publisher(Float32MultiArray, "/policy/commands", control_qos)
        self.create_timer(1.0 / args.publish_rate, self.publish_command)
        self.create_timer(1.0 / min(args.record_rate, args.publish_rate), self.record_current)

        self.get_logger().info("========================================")
        self.get_logger().info(f"Synchronous sweep motors: {self.motor_ids}")
        self.get_logger().info(f"Frequency: {args.frequency:.3f} Hz, amplitude: {args.amplitude:.3f}")
        self.get_logger().info(f"Signs: {self.signs}; command indices: {[m - 1 for m in self.motor_ids]}")
        self.get_logger().info(f"Publish/record: {args.publish_rate:.1f}/{min(args.record_rate, args.publish_rate):.1f} Hz")
        self.get_logger().info(f"Duration: {args.duration:.2f}s; pre-hold: {args.pre_hold_sec:.2f}s")
        self.get_logger().info(f"Output CSV: {self.output_file}")
        self.get_logger().info("Other /policy/commands entries stay at 0.0")
        self.get_logger().info("========================================")

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def phase_elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_start_time).nanoseconds * 1e-9

    def start_enter_wait_thread(self) -> None:
        if self.enter_thread is not None:
            return

        def wait_for_enter() -> None:
            try:
                input("Pre-hold is active. Press Enter to start the formal sweep...")
            except EOFError:
                self.get_logger().warning("stdin closed while waiting for Enter; starting sweep.")
            self.enter_event.set()

        self.enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        self.enter_thread.start()

    def start_sweep(self) -> None:
        self.phase = "sweep"
        self.start_time = self.get_clock().now()
        self.sweep_started = True
        self.get_logger().info("Starting formal two-motor sweep now.")

    def joint_state_callback(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if not name.startswith("motor_"):
                continue
            try:
                motor_id = int(name[len("motor_") :])
            except ValueError:
                continue
            if motor_id < 1 or motor_id > MOTOR_COUNT:
                continue
            idx = motor_id - 1
            if i < len(msg.position):
                self.positions[idx] = float(msg.position[i])
            if i < len(msg.velocity):
                self.velocities[idx] = float(msg.velocity[i])
            if i < len(msg.effort):
                self.torques[idx] = float(msg.effort[i])

    def publish_zero(self) -> None:
        self.commands = [0.0] * MOTOR_COUNT
        self.pub.publish(Float32MultiArray(data=self.commands))

    def publish_command(self) -> None:
        if self.finished:
            return

        if self.phase == "pre_hold":
            self.publish_zero()
            if self.phase_elapsed() >= self.args.pre_hold_sec:
                if self.args.wait_for_enter_after_hold:
                    self.phase = "wait_enter"
                    self.get_logger().info("Pre-hold complete. Holding cmd=0 until Enter is pressed.")
                    self.start_enter_wait_thread()
                else:
                    self.start_sweep()
            return

        if self.phase == "wait_enter":
            self.publish_zero()
            if self.enter_event.is_set():
                self.start_sweep()
            return

        t = self.elapsed()
        if t >= self.args.duration:
            self.publish_zero()
            self.record_current()
            self.save_csv()
            self.finished = True
            self.get_logger().info("Sweep finished. Published zero command and shutting down.")
            return

        self.commands = [0.0] * MOTOR_COUNT
        signal = self.args.amplitude * math.sin(2.0 * math.pi * self.args.frequency * t)
        for motor_id, sign in zip(self.motor_ids, self.signs):
            self.commands[motor_id - 1] = sign * signal
        self.pub.publish(Float32MultiArray(data=self.commands))

        if int(t * self.args.publish_rate) % max(1, int(self.args.publish_rate)) == 0:
            terms = ", ".join(f"cmd[{motor_id - 1}]={self.commands[motor_id - 1]:+.4f}" for motor_id in self.motor_ids)
            self.get_logger().info(f"t={t:.2f}/{self.args.duration:.2f}s freq={self.args.frequency:.3f}Hz {terms}")

    def command_for_motor(self, motor_id: int) -> float:
        return self.commands[motor_id - 1] if 1 <= motor_id <= MOTOR_COUNT else 0.0

    def record_current(self) -> None:
        if self.finished or not self.sweep_started or self.phase != "sweep":
            return
        row = [self.elapsed()]
        for motor_id in range(1, MOTOR_COUNT + 1):
            idx = motor_id - 1
            row.extend([self.command_for_motor(motor_id), self.positions[idx], self.velocities[idx], self.torques[idx]])
        self.records.append(row)

    def save_csv(self) -> None:
        header = ["timestamp"]
        for motor_id in range(1, MOTOR_COUNT + 1):
            header.extend(
                [
                    f"motor_{motor_id}_cmd_pos",
                    f"motor_{motor_id}_act_pos",
                    f"motor_{motor_id}_act_vel",
                    f"motor_{motor_id}_act_torque",
                ]
            )
        with self.output_file.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            writer.writerows(self.records)
        self.get_logger().info(f"Wrote CSV: {self.output_file} ({len(self.records)} rows)")


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = DualMotorSweepNode(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted. Publishing zero command and saving partial data.")
        node.publish_zero()
        node.save_csv()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
