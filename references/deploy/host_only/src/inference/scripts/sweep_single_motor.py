#!/usr/bin/env python3
"""Single-motor chirp publisher and recorder for DR002.

Publishes a full 6-element /policy/commands vector. Only one selected motor
receives the chirp signal; the other five commands stay at zero.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

try:
    from motors.srv import SetMotorGains
except ImportError:
    SetMotorGains = None


MOTOR_COUNT = 6
DEFAULT_MOTOR_NAMES = [
    "left_thigh_joint",
    "left_calf_joint",
    "left_foot_joint",
    "right_thigh_joint",
    "right_calf_joint",
    "right_foot_joint",
]
WHEEL_INDICES = {2, 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep one DR002 motor while holding the others at zero command.")
    parser.add_argument("--motor-index", type=int, required=True, help="0-based command index in /policy/commands.")
    parser.add_argument("--motor-id", type=int, required=True, help="Physical motor id, used for CSV labels.")
    parser.add_argument("--motor-name", type=str, required=True, help="Human-readable motor/joint name.")
    parser.add_argument("--start-frequency", type=float, default=0.1, help="Start frequency in Hz.")
    parser.add_argument("--end-frequency", type=float, default=10.0, help="End frequency in Hz.")
    parser.add_argument("--amplitude", type=float, default=0.3, help="Chirp amplitude. Legs use relative position; wheels use velocity target.")
    parser.add_argument("--publish-rate", type=float, default=400.0, help="Command publish rate in Hz.")
    parser.add_argument("--record-rate", type=float, default=400.0, help="CSV record rate in Hz.")
    parser.add_argument("--duration", type=float, default=40.0, help="Sweep duration in seconds.")
    parser.add_argument("--pre-hold-sec", type=float, default=0.0, help="Publish zero command for this many seconds before starting the sweep.")
    parser.add_argument("--wait-for-enter-after-hold", action="store_true", help="Keep publishing zero command after pre-hold until Enter is pressed.")
    parser.add_argument("--amplitude-ramp-sec", type=float, default=0.0, help="Ramp sweep amplitude from 0 to full scale over this many seconds.")
    parser.add_argument(
        "--constant-command",
        type=float,
        default=None,
        help="Publish this constant command instead of a chirp. Useful for command-to-motor mapping checks.",
    )
    parser.add_argument("--output-file", type=str, default="", help="CSV output path. Empty means auto-generate under plots/.")
    parser.add_argument("--no-plot", action="store_true", help="Do not invoke plot_sweep_frequency.py after saving CSV.")
    parser.add_argument(
        "--sign",
        type=float,
        default=0.0,
        help="Signal sign override. 0 means auto: left indices 0-2 negative, right indices 3-5 positive.",
    )
    parser.add_argument("--dynamic-gains", action="store_true", help="Set runtime kp/kd through /set_motor_gains before sweeping.")
    parser.add_argument("--sweep-kp", type=float, default=None, help="Kp for the swept motor. Required for position motors with --dynamic-gains.")
    parser.add_argument("--sweep-kd", type=float, default=None, help="Kd for the swept motor. Required with --dynamic-gains.")
    parser.add_argument("--hold-kp", type=float, default=8.0, help="Kp for non-swept position motors when --dynamic-gains is used.")
    parser.add_argument("--hold-kd", type=float, default=0.8, help="Kd for non-swept position motors when --dynamic-gains is used.")
    parser.add_argument("--wheel-hold-kd", type=float, default=0.8, help="Kd for non-swept wheel velocity motors when --dynamic-gains is used.")
    parser.add_argument(
        "--gains-motor-ids",
        type=str,
        default="",
        help="Comma-separated physical motor ids for /set_motor_gains. Empty means all 1..6.",
    )
    parser.add_argument("--gains-service-timeout", type=float, default=5.0, help="Seconds to wait for /set_motor_gains.")
    parser.add_argument("--no-restore-gains-on-exit", action="store_true", help="Do not restore startup gains after the sweep.")
    return parser.parse_args()


def default_output_file(motor_id: int, motor_name: str) -> Path:
    root = os.environ.get("DEPLOY_CPP_ROOT")
    plots_dir = Path(root) / "plots" if root else Path.cwd() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1.0) * 1000)
    safe_name = motor_name.replace("/", "_").replace(" ", "_")
    return plots_dir / f"sweep_motor_{motor_id}_{safe_name}_{stamp}_{ms:03d}.csv"


class SingleMotorSweepNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__(f"sweep_motor_{args.motor_id}_{args.motor_name}")
        if not 0 <= args.motor_index < MOTOR_COUNT:
            raise ValueError(f"--motor-index must be in [0, {MOTOR_COUNT - 1}], got {args.motor_index}")
        if args.start_frequency <= 0 or args.end_frequency <= 0 or args.start_frequency > args.end_frequency:
            raise ValueError("--start-frequency and --end-frequency must be positive and ordered")
        if args.amplitude <= 0 or args.publish_rate <= 0 or args.record_rate <= 0 or args.duration <= 0:
            raise ValueError("--amplitude, --publish-rate, --record-rate and --duration must be positive")
        if args.pre_hold_sec < 0 or args.amplitude_ramp_sec < 0:
            raise ValueError("--pre-hold-sec and --amplitude-ramp-sec must be non-negative")

        self.args = args
        self.output_file = Path(args.output_file) if args.output_file else default_output_file(args.motor_id, args.motor_name)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.sign = args.sign if abs(args.sign) > 1e-6 else (-1.0 if args.motor_index < 3 else 1.0)
        self.commands: List[float] = [0.0] * MOTOR_COUNT
        self.positions: List[float] = [0.0] * MOTOR_COUNT
        self.velocities: List[float] = [0.0] * MOTOR_COUNT
        self.torques: List[float] = [0.0] * MOTOR_COUNT
        self.records: list[list[float]] = []
        self.start_time = self.get_clock().now()
        self.phase_start_time = self.start_time
        self.finished = False
        self.commanding_started = not args.dynamic_gains
        self.sweep_started = False
        self.gains_applied = False
        self.enter_event = threading.Event()
        self.enter_thread: threading.Thread | None = None
        self.phase = "sweep"
        if args.pre_hold_sec > 0.0 or args.wait_for_enter_after_hold:
            self.phase = "pre_hold"

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
        self.sub = self.create_subscription(JointState, "/policy/joint_states", self.joint_state_callback, sensor_qos)
        self.pub = self.create_publisher(Float32MultiArray, "/policy/commands", control_qos)
        self.gains_client = None
        if args.dynamic_gains:
            if SetMotorGains is None:
                raise RuntimeError("Cannot import motors.srv.SetMotorGains. Source the motors workspace after rebuilding it.")
            self.gains_client = self.create_client(SetMotorGains, "/set_motor_gains")

        self.publish_timer = self.create_timer(1.0 / args.publish_rate, self.publish_command)
        self.record_timer = self.create_timer(1.0 / min(args.record_rate, args.publish_rate), self.record_current)
        if self.commanding_started and self.phase == "sweep":
            self.start_sweep()

        self.get_logger().info("========================================")
        self.get_logger().info(f"Single motor sweep: motor_id={args.motor_id}, index={args.motor_index}, name={args.motor_name}")
        self.get_logger().info(f"Frequency: {args.start_frequency:.2f} Hz -> {args.end_frequency:.2f} Hz")
        if args.constant_command is None:
            self.get_logger().info(f"Amplitude: {args.amplitude:.3f}, sign={self.sign:+.0f}")
        else:
            self.get_logger().info(f"Constant command: {args.constant_command:+.3f} rad")
        self.get_logger().info(f"Publish/record: {args.publish_rate:.1f}/{min(args.record_rate, args.publish_rate):.1f} Hz")
        self.get_logger().info(f"Duration: {args.duration:.2f} s")
        if args.pre_hold_sec > 0.0 or args.wait_for_enter_after_hold:
            self.get_logger().info(
                f"Pre-hold: cmd=0 for {args.pre_hold_sec:.2f}s"
                + (" then wait for Enter" if args.wait_for_enter_after_hold else "")
            )
        if args.amplitude_ramp_sec > 0.0 and args.constant_command is None:
            self.get_logger().info(f"Amplitude ramp: {args.amplitude_ramp_sec:.2f}s")
        self.get_logger().info(f"Output CSV: {self.output_file}")
        self.get_logger().info("Other five /policy/commands entries stay at 0.0")
        if args.dynamic_gains:
            self.get_logger().info(
                f"Dynamic gains enabled: sweep_kp={args.sweep_kp}, sweep_kd={args.sweep_kd}, "
                f"hold_kp={args.hold_kp}, hold_kd={args.hold_kd}, wheel_hold_kd={args.wheel_hold_kd}"
            )
        self.get_logger().info("========================================")

    def build_sweep_gains(self) -> tuple[list[float], list[float]]:
        if self.args.sweep_kd is None:
            raise ValueError("--sweep-kd is required with --dynamic-gains")
        if self.args.motor_index not in WHEEL_INDICES and self.args.sweep_kp is None:
            raise ValueError("--sweep-kp is required for position motors with --dynamic-gains")

        kp = [self.args.hold_kp] * MOTOR_COUNT
        kd = [self.args.hold_kd] * MOTOR_COUNT
        for index in WHEEL_INDICES:
            kp[index] = 0.0
            kd[index] = self.args.wheel_hold_kd

        if self.args.motor_index in WHEEL_INDICES:
            kp[self.args.motor_index] = 0.0
        else:
            kp[self.args.motor_index] = float(self.args.sweep_kp)
        kd[self.args.motor_index] = float(self.args.sweep_kd)
        return kp, kd

    def call_set_motor_gains(self, *, restore_defaults: bool, kp: list[float] | None = None, kd: list[float] | None = None) -> None:
        if self.gains_client is None:
            return
        if not self.gains_client.wait_for_service(timeout_sec=self.args.gains_service_timeout):
            raise RuntimeError("/set_motor_gains service is not available")

        req = SetMotorGains.Request()
        req.restore_defaults = restore_defaults
        if not restore_defaults:
            if self.args.gains_motor_ids.strip():
                motor_ids = [int(item.strip()) for item in self.args.gains_motor_ids.split(",") if item.strip()]
                req.motor_ids = motor_ids
                req.kp = [float(self.gain_for_motor(kp, motor_id)) for motor_id in motor_ids]
                req.kd = [float(self.gain_for_motor(kd, motor_id)) for motor_id in motor_ids]
            else:
                req.motor_ids = list(range(1, MOTOR_COUNT + 1))
                req.kp = [float(value) for value in kp]
                req.kd = [float(value) for value in kd]

        future = self.gains_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.gains_service_timeout)
        if not future.done():
            raise RuntimeError("/set_motor_gains request timed out")
        result = future.result()
        if result is None or not result.success:
            message = "no response" if result is None else result.message
            raise RuntimeError(f"/set_motor_gains failed: {message}")
        self.get_logger().info(result.message)

    def apply_dynamic_gains(self) -> None:
        if not self.args.dynamic_gains:
            return
        kp, kd = self.build_sweep_gains()
        self.call_set_motor_gains(restore_defaults=False, kp=kp, kd=kd)
        self.gains_applied = True
        self.commanding_started = True
        self.phase_start_time = self.get_clock().now()
        if self.args.pre_hold_sec > 0.0 or self.args.wait_for_enter_after_hold:
            self.phase = "pre_hold"
            self.sweep_started = False
        else:
            self.start_sweep()
        self.get_logger().info(f"Applied sweep gains: kp={kp}, kd={kd}")

    def restore_dynamic_gains(self) -> None:
        if not self.args.dynamic_gains or not self.gains_applied or self.args.no_restore_gains_on_exit:
            return
        self.call_set_motor_gains(restore_defaults=True)
        self.gains_applied = False
        self.get_logger().info("Restored startup motor gains")

    def gain_for_motor(self, values: list[float], motor_id: int) -> float:
        if motor_id == self.args.motor_id:
            return values[self.args.motor_index]
        return values[motor_id - 1]

    def command_for_motor(self, motor_id: int) -> float:
        if motor_id == self.args.motor_id:
            return self.commands[self.args.motor_index]
        return 0.0

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def phase_elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_start_time).nanoseconds * 1e-9

    def start_sweep(self) -> None:
        self.phase = "sweep"
        self.start_time = self.get_clock().now()
        self.sweep_started = True
        self.get_logger().info("Starting formal sweep now.")

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

    def joint_state_callback(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            idx = self.name_to_index(name)
            if idx is None:
                continue
            if i < len(msg.position):
                self.positions[idx] = float(msg.position[i])
            if i < len(msg.velocity):
                self.velocities[idx] = float(msg.velocity[i])
            if i < len(msg.effort):
                self.torques[idx] = float(msg.effort[i])

    @staticmethod
    def name_to_index(name: str) -> int | None:
        if name.startswith("motor_"):
            try:
                motor_id = int(name[len("motor_") :])
            except ValueError:
                return None
            if 1 <= motor_id <= MOTOR_COUNT:
                return motor_id - 1
        if name in DEFAULT_MOTOR_NAMES:
            return DEFAULT_MOTOR_NAMES.index(name)
        return None

    def current_frequency(self, t: float) -> float:
        ratio = min(max(t / self.args.duration, 0.0), 1.0)
        return self.args.start_frequency + (self.args.end_frequency - self.args.start_frequency) * ratio

    def chirp_signal(self, t: float) -> float:
        if self.args.constant_command is not None:
            return float(self.args.constant_command)
        phase = self.args.start_frequency * t + 0.5 * (self.args.end_frequency - self.args.start_frequency) * t * t / self.args.duration
        ramp = 1.0
        if self.args.amplitude_ramp_sec > 0.0:
            ramp = min(max(t / self.args.amplitude_ramp_sec, 0.0), 1.0)
        return self.sign * self.args.amplitude * ramp * math.sin(2.0 * math.pi * phase)

    def publish_command(self) -> None:
        if self.finished or not self.commanding_started:
            return

        if self.phase == "pre_hold":
            self.commands = [0.0] * MOTOR_COUNT
            self.pub.publish(Float32MultiArray(data=self.commands))
            if self.phase_elapsed() >= self.args.pre_hold_sec:
                if self.args.wait_for_enter_after_hold:
                    self.phase = "wait_enter"
                    self.get_logger().info("Pre-hold complete. Holding cmd=0 until Enter is pressed.")
                    self.start_enter_wait_thread()
                else:
                    self.start_sweep()
            return

        if self.phase == "wait_enter":
            self.commands = [0.0] * MOTOR_COUNT
            self.pub.publish(Float32MultiArray(data=self.commands))
            if self.enter_event.is_set():
                self.start_sweep()
            return

        if not self.sweep_started:
            return

        t = self.elapsed()
        if t >= self.args.duration:
            self.commands = [0.0] * MOTOR_COUNT
            self.pub.publish(Float32MultiArray(data=self.commands))
            self.record_current()
            self.save_csv()
            if not self.args.no_plot:
                self.plot_csv()
            self.finished = True
            self.get_logger().info("Sweep finished. Published zero command and shutting down.")
            return

        self.commands = [0.0] * MOTOR_COUNT
        self.commands[self.args.motor_index] = self.chirp_signal(t)
        self.pub.publish(Float32MultiArray(data=self.commands))

        if int(t * self.args.publish_rate) % max(1, int(self.args.publish_rate)) == 0:
            self.get_logger().info(
                f"t={t:.2f}/{self.args.duration:.2f}s "
                f"freq={self.current_frequency(t):.3f}Hz "
                f"cmd[{self.args.motor_index}]={self.commands[self.args.motor_index]:+.4f}"
            )

    def record_current(self) -> None:
        if self.finished or not self.sweep_started or self.phase != "sweep":
            return
        row = [self.elapsed()]
        for i in range(MOTOR_COUNT):
            motor_id = i + 1
            row.extend([self.command_for_motor(motor_id), self.positions[i], self.velocities[i], self.torques[i]])
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
        with self.output_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.records)
        self.get_logger().info(f"Wrote CSV: {self.output_file} ({len(self.records)} rows)")

    def plot_csv(self) -> None:
        script = Path(__file__).resolve().parent / "plot_sweep_frequency.py"
        if not script.exists():
            self.get_logger().warning(f"Plot script not found: {script}")
            return
        pdf = self.output_file.with_name(self.output_file.stem + "_plot.pdf")
        result = subprocess.run([sys.executable, str(script), str(self.output_file), "-o", str(pdf)], check=False)
        if result.returncode == 0:
            self.get_logger().info(f"Wrote PDF: {pdf}")
        else:
            self.get_logger().warning(f"Plot script failed with return code {result.returncode}")


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = SingleMotorSweepNode(args)
    try:
        node.apply_dynamic_gains()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted. Publishing zero command and saving partial data.")
        node.commands = [0.0] * MOTOR_COUNT
        node.pub.publish(Float32MultiArray(data=node.commands))
        node.save_csv()
    finally:
        try:
            node.restore_dynamic_gains()
        except Exception as exc:
            node.get_logger().error(f"Failed to restore motor gains: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
