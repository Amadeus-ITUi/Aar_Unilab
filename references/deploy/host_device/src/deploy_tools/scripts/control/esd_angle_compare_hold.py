#!/usr/bin/env python3
"""Hold a fixed policy-coordinate pose for the ESD angle comparison test."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from esd_link_msgs.msg import LinkStatus, LowerState, MaintenanceCommand
from esd_link_msgs.srv import SetControlMode
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
import yaml

from esd_control_timeouts import CONTROL_MODE_SERVICE_TIMEOUT_SECONDS
from esd_safety_limits import LEG_POSITION_LIMITS


POSITION_PORTS = (1, 2, 4, 5)
WHEEL_PORTS = (3, 6)
ALLOWED_POLICY_TARGETS = (-0.1, 0.0, 0.1)


def policy_target(value: str | float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("policy-q must be finite")
    for allowed in ALLOWED_POLICY_TARGETS:
        if math.isclose(parsed, allowed, rel_tol=0.0, abs_tol=1e-9):
            return allowed
    raise argparse.ArgumentTypeError("policy-q must be exactly -0.1, 0, or +0.1 rad")


def load_gains(config_path: Path) -> tuple[list[float], list[float]]:
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    parameters = document["esd_link_bridge_node"]["ros__parameters"]
    kp = [float(value) for value in parameters["kp"]]
    kd = [float(value) for value in parameters["kd"]]
    wing_kp = [float(value) for value in parameters["wing_kp"]]
    wing_kd = [float(value) for value in parameters["wing_kd"]]
    if not (len(kp) == len(kd) == 6 and len(wing_kp) == len(wing_kd) == 2):
        raise ValueError("bridge Kp/Kd arrays must cover P1-P8")
    all_gains = kp + kd + wing_kp + wing_kd
    if not all(math.isfinite(value) and value >= 0.0 for value in all_gains):
        raise ValueError("bridge Kp/Kd entries must be finite and non-negative")
    if not all(math.isclose(kp[port - 1], 0.0, abs_tol=1e-9) for port in WHEEL_PORTS):
        raise ValueError("P3/P6 must retain Kp=0 for zero-speed control")
    return kp + wing_kp, kd + wing_kd


def make_command(
    state: LowerState,
    target: float,
    kp: list[float],
    kd: list[float],
) -> MaintenanceCommand:
    target = policy_target(target)
    if list(state.port_id) != list(range(1, 9)):
        raise ValueError(f"unexpected port layout: {list(state.port_id)}")
    if len(kp) != 8 or len(kd) != 8:
        raise ValueError("Kp/Kd must contain exactly eight entries")
    for port in POSITION_PORTS:
        lower, upper = LEG_POSITION_LIMITS[port]
        if target < lower or target > upper:
            raise ValueError(f"P{port} target {target:+.4f} outside [{lower}, {upper}]")

    command = MaintenanceCommand()
    command.mode = MaintenanceCommand.MANUAL_TEST
    command.port_id = list(range(1, 9))
    command.position_rad = [float(value) for value in state.position_rad]
    command.velocity_rad_s = [0.0] * 8
    command.kp = list(kp)
    command.kd = list(kd)
    command.effort_nm = [0.0] * 8
    for port in POSITION_PORTS:
        command.position_rad[port - 1] = target
    return command


class AngleCompareHold(Node):
    def __init__(self, target: float, kp: list[float], kd: list[float]) -> None:
        super().__init__("esd_angle_compare_hold")
        self.target = target
        self.kp = kp
        self.kd = kd
        self.state: LowerState | None = None
        self.link: LinkStatus | None = None
        self.command: MaintenanceCommand | None = None
        self.streaming = False
        self.last_source_sequence: int | None = None
        self.last_print = 0.0
        self.error = ""
        self.publisher = self.create_publisher(
            MaintenanceCommand, "/maintenance/commands", 1)
        self.create_subscription(
            LowerState, "/lower/state", self._state, qos_profile_sensor_data)
        self.create_subscription(
            LinkStatus,
            "/lower/link_status",
            self._link,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.mode_client = self.create_client(
            SetControlMode, "/lower/set_control_mode")

    def _link(self, msg: LinkStatus) -> None:
        self.link = msg
        if self.streaming and (
            not msg.session_valid
            or msg.link_state == LinkStatus.FAULT
            or msg.control_mode not in (
                SetControlMode.Request.MANUAL_TEST,
                SetControlMode.Request.DISABLED,
            )
            or msg.fault_flags
            or msg.offline_port_mask
        ):
            self.error = (
                f"link invalid: mode={msg.control_mode}, fault=0x{msg.fault_flags:x}, "
                f"offline=0x{msg.offline_port_mask:x}"
            )

    @staticmethod
    def state_error(msg: LowerState) -> str | None:
        if msg.schema_id != 1 or msg.active_port_mask != 0x1FE:
            return f"schema/layout invalid: schema={msg.schema_id}, mask=0x{msg.active_port_mask:x}"
        if msg.fault_flags or msg.offline_port_mask:
            return f"lower fault=0x{msg.fault_flags:x}, offline=0x{msg.offline_port_mask:x}"
        if msg.imu_valid_mask & 0x07 != 0x07:
            return f"IMU valid mask={msg.imu_valid_mask}"
        if list(msg.port_id) != list(range(1, 9)):
            return f"port layout={list(msg.port_id)}"
        if any((mask & 0x03) != 0x03 for mask in msg.valid_mask):
            return f"valid mask={list(msg.valid_mask)}"
        values = list(msg.position_rad) + list(msg.velocity_rad_s)
        if not all(math.isfinite(value) for value in values):
            return "motor feedback contains a non-finite value"
        if msg.control_state in (5, 6):
            return f"lower control_state={msg.control_state}"
        return None

    def _state(self, msg: LowerState) -> None:
        previous = self.state.state_sample_seq if self.state is not None else None
        self.state = msg
        if not self.streaming or previous == msg.state_sample_seq:
            return
        error = self.state_error(msg)
        if error is not None:
            self.error = error
            return
        assert self.command is not None
        if self.last_source_sequence != msg.state_sample_seq:
            command = copy.deepcopy(self.command)
            command.header.stamp = self.get_clock().now().to_msg()
            command.header.frame_id = "base_link"
            command.source_state_sample_seq = msg.state_sample_seq
            self.publisher.publish(command)
            self.last_source_sequence = msg.state_sample_seq
        now = time.monotonic()
        if now - self.last_print >= 1.0 / 30.0:
            q = msg.position_rad
            print(
                f"target={self.target:+.3f} | "
                f"P1={q[0]:+.6f} P2={q[1]:+.6f} "
                f"P4={q[3]:+.6f} P5={q[4]:+.6f}",
                flush=True,
            )
            self.last_print = now

    def wait_ready(self, timeout: float = 15.0) -> LowerState:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.state is None or self.link is None:
                continue
            if (
                self.state_error(self.state) is None
                and self.state.control_state == 2
                and self.link.session_valid
                and self.link.control_mode == SetControlMode.Request.DISABLED
                and self.link.device_control_state == 2
                and self.link.fault_flags == 0
                and self.link.offline_port_mask == 0
            ):
                return self.state
        raise RuntimeError("未获得 P1-P8/IMU 有效且 upper/lower DISABLED 的状态")

    def set_mode(self, mode: int) -> None:
        timeout = CONTROL_MODE_SERVICE_TIMEOUT_SECONDS
        if not self.mode_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("/lower/set_control_mode unavailable")
        request = SetControlMode.Request()
        request.target_mode = mode
        future = self.mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        result = future.result()
        if result is None or not result.success or result.actual_mode != mode:
            message = "timeout" if result is None else result.message
            raise RuntimeError(f"control mode {mode} rejected: {message}")


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(get_package_share_directory("esd_link_bridge"))
        / "config/esd_link_bridge.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Hold P1/P2/P4/P5 at a fixed policy angle until Ctrl+C")
    parser.add_argument("--policy-q", required=True, type=policy_target)
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kp, kd = load_gains(args.config)
    rclpy.init()
    node = AngleCompareHold(args.policy_q, kp, kd)
    armed = False
    try:
        state = node.wait_ready()
        node.command = make_command(state, args.policy_q, kp, kd)
        print(f"Kp={kp}; Kd={kd}", flush=True)
        print(
            f"P1/P2/P4/P5 policy_q={args.policy_q:+.3f} rad; "
            "P3/P6 velocity=0; P7/P8 hold current. Ctrl+C 结束。",
            flush=True,
        )
        node.set_mode(SetControlMode.Request.MANUAL_TEST)
        armed = True
        node.streaming = True
        while rclpy.ok() and not node.error:
            rclpy.spin_once(node, timeout_sec=0.01)
        if node.error:
            raise RuntimeError(node.error)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", flush=True)
        return 1
    finally:
        node.streaming = False
        if armed:
            try:
                node.set_mode(SetControlMode.Request.DISABLED)
                print("P1-P8 整体失能已确认。", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] 整体失能未确认：{exc}；请立即硬件断电。", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
