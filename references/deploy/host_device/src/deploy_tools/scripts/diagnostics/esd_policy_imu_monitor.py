#!/usr/bin/env python3
"""Print the final native ESD IMU observation consumed by the policy."""

from __future__ import annotations

import math
import time

import rclpy
from esd_link_msgs.msg import LinkStatus, LowerState
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)


PRINT_HZ = 10.0


def inference_compatible_error(msg: LowerState) -> str | None:
    """Mirror the native LowerState acceptance gates in lab_inference_node."""
    if msg.schema_id != 1:
        return f"schema_id={msg.schema_id}"
    if msg.active_port_mask != 0x1FE:
        return f"active_port_mask=0x{msg.active_port_mask:x}"
    if msg.offline_port_mask:
        return f"offline_port_mask=0x{msg.offline_port_mask:x}"
    if msg.fault_flags:
        return f"fault_flags=0x{msg.fault_flags:x}"
    if msg.imu_valid_mask & 0x07 != 0x07:
        return f"imu_valid_mask={msg.imu_valid_mask}"
    found = 0
    for index in range(8):
        port = int(msg.port_id[index])
        if port < 1 or port > 8 or (int(msg.valid_mask[index]) & 0x03) != 0x03:
            return f"invalid P1-P8 layout/valid mask at index {index}"
        if not (
            math.isfinite(float(msg.position_rad[index]))
            and math.isfinite(float(msg.velocity_rad_s[index]))
        ):
            return f"non-finite actuator feedback at index {index}"
        found |= 1 << port
    if found != 0x1FE:
        return f"port mask from layout=0x{found:x}"
    angular, gravity = final_policy_imu(msg)
    if not all(math.isfinite(value) for value in angular + gravity):
        return "non-finite final policy IMU observation"
    return None


def final_policy_imu(
    msg: LowerState,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return exactly the two vectors copied by the native inference callback."""
    imu = msg.imu
    angular = (
        float(imu.angular_velocity.x),
        float(imu.angular_velocity.y),
        float(imu.angular_velocity.z),
    )
    gravity = (
        float(imu.angular_velocity_covariance[0]),
        float(imu.angular_velocity_covariance[1]),
        float(imu.angular_velocity_covariance[2]),
    )
    return angular, gravity


def read_only_link_error(msg: LinkStatus) -> str | None:
    if not msg.session_valid:
        return None
    if (
        msg.control_mode != 0
        or msg.device_control_state != 2
        or msg.fault_flags
        or msg.offline_port_mask
    ):
        return (
            "链路不是 READ_ONLY/DISABLED："
            f"upper={msg.control_mode}, lower={msg.device_control_state}, "
            f"fault=0x{msg.fault_flags:x}, offline=0x{msg.offline_port_mask:x}"
        )
    return None


class PolicyImuMonitor(Node):
    def __init__(self) -> None:
        super().__init__("esd_policy_imu_monitor")
        self.link: LinkStatus | None = None
        self.last_print = 0.0
        self.samples = 0
        self.error = ""
        self.started = False
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

    def _link(self, msg: LinkStatus) -> None:
        self.link = msg
        error = read_only_link_error(msg)
        if error is not None:
            self.error = error

    def _state(self, msg: LowerState) -> None:
        if self.error:
            return
        error = inference_compatible_error(msg)
        if error is not None:
            if self.started or (self.link is not None and self.link.session_valid):
                self.error = error
            return
        if msg.control_state != 2:
            if self.started:
                self.error = f"lower control_state={msg.control_state}, expected DISABLED(2)"
            return
        if self.link is None or not self.link.session_valid:
            return
        if self.link.control_mode != 0 or self.link.device_control_state != 2:
            return
        self.started = True
        self.samples += 1
        now = time.monotonic()
        if now - self.last_print < 1.0 / PRINT_HZ:
            return
        angular, gravity = final_policy_imu(msg)
        print(
            "base_ang_vel = "
            f"[{angular[0]:+.6f}, {angular[1]:+.6f}, {angular[2]:+.6f}]    "
            "projected_gravity = "
            f"[{gravity[0]:+.6f}, {gravity[1]:+.6f}, {gravity[2]:+.6f}]",
            flush=True,
        )
        self.last_print = now


def main() -> int:
    rclpy.init()
    node = PolicyImuMonitor()
    print(
        "只读监视 /lower/state：以下数值与 use_lower_state=true 时 actor 观测前 6 维一致。",
        flush=True,
    )
    print("保持 upper/lower DISABLED；Ctrl+C 结束。", flush=True)
    try:
        startup_deadline = time.monotonic() + 15.0
        while rclpy.ok() and not node.error:
            rclpy.spin_once(node, timeout_sec=0.05)
            if not node.started and time.monotonic() >= startup_deadline:
                node.error = "15 秒内未收到可送入策略的有效 LowerState"
        if node.error:
            print(f"[ERROR] {node.error}", flush=True)
            return 1
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        samples = node.samples
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"监视结束：接收有效状态 {samples} 帧。", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
