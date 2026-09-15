#!/usr/bin/env python3
"""Read-only CAN status publisher for Foxglove debugging.

This node sends only protocol 0x02 status requests. It never sends enable,
disable, MIT, mode-change, or zero-setting frames.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass

import can
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState


P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -50.0, 50.0
T_MIN, T_MAX = -14.0, 14.0


@dataclass
class MotorState:
    position: float = math.nan
    velocity: float = math.nan
    torque: float = math.nan
    temperature: float = math.nan
    received_at: float = 0.0


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip(), 0) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_can_id(comm_type: int, motor_id: int, data_field: int = 0) -> int:
    return ((comm_type & 0x1F) << 24) | ((data_field & 0xFFFF) << 8) | (motor_id & 0xFF)


def uint_to_float(value: int, low: float, high: float, bits: int = 16) -> float:
    return float(value) * (high - low) / ((1 << bits) - 1) + low


def wrap_position_to_limits(value: float, pos_min: float, pos_max: float) -> float:
    period = pos_max - pos_min
    if period <= 0.0:
        return value
    wrapped = math.fmod(value - pos_min, period)
    if wrapped < 0.0:
        wrapped += period
    return pos_min + wrapped


def parse_status(message: can.Message, motor_id: int) -> MotorState | None:
    if not message.is_extended_id:
        return None

    can_id = message.arbitration_id & 0x1FFFFFFF
    comm_type = (can_id >> 24) & 0x1F
    low_id = can_id & 0xFF
    mid_id = (can_id >> 8) & 0xFF

    # Status feedback is 0x0200MM00. Ignore our request 0x020000MM.
    if comm_type != 0x02 or low_id != 0x00 or mid_id != motor_id:
        return None

    data = bytes(message.data).ljust(8, b"\x00")
    pos_u = (data[0] << 8) | data[1]
    vel_u = (data[2] << 8) | data[3]
    torque_u = (data[4] << 8) | data[5]
    temperature_u = (data[6] << 8) | data[7]
    return MotorState(
        position=uint_to_float(pos_u, P_MIN, P_MAX),
        velocity=uint_to_float(vel_u, V_MIN, V_MAX),
        torque=uint_to_float(torque_u, T_MIN, T_MAX),
        temperature=temperature_u * 0.1,
        received_at=time.monotonic(),
    )


class ReadOnlyMotorStateNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("readonly_motor_state_node")

        self.channel = args.channel
        self.motor_ids = parse_csv_ints(args.motor_ids)
        self.default_angles = parse_csv_floats(args.default_angles)
        self.flipped = [bool(int(value)) for value in parse_csv_ints(args.flipped)]
        self.request_rate = args.request_rate
        self.publish_rate = args.publish_rate
        self.status_timeout = args.status_timeout

        if len(self.motor_ids) != 6:
            raise ValueError("--motor-ids must contain exactly six motor ids")
        if len(self.default_angles) != 6 or len(self.flipped) != 6:
            raise ValueError("--default-angles and --flipped must contain six values")
        if self.request_rate <= 0.0 or self.publish_rate <= 0.0:
            raise ValueError("--request-rate and --publish-rate must be positive")

        self.bus = can.Bus(interface="socketcan", channel=self.channel)
        self.states = {motor_id: MotorState() for motor_id in self.motor_ids}
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_error = ""
        self.last_warning = 0.0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(JointState, "/policy/joint_states", qos)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate, self.publish_joint_state)

        self.receive_thread = threading.Thread(target=self.receive_loop, name="can_status_rx", daemon=True)
        self.request_thread = threading.Thread(target=self.request_loop, name="can_status_tx", daemon=True)
        self.receive_thread.start()
        self.request_thread.start()

        self.get_logger().warning(
            f"READ-ONLY motor monitor started: CAN={self.channel} motors={self.motor_ids} "
            f"request_rate={self.request_rate:.1f}Hz publish_rate={self.publish_rate:.1f}Hz"
        )
        self.get_logger().warning(
            "Only 0x02 status requests are transmitted; no 0x01/0x03/0x04/0x06 frames are generated."
        )

    def request_loop(self) -> None:
        interval = 1.0 / (self.request_rate * len(self.motor_ids))
        while not self.stop_event.is_set():
            for motor_id in self.motor_ids:
                if self.stop_event.is_set():
                    return
                try:
                    self.bus.send(
                        can.Message(
                            arbitration_id=build_can_id(0x02, motor_id),
                            data=bytes(8),
                            is_extended_id=True,
                        )
                    )
                except Exception as exc:  # python-can backend errors are runtime failures
                    self.last_error = f"CAN request failed: {exc}"
                self.stop_event.wait(interval)

    def receive_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                message = self.bus.recv(timeout=0.1)
            except Exception as exc:
                self.last_error = f"CAN receive failed: {exc}"
                time.sleep(0.1)
                continue
            if message is None:
                continue
            motor_id = (message.arbitration_id >> 8) & 0xFF
            if motor_id not in self.states:
                continue
            state = parse_status(message, motor_id)
            if state is not None:
                with self.state_lock:
                    self.states[motor_id] = state

    def publish_joint_state(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"

        with self.state_lock:
            states = {motor_id: self.states[motor_id] for motor_id in self.motor_ids}

        message.name = [f"motor_{motor_id}" for motor_id in self.motor_ids]
        for index, motor_id in enumerate(self.motor_ids):
            state = states[motor_id]
            sign = -1.0 if self.flipped[index] else 1.0
            signed_position = sign * state.position
            signed_velocity = sign * state.velocity
            signed_torque = sign * state.torque
            policy_position = signed_position - self.default_angles[index]
            if index in (2, 5):
                policy_position = wrap_position_to_limits(policy_position, -6.28, 6.28)
            message.position.append(policy_position)
            message.velocity.append(signed_velocity)
            message.effort.append(signed_torque)

            if state.received_at <= 0.0 or time.monotonic() - state.received_at > self.status_timeout:
                message.position[-1] = math.nan
                message.velocity[-1] = math.nan
                message.effort[-1] = math.nan

        self.publisher.publish(message)

        if self.last_error and time.monotonic() - self.last_warning > 5.0:
            self.get_logger().error(self.last_error)
            self.last_warning = time.monotonic()

    def close(self) -> None:
        self.stop_event.set()
        self.request_thread.join(timeout=1.0)
        self.receive_thread.join(timeout=1.0)
        self.bus.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read motor states without enabling or controlling motors.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor-ids", default="1,2,3,4,5,6")
    parser.add_argument("--request-rate", type=float, default=20.0, help="Complete six-motor request cycles per second.")
    parser.add_argument("--publish-rate", type=float, default=50.0)
    parser.add_argument("--status-timeout", type=float, default=1.0)
    parser.add_argument("--default-angles", default="-0.92020,0.98338,0.0,-0.92020,0.98338,0.0")
    parser.add_argument("--flipped", default="1,0,1,0,1,0")
    args = parser.parse_args()

    rclpy.init()
    node = None
    try:
        node = ReadOnlyMotorStateNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
