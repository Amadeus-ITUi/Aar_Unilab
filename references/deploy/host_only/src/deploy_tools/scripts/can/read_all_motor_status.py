#!/usr/bin/env python3
"""Read encoder/status for motors 1-6 without enabling or motion commands.

Default mode is passive listen-only from user space: it does not transmit any
CAN frame. Add --request to send 0x02 status requests.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import can


P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -50.0, 50.0
T_MIN, T_MAX = -14.0, 14.0
DEFAULT_MAPPING = {
    "can0": [1, 2, 3, 4, 5, 6],
}


def build_can_id(cmd_type: int, motor_id: int, data_field: int = 0) -> int:
    return ((cmd_type & 0x1F) << 24) | ((data_field & 0xFFFF) << 8) | (motor_id & 0xFF)


def uint_to_float(x: int, x_min: float, x_max: float, bits: int = 16) -> float:
    return float(x) * (x_max - x_min) / ((1 << bits) - 1) + x_min


def ensure_socketcan_up(channel: str) -> None:
    result = subprocess.run(
        ["ip", "-details", "link", "show", channel],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"找不到 SocketCAN 接口 {channel}: {result.stderr.strip()}")
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    if "UP" not in first_line:
        raise RuntimeError(f"{channel} 未处于 UP 状态:\n{result.stdout}")


def motors_node_running() -> bool | None:
    try:
        result = subprocess.run(
            ["ros2", "node", "list"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return any(line.strip() == "/motors_node" for line in result.stdout.splitlines())


def parse_motor_status(msg: can.Message, motor_id: int) -> tuple[float, float, float, float] | None:
    if not msg.is_extended_id:
        return None
    can_id = msg.arbitration_id & 0x1FFFFFFF
    comm_type = (can_id >> 24) & 0x1F
    if comm_type != 0x02:
        return None

    low_id = can_id & 0xFF
    mid_id = (can_id >> 8) & 0xFF
    if not (low_id == 0x00 and mid_id == motor_id):
        return None

    data = bytes(msg.data).ljust(8, b"\x00")
    pos_u = (data[0] << 8) | data[1]
    vel_u = (data[2] << 8) | data[3]
    tor_u = (data[4] << 8) | data[5]
    tmp_u = (data[6] << 8) | data[7]
    return (
        uint_to_float(pos_u, P_MIN, P_MAX),
        uint_to_float(vel_u, V_MIN, V_MAX),
        uint_to_float(tor_u, T_MIN, T_MAX),
        tmp_u * 0.1,
    )


def drain_bus(bus: can.BusABC, seconds: float) -> None:
    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        bus.recv(timeout=0.01)


def wait_status(bus: can.BusABC, motor_id: int, timeout: float) -> tuple[float, float, float, float] | None:
    end_time = time.monotonic() + timeout
    while time.monotonic() < end_time:
        msg = bus.recv(timeout=0.05)
        if msg is None:
            continue
        parsed = parse_motor_status(msg, motor_id)
        if parsed is not None:
            return parsed
    return None


def request_status(bus: can.BusABC, channel: str, motor_id: int, timeout: float) -> tuple[float, float, float, float] | None:
    request_id = build_can_id(0x02, motor_id)
    bus.send(can.Message(arbitration_id=request_id, data=bytes(8), is_extended_id=True))
    print(f"[TX] {channel} motor={motor_id} request id=0x{request_id:08X}")
    end_time = time.monotonic() + timeout
    while time.monotonic() < end_time:
        msg = bus.recv(timeout=0.05)
        if msg is None:
            continue
        parsed = parse_motor_status(msg, motor_id)
        if parsed is not None:
            return parsed
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Read motors 1-6 encoder/status without motion excitation.")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--drain", type=float, default=0.2)
    parser.add_argument("--request", action="store_true", help="Actively send 0x02 status requests.")
    args = parser.parse_args()

    for channel in DEFAULT_MAPPING:
        ensure_socketcan_up(channel)

    buses: dict[str, can.BusABC] = {}
    try:
        for channel in DEFAULT_MAPPING:
            buses[channel] = can.Bus(interface="socketcan", channel=channel)
            drain_bus(buses[channel], args.drain)

        ok = True
        print("=== READ ALL MOTOR ENCODERS ONLY ===")
        print("当前映射: can0 -> 1/2/3/4/5/6")
        if args.request:
            print("主动发送 0x02 状态请求；不 enable，不发送 0x01 MIT 控制帧。")
        else:
            print("被动监听反馈；不发送任何 CAN 帧。若需要主动请求，添加 --request。")
            running = motors_node_running()
            if running is False:
                print("[INFO] 未检测到 /motors_node；如果没有其他程序在发请求/控制帧，被动监听会全部 timeout。")
        for channel, motor_ids in DEFAULT_MAPPING.items():
            bus = buses[channel]
            for motor_id in motor_ids:
                if args.request:
                    status = request_status(bus, channel, motor_id, args.timeout)
                else:
                    status = wait_status(bus, motor_id, args.timeout)
                if status is None:
                    print(f"[RX] motor {motor_id} on {channel}: timeout")
                    ok = False
                    continue
                pos, vel, torque, temp = status
                print(
                    f"[RX] motor {motor_id} on {channel}: "
                    f"pos={pos:+.6f} rad ({pos * 57.295779513:+.3f} deg), "
                    f"vel={vel:+.6f} rad/s, torque={torque:+.6f} Nm, temp={temp:.1f} C"
                )
        return 0 if ok else 2
    finally:
        for bus in buses.values():
            bus.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
