#!/usr/bin/env python3
"""Read one motor status without enabling or motion commands.

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


def build_can_id(cmd_type: int, motor_id: int, master_id: int = 0) -> int:
    return ((cmd_type & 0x1F) << 24) | ((master_id & 0xFFFF) << 8) | (motor_id & 0xFF)


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
        print(f"[ERROR] 找不到 SocketCAN 接口 {channel}: {result.stderr.strip()}")
        sys.exit(2)
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    if "UP" not in first_line:
        print(f"[ERROR] {channel} 未处于 UP 状态。")
        print(result.stdout)
        sys.exit(2)


def parse_motor_status(msg: can.Message, motor_id: int) -> tuple[float, float, float, float] | None:
    if not msg.is_extended_id:
        return None
    can_id = msg.arbitration_id & 0x1FFFFFFF
    comm_type = (can_id >> 24) & 0x1F
    if comm_type != 0x02:
        return None

    # Real feedback is observed as 0x0200MM00. Ignore our own request
    # 0x020000MM and any stale/foreign frame.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Read motor encoder/status without motion excitation.")
    parser.add_argument("--channel", default="can1")
    parser.add_argument("--motor-id", type=int, default=1)
    parser.add_argument("--master-id", type=lambda text: int(text, 0), default=0)
    parser.add_argument("--timeout", type=float, default=1.5)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--request", action="store_true", help="Actively send 0x02 status requests.")
    args = parser.parse_args()

    ensure_socketcan_up(args.channel)
    bus = can.Bus(interface="socketcan", channel=args.channel)
    try:
        request_id = build_can_id(0x02, args.motor_id, args.master_id)
        print("=== read motor status only ===")
        print(f"channel={args.channel} motor_id={args.motor_id} master_id={args.master_id}")
        if args.request:
            print("主动发送 0x02 状态请求；不 enable，不发送 0x01 MIT 控制帧。")
        else:
            print("被动监听反馈；不发送任何 CAN 帧。若需要主动请求，添加 --request。")

        if args.request:
            for _ in range(max(1, args.requests)):
                bus.send(can.Message(arbitration_id=request_id, data=bytes(8), is_extended_id=True))
                print(f"[TX] status request id=0x{request_id:08X} data=0000000000000000")
                time.sleep(0.02)

        end_time = time.monotonic() + args.timeout
        while time.monotonic() < end_time:
            msg = bus.recv(timeout=0.05)
            if msg is None:
                continue
            parsed = parse_motor_status(msg, args.motor_id)
            if parsed is None:
                continue
            pos, vel, torque, temp = parsed
            print(f"[RX] pos={pos:+.6f} rad vel={vel:+.6f} rad/s torque={torque:+.6f} Nm temp={temp:.1f} C")
            print(f"[RX] encoder angle = {pos:+.6f} rad = {pos * 57.295779513:+.3f} deg")
            return 0

        print("[ERROR] 超时，没有收到该电机状态反馈。")
        if not args.request:
            print("[HINT] 当前是被动监听模式；如果没有其他节点让电机回帧，这里会超时。")
        return 1
    finally:
        bus.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
