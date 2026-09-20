#!/usr/bin/env python3
"""Read motor firmware version with the vendor 0x04 / 0xC4 query frame.

This script only sends firmware-version read requests. It does not enable motors
and does not send MIT control frames.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import can


DEFAULT_MAPPING = {
    "can0": [1, 2, 3, 4, 5, 6],
}


def build_can_id(cmd_type: int, motor_id: int, master_id: int = 0) -> int:
    return ((cmd_type & 0x1F) << 24) | ((master_id & 0xFFFF) << 8) | (motor_id & 0xFF)


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


def parse_firmware_reply(msg: can.Message, motor_id: int, master_id: int) -> tuple[str, bytes] | None:
    if not msg.is_extended_id:
        return None

    can_id = msg.arbitration_id & 0x1FFFFFFF
    comm_type = (can_id >> 24) & 0x1F
    if comm_type != 0x02:
        return None

    low_id = can_id & 0xFF
    mid_id = (can_id >> 8) & 0xFF
    if mid_id != (motor_id & 0xFF) or low_id != (master_id & 0xFF):
        return None

    data = bytes(msg.data).ljust(8, b"\x00")
    if data[0] != 0x00 or data[1] != 0xC4:
        return None

    # According to the manual page: byte[3]..byte[6] are the firmware version,
    # ordered from high to low. Keep byte[2] too because many firmwares use it
    # as a fixed marker, commonly 0x56.
    version_bytes = data[3:7]
    version_hex = ".".join(f"{b:02X}" for b in version_bytes)
    return version_hex, data


def drain_bus(bus: can.BusABC, seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if bus.recv(timeout=0.01) is None:
            continue


def query_one(bus: can.BusABC, channel: str, motor_id: int, master_id: int, timeout: float) -> bool:
    request_id = build_can_id(0x04, motor_id, master_id)
    payload = bytes([0x00, 0xC4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    drain_bus(bus)
    bus.send(can.Message(arbitration_id=request_id, data=payload, is_extended_id=True))
    print(f"[TX] {channel} motor={motor_id} id=0x{request_id:08X} data={payload.hex()}")

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        msg = bus.recv(timeout=0.05)
        if msg is None:
            continue
        parsed = parse_firmware_reply(msg, motor_id, master_id)
        if parsed is None:
            continue
        version_hex, data = parsed
        print(
            f"[RX] {channel} motor={motor_id}: firmware={version_hex} "
            f"raw_data={data.hex()} raw_id=0x{msg.arbitration_id:08X}"
        )
        return True

    print(f"[RX] {channel} motor={motor_id}: timeout")
    return False


def parse_mapping(text: str) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for group in text.split(","):
        if not group.strip():
            continue
        channel, ids_text = group.split(":", 1)
        mapping[channel.strip()] = [int(item, 0) for item in ids_text.split("/") if item.strip()]
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Read firmware versions from RS motors.")
    parser.add_argument("--channel", help="Query one SocketCAN channel, e.g. can0.")
    parser.add_argument("--motor-id", type=lambda text: int(text, 0), help="Query one motor id.")
    parser.add_argument("--master-id", type=lambda text: int(text, 0), default=0)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--mapping",
        default="can0:1/2/3/4/5/6",
        help="Default: can0:1/2/3/4/5/6",
    )
    args = parser.parse_args()

    if (args.channel is None) != (args.motor_id is None):
        print("[ERROR] --channel 和 --motor-id 需要同时提供，或都不提供以查询默认 6 个电机。")
        return 2

    mapping = {args.channel: [args.motor_id]} if args.channel else parse_mapping(args.mapping)

    print("=== read motor firmware only ===")
    print("只发送版本号读取帧 0x04 + data[0:2]=00 c4；不 enable，不发送 MIT 控制帧。")
    print(f"master_id={args.master_id}")

    buses: dict[str, can.BusABC] = {}
    try:
        for channel in mapping:
            ensure_socketcan_up(channel)
            buses[channel] = can.Bus(interface="socketcan", channel=channel)

        ok = True
        for channel, motor_ids in mapping.items():
            for motor_id in motor_ids:
                if not query_one(buses[channel], channel, motor_id, args.master_id, args.timeout):
                    ok = False
                time.sleep(0.05)
        return 0 if ok else 1
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2
    finally:
        for bus in buses.values():
            bus.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
