#!/usr/bin/env python3
"""Decode RS motor CAN frames from candump output.

This is a read-only/offline decoder. It does not open SocketCAN and does not
transmit any CAN frame.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -50.0, 50.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

MOTOR_TYPES = {
    1: "RS05",
    2: "RS00",
    3: "RS05",
    4: "RS05",
    5: "RS00",
    6: "RS05",
}

TORQUE_RANGE = {
    "RS00": (-14.0, 14.0),
    "RS05": (-5.5, 5.5),
}

COMM_NAMES = {
    0x01: "MIT控制",
    0x02: "状态/反馈",
    0x03: "使能",
    0x04: "失能/读版本",
    0x06: "设置零点",
    0x11: "读参数",
    0x12: "写参数",
    0x15: "特殊/保护帧",
}

FAULT_BITS = [
    (16, "欠压"),
    (17, "驱动故障"),
    (18, "过温"),
    (19, "磁编码故障"),
    (20, "堵转/过载"),
    (21, "未标定"),
]

MODE_NAMES = {
    0: "Reset",
    1: "Calibration",
    2: "Motor",
    3: "Reserved",
}


LINE_RE = re.compile(
    r"(?:\((?P<time>[-+0-9.]+)\)\s+)?"
    r"(?P<iface>\S+)\s+"
    r"(?P<canid>[0-9A-Fa-f]{3,8})\s+"
    r"\[(?P<dlc>\d+)\]\s*"
    r"(?P<data>(?:[0-9A-Fa-f]{2}\s*)*)"
)


def uint_to_float(value: int, low: float, high: float, bits: int = 16) -> float:
    return float(value) * (high - low) / ((1 << bits) - 1) + low


def motor_type(motor_id: int, overrides: dict[int, str]) -> str:
    return overrides.get(motor_id, MOTOR_TYPES.get(motor_id, "RS05")).upper()


def torque_limits(motor_id: int, overrides: dict[int, str]) -> tuple[float, float]:
    return TORQUE_RANGE.get(motor_type(motor_id, overrides), TORQUE_RANGE["RS05"])


def parse_types(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    if not text:
        return result
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        result[int(key, 0)] = value.strip().upper()
    return result


def parse_line(line: str) -> tuple[str | None, str, int, bytes] | None:
    match = LINE_RE.search(line)
    if not match:
        return None
    time_text = match.group("time")
    iface = match.group("iface")
    can_id = int(match.group("canid"), 16)
    dlc = int(match.group("dlc"))
    data_text = match.group("data").strip()
    data = bytes(int(part, 16) for part in data_text.split()) if data_text else b""
    return time_text, iface, can_id, data[:dlc].ljust(8, b"\x00")


def flags_text(can_id: int) -> str:
    flags = [name for bit, name in FAULT_BITS if can_id & (1 << bit)]
    return ",".join(flags) if flags else "无"


def mode_text(can_id: int) -> str:
    mode = (can_id >> 22) & 0x3
    return f"{mode}:{MODE_NAMES.get(mode, '?')}"


def decode_feedback(can_id: int, data: bytes, overrides: dict[int, str]) -> str:
    low_id = can_id & 0xFF
    mid_id = (can_id >> 8) & 0xFF

    if mid_id == 0 and low_id != 0:
        return f"TX 状态请求 target={low_id}"

    motor_id = mid_id
    if data[0] == 0x00 and data[1] == 0xC4:
        version = ".".join(f"{byte:02X}" for byte in data[3:7])
        return (
            f"RX 版本回复 motor={motor_id} master={low_id} marker=0x{data[2]:02X} "
            f"firmware={version} mode={mode_text(can_id)} fault={flags_text(can_id)}"
        )

    pos_u = (data[0] << 8) | data[1]
    vel_u = (data[2] << 8) | data[3]
    tor_u = (data[4] << 8) | data[5]
    tmp_u = (data[6] << 8) | data[7]
    t_min, t_max = torque_limits(motor_id, overrides)

    pos = uint_to_float(pos_u, P_MIN, P_MAX)
    vel = uint_to_float(vel_u, V_MIN, V_MAX)
    torque = uint_to_float(tor_u, t_min, t_max)
    temp = tmp_u * 0.1
    return (
        f"RX 状态 motor={motor_id} type={motor_type(motor_id, overrides)} "
        f"pos={pos:+.4f}rad ({pos * 57.295779513:+.2f}deg) "
        f"vel={vel:+.4f}rad/s torque={torque:+.4f}Nm temp={temp:.1f}C "
        f"mode={mode_text(can_id)} fault={flags_text(can_id)}"
    )


def decode_mit(can_id: int, data: bytes, overrides: dict[int, str]) -> str:
    motor_id = can_id & 0xFF
    torque_u = (can_id >> 8) & 0xFFFF
    pos_u = (data[0] << 8) | data[1]
    vel_u = (data[2] << 8) | data[3]
    kp_u = (data[4] << 8) | data[5]
    kd_u = (data[6] << 8) | data[7]
    t_min, t_max = torque_limits(motor_id, overrides)
    return (
        f"TX MIT target={motor_id} type={motor_type(motor_id, overrides)} "
        f"pos={uint_to_float(pos_u, P_MIN, P_MAX):+.4f}rad "
        f"vel={uint_to_float(vel_u, V_MIN, V_MAX):+.4f}rad/s "
        f"kp={uint_to_float(kp_u, KP_MIN, KP_MAX):.3f} "
        f"kd={uint_to_float(kd_u, KD_MIN, KD_MAX):.3f} "
        f"torque_field={uint_to_float(torque_u, t_min, t_max):+.4f}Nm"
    )


def decode_param(prefix: str, can_id: int, data: bytes) -> str:
    target = can_id & 0xFF
    data_field = (can_id >> 8) & 0xFFFF
    index = data[0] | (data[1] << 8)
    raw_value = data[4:8].hex()
    return f"{prefix} target={target} data_field=0x{data_field:04X} index=0x{index:04X} raw_value={raw_value}"


def decode_frame(iface: str, can_id: int, data: bytes, overrides: dict[int, str]) -> str:
    comm = (can_id >> 24) & 0x1F
    low_id = can_id & 0xFF
    mid_id = (can_id >> 8) & 0xFF
    name = COMM_NAMES.get(comm, f"未知0x{comm:02X}")

    if comm == 0x01:
        detail = decode_mit(can_id, data, overrides)
    elif comm == 0x02:
        detail = decode_feedback(can_id, data, overrides)
    elif comm == 0x03:
        detail = f"TX 使能 target={low_id} master/data_field=0x{((can_id >> 8) & 0xFFFF):04X}"
    elif comm == 0x04:
        if data[0] == 0x00 and data[1] == 0xC4:
            detail = f"TX 读版本 target={low_id} data=00c4..."
        else:
            detail = f"TX 失能 target={low_id} master/data_field=0x{((can_id >> 8) & 0xFFFF):04X}"
    elif comm == 0x06:
        detail = f"TX 设置零点 target={low_id} payload={data.hex()}"
    elif comm == 0x11:
        detail = decode_param("TX/ RX 读参数", can_id, data)
    elif comm == 0x12:
        detail = decode_param("TX 写参数", can_id, data)
    elif comm == 0x15:
        detail = (
            f"特殊/保护帧 mid_id={mid_id} low_id={low_id} "
            f"mode={mode_text(can_id)} fault={flags_text(can_id)} data={data.hex()}"
        )
    else:
        detail = f"raw mid_id={mid_id} low_id={low_id} data={data.hex()}"

    return f"{iface} id=0x{can_id:08X} comm=0x{comm:02X}({name}) {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode RS motor candump lines.")
    parser.add_argument("path", nargs="?", help="candump.log path; omit to read stdin")
    parser.add_argument("--types", default="", help="Override motor types, e.g. 2:RS00,5:RS00")
    parser.add_argument("--only-feedback", action="store_true", help="Only print comm_type 0x02 frames")
    parser.add_argument("--comm", type=lambda text: int(text, 0), help="Only print this comm_type, e.g. 0x15")
    parser.add_argument("--limit", type=int, default=0, help="Max decoded frames to print")
    parser.add_argument("--summary", action="store_true", help="Print frame count summary at the end")
    parser.add_argument("--summary-only", action="store_true", help="Only print the final frame count summary")
    args = parser.parse_args()

    overrides = parse_types(args.types)
    counters: Counter[tuple[str, int]] = Counter()
    printed = 0

    if args.path:
        fp = Path(args.path).open("r", errors="replace")
    else:
        fp = sys.stdin

    try:
        for line in fp:
            parsed = parse_line(line)
            if parsed is None:
                continue
            time_text, iface, can_id, data = parsed
            comm = (can_id >> 24) & 0x1F
            counters[(iface, comm)] += 1
            if args.summary_only:
                continue
            if args.comm is not None and comm != args.comm:
                continue
            if args.only_feedback and comm != 0x02:
                continue
            prefix = f"({time_text}) " if time_text else ""
            print(prefix + decode_frame(iface, can_id, data, overrides))
            printed += 1
            if args.limit and printed >= args.limit:
                break
    finally:
        if fp is not sys.stdin:
            fp.close()

    if args.summary or args.summary_only:
        print("\n=== summary ===")
        for (iface, comm), count in sorted(counters.items()):
            print(f"{iface} comm=0x{comm:02X}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
