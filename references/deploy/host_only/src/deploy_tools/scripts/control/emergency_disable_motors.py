#!/usr/bin/env python3
"""Repeatedly send the RS-series disable command to a set of motors.

This is the final shutdown fallback used after all motor-control processes have
stopped.  A successful SocketCAN send confirms that the kernel accepted each
frame; the RS protocol does not provide a disable acknowledgement.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import can


DISABLE_COMMAND = 0x04


def build_can_id(command: int, motor_id: int, master_id: int = 0) -> int:
    return ((command & 0x1F) << 24) | ((master_id & 0xFFFF) << 8) | (motor_id & 0xFF)


def parse_motor_ids(text: str) -> list[int]:
    ids: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        motor_id = int(item, 0)
        if not 1 <= motor_id <= 0xFF:
            raise argparse.ArgumentTypeError(f"motor ID 超出范围: {motor_id}")
        if motor_id not in ids:
            ids.append(motor_id)
    if not ids:
        raise argparse.ArgumentTypeError("至少需要一个 motor ID")
    return ids


def socketcan_is_up(channel: str) -> bool:
    result = subprocess.run(
        ["ip", "-details", "link", "show", channel],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] 找不到 SocketCAN 接口 {channel}: {result.stderr.strip()}", file=sys.stderr)
        return False
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    if "UP" not in first_line:
        print(f"[ERROR] SocketCAN 接口 {channel} 未处于 UP 状态", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="重复失能 RS 系列电机（通信类型 0x04）")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor-ids", type=parse_motor_ids, default=parse_motor_ids("1,2,3,4,5,6,7,8"))
    parser.add_argument("--master-id", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true", help="只打印帧，不访问 CAN 总线")
    args = parser.parse_args()

    if not 1 <= args.retries <= 100:
        parser.error("--retries 必须在 1..100 之间")
    if not 0.0 <= args.interval <= 1.0:
        parser.error("--interval 必须在 0..1 秒之间")
    if not 0 <= args.master_id <= 0xFFFF:
        parser.error("--master-id 必须在 0..0xffff 之间")

    frames = {
        motor_id: build_can_id(DISABLE_COMMAND, motor_id, args.master_id)
        for motor_id in args.motor_ids
    }
    print(
        f"失能目标: channel={args.channel} motors={','.join(map(str, args.motor_ids))} "
        f"retries={args.retries} interval={args.interval:.3f}s"
    )

    if args.dry_run:
        for motor_id, arbitration_id in frames.items():
            print(f"[DRY-RUN] motor={motor_id} id=0x{arbitration_id:08X} data=0000000000000000")
        return 0

    if not socketcan_is_up(args.channel):
        return 2

    failures: dict[int, int] = {motor_id: 0 for motor_id in args.motor_ids}
    bus = can.Bus(interface="socketcan", channel=args.channel)
    try:
        for attempt in range(args.retries):
            for motor_id, arbitration_id in frames.items():
                try:
                    bus.send(
                        can.Message(
                            arbitration_id=arbitration_id,
                            data=bytes(8),
                            is_extended_id=True,
                        ),
                        timeout=0.1,
                    )
                except can.CanError as exc:
                    failures[motor_id] += 1
                    print(
                        f"[ERROR] motor={motor_id} 第 {attempt + 1}/{args.retries} 次失能发送失败: {exc}",
                        file=sys.stderr,
                    )
            if attempt + 1 < args.retries:
                time.sleep(args.interval)
    finally:
        bus.shutdown()

    failed = [motor_id for motor_id, count in failures.items() if count == args.retries]
    if failed:
        print(f"[ERROR] 以下电机没有任何一次失能帧发送成功: {failed}", file=sys.stderr)
        return 1
    partial = {motor_id: count for motor_id, count in failures.items() if count}
    if partial:
        print(f"[WARN] 部分发送失败但每个电机至少成功一次: {partial}")
    print("[OK] 所有目标电机的重复失能帧均已提交到 SocketCAN")
    print("[NOTE] RS 协议无失能确认帧；仍须依靠急停/断电作为最终安全保障")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
