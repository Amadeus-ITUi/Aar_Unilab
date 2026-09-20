#!/usr/bin/env python3
"""Continuously read left/right calf feedback with CAN 0x02 only.

This script talks directly to SocketCAN. It never starts ROS, sends enable or
MIT control frames, calls services, or publishes anything into the policy.
Motor 2 is not flipped and motor 5 is flipped, matching motors.yaml. The
printed values therefore use the same sign conversion as the policy topic.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass

import can


MOTOR_IDS = (2, 5)
DEFAULT_CALF_ANGLE = 0.98338
FLIPPED = {2: False, 5: True}
P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -50.0, 50.0
T_MIN, T_MAX = -14.0, 14.0


@dataclass(frozen=True)
class Status:
    raw_position: float
    policy_position: float
    velocity: float
    torque: float


def build_can_id(comm_type: int, motor_id: int, master_id: int = 0) -> int:
    return ((comm_type & 0x1F) << 24) | ((master_id & 0xFFFF) << 8) | (motor_id & 0xFF)


def uint_to_float(value: int, low: float, high: float) -> float:
    return float(value) * (high - low) / 65535.0 + low


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


def parse_status(
    message: can.Message,
    motor_id: int,
    default_angle: float,
    flipped: bool,
) -> Status | None:
    if not message.is_extended_id:
        return None

    can_id = message.arbitration_id & 0x1FFFFFFF
    comm_type = (can_id >> 24) & 0x1F
    response_motor_id = (can_id >> 8) & 0xFF
    response_low_id = can_id & 0xFF

    # Feedback is 0x0200MM00. Our request is 0x020000MM, so ignore it.
    if comm_type != 0x02 or response_motor_id != motor_id or response_low_id != 0:
        return None

    data = bytes(message.data).ljust(8, b"\x00")
    raw_position = uint_to_float((data[0] << 8) | data[1], P_MIN, P_MAX)
    raw_velocity = uint_to_float((data[2] << 8) | data[3], V_MIN, V_MAX)
    raw_torque = uint_to_float((data[4] << 8) | data[5], T_MIN, T_MAX)
    sign = -1.0 if flipped else 1.0

    return Status(
        raw_position=raw_position,
        policy_position=sign * raw_position - default_angle,
        velocity=sign * raw_velocity,
        torque=sign * raw_torque,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly poll motor 2 and 5 with CAN 0x02 without enabling them."
    )
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--master-id", type=lambda value: int(value, 0), default=0)
    parser.add_argument(
        "--request-hz",
        type=float,
        default=1000.0,
        help="total 0x02 request-frame rate, round-robin across motors 2 and 5",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=400.0,
        help="maximum print rate per motor; 0 prints every received feedback frame",
    )
    parser.add_argument("--default-angle", type=float, default=DEFAULT_CALF_ANGLE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.request_hz <= 0.0:
        print("[ERROR] --request-hz 必须大于 0", file=sys.stderr)
        return 2
    if args.print_hz < 0.0:
        print("[ERROR] --print-hz 不能小于 0", file=sys.stderr)
        return 2

    try:
        ensure_socketcan_up(args.channel)
        bus = can.Bus(interface="socketcan", channel=args.channel)
    except Exception as exc:
        print(f"[ERROR] CAN 初始化失败: {exc}", file=sys.stderr)
        return 2

    request_messages = {
        motor_id: can.Message(
            arbitration_id=build_can_id(0x02, motor_id, args.master_id),
            data=bytes(8),
            is_extended_id=True,
        )
        for motor_id in MOTOR_IDS
    }
    last_print = {motor_id: 0.0 for motor_id in MOTOR_IDS}
    last_feedback = {motor_id: 0.0 for motor_id in MOTOR_IDS}
    print_period = 0.0 if args.print_hz == 0.0 else 1.0 / args.print_hz
    request_period = 1.0 / args.request_hz
    next_request = time.perf_counter()
    request_index = 0
    sent_count = 0

    print(
        f"[INFO] {args.channel}: 只发送 motor 2/5 的 CAN 0x02 状态请求；"
        "不发送使能、MIT 控制、清零或策略消息。",
        file=sys.stderr,
    )
    print(
        f"[INFO] 请求总频率={args.request_hz:.1f}Hz，打印频率="
        f"{'每帧' if args.print_hz == 0.0 else f'{args.print_hz:.1f}Hz'}；"
        f"策略位置=方向转换后位置 - {args.default_angle:.3f}rad。",
        file=sys.stderr,
    )
    print(
        "[INFO] 方向映射: motor_2 未翻转；motor_5 翻转。",
        file=sys.stderr,
    )

    try:
        while True:
            now = time.perf_counter()
            if now >= next_request:
                motor_id = MOTOR_IDS[request_index % len(MOTOR_IDS)]
                bus.send(request_messages[motor_id])
                request_index += 1
                sent_count += 1
                next_request += request_period
                if next_request < now - request_period:
                    next_request = now + request_period

            message = bus.recv(timeout=0.001)
            if message is not None:
                motor_id = (message.arbitration_id >> 8) & 0xFF
                if motor_id in MOTOR_IDS:
                    status = parse_status(
                        message,
                        motor_id,
                        args.default_angle,
                        FLIPPED[motor_id],
                    )
                    if status is not None:
                        feedback_now = time.perf_counter()
                        last_feedback[motor_id] = feedback_now
                        if feedback_now - last_print[motor_id] >= print_period:
                            side = "left_calf" if motor_id == 2 else "right_calf"
                            print(
                                f"motor_{motor_id}({side}) "
                                f"position={status.policy_position:+.6f} "
                                f"velocity={status.velocity:+.6f} "
                                f"torque={status.torque:+.6f}",
                                flush=True,
                            )
                            last_print[motor_id] = feedback_now

            # Keep the loop responsive without adding another CAN frame.
            sleep_time = next_request - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(min(sleep_time, 0.0005))
    except KeyboardInterrupt:
        print(f"[INFO] 已停止，只读请求帧={sent_count}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"[ERROR] CAN 运行失败: {exc}", file=sys.stderr)
        return 1
    finally:
        bus.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
