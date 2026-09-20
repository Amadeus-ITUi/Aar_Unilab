#!/usr/bin/env python3
"""Sweep motor 7 and motor 8 together at 200 Hz.

Motor 7 is the right motor: -5 deg to +45 deg.
Motor 8 is the left motor:  -45 deg to +5 deg.

The script does not set hardware zero. It enables only motors 7 and 8, switches
them to MIT/move-control mode, ramps each target to its sweep center, then runs
a mirrored sine sweep.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import can


P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -50.0, 50.0
T_MIN, T_MAX = -14.0, 14.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0
IDX_RUN_MODE = 0x7005
RUN_MODE_MIT = 0
DEG2RAD = math.pi / 180.0


@dataclass
class MotorPlan:
    motor_id: int
    name: str
    center: float
    amplitude: float
    sign: float
    target: float = 0.0
    pos: float | None = None
    vel: float | None = None
    torque: float | None = None
    temp: float | None = None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_can_id(cmd_type: int, motor_id: int, data_field: int = 0) -> int:
    return ((cmd_type & 0x1F) << 24) | ((data_field & 0xFFFF) << 8) | (motor_id & 0xFF)


def float_to_uint(value: float, low: float, high: float, bits: int = 16) -> int:
    value = clamp(value, low, high)
    return int((value - low) * ((1 << bits) - 1) / (high - low))


def uint_to_float(value: int, low: float, high: float, bits: int = 16) -> float:
    return float(value) * (high - low) / ((1 << bits) - 1) + low


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
        print(f"[ERROR] {channel} 未处于 UP 状态:\n{result.stdout}")
        sys.exit(2)


def send_frame(bus: can.BusABC, cmd_type: int, motor_id: int, data_field: int, payload: bytes) -> None:
    bus.send(
        can.Message(
            arbitration_id=build_can_id(cmd_type, motor_id, data_field),
            data=payload.ljust(8, b"\x00")[:8],
            is_extended_id=True,
        )
    )


def encode_set_param_u8(index: int, value: int) -> bytes:
    payload = bytearray(8)
    payload[0] = index & 0xFF
    payload[1] = (index >> 8) & 0xFF
    payload[4] = value & 0xFF
    return bytes(payload)


def encode_mit(pos: float, vel: float, kp: float, kd: float) -> bytes:
    p_uint = float_to_uint(pos, P_MIN, P_MAX)
    v_uint = float_to_uint(vel, V_MIN, V_MAX)
    kp_uint = float_to_uint(kp, KP_MIN, KP_MAX)
    kd_uint = float_to_uint(kd, KD_MIN, KD_MAX)
    return bytes(
        [
            (p_uint >> 8) & 0xFF,
            p_uint & 0xFF,
            (v_uint >> 8) & 0xFF,
            v_uint & 0xFF,
            (kp_uint >> 8) & 0xFF,
            kp_uint & 0xFF,
            (kd_uint >> 8) & 0xFF,
            kd_uint & 0xFF,
        ]
    )


def parse_motor_status(msg: can.Message, motor_id: int) -> tuple[float, float, float, float] | None:
    if not msg.is_extended_id:
        return None
    can_id = msg.arbitration_id & 0x1FFFFFFF
    if ((can_id >> 24) & 0x1F) != 0x02:
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


def request_status(bus: can.BusABC, motor_id: int, master_id: int, timeout: float) -> tuple[float, float, float, float] | None:
    send_frame(bus, 0x02, motor_id, master_id, bytes(8))
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        msg = bus.recv(timeout=0.02)
        if msg is None:
            continue
        parsed = parse_motor_status(msg, motor_id)
        if parsed is not None:
            return parsed
    return None


def enable_motor(bus: can.BusABC, motor_id: int, master_id: int) -> None:
    for _ in range(5):
        send_frame(bus, 0x03, motor_id, master_id, bytes(8))
        time.sleep(0.01)
    time.sleep(0.1)
    send_frame(bus, 0x12, motor_id, master_id, encode_set_param_u8(IDX_RUN_MODE, RUN_MODE_MIT))
    time.sleep(0.05)


def disable_motor(bus: can.BusABC, motor_id: int, master_id: int) -> None:
    send_frame(bus, 0x04, motor_id, master_id, bytes(8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep right motor 7 and left motor 8 together.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--master-id", type=lambda text: int(text, 0), default=0)
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque", type=float, default=0.0)
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--pre-hold", type=float, default=2.0)
    parser.add_argument("--return-speed", type=float, default=0.35)
    parser.add_argument("--duration", type=float, default=0.0, help="Sweep duration in seconds. 0 means run until Ctrl+C.")
    parser.add_argument("--log", default="", help="CSV log path. Empty means auto path under logs/.")
    parser.add_argument("--status-timeout", type=float, default=1.0)
    parser.add_argument("--keep-enabled-on-exit", action="store_true")
    return parser.parse_args()


def update_feedback(msg: can.Message, motors: list[MotorPlan]) -> None:
    for motor in motors:
        parsed = parse_motor_status(msg, motor.motor_id)
        if parsed is None:
            continue
        motor.pos, motor.vel, motor.torque, motor.temp = parsed
        return


def fmt_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.9f}"


def make_log_path(requested: str) -> str:
    if requested:
        return requested
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join("logs", f"sweep_motor78_{stamp}.csv")


def main() -> int:
    args = parse_args()
    if args.hz <= 0 or args.freq <= 0 or args.freq > 3.0 or args.return_speed <= 0:
        print("[ERROR] hz 和 return-speed 必须 > 0，扑翼 freq 必须在 (0, 3] Hz")
        return 2
    if args.pre_hold < 0 or args.duration < 0:
        print("[ERROR] pre-hold 和 duration 不能为负")
        return 2
    if not all(math.isfinite(x) for x in (args.kp, args.kd, args.torque, args.freq, args.return_speed)):
        print("[ERROR] 参数不能是 NaN/Inf")
        return 2

    motors = [
        MotorPlan(motor_id=7, name="right", center=20.0 * DEG2RAD, amplitude=25.0 * DEG2RAD, sign=1.0),
        MotorPlan(motor_id=8, name="left", center=-20.0 * DEG2RAD, amplitude=25.0 * DEG2RAD, sign=-1.0),
    ]

    ensure_socketcan_up(args.channel)
    bus = can.Bus(interface="socketcan", channel=args.channel)
    period = 1.0 / args.hz
    max_step = args.return_speed * period
    torque_uint = float_to_uint(args.torque, T_MIN, T_MAX)
    sent = 0
    last_log = time.perf_counter()
    center_reached_time = None
    sweep_start = None
    stop_after_center = False
    log_path = make_log_path(args.log)

    print("=== motor 7/8 mirrored sweep ===")
    print(f"channel={args.channel} hz={args.hz:.1f} kp={args.kp:.3f} kd={args.kd:.3f} freq={args.freq:.3f}Hz")
    print("motor7/right: -5 deg to +45 deg, center=+20 deg, amplitude=25 deg")
    print("motor8/left : -45 deg to +5 deg, center=-20 deg, amplitude=25 deg")
    print("不设置零点；只 enable 7/8，先回各自 center，再同步正弦扫频。")
    print(f"CSV log: {log_path}")

    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "t",
                "phase",
                "sent",
                "m7_target",
                "m7_pos",
                "m7_vel",
                "m7_torque",
                "m7_temp",
                "m8_target",
                "m8_pos",
                "m8_vel",
                "m8_torque",
                "m8_temp",
            ]
        )

        for motor in motors:
            status = request_status(bus, motor.motor_id, args.master_id, args.status_timeout)
            if status is None:
                print(f"[ERROR] 启动前没有读到 {motor.motor_id} 号电机状态，停止。")
                return 1
            motor.pos, motor.vel, motor.torque, motor.temp = status
            motor.target = motor.pos
            print(
                f"[INIT] motor={motor.motor_id} {motor.name} "
                f"pos={motor.pos:+.6f} rad vel={motor.vel:+.6f} rad/s torque={motor.torque:+.6f} Nm temp={motor.temp:.1f} C"
            )

        for motor in motors:
            enable_motor(bus, motor.motor_id, args.master_id)
        print("[INFO] motors enabled, ramping targets to centers.")

        next_tick = time.perf_counter()
        control_start = next_tick
        while True:
            now = time.perf_counter()
            if now >= next_tick:
                if sweep_start is not None:
                    elapsed = now - sweep_start
                    done = args.duration > 0 and elapsed >= args.duration
                    for motor in motors:
                        if done:
                            motor.target = motor.center
                            stop_after_center = True
                        else:
                            motor.target = motor.center + motor.sign * motor.amplitude * math.sin(2.0 * math.pi * args.freq * elapsed)
                else:
                    all_centered = True
                    for motor in motors:
                        error = motor.target - motor.center
                        if abs(error) <= max_step:
                            motor.target = motor.center
                        else:
                            motor.target -= math.copysign(max_step, error)
                            all_centered = False
                    if all_centered:
                        if center_reached_time is None:
                            center_reached_time = now
                        if now - center_reached_time >= args.pre_hold:
                            sweep_start = now
                            print("[INFO] sweep started.")

                for motor in motors:
                    payload = encode_mit(motor.target, 0.0, args.kp, args.kd)
                    send_frame(bus, 0x01, motor.motor_id, torque_uint, payload)
                    sent += 1
                phase = "sweep" if sweep_start is not None and not stop_after_center else "return_or_hold"
                writer.writerow(
                    [
                        f"{now - control_start:.9f}",
                        phase,
                        sent,
                        f"{motors[0].target:.9f}",
                        fmt_optional(motors[0].pos),
                        fmt_optional(motors[0].vel),
                        fmt_optional(motors[0].torque),
                        fmt_optional(motors[0].temp),
                        f"{motors[1].target:.9f}",
                        fmt_optional(motors[1].pos),
                        fmt_optional(motors[1].vel),
                        fmt_optional(motors[1].torque),
                        fmt_optional(motors[1].temp),
                    ]
                )
                next_tick += period
                if stop_after_center:
                    print("[INFO] sweep duration finished, targets returned to centers.")
                    break

            msg = bus.recv(timeout=0.001)
            if msg is not None:
                update_feedback(msg, motors)

            if now - last_log >= 1.0:
                parts = []
                for motor in motors:
                    pos_text = "nan" if motor.pos is None else f"{motor.pos:+.6f}"
                    parts.append(f"m{motor.motor_id}: target={motor.target:+.6f} pos={pos_text}")
                print(f"[STAT] sent={sent} " + " | ".join(parts))
                last_log = now

            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.001))
            elif sleep_time < -period:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received.")
        return 0
    finally:
        try:
            log_file.close()
            print(f"[INFO] CSV saved: {log_path}")
        except Exception:
            pass
        if not args.keep_enabled_on_exit:
            for motor in motors:
                disable_motor(bus, motor.motor_id, args.master_id)
            print("[INFO] motors disabled on exit.")
        bus.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
