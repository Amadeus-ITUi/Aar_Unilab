#!/usr/bin/env python3
"""Sweep one or two P1-P6 targets through the full-device ESD-Link path."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from esd_link_msgs.srv import SetControlMode

from esd_safety_limits import clamp_position_target, LEG_POSITION_LIMITS, POSITION_LIMITS
from esd_wing_sweep import WingSweep, positive_finite


POSITION_LIMITS = LEG_POSITION_LIMITS
WHEELS = {3, 6}


def csv_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("motor ids must be comma-separated integers") from exc
    invalid = (
        len(result) not in (1, 2)
        or len(set(result)) != len(result)
        or any(item < 1 or item > 6 for item in result)
    )
    if invalid:
        raise argparse.ArgumentTypeError("provide one or two unique motor ids in P1..P6")
    return result


def csv_signs(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("signs must be comma-separated numbers") from exc
    if any(not math.isfinite(item) or abs(item) < 1e-6 for item in result):
        raise argparse.ArgumentTypeError("signs must be finite and non-zero")
    return [1.0 if item > 0 else -1.0 for item in result]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESD-Link single/dual motor chirp; all P1-P8 remain in every command")
    parser.add_argument(
        "--motor-ids", type=csv_ints, required=True,
        help="one or two ids, e.g. 1 or 2,5")
    parser.add_argument(
        "--signs", type=csv_signs, default=None,
        help="optional comma-separated target signs")
    parser.add_argument("--start-frequency", type=positive_finite, default=0.1)
    parser.add_argument("--end-frequency", type=positive_finite, default=5.0)
    parser.add_argument("--amplitude", type=positive_finite, default=0.2)
    parser.add_argument("--duration", type=positive_finite, default=40.0)
    parser.add_argument("--return-speed", type=positive_finite, default=0.10)
    parser.add_argument("--hz", type=positive_finite, default=200.0)
    parser.add_argument("--kp", type=positive_finite, default=2.0)
    parser.add_argument("--kd", type=positive_finite, default=0.1)
    parser.add_argument("--button-index", type=int, default=0)
    parser.add_argument("--confirm-timeout", type=positive_finite, default=120.0)
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    if args.hz > 250.0:
        parser.error("ESD-Link publish rate must be <= 250 Hz")
    if args.start_frequency > args.end_frequency:
        parser.error("start frequency must not exceed end frequency")
    if args.signs is None:
        args.signs = [-1.0 if motor <= 3 else 1.0 for motor in args.motor_ids]
    if len(args.signs) != len(args.motor_ids):
        parser.error("--signs length must match --motor-ids")
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = WingSweep(args)
    armed = False
    try:
        state = node.wait_state()
        node.initialize(state)
        initial = list(node.command.position_rad)
        recovery = list(initial)
        for port in POSITION_LIMITS:
            recovery[port - 1] = clamp_position_target(port, recovery[port - 1])
        centers = {}
        for motor in args.motor_ids:
            if motor in POSITION_LIMITS:
                low, high = POSITION_LIMITS[motor]
                center = clamp_position_target(
                    motor, initial[motor - 1], margin=args.amplitude)
                centers[motor] = center
                recovery[motor - 1] = center
                assert center - args.amplitude >= low
                assert center + args.amplitude <= high
            else:
                centers[motor] = initial[motor - 1]

        node.wait_button("P1-P8 全在线且 DISABLED。确认台架和硬件断电手段后，第一次按 A 整体使能。")
        node.set_mode(SetControlMode.Request.SWEEP)
        armed = True
        node.recording = True
        for motor in args.motor_ids:
            node.command.kp[motor - 1] = 0.0 if motor in WHEELS else args.kp
            node.command.kd[motor - 1] = args.kd

        distance = max(
            abs(initial[port - 1] - recovery[port - 1])
            for port in POSITION_LIMITS
        )
        ramp_duration = max(1.0, distance / args.return_speed)

        def recover(elapsed: float, total: float) -> None:
            alpha = min(elapsed / total, 1.0)
            for port in POSITION_LIMITS:
                index = port - 1
                node.command.position_rad[index] = initial[index] + (
                    recovery[index] - initial[index]) * alpha

        node.run_phase(ramp_duration, recover)
        for port in POSITION_LIMITS:
            node.command.position_rad[port - 1] = recovery[port - 1]
        node.hold_for_button("越界关节已只向内回收到安全区；检查全端口后第二次按 A。")
        node.hold_for_button("最终检查完成；第三次按 A 开始正式 chirp。")

        def chirp(elapsed: float, total: float) -> None:
            phase = (
                args.start_frequency * elapsed
                + 0.5 * (args.end_frequency - args.start_frequency)
                * elapsed * elapsed / total
            )
            value = args.amplitude * math.sin(2.0 * math.pi * phase)
            for motor, sign in zip(args.motor_ids, args.signs):
                if motor in WHEELS:
                    node.command.velocity_rad_s[motor - 1] = sign * value
                else:
                    node.command.position_rad[motor - 1] = centers[motor] + sign * value

        node.run_phase(args.duration, chirp)
        for motor in args.motor_ids:
            node.command.position_rad[motor - 1] = centers[motor]
            node.command.velocity_rad_s[motor - 1] = 0.0
        node.run_phase(0.25, lambda _elapsed, _total: None)
        node.set_mode(SetControlMode.Request.DISABLED)
        armed = False
        print("P1-P8 整体失能已确认。", flush=True)
        if not args.log:
            selected = "_".join(str(motor) for motor in args.motor_ids)
            args.log = f"logs/esd_sweep_motor_{selected}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path = node.save()
        print(f"选择端口扫频完成：{path}，rows={len(node.rows)}；温度=unavailable", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", flush=True)
        return 1
    finally:
        if armed:
            try:
                node.set_mode(SetControlMode.Request.DISABLED)
                print("P1-P8 整体失能已确认。", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] 整体失能未确认：{exc}；请立即使用硬件断电。", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
