#!/usr/bin/env python3
"""Guided first-enable test that holds the current P1-P8 pose at low gains."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from esd_link_msgs.srv import SetControlMode

from esd_wing_sweep import WingSweep, positive_finite


TRACKED_POSITION_INDICES = (0, 1, 3, 4, 6, 7)
LOW_KP = (0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0)
LOW_KD = (0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1)


def wait_for_authorization(node: WingSweep, operator_approved: bool) -> None:
    prompt = (
        "确认样机固定、P1-P8 无干涉、硬件断电可立即操作；"
        "开始后整体低增益保持当前位。"
    )
    if not operator_approved:
        node.wait_button(prompt + "释放后按 A 开始。")
        return
    print(prompt, flush=True)
    print(
        "已通过 --operator-approved 记录现场授权；3 秒后开始，"
        "可随时 Ctrl-C 或硬件断电。",
        flush=True,
    )
    deadline = time.monotonic() + 3.0
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        health_error = node.state_health_error()
        if health_error is not None:
            raise RuntimeError("授权倒计时期间状态失效：" + health_error)


def verify_hold_state(
    initial: list[float],
    current: list[float],
    velocities: list[float],
    position_tolerance: float,
    velocity_limit: float,
) -> None:
    for index in TRACKED_POSITION_INDICES:
        deviation = abs(current[index] - initial[index])
        if not math.isfinite(deviation) or deviation > position_tolerance:
            raise RuntimeError(
                f"P{index + 1} 位置偏移 {deviation:.4f} rad 超过 "
                f"{position_tolerance:.4f} rad"
            )
    for index, velocity in enumerate(velocities):
        if not math.isfinite(velocity) or abs(velocity) > velocity_limit:
            raise RuntimeError(
                f"P{index + 1} 速度 {velocity:.4f} rad/s 超过 "
                f"{velocity_limit:.4f} rad/s"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESD-Link guided low-gain current-pose hold for all P1-P8")
    parser.add_argument("--duration", type=positive_finite, default=5.0)
    parser.add_argument("--hz", type=positive_finite, default=200.0)
    parser.add_argument("--position-tolerance", type=positive_finite, default=0.08)
    parser.add_argument("--velocity-limit", type=positive_finite, default=2.0)
    parser.add_argument("--button-index", type=int, default=0)
    parser.add_argument("--confirm-timeout", type=positive_finite, default=120.0)
    parser.add_argument(
        "--operator-approved", action="store_true",
        help="record explicit onsite approval and use a 3-second countdown instead of A")
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    if args.hz > 250.0:
        parser.error("ESD-Link publish rate must be <= 250 Hz")
    # WingSweep owns these attributes even though this tool never chirps.
    args.kp = LOW_KP[6]
    args.kd = LOW_KD[6]
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = WingSweep(args)
    node.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
    armed = False
    saved = False
    try:
        state = node.wait_state()
        node.initialize(state)
        initial = list(node.command.position_rad)
        node.command.mode = node.command.MANUAL_TEST
        node.command.kp = list(LOW_KP)
        node.command.kd = list(LOW_KD)
        print(
            "当前上位机坐标 P1-P8: "
            + ", ".join(f"{value:+.4f}" for value in initial),
            flush=True,
        )
        print(
            "低增益 Kp=" + str(list(LOW_KP)) + " Kd=" + str(list(LOW_KD)),
            flush=True,
        )
        wait_for_authorization(node, args.operator_approved)
        node.set_mode(SetControlMode.Request.MANUAL_TEST)
        armed = True
        node.recording = True

        def hold_check(_elapsed: float, _total: float) -> None:
            current_state = node.healthy_state()
            if current_state is None:
                raise RuntimeError("LowerState 或 bridge 控制状态无效")
            verify_hold_state(
                initial,
                list(current_state.position_rad),
                list(current_state.velocity_rad_s),
                args.position_tolerance,
                args.velocity_limit,
            )

        node.run_phase(args.duration, hold_check)
        node.set_mode(SetControlMode.Request.DISABLED)
        armed = False
        print("P1-P8 整体失能已确认。", flush=True)
        if not args.log:
            args.log = f"logs/esd_hold_current_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path = node.save()
        saved = True
        print(
            f"当前位低增益保持通过：duration={args.duration:.2f}s, "
            f"rows={len(node.rows)}, log={path}",
            flush=True,
        )
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
                print(
                    f"[ERROR] 整体失能未确认：{exc}；请立即使用硬件断电。",
                    flush=True,
                )
        if node.rows and not saved:
            try:
                if not args.log:
                    args.log = f"logs/esd_hold_current_failed_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                path = node.save()
                print(f"失败过程已保存：{path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] 失败记录保存失败：{exc}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
