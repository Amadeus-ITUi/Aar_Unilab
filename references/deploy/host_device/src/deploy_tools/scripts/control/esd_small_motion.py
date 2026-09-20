#!/usr/bin/env python3
"""Guided A2 positive/negative small-motion check for one ESD-Link port."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from esd_link_msgs.srv import SetControlMode

from esd_hold_current import LOW_KD, LOW_KP, TRACKED_POSITION_INDICES
from esd_safety_limits import POSITION_LIMITS
from esd_wing_sweep import WingSweep, positive_finite


POSITION_PORTS = (1, 2, 4, 5, 7, 8)
WHEEL_PORTS = (3, 6)


def position_targets(port: int, initial: float, delta: float) -> tuple[float, float]:
    if port not in POSITION_PORTS:
        raise ValueError(f"P{port} 不是位置型 A2 端口")
    lower, upper = POSITION_LIMITS[port]
    positive = initial + delta
    negative = initial - delta
    if negative < lower or positive > upper:
        raise RuntimeError(
            f"P{port} 当前位 {initial:.4f} rad 的 ±{delta:.4f} rad 动作"
            f"超出软件限位 [{lower:.4f}, {upper:.4f}]"
        )
    return positive, negative


def verify_motion_state(
    initial: list[float],
    current: list[float],
    velocities: list[float],
    selected_port: int,
    position_envelope: float,
    velocity_limit: float,
) -> None:
    selected_index = selected_port - 1
    for index in TRACKED_POSITION_INDICES:
        allowed = position_envelope if index == selected_index else 0.03
        deviation = abs(current[index] - initial[index])
        if not math.isfinite(deviation) or deviation > allowed:
            raise RuntimeError(
                f"P{index + 1} 相对初始位偏移 {deviation:.4f} rad "
                f"超过 {allowed:.4f} rad"
            )
    for index, velocity in enumerate(velocities):
        if not math.isfinite(velocity) or abs(velocity) > velocity_limit:
            raise RuntimeError(
                f"P{index + 1} 速度 {velocity:.4f} rad/s "
                f"超过 {velocity_limit:.4f} rad/s"
            )


def verify_response(port: int, positive_peak: float, negative_peak: float,
                    minimum_response: float) -> None:
    if positive_peak < minimum_response:
        raise RuntimeError(
            f"P{port} 正向反馈仅 {positive_peak:.4f}，"
            f"小于最小响应 {minimum_response:.4f}"
        )
    if negative_peak > -minimum_response:
        raise RuntimeError(
            f"P{port} 反向反馈仅 {negative_peak:.4f}，"
            f"未达到 -{minimum_response:.4f}"
        )


def wait_for_authorization(node: WingSweep, args: argparse.Namespace,
                           prompt: str) -> None:
    if not args.operator_approved:
        node.wait_button(prompt)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESD-Link guided A2 small motion for one P1-P8 port")
    parser.add_argument("--port", type=int, required=True, choices=range(1, 9))
    parser.add_argument("--hz", type=positive_finite, default=200.0)
    parser.add_argument("--position-delta", type=positive_finite, default=0.03)
    parser.add_argument("--wheel-speed", type=positive_finite, default=0.5)
    parser.add_argument("--ramp-duration", type=positive_finite, default=1.5)
    parser.add_argument("--hold-duration", type=positive_finite, default=0.5)
    parser.add_argument("--position-envelope", type=positive_finite, default=0.06)
    parser.add_argument("--velocity-limit", type=positive_finite, default=2.0)
    parser.add_argument("--minimum-position-response", type=positive_finite, default=0.003)
    parser.add_argument("--minimum-velocity-response", type=positive_finite, default=0.05)
    parser.add_argument("--button-index", type=int, default=0)
    parser.add_argument("--confirm-timeout", type=positive_finite, default=120.0)
    parser.add_argument(
        "--operator-approved", action="store_true",
        help="record explicit onsite approval and use a 3-second countdown instead of A")
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    if args.hz > 250.0:
        parser.error("ESD-Link publish rate must be <= 250 Hz")
    if args.position_delta > 0.03:
        parser.error("A2 position delta must be <= 0.03 rad")
    if args.wheel_speed > 0.5:
        parser.error("A2 wheel speed must be <= 0.5 rad/s")
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
        port_index = args.port - 1
        if args.port in POSITION_PORTS:
            positive_target, negative_target = position_targets(
                args.port, initial[port_index], args.position_delta)
            motion_text = f"位置 ±{args.position_delta:.4f} rad"
        else:
            positive_target, negative_target = args.wheel_speed, -args.wheel_speed
            motion_text = f"速度 ±{args.wheel_speed:.4f} rad/s"

        print(
            f"A2 P{args.port} 小动作：{motion_text}；其余端口保持当前位。",
            flush=True,
        )
        print(
            "低增益 Kp=" + str(list(LOW_KP)) + " Kd=" + str(list(LOW_KD)),
            flush=True,
        )
        wait_for_authorization(
            node, args,
            "确认样机固定、该端口两个方向均无干涉、硬件断电可立即操作；"
            f"释放后按 A，工具将整体低增益使能、执行 P{args.port} 正向动作、"
            "自动回位并立即整体失能。"
        )
        node.set_mode(SetControlMode.Request.MANUAL_TEST)
        armed = True
        node.recording = True

        def safety_check() -> None:
            current_state = node.healthy_state()
            if current_state is None:
                raise RuntimeError("LowerState 或 bridge 控制状态无效")
            verify_motion_state(
                initial,
                list(current_state.position_rad),
                list(current_state.velocity_rad_s),
                args.port,
                args.position_envelope,
                args.velocity_limit,
            )

        node.run_phase(0.5, lambda _elapsed, _total: safety_check())

        positive_peak = -math.inf

        def positive_motion(elapsed: float, total: float) -> None:
            nonlocal positive_peak
            safety_check()
            current_state = node.state
            assert current_state is not None
            if args.port in POSITION_PORTS:
                alpha = min(elapsed / total, 1.0)
                node.command.position_rad[port_index] = initial[port_index] + (
                    positive_target - initial[port_index]) * alpha
                response = current_state.position_rad[port_index] - initial[port_index]
            else:
                node.command.velocity_rad_s[port_index] = positive_target
                response = current_state.velocity_rad_s[port_index]
            positive_peak = max(positive_peak, response)

        node.run_phase(args.ramp_duration, positive_motion)
        node.command.position_rad[port_index] = (
            positive_target if args.port in POSITION_PORTS else initial[port_index]
        )
        node.command.velocity_rad_s[port_index] = (
            positive_target if args.port in WHEEL_PORTS else 0.0
        )

        def positive_hold(_elapsed: float, _total: float) -> None:
            nonlocal positive_peak
            safety_check()
            current_state = node.state
            assert current_state is not None
            response = (
                current_state.position_rad[port_index] - initial[port_index]
                if args.port in POSITION_PORTS
                else current_state.velocity_rad_s[port_index]
            )
            positive_peak = max(positive_peak, response)

        node.run_phase(args.hold_duration, positive_hold)

        def return_to_initial(elapsed: float, total: float) -> None:
            safety_check()
            alpha = min(elapsed / total, 1.0)
            if args.port in POSITION_PORTS:
                node.command.position_rad[port_index] = positive_target + (
                    initial[port_index] - positive_target) * alpha
            else:
                node.command.velocity_rad_s[port_index] = positive_target * (1.0 - alpha)

        node.run_phase(args.ramp_duration, return_to_initial)
        node.command.position_rad[port_index] = initial[port_index]
        node.command.velocity_rad_s[port_index] = 0.0
        node.run_phase(0.25, lambda _elapsed, _total: safety_check())
        node.set_mode(SetControlMode.Request.DISABLED)
        armed = False
        node.recording = False
        minimum = (
            args.minimum_position_response
            if args.port in POSITION_PORTS
            else args.minimum_velocity_response
        )
        if positive_peak < minimum:
            raise RuntimeError(
                f"P{args.port} 正向反馈仅 {positive_peak:.4f}，"
                f"小于最小响应 {minimum:.4f}"
            )
        print(
            f"P{args.port} 正向动作完成并已整体失能，反馈峰值 "
            f"{positive_peak:+.4f}。",
            flush=True,
        )
        wait_for_authorization(
            node, args,
            f"确认刚才 P{args.port} 物理正向正确且无异常后按 A；"
            "工具将短时整体使能、执行反向动作、回位并自动失能。"
            "不正确则不要按 A，并说明。"
        )
        node.set_mode(SetControlMode.Request.MANUAL_TEST)
        armed = True
        node.recording = True
        node.run_phase(0.5, lambda _elapsed, _total: safety_check())

        negative_peak = math.inf

        def negative_motion(elapsed: float, total: float) -> None:
            nonlocal negative_peak
            safety_check()
            current_state = node.state
            assert current_state is not None
            if args.port in POSITION_PORTS:
                alpha = min(elapsed / total, 1.0)
                node.command.position_rad[port_index] = initial[port_index] + (
                    negative_target - initial[port_index]) * alpha
                response = current_state.position_rad[port_index] - initial[port_index]
            else:
                node.command.velocity_rad_s[port_index] = negative_target
                response = current_state.velocity_rad_s[port_index]
            negative_peak = min(negative_peak, response)

        node.run_phase(args.ramp_duration, negative_motion)
        node.command.position_rad[port_index] = (
            negative_target if args.port in POSITION_PORTS else initial[port_index]
        )
        node.command.velocity_rad_s[port_index] = (
            negative_target if args.port in WHEEL_PORTS else 0.0
        )

        def negative_hold(_elapsed: float, _total: float) -> None:
            nonlocal negative_peak
            safety_check()
            current_state = node.state
            assert current_state is not None
            response = (
                current_state.position_rad[port_index] - initial[port_index]
                if args.port in POSITION_PORTS
                else current_state.velocity_rad_s[port_index]
            )
            negative_peak = min(negative_peak, response)

        node.run_phase(args.hold_duration, negative_hold)

        def final_return(elapsed: float, total: float) -> None:
            safety_check()
            alpha = min(elapsed / total, 1.0)
            if args.port in POSITION_PORTS:
                node.command.position_rad[port_index] = negative_target + (
                    initial[port_index] - negative_target) * alpha
            else:
                node.command.velocity_rad_s[port_index] = negative_target * (1.0 - alpha)

        node.run_phase(args.ramp_duration, final_return)
        node.command.position_rad[port_index] = initial[port_index]
        node.command.velocity_rad_s[port_index] = 0.0
        node.run_phase(0.25, lambda _elapsed, _total: safety_check())
        node.set_mode(SetControlMode.Request.DISABLED)
        armed = False
        node.recording = False
        verify_response(args.port, positive_peak, negative_peak, minimum)
        print(
            f"P{args.port} 反向动作完成并已整体失能，反馈峰值 "
            f"{negative_peak:+.4f}。",
            flush=True,
        )
        if not args.log:
            args.log = (
                f"logs/esd_small_motion_p{args.port}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
        path = node.save()
        saved = True
        print(
            f"A2 P{args.port} 小动作完成：positive={positive_peak:+.4f}, "
            f"negative={negative_peak:+.4f}, rows={len(node.rows)}, log={path}",
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
                    args.log = (
                        f"logs/esd_small_motion_p{args.port}_failed_"
                        f"{time.strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                path = node.save()
                print(f"失败过程已保存：{path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] 失败记录保存失败：{exc}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
