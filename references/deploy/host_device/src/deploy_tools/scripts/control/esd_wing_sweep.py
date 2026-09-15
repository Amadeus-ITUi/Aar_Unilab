#!/usr/bin/env python3
"""Guided P7/P8 sweep over ESD-Link while holding P1-P6.

All commands cover P1-P8.  "Wing sweep" describes which targets vary; it
never means that the frozen lower firmware powers only P7/P8.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import rclpy
from esd_link_msgs.msg import LinkStatus, LowerState, MaintenanceCommand
from esd_link_msgs.srv import SetControlMode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy

from esd_control_timeouts import CONTROL_MODE_SERVICE_TIMEOUT_SECONDS

from esd_safety_limits import clamp_position_target, WING_POSITION_LIMITS

# P7/P8 are the only position targets changed by this tool.  Keep port IDs
# separate from the ordered limit pairs: WING_POSITION_LIMITS is a tuple of
# ``(minimum, maximum)`` pairs, not a mapping keyed by port number.
WING_PORTS = (7, 8)


class WingSweep(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("esd_wing_sweep")
        self.args = args
        self.state: LowerState | None = None
        self.link_status: LinkStatus | None = None
        self.button_armed = False
        self.button_event = False
        self.recording = False
        self.rows: list[list[object]] = []
        self.started_ns = time.monotonic_ns()
        self.last_tx_source_sequence: int | None = None
        self.command = MaintenanceCommand()
        self.command.mode = MaintenanceCommand.SWEEP
        self.command.port_id = list(range(1, 9))
        self.command.position_rad = [0.0] * 8
        self.command.velocity_rad_s = [0.0] * 8
        self.command.kp = [2.0, 8.0, 0.0, 2.0, 8.0, 0.0, args.kp, args.kp]
        self.command.kd = [0.1, 0.8, 0.2, 0.1, 0.8, 0.2, args.kd, args.kd]
        self.command.effort_nm = [0.0] * 8
        self.publisher = self.create_publisher(MaintenanceCommand, "/maintenance/commands", 1)
        self.create_subscription(LowerState, "/lower/state", self._state_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(LinkStatus, "/lower/link_status", self._link_status_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Joy, "/joy", self._joy_callback, 10)
        self.mode_client = self.create_client(SetControlMode, "/lower/set_control_mode")

    def _state_callback(self, msg: LowerState) -> None:
        previous = self.state.state_sample_seq if self.state is not None else None
        self.state = msg
        if not self.recording or previous == msg.state_sample_seq:
            return
        elapsed = (time.monotonic_ns() - self.started_ns) / 1e9
        row: list[object] = [
            f"{elapsed:.9f}", msg.host_monotonic_ns, msg.device_sample_time_us,
            msg.state_sample_seq, self.command.source_state_sample_seq,
            msg.last_applied_command_seq, msg.command_status_flags,
        ]
        for index in range(8):
            row.extend([
                self.command.position_rad[index], self.command.velocity_rad_s[index],
                self.command.kp[index], self.command.kd[index],
                self.command.effort_nm[index], msg.position_rad[index],
                msg.velocity_rad_s[index], msg.effort_nm[index], "unavailable",
            ])
        self.rows.append(row)

    def _joy_callback(self, msg: Joy) -> None:
        pressed = (
            len(msg.buttons) > self.args.button_index
            and msg.buttons[self.args.button_index] == 1
        )
        if not self.button_armed:
            self.button_armed = not pressed
        elif pressed:
            self.button_event = True
            self.button_armed = False

    def _link_status_callback(self, msg: LinkStatus) -> None:
        self.link_status = msg

    def state_health_error(self) -> str | None:
        state = self.state
        if state is None:
            return "尚未收到 LowerState"
        if self.link_status is not None and (
            self.link_status.link_state == LinkStatus.FAULT
            or self.link_status.control_mode == 5
        ):
            return "bridge control mode FAULT"
        if state.imu_valid_mask & 0x07 != 0x07:
            return f"IMU valid mask={state.imu_valid_mask}"
        if state.offline_port_mask:
            return f"offline_port_mask=0x{state.offline_port_mask:x}"
        if state.fault_flags:
            return f"fault_flags=0x{state.fault_flags:x}"
        if list(state.port_id) != list(range(1, 9)):
            return f"port_id layout={list(state.port_id)}"
        if any((valid & 0x03) != 0x03 for valid in state.valid_mask):
            return f"valid_mask={list(state.valid_mask)}"
        return None

    def healthy_state(self) -> LowerState | None:
        return self.state if self.state_health_error() is None else None

    def wait_state(self, timeout: float = 10.0) -> LowerState:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            state = self.healthy_state()
            if state is not None and state.control_state == 2:
                return state
        raise TimeoutError("未获得 P1-P8、IMU 全部有效且处于 DISABLED 的新 LowerState")

    def wait_button(self, prompt: str) -> None:
        print(prompt, flush=True)
        deadline = time.monotonic() + self.args.confirm_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            health_error = self.state_health_error()
            if health_error is not None:
                raise RuntimeError("等待确认期间 ESD-Link 状态失效：" + health_error)
            if self.button_event:
                self.button_event = False
                return
        raise TimeoutError("等待手柄确认超时")

    def set_mode(
        self,
        mode: int,
        timeout: float = CONTROL_MODE_SERVICE_TIMEOUT_SECONDS,
    ) -> None:
        if not self.mode_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("/lower/set_control_mode 不可用")
        request = SetControlMode.Request()
        request.target_mode = mode
        future = self.mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError("模式切换失败：" + ("timeout" if result is None else result.message))

    def initialize(self, state: LowerState) -> None:
        for index in range(8):
            self.command.position_rad[index] = state.position_rad[index]

    def publish(self) -> bool:
        health_error = self.state_health_error()
        if health_error is not None:
            raise RuntimeError("拒绝发送：" + health_error)
        state = self.state
        assert state is not None
        if self.last_tx_source_sequence == state.state_sample_seq:
            return False
        self.command.header.stamp = self.get_clock().now().to_msg()
        self.command.header.frame_id = "base_link"
        self.command.source_state_sample_seq = state.state_sample_seq
        self.publisher.publish(self.command)
        self.last_tx_source_sequence = state.state_sample_seq
        return True

    def run_phase(self, duration: float, update) -> None:
        period = 1.0 / self.args.hz
        start = time.monotonic()
        next_send = start
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= duration:
                break
            if now >= next_send:
                update(elapsed, duration)
                self.publish()
                next_send += period
                if next_send < now - period:
                    next_send = now + period

    def hold_for_button(self, prompt: str, update=None) -> None:
        print(prompt, flush=True)
        period = 1.0 / self.args.hz
        deadline = time.monotonic() + self.args.confirm_timeout
        next_send = time.monotonic()
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.001)
            now = time.monotonic()
            if now >= next_send:
                if update is not None:
                    update()
                self.publish()
                next_send += period
            if self.button_event:
                self.button_event = False
                return
        raise TimeoutError("等待手柄确认超时")

    def save(self) -> Path:
        output = self.args.log or f"logs/sweep_motor78_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ["t", "host_monotonic_ns", "device_sample_time_us", "state_sample_seq",
                  "source_state_sample_seq", "last_applied_command_seq", "command_status_flags"]
        for port in range(1, 9):
            header.extend([f"p{port}_cmd_q", f"p{port}_cmd_dq", f"p{port}_kp",
                           f"p{port}_kd", f"p{port}_cmd_tau", f"p{port}_q",
                           f"p{port}_dq", f"p{port}_tau", f"p{port}_temperature"])
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.rows)
        return path


def positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须为有限正数")
    return parsed


def build_recovery_targets(initial: list[float], centers: list[float]) -> list[float]:
    """Hold P1-P6 and move only P7/P8 to their configured sweep centers."""
    if len(initial) != 8 or len(centers) != 2:
        raise ValueError("initial must contain P1-P8 and centers must contain P7/P8")
    recovery = list(initial)
    for index, port in enumerate(WING_PORTS):
        recovery[port - 1] = clamp_position_target(port, centers[index])
    return recovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESD-Link P7/P8 mirrored sweep")
    parser.add_argument("--hz", type=positive_finite, default=200.0)
    parser.add_argument("--kp", type=positive_finite, default=1.0)
    parser.add_argument("--kd", type=positive_finite, default=0.1)
    parser.add_argument("--freq", type=positive_finite, default=1.0)
    parser.add_argument("--duration", type=positive_finite, default=30.0)
    parser.add_argument("--return-speed", type=positive_finite, default=0.35)
    parser.add_argument("--center-deg", nargs=2, type=float, default=(20.0, -20.0))
    parser.add_argument("--amplitude-deg", nargs=2, type=positive_finite, default=(20.0, 20.0))
    parser.add_argument("--button-index", type=int, default=0)
    parser.add_argument("--confirm-timeout", type=positive_finite, default=120.0)
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    if args.hz > 250.0 or args.freq > 3.0:
        parser.error("hz 上限为 250，freq 必须 <= 3 Hz")
    centers = [math.radians(value) for value in args.center_deg]
    amplitudes = [math.radians(value) for value in args.amplitude_deg]
    sweep_ranges = zip(centers, amplitudes, WING_POSITION_LIMITS)
    for index, (center, amplitude, limits) in enumerate(sweep_ranges, 7):
        if center - amplitude < limits[0] or center + amplitude > limits[1]:
            parser.error(f"P{index} 扫频范围超出软件限位 {limits}")
    args.centers = centers
    args.amplitudes = amplitudes
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
        recovery = build_recovery_targets(initial, args.centers)
        node.wait_button("P1-P8 全在线且 DISABLED。确认台架和硬件断电手段后，第一次按 A 整体使能。")
        node.set_mode(SetControlMode.Request.SWEEP)
        armed = True
        node.recording = True
        distance = max(
            abs(initial[port - 1] - recovery[port - 1])
            for port in WING_PORTS
        )
        ramp_duration = max(1.0, distance / args.return_speed)

        def ramp(elapsed: float, total: float) -> None:
            alpha = min(elapsed / total, 1.0)
            for port in WING_PORTS:
                index = port - 1
                node.command.position_rad[index] = initial[index] + (
                    recovery[index] - initial[index]) * alpha

        node.run_phase(ramp_duration, ramp)
        for port in WING_PORTS:
            node.command.position_rad[port - 1] = recovery[port - 1]
        node.hold_for_button("越界关节已只向内回收到安全区，P7/P8 已到中心；检查后第二次按 A。")
        node.hold_for_button("保持正常，第三次按 A 开始扫频。")

        def sweep(elapsed: float, _total: float) -> None:
            sine = math.sin(2.0 * math.pi * args.freq * elapsed)
            node.command.position_rad[6] = args.centers[0] + args.amplitudes[0] * sine
            node.command.position_rad[7] = args.centers[1] - args.amplitudes[1] * sine

        node.run_phase(args.duration, sweep)
        for index in range(2):
            node.command.position_rad[6 + index] = args.centers[index]
        node.run_phase(0.25, lambda _elapsed, _total: None)
        node.set_mode(SetControlMode.Request.DISABLED)
        armed = False
        print("P1-P8 整体失能已确认。", flush=True)
        path = node.save()
        print(f"P7/P8 扫频完成：{path}，rows={len(node.rows)}；温度=unavailable", flush=True)
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
