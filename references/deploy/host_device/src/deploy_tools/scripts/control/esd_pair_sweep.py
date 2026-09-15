#!/usr/bin/env python3
"""Guided 1/4, 2/5 or 3/6 sweep through the frozen ESD-Link bridge.

The selected pair is the only target that varies. Every transmitted command
still covers P1-P8, and feedback rows are emitted only for new LowerState
samples. Starting this tool never sets zero; entering SWEEP requires the first
operator A-button confirmation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from pathlib import Path

import rclpy
import yaml
from esd_link_msgs.msg import LinkStatus, LowerState, MaintenanceCommand
from esd_link_msgs.srv import SetControlMode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy

from esd_control_timeouts import CONTROL_MODE_SERVICE_TIMEOUT_SECONDS
from esd_safety_limits import (
    clamp_position_target,
    LEG_POSITION_LIMITS,
    POSITION_LIMITS,
    pose_limit_error,
)

SUPPORTED_PAIRS = {(1, 4), (2, 5), (3, 6)}
VALIDATED_WHEEL_SWEEP_AMPLITUDE_RAD_S = 5.0


def finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def gain_filename_token(value: float) -> str:
    """Format a non-negative gain for stable filenames (0.05 -> 0p05)."""
    number = finite(value, "gain filename value")
    if number < 0.0:
        raise ValueError("gain filename value must be non-negative")
    if math.isclose(number, 0.0, abs_tol=1e-12):
        number = 0.0
    return f"{number:.9g}".replace(".", "p")


class SweepNode(Node):
    def __init__(self, config: dict, config_path: Path) -> None:
        super().__init__("esd_pair_sweep")
        self.config = config
        self.config_path = config_path
        self.state: LowerState | None = None
        self.link_status: LinkStatus | None = None
        self.state_received_at = 0.0
        self.last_recorded_seq: int | None = None
        self.button = int(config.get("joy", {}).get("confirm_button", 0))
        self.last_pressed = False
        self.button_armed = False
        self.button_event = False
        self.command = MaintenanceCommand()
        self.command.mode = MaintenanceCommand.SWEEP
        self.command.port_id = list(range(1, 9))
        self.command.position_rad = [0.0] * 8
        self.command.velocity_rad_s = [0.0] * 8
        self.command.kp = [0.0] * 8
        self.command.kd = [0.0] * 8
        self.command.effort_nm = [0.0] * 8
        self.tx_snapshot = copy.deepcopy(self.command)
        self.last_transmitted_command = copy.deepcopy(self.command)
        self.streaming = False
        self.tx_period = 0.0
        self.next_tx_time = 0.0
        self.last_tx_source_sequence: int | None = None
        self.stream_error: Exception | None = None
        self.records: list[list[object]] = []
        self.recording = False
        self.create_subscription(LowerState, "/lower/state", self._state, qos_profile_sensor_data)
        self.create_subscription(
            LinkStatus, "/lower/link_status", self._link_status, qos_profile_sensor_data)
        self.create_subscription(Joy, "/joy", self._joy, qos_profile_sensor_data)
        self.publisher = self.create_publisher(MaintenanceCommand, "/maintenance/commands", 1)
        self.mode_client = self.create_client(SetControlMode, "/lower/set_control_mode")

    def _joy(self, msg: Joy) -> None:
        pressed = 0 <= self.button < len(msg.buttons) and msg.buttons[self.button] == 1
        if not pressed:
            self.button_armed = True
        if self.button_armed and pressed and not self.last_pressed:
            self.button_event = True
            self.button_armed = False
        self.last_pressed = pressed

    def _link_status(self, msg: LinkStatus) -> None:
        self.link_status = msg

    def _state(self, msg: LowerState) -> None:
        previous_sequence = self.state.state_sample_seq if self.state is not None else None
        is_new_state = previous_sequence != msg.state_sample_seq
        self.state = msg
        self.state_received_at = time.monotonic()
        if self.streaming and is_new_state and self.state_received_at >= self.next_tx_time:
            try:
                self._publish_for_state(msg)
            except Exception as exc:  # noqa: BLE001
                self.stream_error = exc
                self.streaming = False
            else:
                self.next_tx_time += self.tx_period
                if self.next_tx_time < self.state_received_at - self.tx_period:
                    self.next_tx_time = self.state_received_at + self.tx_period
        if not self.recording or self.last_recorded_seq == msg.state_sample_seq:
            return
        self.last_recorded_seq = msg.state_sample_seq
        command = self.last_transmitted_command
        row: list[object] = [
            time.time(), msg.host_monotonic_ns, msg.device_sample_time_us,
            msg.state_sample_seq, command.source_state_sample_seq,
            msg.last_applied_command_seq, msg.command_status_flags,
        ]
        feedback = {msg.port_id[i]: i for i in range(8)}
        for port in range(1, 9):
            i = port - 1
            f = feedback.get(port)
            row.extend([
                command.position_rad[i], command.velocity_rad_s[i],
                command.kp[i], command.kd[i], command.effort_nm[i],
                msg.position_rad[f] if f is not None else math.nan,
                msg.velocity_rad_s[f] if f is not None else math.nan,
                msg.effort_nm[f] if f is not None else math.nan,
                "unavailable",
            ])
        self.records.append(row)

    def state_health_error(self) -> str | None:
        msg = self.state
        if msg is None:
            return "尚未收到 LowerState"
        if self.link_status is not None and (
            self.link_status.link_state == LinkStatus.FAULT
            or self.link_status.control_mode == 5
        ):
            return "bridge control mode FAULT"
        if msg.schema_id != 1:
            return f"schema_id={msg.schema_id}"
        if msg.active_port_mask != 0x1FE:
            return f"active_port_mask=0x{msg.active_port_mask:x}"
        if msg.offline_port_mask:
            return f"offline_port_mask=0x{msg.offline_port_mask:x}"
        if msg.fault_flags:
            return f"fault_flags=0x{msg.fault_flags:x}"
        if msg.control_state in (5, 6):
            return f"lower control_state={msg.control_state}"
        if msg.imu_valid_mask & 0x07 != 0x07:
            return f"imu_valid_mask={msg.imu_valid_mask}"
        if sorted(msg.port_id) != list(range(1, 9)):
            return f"port_id={list(msg.port_id)}"
        if any((mask & 0x03) != 0x03 for mask in msg.valid_mask):
            return f"valid_mask={list(msg.valid_mask)}"
        return None

    def healthy_state(self) -> LowerState | None:
        return self.state if self.state_health_error() is None else None

    def wait_for_state(self, timeout: float = 10.0) -> LowerState:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            state = self.healthy_state()
            if state is not None:
                return state
        raise RuntimeError("ESD-Link P1-P8/IMU feedback is not valid")

    def wait_button(self, prompt: str) -> None:
        print(prompt, flush=True)
        timeout = finite(self.config.get("joy", {}).get("timeout_sec", 120.0), "joy.timeout_sec")
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            health_error = self.state_health_error()
            if health_error is not None:
                raise RuntimeError("确认期间 ESD-Link 状态失效：" + health_error)
            if self.button_event:
                self.button_event = False
                return
        raise TimeoutError("gamepad confirmation timeout")

    def set_mode(
        self,
        mode: int,
        timeout: float = CONTROL_MODE_SERVICE_TIMEOUT_SECONDS,
    ) -> None:
        if not self.mode_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("/lower/set_control_mode unavailable")
        request = SetControlMode.Request()
        request.target_mode = mode
        future = self.mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        result = future.result()
        if result is None or not result.success or result.actual_mode != mode:
            message = "timeout" if result is None else result.message
            raise RuntimeError(f"control mode {mode} rejected: {message}")

    def initialize_command(self, state: LowerState) -> None:
        by_port = {state.port_id[i]: i for i in range(8)}
        for port in range(1, 9):
            source = by_port[port]
            index = port - 1
            self.command.position_rad[index] = state.position_rad[source]
            self.command.velocity_rad_s[index] = 0.0
            self.command.kp[index] = 10.0 if port > 6 else (0.0 if port in (3, 6) else 1.0)
            self.command.kd[index] = 0.5 if port > 6 else 0.1
        self.refresh_command_snapshot()

    def refresh_command_snapshot(self) -> None:
        # Replace one complete eight-port target atomically. The next selected
        # LowerState callback consumes this snapshot without a timer heartbeat.
        self.tx_snapshot = copy.deepcopy(self.command)

    def _publish_for_state(self, state: LowerState) -> bool:
        if self.state_health_error() is not None:
            raise RuntimeError("拒绝维护命令：" + str(self.state_health_error()))
        if self.last_tx_source_sequence == state.state_sample_seq:
            return False
        command = copy.deepcopy(self.tx_snapshot)
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_link"
        command.source_state_sample_seq = state.state_sample_seq
        self.publisher.publish(command)
        self.last_tx_source_sequence = state.state_sample_seq
        self.last_transmitted_command = command
        return True

    def start_streaming(self, rate: float) -> None:
        if self.streaming:
            raise RuntimeError("maintenance stream already running")
        self.stream_error = None
        self.tx_period = 1.0 / rate
        self.next_tx_time = 0.0
        self.last_tx_source_sequence = None
        self.streaming = True

    def stop_streaming(self) -> None:
        self.streaming = False

    def check_stream(self) -> None:
        if self.stream_error is not None:
            raise RuntimeError("状态驱动发令已停止：" + str(self.stream_error))

    def run_phase(self, duration: float, rate: float, update) -> None:
        start = time.perf_counter()
        deadline = start
        period = 1.0 / rate
        while rclpy.ok():
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= duration:
                break
            rclpy.spin_once(self, timeout_sec=0.0)
            self.check_stream()
            now = time.perf_counter()
            elapsed = now - start
            if now >= deadline:
                update(elapsed, duration)
                self.refresh_command_snapshot()
                deadline += period
                if deadline < now:
                    deadline = now + period
            time.sleep(min(max(deadline - time.perf_counter(), 0.0), 0.001))

    def hold_until_button(self, prompt: str) -> None:
        print(prompt, flush=True)
        timeout = finite(self.config.get("joy", {}).get("timeout_sec", 120.0), "joy.timeout_sec")
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.001)
            self.check_stream()
            health_error = self.state_health_error()
            if health_error is not None:
                raise RuntimeError("保持期间 ESD-Link 状态失效：" + health_error)
            if self.button_event:
                self.button_event = False
                return
        raise TimeoutError("gamepad confirmation timeout")

    def save(self, output_value: str) -> Path:
        output = Path(output_value.replace("{timestamp}", time.strftime("%Y%m%d_%H%M%S")))
        if not output.is_absolute():
            output = self.config_path.parent / output
        output.parent.mkdir(parents=True, exist_ok=True)
        header = ["wall_time", "host_monotonic_ns", "device_sample_time_us", "state_sample_seq",
                  "source_state_sample_seq", "last_applied_command_seq", "command_status_flags"]
        for port in range(1, 9):
            header.extend([f"p{port}_cmd_q", f"p{port}_cmd_dq", f"p{port}_kp",
                           f"p{port}_kd", f"p{port}_cmd_tau", f"p{port}_q",
                           f"p{port}_dq", f"p{port}_tau", f"p{port}_temperature"])
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.records)
        return output


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    pair = tuple(int(value) for value in config.get("swept_motor_ids", []))
    if pair not in SUPPORTED_PAIRS:
        raise ValueError("swept_motor_ids must be [1,4], [2,5], or [3,6]")
    rate = finite(config.get("sweep", {}).get("publish_rate_hz", 200.0), "publish_rate_hz")
    if rate <= 0.0 or rate > 250.0:
        raise ValueError("ESD-Link sweep rate must be in (0, 250] Hz")
    if pair == (3, 6):
        amplitude = finite(
            config.get("sweep", {}).get("velocity_amplitude_rad_s", 0.0),
            "velocity_amplitude_rad_s",
        )
        if amplitude <= 0.0 or amplitude > VALIDATED_WHEEL_SWEEP_AMPLITUDE_RAD_S:
            raise ValueError(
                "P3/P6 ESD-Link sweep amplitude must be in (0, "
                f"{VALIDATED_WHEEL_SWEEP_AMPLITUDE_RAD_S}] rad/s"
            )
    return config


def apply_sweep_overrides(
    config: dict,
    sweep_kp: float | None = None,
    sweep_kd: float | None = None,
    output_file: str | None = None,
) -> dict:
    """Apply one-run identification settings without changing the base YAML."""
    result = copy.deepcopy(config)
    pair = tuple(int(value) for value in result.get("swept_motor_ids", []))
    if pair not in SUPPORTED_PAIRS:
        raise ValueError("swept_motor_ids must be [1,4], [2,5], or [3,6]")
    sweep = result.setdefault("sweep", {})
    if sweep_kp is not None:
        kp = finite(sweep_kp, "sweep kp")
        if kp < 0.0:
            raise ValueError("sweep kp must be non-negative")
        if pair == (3, 6) and not math.isclose(kp, 0.0, abs_tol=1e-9):
            raise ValueError("P3/P6 D-only velocity sweep requires Kp=0")
        sweep["kp"] = kp
    if sweep_kd is not None:
        kd = finite(sweep_kd, "sweep kd")
        if kd <= 0.0:
            raise ValueError("sweep kd must be positive")
        sweep["kd"] = kd
    if output_file is not None:
        if not output_file.strip():
            raise ValueError("output file must not be empty")
        sweep["output_file"] = output_file
    elif sweep_kp is not None or sweep_kd is not None:
        effective_kp = finite(sweep.get("kp", 0.0), "effective sweep kp")
        effective_kd = finite(sweep.get("kd", 0.1), "effective sweep kd")
        pair_name = "".join(str(port) for port in pair)
        sweep["output_file"] = (
            f"logs/motor{pair_name}_{{timestamp}}_"
            f"kp{gain_filename_token(effective_kp)}_"
            f"kd{gain_filename_token(effective_kd)}.csv"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Guided ESD-Link full-device pair sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--test-flow", action="store_true",
        help="run an 8-second low-amplitude chirp before the configured chirp")
    parser.add_argument(
        "--sweep-kp", type=float, default=None,
        help="override only the swept pair Kp for this run")
    parser.add_argument(
        "--sweep-kd", type=float, default=None,
        help="override only the swept pair Kd for this run")
    parser.add_argument(
        "--output-file", default=None,
        help=("override the output CSV path; {timestamp} is expanded. "
              "When omitted with gain overrides, pair and gains are named automatically"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = apply_sweep_overrides(
        load_config(config_path), args.sweep_kp, args.sweep_kd, args.output_file)
    rclpy.init()
    node = SweepNode(config, config_path)
    pair = [int(value) for value in config["swept_motor_ids"]]
    armed = False
    try:
        state = node.wait_for_state()
        node.initialize_command(state)
        hold_ids = [int(value) for value in config["hold_motor_ids"]]
        sweep = config["sweep"]
        return_cfg = config["return_to_def_pos"]
        rate = finite(sweep.get("publish_rate_hz", 200.0), "publish_rate_hz")
        base = [finite(value, "sweep base") for value in config["sweep_base_def_pos_policy_rad"]]
        offset = [
            finite(value, "zero offset")
            for value in config["zero_reference_offset_policy_rad"]
        ]
        target_pose = [left + right for left, right in zip(base, offset)]
        initial = list(node.command.position_rad)
        recovery = target_pose + [initial[6], initial[7]]
        for port in POSITION_LIMITS:
            recovery[port - 1] = clamp_position_target(port, recovery[port - 1])
        error = pose_limit_error(recovery)
        if error is not None:
            raise RuntimeError(f"扫频目标姿态非法：{error}；保持 DISABLED")
        velocity_mode = str(sweep.get("control_mode", "position")) == "velocity"
        amplitude = finite(
            sweep.get("velocity_amplitude_rad_s" if velocity_mode else "amplitude_rad", 0.2),
            "amplitude",
        )
        if not velocity_mode:
            for motor in pair:
                low, high = LEG_POSITION_LIMITS[motor]
                center = target_pose[motor - 1]
                if center - amplitude < low or center + amplitude > high:
                    raise RuntimeError(
                        f"P{motor} 扫频范围 [{center - amplitude:+.4f}, "
                        f"{center + amplitude:+.4f}] 超出软件限位 "
                        f"[{low:+.4f}, {high:+.4f}]；保持 DISABLED"
                    )
        node.wait_button(
            "P1-P8 已全部在线且仍为 DISABLED。确认台架和硬件断电手段后，释放再第一次按 A：整体使能并进入 SWEEP。"
        )
        node.set_mode(SetControlMode.Request.SWEEP)
        armed = True
        node.recording = True
        node.start_streaming(rate)
        return_kp = finite(return_cfg.get("kp", 1.0), "return kp")
        return_kd = finite(return_cfg.get("kd", 0.1), "return kd")
        hold_kp = [finite(value, "hold kp") for value in config["hold"]["kp"]]
        hold_kd = [finite(value, "hold kd") for value in config["hold"]["kd"]]
        for motor, kp, kd in zip(hold_ids, hold_kp, hold_kd):
            node.command.kp[motor - 1] = kp
            node.command.kd[motor - 1] = kd
        for motor in pair:
            node.command.kp[motor - 1] = 0.0 if motor in (3, 6) else return_kp
            node.command.kd[motor - 1] = return_kd
        node.refresh_command_snapshot()
        distance = max(
            abs(initial[port - 1] - recovery[port - 1])
            for port in POSITION_LIMITS
        )
        duration = max(finite(return_cfg.get("min_duration_sec", 1.0), "minimum duration"),
                       distance / finite(return_cfg.get("speed_rad_s", 0.1), "return speed"))

        def return_update(elapsed: float, total: float) -> None:
            alpha = min(elapsed / total, 1.0)
            for port in POSITION_LIMITS:
                index = port - 1
                node.command.position_rad[index] = initial[index] + (
                    recovery[index] - initial[index]) * alpha
                node.command.velocity_rad_s[index] = 0.0

        node.run_phase(duration, rate, return_update)
        for port in POSITION_LIMITS:
            node.command.position_rad[port - 1] = recovery[port - 1]
        node.refresh_command_snapshot()
        node.hold_until_button(
            "全部位置关节已收至正式扫频姿态。检查后第二次按 A 开始低幅 chirp。",
        )

        f0 = finite(sweep.get("start_frequency_hz", 0.1), "start frequency")
        f1 = finite(sweep.get("end_frequency_hz", 5.0), "end frequency")
        sweep_duration = finite(sweep.get("duration_sec", 40.0), "sweep duration")
        signs = [finite(value, "sweep sign") for value in sweep["signs"]]
        for motor in pair:
            node.command.kp[motor - 1] = (
                0.0 if velocity_mode
                else finite(sweep.get("kp", 2.0), "sweep kp")
            )
            node.command.kd[motor - 1] = finite(
                sweep.get("kd", 0.1), "sweep kd")
        node.refresh_command_snapshot()

        def sweep_update(elapsed: float, total: float) -> None:
            phase = f0 * elapsed + 0.5 * (f1 - f0) * elapsed * elapsed / total
            value = amplitude * math.sin(2.0 * math.pi * phase)
            for motor, direction in zip(pair, signs):
                if velocity_mode:
                    node.command.velocity_rad_s[motor - 1] = direction * value
                else:
                    node.command.position_rad[motor - 1] = (
                        target_pose[motor - 1] + direction * value
                    )

        if args.test_flow:
            smoke_amplitude = min(amplitude, 0.5 if velocity_mode else 0.03)
            smoke_f1 = min(f1, 1.0)

            def smoke_update(elapsed: float, total: float) -> None:
                phase = f0 * elapsed + 0.5 * (smoke_f1 - f0) * elapsed * elapsed / total
                value = smoke_amplitude * math.sin(2.0 * math.pi * phase)
                for motor, direction in zip(pair, signs):
                    if velocity_mode:
                        node.command.velocity_rad_s[motor - 1] = direction * value
                    else:
                        node.command.position_rad[motor - 1] = (
                            target_pose[motor - 1] + direction * value
                        )

            print("开始 8 秒低幅 chirp。", flush=True)
            node.run_phase(8.0, rate, smoke_update)
            for motor in pair:
                node.command.position_rad[motor - 1] = target_pose[motor - 1]
                node.command.velocity_rad_s[motor - 1] = 0.0
            node.refresh_command_snapshot()
            node.run_phase(0.25, rate, lambda _elapsed, _total: None)
            node.hold_until_button(
                "低幅 chirp 完成，当前保持正式扫频姿态；确认正常后第三次按 A 开始正式 chirp。",
            )
        else:
            node.hold_until_button("保持正常。第三次按 A 开始正式 chirp。")

        print(f"开始 {sweep_duration:.1f} 秒正式 chirp。", flush=True)
        node.run_phase(sweep_duration, rate, sweep_update)
        for motor in pair:
            node.command.position_rad[motor - 1] = target_pose[motor - 1]
            node.command.velocity_rad_s[motor - 1] = 0.0
        node.refresh_command_snapshot()
        node.run_phase(0.2, rate, lambda _elapsed, _total: None)
        print("正式 chirp 完成，正在整体失能并等待 SAFE_DAMPING 结束。", flush=True)
        node.stop_streaming()
        node.set_mode(SetControlMode.Request.DISABLED)
        armed = False
        print("P1-P8 整体失能已确认。", flush=True)
        output = node.save(str(sweep.get("output_file", "logs/esd_sweep_{timestamp}.csv")))
        print(f"扫频完成，CSV={output}，rows={len(node.records)}；温度字段为 unavailable。", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", flush=True)
        if node.records:
            failed_output = node.save(
                f"logs/sweep_motor{''.join(str(motor) for motor in pair)}_failed_"
                f"{time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            print(f"失败前数据已保存：{failed_output}", flush=True)
        return 1
    finally:
        node.stop_streaming()
        if armed:
            try:
                node.set_mode(SetControlMode.Request.DISABLED)
                print("P1-P8 整体失能已确认。", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] 无法确认整体失能：{exc}；请立即使用硬件断电。", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
