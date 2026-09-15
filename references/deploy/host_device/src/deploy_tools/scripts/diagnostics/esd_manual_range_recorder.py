#!/usr/bin/env python3
"""Record manually exercised P1-P8 ranges while ESD-Link remains disabled."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import time

import rclpy
from esd_link_msgs.msg import LinkStatus, LowerState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


EXPECTED_PORTS = list(range(1, 9))
CONTINUOUS_WHEEL_PORTS = (3, 6)
LEG_DEFAULT_ANGLES = (-0.92020, 0.98338, 0.0, -0.92020, 0.98338, 0.0)
PORT_SIGNS = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0)


def lower_position(port: int, policy_position: float) -> float:
    index = port - 1
    default = LEG_DEFAULT_ANGLES[index] if port <= 6 else 0.0
    return PORT_SIGNS[index] * (policy_position + default)


def lower_range(port: int, policy_minimum: float,
                policy_maximum: float) -> tuple[float, float]:
    converted = (
        lower_position(port, policy_minimum),
        lower_position(port, policy_maximum),
    )
    return min(converted), max(converted)


@dataclass
class RangeStatistics:
    samples: int = 0
    minimum: list[float] = field(default_factory=lambda: [math.inf] * 8)
    maximum: list[float] = field(default_factory=lambda: [-math.inf] * 8)
    max_abs_velocity: list[float] = field(default_factory=lambda: [0.0] * 8)
    max_abs_effort: list[float] = field(default_factory=lambda: [0.0] * 8)

    def update(self, position: list[float], velocity: list[float],
               effort: list[float]) -> None:
        if not (len(position) == len(velocity) == len(effort) == 8):
            raise ValueError("range samples must contain exactly eight ports")
        values = position + velocity + effort
        if not all(math.isfinite(value) for value in values):
            raise ValueError("range sample contains a non-finite value")
        self.samples += 1
        for index in range(8):
            self.minimum[index] = min(self.minimum[index], position[index])
            self.maximum[index] = max(self.maximum[index], position[index])
            self.max_abs_velocity[index] = max(
                self.max_abs_velocity[index], abs(velocity[index]))
            self.max_abs_effort[index] = max(
                self.max_abs_effort[index], abs(effort[index]))


def validate_disabled_state(msg: LowerState) -> str | None:
    if msg.control_state != 2:
        return f"lower control_state={msg.control_state}, expected DISABLED(2)"
    if msg.fault_flags:
        return f"fault_flags=0x{msg.fault_flags:x}"
    if msg.offline_port_mask:
        return f"offline_port_mask=0x{msg.offline_port_mask:x}"
    if msg.imu_valid_mask & 0x07 != 0x07:
        return f"IMU valid mask={msg.imu_valid_mask}"
    if list(msg.port_id) != EXPECTED_PORTS:
        return f"port_id layout={list(msg.port_id)}"
    if any((mask & 0x03) != 0x03 for mask in msg.valid_mask):
        return f"valid_mask={list(msg.valid_mask)}"
    values = list(msg.position_rad) + list(msg.velocity_rad_s) + list(msg.effort_nm)
    if not all(math.isfinite(value) for value in values):
        return "motor feedback contains a non-finite value"
    return None


def counter_delta(start: int, end: int) -> int:
    return end - start if end >= start else end


class ManualRangeRecorder(Node):
    def __init__(self, writer: csv.writer):
        super().__init__("esd_manual_range_recorder")
        self.writer = writer
        self.statistics = RangeStatistics()
        self.last_state: LowerState | None = None
        self.last_sequence: int | None = None
        self.first_sequence: int | None = None
        self.recorder_gaps = 0
        self.started_ns = 0
        self.error = ""
        self.link_start: LinkStatus | None = None
        self.link_end: LinkStatus | None = None
        self.recording = False
        self.create_subscription(
            LowerState,
            "/lower/state",
            self._state_callback,
            QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            LinkStatus,
            "/lower/link_status",
            self._link_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

    def _link_callback(self, msg: LinkStatus) -> None:
        self.link_end = msg
        if self.link_start is None and msg.session_valid:
            self.link_start = msg
        if not self.recording:
            return
        if msg.control_mode != 0 or msg.device_control_state != 2:
            self.error = (
                f"link left READ_ONLY/DISABLED: upper={msg.control_mode}, "
                f"lower={msg.device_control_state}"
            )
        elif msg.fault_flags or msg.offline_port_mask:
            self.error = (
                f"link fault=0x{msg.fault_flags:x}, "
                f"offline=0x{msg.offline_port_mask:x}"
            )
        elif self.link_start is not None and (
            msg.last_command_sequence != self.link_start.last_command_sequence
            or msg.last_applied_command_sequence
            != self.link_start.last_applied_command_sequence
        ):
            self.error = "command sequence changed during passive range recording"

    def _state_callback(self, msg: LowerState) -> None:
        self.last_state = msg
        if not self.recording or msg.state_sample_seq == self.last_sequence:
            return
        state_error = validate_disabled_state(msg)
        if state_error is not None:
            self.error = state_error
            return
        if self.first_sequence is None:
            self.first_sequence = msg.state_sample_seq
        if self.last_sequence is not None:
            delta = (msg.state_sample_seq - self.last_sequence) & 0xFFFFFFFF
            if 1 < delta < 0x80000000:
                self.recorder_gaps += delta - 1
        self.last_sequence = msg.state_sample_seq
        position = list(msg.position_rad)
        velocity = list(msg.velocity_rad_s)
        effort = list(msg.effort_nm)
        self.statistics.update(position, velocity, effort)
        elapsed = (time.monotonic_ns() - self.started_ns) / 1e9
        row: list[object] = [
            f"{elapsed:.9f}",
            msg.host_monotonic_ns,
            msg.device_sample_time_us,
            msg.state_sample_seq,
            msg.control_state,
            msg.imu_valid_mask,
            msg.fault_flags,
            msg.offline_port_mask,
        ]
        for index in range(8):
            row.extend([
                msg.valid_mask[index],
                f"{position[index]:.9f}",
                f"{lower_position(index + 1, position[index]):.9f}",
                f"{velocity[index]:.9f}",
                f"{effort[index]:.9f}",
            ])
        self.writer.writerow(row)

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.last_state is None or self.link_end is None:
                continue
            state_error = validate_disabled_state(self.last_state)
            link = self.link_end
            if (
                state_error is None
                and link.session_valid
                and link.control_mode == 0
                and link.device_control_state == 2
                and link.fault_flags == 0
                and link.offline_port_mask == 0
            ):
                return
        raise RuntimeError("未获得 P1-P8/IMU 有效且 upper/lower DISABLED 的状态")


def make_header() -> list[str]:
    header = [
        "t",
        "host_monotonic_ns",
        "device_sample_time_us",
        "state_sample_seq",
        "control_state",
        "imu_valid_mask",
        "fault_flags",
        "offline_port_mask",
    ]
    for port in EXPECTED_PORTS:
        header.extend([
            f"p{port}_valid",
            f"p{port}_q_policy_rad",
            f"p{port}_q_lower_rad",
            f"p{port}_dq_policy_rad_s",
            f"p{port}_effort_policy_nm",
        ])
    return header


def dashboard_text(node: ManualRangeRecorder, elapsed: float,
                   duration: float) -> str:
    state = node.last_state
    stats = node.statistics
    link = node.link_end
    target = "until Ctrl-C" if duration == 0.0 else f"{duration:.0f}s"
    lines = [
        "ESD-Link P1-P8 人工限位监视器  [纯只读 / 禁止使能]",
        (
            f"elapsed {elapsed:7.1f}s / {target:<12}  "
            f"samples {stats.samples:8d}  recorder gaps {node.recorder_gaps}"
        ),
    ]
    if link is not None:
        lines.append(
            f"upper={link.control_mode} lower={link.device_control_state} "
            f"rate={link.state_rate_hz:6.1f}Hz age={link.latest_state_age_ms:6.2f}ms "
            f"source_gaps={link.sequence_gaps} COBS={link.cobs_errors} "
            f"CRC={link.crc_errors} cmd={link.last_command_sequence}/"
            f"{link.last_applied_command_sequence}"
        )
    lines.extend([
        "",
        "端口  类型    当前策略值      历史最小      历史最大      跨度"
        "       下位机原始值      原始最小      原始最大      当前速度",
        "----  ------  ------------  ------------  ------------  ---------"
        "  --------------  ------------  ------------  ------------",
    ])
    if state is None or stats.samples == 0:
        lines.append("等待有效 LowerState...")
    else:
        positions = list(state.position_rad)
        velocities = list(state.velocity_rad_s)
        for index, port in enumerate(EXPECTED_PORTS):
            raw_minimum, raw_maximum = lower_range(
                port, stats.minimum[index], stats.maximum[index])
            kind = "轮子" if port in CONTINUOUS_WHEEL_PORTS else "关节"
            lines.append(
                f"P{port:<3}  {kind:<6}  {positions[index]:+12.6f}  "
                f"{stats.minimum[index]:+12.6f}  {stats.maximum[index]:+12.6f}  "
                f"{stats.maximum[index] - stats.minimum[index]:9.6f}  "
                f"{lower_position(port, positions[index]):+14.6f}  "
                f"{raw_minimum:+12.6f}  {raw_maximum:+12.6f}  "
                f"{velocities[index]:+12.6f}"
            )
    lines.extend([
        "",
        "说明：策略值用于 bridge/RL；下位机原始值是冻结协议中的编码器侧坐标。",
        "P3/P6 连续旋转，min/max 仅表示本次累计转动；Ctrl-C 可随时保存并退出。",
    ])
    if node.error:
        lines.append("ERROR: " + node.error)
    return "\n".join(lines)


def write_summary(path: Path, node: ManualRangeRecorder, elapsed: float,
                  interrupted: bool) -> None:
    stats = node.statistics
    lines = [
        "ESD-Link manual range recording summary",
        f"elapsed_seconds={elapsed:.6f}",
        f"samples={stats.samples}",
        f"first_state_sample_seq={node.first_sequence}",
        f"last_state_sample_seq={node.last_sequence}",
        f"ros_recorder_sequence_gaps={node.recorder_gaps}",
        f"stopped_by_ctrl_c={str(interrupted).lower()}",
        f"error={node.error or 'none'}",
        "coordinate=bridge policy/base_link",
        "P3/P6_note=continuous wheels; observed position span is not a mechanical limit",
        "",
        "port,current_policy_rad,min_policy_rad,max_policy_rad,span_policy_rad,"
        "current_lower_rad,min_lower_rad,max_lower_rad,"
        "max_abs_velocity_rad_s,max_abs_effort_nm",
    ]
    current = list(node.last_state.position_rad) if node.last_state is not None else [math.nan] * 8
    for index, port in enumerate(EXPECTED_PORTS):
        raw_minimum, raw_maximum = lower_range(
            port, stats.minimum[index], stats.maximum[index])
        lines.append(
            f"P{port},{current[index]:.9f},{stats.minimum[index]:.9f},"
            f"{stats.maximum[index]:.9f},"
            f"{stats.maximum[index] - stats.minimum[index]:.9f},"
            f"{lower_position(port, current[index]):.9f},"
            f"{raw_minimum:.9f},{raw_maximum:.9f},"
            f"{stats.max_abs_velocity[index]:.9f},"
            f"{stats.max_abs_effort[index]:.9f}"
        )
    if node.link_start is not None and node.link_end is not None:
        start = node.link_start
        end = node.link_end
        lines.extend([
            "",
            f"link_sequence_gaps_delta={counter_delta(start.sequence_gaps, end.sequence_gaps)}",
            f"cobs_errors_delta={counter_delta(start.cobs_errors, end.cobs_errors)}",
            f"crc_errors_delta={counter_delta(start.crc_errors, end.crc_errors)}",
            "rejected_commands_delta="
            f"{counter_delta(start.rejected_commands, end.rejected_commands)}",
            f"watchdog_events_delta={counter_delta(start.watchdog_events, end.watchdog_events)}",
            "control_fault_events_delta="
            f"{counter_delta(start.control_fault_events, end.control_fault_events)}",
            f"last_command_sequence={end.last_command_sequence}",
            f"last_applied_command_sequence={end.last_applied_command_sequence}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record P1-P8 manual travel while ESD-Link stays DISABLED")
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="record seconds; 0 records until Ctrl-C",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="default: logs/manual_range_TIMESTAMP",
    )
    parser.add_argument(
        "--ui-hz",
        type=float,
        default=10.0,
        help="interactive terminal refresh rate; maximum 30 Hz",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="disable the ANSI dashboard and print a summary every 5 seconds",
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0.0:
        parser.error("duration must be finite and >= 0")
    if not math.isfinite(args.ui_hz) or args.ui_hz <= 0.0 or args.ui_hz > 30.0:
        parser.error("ui-hz must be finite and in (0, 30]")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out_dir) if args.out_dir else Path(
        f"logs/manual_range_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lower_state_ranges.csv"
    summary_path = output_dir / "summary.txt"
    interrupted = False
    use_dashboard = sys.stdout.isatty() and not args.plain
    start = time.monotonic()
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(make_header())
        stream.flush()
        rclpy.init()
        node = ManualRangeRecorder(writer)
        try:
            node.wait_until_ready()
            node.started_ns = time.monotonic_ns()
            node.recording = True
            start = time.monotonic()
            print(
                "已确认 upper/lower DISABLED；脚本不发布命令、不调用服务。"
                "请缓慢手动移动 P1-P8 到计划测量的两端。",
                flush=True,
            )
            print(
                "P3/P6 为连续旋转轮，其位置范围不会作为机械限位。"
                "可随时 Ctrl-C 提前结束并保存。",
                flush=True,
            )
            next_report = start + 5.0
            next_render = start
            first_render = True
            if use_dashboard:
                sys.stdout.write("\033[?25l")
                sys.stdout.flush()
            while rclpy.ok() and (
                args.duration == 0.0
                or time.monotonic() - start < args.duration
            ):
                rclpy.spin_once(node, timeout_sec=0.02)
                if node.error:
                    raise RuntimeError(node.error)
                now = time.monotonic()
                if use_dashboard and now >= next_render:
                    prefix = "\033[2J\033[H" if first_render else "\033[H"
                    sys.stdout.write(
                        prefix
                        + dashboard_text(node, now - start, args.duration)
                        + "\033[J"
                    )
                    sys.stdout.flush()
                    first_render = False
                    next_render = now + 1.0 / args.ui_hz
                    if now >= next_report:
                        stream.flush()
                        next_report = now + 5.0
                elif not use_dashboard and now >= next_report:
                    spans = [
                        node.statistics.maximum[i] - node.statistics.minimum[i]
                        for i in range(8)
                    ]
                    print(
                        f"记录 {now - start:.1f}s / {node.statistics.samples} 帧；"
                        "范围 " + ", ".join(
                            f"P{i + 1}={span:.3f}" for i, span in enumerate(spans)
                        ),
                        flush=True,
                    )
                    stream.flush()
                    next_report = now + 5.0
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:  # noqa: BLE001
            node.error = str(exc)
            print(f"[ERROR] {exc}", flush=True)
        finally:
            node.recording = False
            elapsed = time.monotonic() - start
            stream.flush()
            write_summary(summary_path, node, elapsed, interrupted)
            if use_dashboard:
                sys.stdout.write(
                    "\033[H" + dashboard_text(node, elapsed, args.duration)
                    + "\033[J\033[?25h\n"
                )
                sys.stdout.flush()
            samples = node.statistics.samples
            error = node.error
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    print(f"记录完成：samples={samples}, csv={csv_path}, summary={summary_path}", flush=True)
    return 1 if error or samples == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
