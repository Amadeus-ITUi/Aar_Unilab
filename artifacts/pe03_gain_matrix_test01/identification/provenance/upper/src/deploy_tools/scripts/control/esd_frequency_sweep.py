#!/usr/bin/env python3
"""Profile-driven guided frequency sweep over the ESD-Link maintenance path."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Callable

import rclpy
import yaml
from esd_link_msgs.msg import ActuatorCommand, LinkStatus, RobotState
from esd_link_msgs.srv import SetControlMode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

from deployment_selection import DeploymentSelection, load_active_deployment
from esd_control_timeouts import CONTROL_MODE_SERVICE_TIMEOUT_SECONDS
from sweep_configuration import (
    SweepCatalog,
    SweepGroup,
    SweepTarget,
    apply_gain_overrides,
    load_sweep_catalog,
    select_group,
)


def find_deploy_root(configured: str | None = None) -> Path:
    if configured:
        return Path(configured).resolve(strict=True)
    environment = os.environ.get("WE11_DEPLOY_ROOT")
    if environment:
        return Path(environment).resolve(strict=True)
    for parent in Path.cwd().resolve().parents:
        if (parent / "config/active_deployment.yaml").is_file() and (parent / "src").is_dir():
            return parent
    current = Path.cwd().resolve()
    if (current / "config/active_deployment.yaml").is_file():
        return current
    raise RuntimeError("cannot locate Deploy root; pass --root or set WE11_DEPLOY_ROOT")


def resolve_catalog(
    root: Path, deployment_id: str | None
) -> tuple[DeploymentSelection, SweepCatalog]:
    selection = load_active_deployment(root, deployment_id=deployment_id)
    catalog = load_sweep_catalog(
        selection.robot_profile_path, selection.sweep_profile_path)
    expected_robot = selection.deployment_id.split("/", 1)[0]
    if catalog.robot_id != expected_robot:
        raise ValueError(
            f"sweep robot_id {catalog.robot_id!r} does not match deployment {expected_robot!r}")
    return selection, catalog


def target_label(targets: tuple[SweepTarget, ...]) -> str:
    return "+".join(target.joint_name for target in targets)


def chirp_value(
    elapsed: float,
    duration: float,
    start_frequency_hz: float,
    end_frequency_hz: float,
) -> float:
    phase_cycles = (
        start_frequency_hz * elapsed
        + 0.5 * (end_frequency_hz - start_frequency_hz)
        * elapsed * elapsed / duration
    )
    return math.sin(2.0 * math.pi * phase_cycles)


def _raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


class FrequencySweepNode(Node):
    def __init__(
        self,
        catalog: SweepCatalog,
        group: SweepGroup,
        targets: tuple[SweepTarget, ...],
    ) -> None:
        super().__init__("esd_frequency_sweep")
        self.catalog = catalog
        self.group = group
        self.targets = targets
        self.state: RobotState | None = None
        self.link_status: LinkStatus | None = None
        self.state_received_at = 0.0
        self.command = ActuatorCommand()
        self.command.owner = ActuatorCommand.OWNER_MAINTENANCE
        self.command.mode = ActuatorCommand.MODE_SWEEP
        self.command.port_id = [item.port_id for item in catalog.actuators]
        count = len(catalog.actuators)
        self.command.position_rad = [0.0] * count
        self.command.velocity_rad_s = [0.0] * count
        self.command.kp = [0.0] * count
        self.command.kd = [0.0] * count
        self.command.effort_nm = [0.0] * count
        self.tx_snapshot = copy.deepcopy(self.command)
        self.last_transmitted_command = copy.deepcopy(self.command)
        self.last_tx_source_sequence: int | None = None
        self.allow_sweep_transition = False
        self.streaming = False
        self.tx_period = 0.0
        self.next_tx_time = 0.0
        self.stream_error: Exception | None = None
        self.recording = False
        self.records: list[list[object]] = []
        self.last_recorded_seq: int | None = None
        self.record_period = 1.0 / float(catalog.defaults["record_rate_hz"])
        self.next_record_time = 0.0
        self.publisher = self.create_publisher(
            ActuatorCommand, "/control/maintenance_command", 1)
        self.create_subscription(
            RobotState, "/robot/state", self._state_callback, qos_profile_sensor_data)
        self.create_subscription(
            LinkStatus, "/lower/link_status", self._link_status_callback,
            qos_profile_sensor_data)
        self.mode_client = self.create_client(SetControlMode, "/lower/set_control_mode")

    def _link_status_callback(self, message: LinkStatus) -> None:
        self.link_status = message

    def _state_callback(self, message: RobotState) -> None:
        previous = self.state.state_sample_seq if self.state is not None else None
        is_new = previous != message.state_sample_seq
        self.state = message
        now = time.monotonic()
        self.state_received_at = now
        if self.streaming and is_new and now >= self.next_tx_time:
            try:
                self._publish_for_state(message)
            except Exception as exc:  # noqa: BLE001
                self.stream_error = exc
                self.streaming = False
            else:
                self.next_tx_time += self.tx_period
                if self.next_tx_time < now - self.tx_period:
                    self.next_tx_time = now + self.tx_period
        if (
            not self.recording
            or not is_new
            or self.last_recorded_seq == message.state_sample_seq
            or now < self.next_record_time
        ):
            return
        self.last_recorded_seq = message.state_sample_seq
        self.next_record_time = now + self.record_period
        self._record(message)

    def _record(self, state: RobotState) -> None:
        command = self.last_transmitted_command
        feedback = {port: index for index, port in enumerate(state.port_id)}
        command_index = {port: index for index, port in enumerate(command.port_id)}
        row: list[object] = [
            time.time(), state.host_monotonic_ns, state.device_sample_time_us,
            state.state_sample_seq, command.source_state_sample_seq,
            state.last_applied_command_seq, state.command_status_flags,
        ]
        for actuator in self.catalog.actuators:
            port = actuator.port_id
            ci = command_index[port]
            fi = feedback.get(port)
            row.extend([
                command.position_rad[ci], command.velocity_rad_s[ci],
                command.kp[ci], command.kd[ci], command.effort_nm[ci],
                state.position_rad[fi] if fi is not None else math.nan,
                state.velocity_rad_s[fi] if fi is not None else math.nan,
                state.effort_nm[fi] if fi is not None else math.nan,
                "unavailable",
            ])
        self.records.append(row)

    def state_health_error(
        self,
        *,
        require_disabled: bool = False,
        allow_sweep_transition: bool = False,
    ) -> str | None:
        state = self.state
        status = self.link_status
        if state is None:
            return "尚未收到 RobotState"
        if status is None:
            return "尚未收到 LinkStatus"
        if status.link_state == LinkStatus.FAULT or status.control_mode == 5:
            return "bridge control mode FAULT"
        expected_link_state = LinkStatus.READ_ONLY if require_disabled else LinkStatus.READY
        acceptable_link_states = (
            (LinkStatus.READ_ONLY, LinkStatus.READY)
            if allow_sweep_transition
            else (expected_link_state,)
        )
        if status.link_state not in acceptable_link_states:
            expected_name = "READ_ONLY" if require_disabled else "READY"
            return (
                f"bridge link_state={status.link_state}, expected {expected_name}")
        expected_mode = (
            SetControlMode.Request.DISABLED
            if require_disabled
            else SetControlMode.Request.SWEEP
        )
        acceptable_modes = (
            (SetControlMode.Request.DISABLED, SetControlMode.Request.SWEEP)
            if allow_sweep_transition
            else (expected_mode,)
        )
        if status.control_mode not in acceptable_modes:
            return (
                f"bridge control_mode={status.control_mode}, "
                f"expected {expected_mode}")
        if status.schema_id != self.catalog.schema_id:
            return f"LinkStatus schema_id={status.schema_id}"
        if status.config_fingerprint != self.catalog.lower_config_fingerprint:
            return (
                f"lower fingerprint=0x{status.config_fingerprint:08x}, expected "
                f"0x{self.catalog.lower_config_fingerprint:08x}")
        if status.active_port_mask != self.catalog.active_port_mask:
            return f"LinkStatus active_port_mask=0x{status.active_port_mask:x}"
        if state.profile_id != self.catalog.profile_id:
            return f"profile_id={state.profile_id!r}, expected {self.catalog.profile_id!r}"
        if state.schema_id != self.catalog.schema_id:
            return f"RobotState schema_id={state.schema_id}"
        if state.active_port_mask != self.catalog.active_port_mask:
            return f"RobotState active_port_mask=0x{state.active_port_mask:x}"
        count = len(self.catalog.actuators)
        arrays = (
            state.port_id, state.joint_name, state.valid_mask,
            state.position_rad, state.velocity_rad_s, state.effort_nm,
        )
        if any(len(value) != count for value in arrays):
            return "RobotState actuator array lengths do not match selected profile"
        live = {port: name for port, name in zip(state.port_id, state.joint_name)}
        expected = {item.port_id: item.joint_name for item in self.catalog.actuators}
        if live != expected:
            return f"RobotState port/joint table={live}, expected={expected}"
        if state.offline_port_mask or status.offline_port_mask:
            return f"offline_port_mask=0x{state.offline_port_mask | status.offline_port_mask:x}"
        combined_faults = state.fault_flags | status.fault_flags
        if combined_faults and not (
            allow_sweep_transition and combined_faults & ~0x02 == 0
        ):
            return f"fault_flags=0x{combined_faults:x}"
        if not state.control_allowed or not state.calibrated:
            return "RobotState is not authorized and calibrated for control"
        if state.control_state == 6 or (
            state.control_state == 5 and not allow_sweep_transition
        ):
            return f"lower control_state={state.control_state}"
        if (
            self.state_received_at > 0.0
            and time.monotonic() - self.state_received_at
            > float(self.catalog.defaults["state_freshness_sec"])
        ):
            return "RobotState feedback is stale"
        if require_disabled and state.control_state != 2:
            return f"lower control_state={state.control_state}, expected DISABLED(2)"
        if state.imu_valid_mask & 0x07 != 0x07:
            return f"imu_valid_mask={state.imu_valid_mask}"
        if any((mask & 0x03) != 0x03 for mask in state.valid_mask):
            return f"valid_mask={list(state.valid_mask)}"
        return None

    def wait_for_disabled_state(self) -> RobotState:
        timeout = float(self.catalog.defaults["state_timeout_sec"])
        deadline = time.monotonic() + timeout
        last_error = "尚未收到状态"
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            error = self.state_health_error(require_disabled=True)
            if error is None:
                assert self.state is not None
                return self.state
            last_error = error
        raise RuntimeError(f"未获得所选样机的健康 DISABLED 状态：{last_error}")

    def wait_for_sweep_state(self) -> RobotState:
        timeout = float(self.catalog.defaults["state_timeout_sec"])
        deadline = time.monotonic() + timeout
        last_error = "尚未收到状态"
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            error = self.state_health_error()
            if error is None:
                self.allow_sweep_transition = False
                assert self.state is not None
                return self.state
            transition_error = self.state_health_error(allow_sweep_transition=True)
            if transition_error is not None:
                raise RuntimeError(
                    "进入 SWEEP 期间状态失效：" + transition_error)
            last_error = error
        raise RuntimeError(f"进入 SWEEP 后状态未就绪：{last_error}")

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
        if result is None or not result.success or result.actual_mode != mode:
            detail = "timeout" if result is None else result.message
            raise RuntimeError(f"control mode {mode} rejected: {detail}")

    def initialize_command(self, state: RobotState) -> list[float]:
        by_port = {port: index for index, port in enumerate(state.port_id)}
        target_ports = {target.port_id for target in self.targets}
        hold_kp = self.catalog.defaults.get("hold_kp")
        hold_kd = self.catalog.defaults.get("hold_kd")
        initial: list[float] = []
        self.command.profile_id = state.profile_id
        self.command.profile_hash = state.profile_hash
        for index, actuator in enumerate(self.catalog.actuators):
            source = by_port[actuator.port_id]
            position = float(state.position_rad[source])
            initial.append(position)
            self.command.position_rad[index] = position
            self.command.velocity_rad_s[index] = 0.0
            if (
                actuator.control == "position"
                and actuator.port_id not in target_ports
                and hold_kp is not None
                and hold_kd is not None
            ):
                self.command.kp[index] = float(hold_kp)
                self.command.kd[index] = float(hold_kd)
            else:
                self.command.kp[index] = (
                    actuator.kp if actuator.control == "position" else 0.0)
                self.command.kd[index] = actuator.kd
            self.command.effort_nm[index] = 0.0
        self.refresh_command_snapshot()
        return initial

    def refresh_command_snapshot(self) -> None:
        self.tx_snapshot = copy.deepcopy(self.command)

    def _publish_for_state(self, state: RobotState) -> bool:
        error = self.state_health_error(
            allow_sweep_transition=self.allow_sweep_transition)
        if error is not None:
            raise RuntimeError("拒绝维护命令：" + error)
        if self.last_tx_source_sequence == state.state_sample_seq:
            return False
        command = copy.deepcopy(self.tx_snapshot)
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_link"
        command.profile_id = state.profile_id
        command.profile_hash = state.profile_hash
        command.source_state_sample_seq = state.state_sample_seq
        self.publisher.publish(command)
        self.last_tx_source_sequence = state.state_sample_seq
        self.last_transmitted_command = command
        return True

    def start_streaming(self) -> None:
        if self.streaming:
            raise RuntimeError("maintenance stream already running")
        self.stream_error = None
        self.tx_period = 1.0 / float(self.catalog.defaults["publish_rate_hz"])
        self.next_tx_time = 0.0
        self.last_tx_source_sequence = None
        self.streaming = True

    def stop_streaming(self) -> None:
        self.streaming = False

    def check_stream(self) -> None:
        if self.stream_error is not None:
            raise RuntimeError("状态驱动发令已停止：" + str(self.stream_error))

    def run_phase(
        self, duration: float, update: Callable[[float, float], None]
    ) -> None:
        start = time.perf_counter()
        next_update = start
        period = self.tx_period
        while rclpy.ok():
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= duration:
                break
            rclpy.spin_once(self, timeout_sec=0.0)
            self.check_stream()
            error = self.state_health_error()
            if error is not None:
                raise RuntimeError("扫频阶段状态失效：" + error)
            now = time.perf_counter()
            elapsed = now - start
            if now >= next_update:
                update(elapsed, duration)
                self.refresh_command_snapshot()
                next_update += period
                if next_update < now:
                    next_update = now + period
            time.sleep(min(max(next_update - time.perf_counter(), 0.0), 0.001))

    def csv_header(self) -> list[str]:
        header = [
            "wall_time", "host_monotonic_ns", "device_sample_time_us",
            "state_sample_seq", "source_state_sample_seq",
            "last_applied_command_seq", "command_status_flags",
        ]
        for actuator in self.catalog.actuators:
            prefix = f"p{actuator.port_id}_{actuator.joint_name}"
            header.extend([
                f"{prefix}_cmd_q", f"{prefix}_cmd_dq", f"{prefix}_kp",
                f"{prefix}_kd", f"{prefix}_cmd_tau", f"{prefix}_q",
                f"{prefix}_dq", f"{prefix}_tau", f"{prefix}_temperature",
            ])
        return header


def output_path(
    root: Path,
    catalog: SweepCatalog,
    group: SweepGroup,
    targets: tuple[SweepTarget, ...],
    override: str | None,
    *,
    failed: bool = False,
) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if override:
        candidate = Path(override.replace("{timestamp}", stamp))
        path = candidate if candidate.is_absolute() else root / candidate
    else:
        selected = "-".join(target.joint_name for target in targets)
        suffix = "_failed" if failed else ""
        path = (
            root / "scripts/sweeps/logs" / catalog.robot_id / group.group_id
            / f"{stamp}_{group.group_id}_{selected}{suffix}.csv"
        )
    if failed and override:
        path = path.with_name(path.stem + "_failed" + path.suffix)
    return path


def save_result(
    path: Path,
    node: FrequencySweepNode,
    selection: DeploymentSelection,
    group: SweepGroup,
    targets: tuple[SweepTarget, ...],
    *,
    completed: bool,
) -> tuple[Path, Path]:
    if path.suffix.lower() != ".csv":
        raise ValueError("output file must use the .csv suffix")
    metadata_path = path.with_suffix(".yaml")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite sweep output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(node.csv_header())
        writer.writerows(node.records)
    state = node.state
    metadata = {
        "format_version": 1,
        "completed": completed,
        "deployment_id": selection.deployment_id,
        "robot_id": node.catalog.robot_id,
        "profile_id": node.catalog.profile_id,
        "profile_hash": int(state.profile_hash) if state is not None else None,
        "lower_config_fingerprint": node.catalog.lower_config_fingerprint,
        "robot_profile": str(node.catalog.robot_profile_path),
        "sweep_profile": str(node.catalog.sweep_profile_path),
        "group_id": group.group_id,
        "mode": group.mode,
        "selected_joints": [target.joint_name for target in targets],
        "selected_ports": [target.port_id for target in targets],
        "single_joint": len(targets) == 1,
        "start_frequency_hz": group.start_frequency_hz,
        "end_frequency_hz": group.end_frequency_hz,
        "duration_sec": group.duration_sec,
        "kp": group.kp,
        "kd": group.kd,
        "hold_kp": node.catalog.defaults.get("hold_kp"),
        "hold_kd": node.catalog.defaults.get("hold_kd"),
        "targets": [
            {
                "joint_name": target.joint_name,
                "port_id": target.port_id,
                "phase_sign": target.phase_sign,
                "amplitude": target.amplitude,
            }
            for target in targets
        ],
        "rows": len(node.records),
    }
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path, metadata_path


def print_catalog(catalog: SweepCatalog) -> None:
    hold_kp = catalog.defaults.get("hold_kp")
    hold_kd = catalog.defaults.get("hold_kd")
    hold_text = (
        "profile defaults" if hold_kp is None or hold_kd is None
        else f"Kp={float(hold_kp):g}/Kd={float(hold_kd):g}"
    )
    print(
        f"robot={catalog.robot_id} profile={catalog.profile_id} "
        f"fingerprint=0x{catalog.lower_config_fingerprint:08x} hold={hold_text}")
    for group in catalog.groups.values():
        state = "enabled" if group.enabled else "disabled"
        targets = ", ".join(
            f"{target.joint_name}=P{target.port_id}" for target in group.targets)
        print(f"{group.group_id}: {state}, {group.mode}, {targets}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile-driven guided ESD-Link frequency sweep")
    parser.add_argument("--root", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--deployment", default=None,
        help="temporary robot_id/contract_id override; defaults to active_deployment")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group", default=None, help="configured sweep group id")
    selection.add_argument(
        "--joint", action="append", dest="joints", default=None,
        help="configured joint name; repeat only for joints in the same group")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="list resolved groups and ports")
    mode.add_argument(
        "--validate-only", action="store_true",
        help="validate profiles and optional selection without starting hardware")
    mode.add_argument(
        "--resolve-robot-profile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sweep-kp", type=float, default=None)
    parser.add_argument("--sweep-kd", type=float, default=None)
    parser.add_argument(
        "--output-file", default=None,
        help="CSV path relative to Deploy root; {timestamp} is expanded")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = find_deploy_root(args.root)
        selection, catalog = resolve_catalog(root, args.deployment)
        if args.resolve_robot_profile:
            print(selection.robot_profile_path)
            return 0
        if args.list:
            print(f"deployment={selection.deployment_id}")
            print_catalog(catalog)
            return 0
        if args.validate_only:
            if args.group is not None or args.joints:
                group, targets = select_group(
                    catalog, group_id=args.group, joint_names=args.joints)
                group = apply_gain_overrides(
                    group, sweep_kp=args.sweep_kp, sweep_kd=args.sweep_kd)
                print(
                    f"valid deployment={selection.deployment_id} group={group.group_id} "
                    f"joints={target_label(targets)} sweep_kp={group.kp:g} "
                    f"sweep_kd={group.kd:g} "
                    f"hold_kp={catalog.defaults.get('hold_kp')} "
                    f"hold_kd={catalog.defaults.get('hold_kd')}")
            else:
                if args.sweep_kp is not None or args.sweep_kd is not None:
                    raise ValueError("gain overrides require --group or --joint")
                print(f"valid deployment={selection.deployment_id} groups={len(catalog.groups)}")
            return 0
        group, targets = select_group(
            catalog, group_id=args.group, joint_names=args.joints)
        group = apply_gain_overrides(
            group, sweep_kp=args.sweep_kp, sweep_kd=args.sweep_kd)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(
        f"deployment={selection.deployment_id} group={group.group_id} "
        f"joints={target_label(targets)} mode={group.mode}", flush=True)
    intended_output = output_path(
        root, catalog, group, targets, args.output_file)
    intended_metadata = intended_output.with_suffix(".yaml")
    if intended_output.suffix.lower() != ".csv":
        print("[ERROR] output file must use the .csv suffix", file=sys.stderr)
        return 2
    if intended_output.exists() or intended_metadata.exists():
        print(
            f"[ERROR] refusing to overwrite sweep output {intended_output}",
            file=sys.stderr,
        )
        return 2
    # Keep the context alive while Ctrl-C/SIGTERM unwinds through finally so
    # the DISABLED service call can still complete and be confirmed.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = FrequencySweepNode(catalog, group, targets)
    armed = False
    completed = False
    saved = False
    try:
        state = node.wait_for_disabled_state()
        initial = node.initialize_command(state)
        port_index = {
            actuator.port_id: index for index, actuator in enumerate(catalog.actuators)}
        name_index = {
            actuator.joint_name: index for index, actuator in enumerate(catalog.actuators)}
        recovery = list(initial)
        for name, position in group.preparation_pose_rad.items():
            recovery[name_index[name]] = position
        print(
            "样机 Profile、端口和下位机 fingerprint 已匹配且当前为 DISABLED；"
            "自动进入 SWEEP 并平滑归位。",
            flush=True,
        )
        node.set_mode(SetControlMode.Request.SWEEP)
        armed = True
        node.allow_sweep_transition = True
        node.recording = True
        node.start_streaming()
        node.wait_for_sweep_state()
        distance = max(
            (abs(before - after) for before, after in zip(initial, recovery)),
            default=0.0,
        )
        ramp_duration = max(
            float(catalog.defaults["return_min_duration_sec"]),
            distance / float(catalog.defaults["return_speed_rad_s"]),
        )

        def ramp_update(elapsed: float, total: float) -> None:
            alpha = min(elapsed / total, 1.0)
            for index, actuator in enumerate(catalog.actuators):
                if actuator.control == "position":
                    node.command.position_rad[index] = (
                        initial[index] + (recovery[index] - initial[index]) * alpha)
                else:
                    node.command.velocity_rad_s[index] = 0.0

        node.run_phase(ramp_duration, ramp_update)
        node.command.position_rad = list(recovery)
        node.refresh_command_snapshot()
        node.run_phase(
            float(catalog.defaults["settle_duration_sec"]),
            lambda _elapsed, _total: None,
        )
        for target in targets:
            index = port_index[target.port_id]
            node.command.kp[index] = group.kp
            node.command.kd[index] = group.kd
        node.refresh_command_snapshot()

        def set_sweep_targets(value: float) -> None:
            for target in targets:
                index = port_index[target.port_id]
                amplitude = target.amplitude
                target_value = target.phase_sign * amplitude * value
                if group.mode == "velocity":
                    node.command.velocity_rad_s[index] = target_value
                else:
                    node.command.position_rad[index] = recovery[index] + target_value

        def formal_update(elapsed: float, total: float) -> None:
            set_sweep_targets(
                chirp_value(
                    elapsed, total,
                    group.start_frequency_hz, group.end_frequency_hz),
            )

        print(
            f"归位完成，自动开始 {group.duration_sec:.1f} 秒正式 chirp。",
            flush=True,
        )
        node.run_phase(group.duration_sec, formal_update)
        set_sweep_targets(0.0)
        node.refresh_command_snapshot()
        node.run_phase(
            float(catalog.defaults["settle_duration_sec"]),
            lambda _elapsed, _total: None,
        )
        node.stop_streaming()
        node.set_mode(SetControlMode.Request.DISABLED)
        armed = False
        completed = True
        csv_path, metadata_path = save_result(
            intended_output, node, selection, group, targets, completed=True)
        saved = True
        print(
            f"扫频完成并已确认整体失能。CSV={csv_path} metadata={metadata_path} "
            f"rows={len(node.records)}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        node.stop_streaming()
        if armed:
            try:
                node.set_mode(SetControlMode.Request.DISABLED)
                print("整体失能已确认。", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[ERROR] 无法确认整体失能：{exc}；请立即使用硬件断电。",
                    file=sys.stderr,
                    flush=True,
                )
        if node.records and not saved:
            try:
                failed_output = output_path(
                    root, catalog, group, targets, args.output_file, failed=True)
                csv_path, metadata_path = save_result(
                    failed_output, node, selection, group, targets,
                    completed=completed)
                print(
                    f"失败前数据已保存：CSV={csv_path} metadata={metadata_path}",
                    flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] 无法保存失败数据：{exc}", file=sys.stderr)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
