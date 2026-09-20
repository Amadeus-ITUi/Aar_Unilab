#!/usr/bin/env python3
"""LEGACY Direct-CAN guided sweep for a configured mirrored motor pair.

One initial 0x02 status request is sent per motor. The script then performs the
normal 1..6 feedback check and enable sequence before sending MIT 0x01 frames
at the configured rate. It never uses motors 7 or 8. The /joy subscriber is
used only for the three manual confirmations.

The supported ESD-Link entrypoint is run_sweep_motor14.sh, which executes
deploy_tools/esd_pair_sweep. This file is retained only for historical CAN
logs and must not be run while the ESD-Link backend is active.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import can
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy


MOTOR_COUNT = 6
SUPPORTED_SWEEP_PAIRS = {(1, 4), (2, 5), (3, 6)}
P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -50.0, 50.0
T_MIN, T_MAX = -14.0, 14.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0
IDX_RUN_MODE = 0x7005
RUN_MODE_MIT = 0


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def float_list(value: Any, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return [finite_float(item, f"{name}[{i}]") for i, item in enumerate(value)]


def int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [int(item) for item in value]
    if any(item < 1 or item > MOTOR_COUNT for item in result):
        raise ValueError(f"{name} may only contain motor ids 1..6")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate motor ids")
    return result


def bool_list(value: Any, name: str, length: int) -> list[bool]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return [bool(item) for item in value]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")

    swept_ids = int_list(config.get("swept_motor_ids"), "swept_motor_ids")
    sweep_pair = tuple(swept_ids)
    if sweep_pair not in SUPPORTED_SWEEP_PAIRS:
        raise ValueError("swept_motor_ids must be [1, 4], [2, 5], or [3, 6]")
    hold_ids = int_list(config.get("hold_motor_ids"), "hold_motor_ids")
    if set(swept_ids) & set(hold_ids):
        raise ValueError("swept_motor_ids and hold_motor_ids must be disjoint")
    if set(swept_ids) | set(hold_ids) != set(range(1, MOTOR_COUNT + 1)):
        raise ValueError("swept_motor_ids + hold_motor_ids must cover motors 1..6")

    config["swept_motor_ids"] = swept_ids
    config["hold_motor_ids"] = hold_ids
    config["motor_ids"] = int_list(config.get("motor_ids", list(range(1, MOTOR_COUNT + 1))), "motor_ids")
    if config["motor_ids"] != list(range(1, MOTOR_COUNT + 1)):
        raise ValueError("motor_ids must be exactly [1, 2, 3, 4, 5, 6]")
    config["sweep_base_def_pos_policy_rad"] = float_list(
        config.get("sweep_base_def_pos_policy_rad"),
        "sweep_base_def_pos_policy_rad",
        MOTOR_COUNT,
    )
    config["zero_reference_offset_policy_rad"] = float_list(
        config.get("zero_reference_offset_policy_rad"),
        "zero_reference_offset_policy_rad",
        MOTOR_COUNT,
    )
    config["flipped_motors"] = bool_list(config.get("flipped_motors"), "flipped_motors", MOTOR_COUNT)

    can_cfg = config.get("can", {})
    joy_cfg = config.get("joy", {})
    return_cfg = config.get("return_to_def_pos", {})
    hold_cfg = config.get("hold", {})
    sweep_cfg = config.get("sweep", {})
    for name, value in (("can", can_cfg), ("joy", joy_cfg), ("return_to_def_pos", return_cfg), ("hold", hold_cfg), ("sweep", sweep_cfg)):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")

    hold_cfg["kp"] = float_list(hold_cfg.get("kp"), "hold.kp", len(hold_ids))
    hold_cfg["kd"] = float_list(hold_cfg.get("kd"), "hold.kd", len(hold_ids))
    sweep_cfg["signs"] = float_list(sweep_cfg.get("signs"), "sweep.signs", len(swept_ids))
    sweep_cfg["signs"] = [1.0 if item >= 0.0 else -1.0 for item in sweep_cfg["signs"]]
    control_mode = str(sweep_cfg.get("control_mode", "position")).strip().lower()
    expected_mode = "velocity" if sweep_pair == (3, 6) else "position"
    if control_mode != expected_mode:
        raise ValueError(
            f"sweep.control_mode must be {expected_mode!r} for motors {swept_ids}"
        )
    sweep_cfg["control_mode"] = control_mode

    publish_rate = finite_float(sweep_cfg.get("publish_rate_hz", 200.0), "sweep.publish_rate_hz")
    record_rate = finite_float(sweep_cfg.get("record_rate_hz", publish_rate), "sweep.record_rate_hz")
    if publish_rate <= 0.0 or record_rate <= 0.0:
        raise ValueError("publish_rate_hz and record_rate_hz must be positive")
    if not math.isclose(publish_rate, 200.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("sweep.publish_rate_hz is fixed at 200.0 Hz")
    if record_rate > publish_rate:
        sweep_cfg["record_rate_hz"] = publish_rate
    if control_mode == "velocity":
        velocity_amplitude = finite_float(
            sweep_cfg.get("velocity_amplitude_rad_s", 5.0),
            "sweep.velocity_amplitude_rad_s",
        )
        if velocity_amplitude <= 0.0 or velocity_amplitude > V_MAX:
            raise ValueError(
                f"sweep.velocity_amplitude_rad_s must be in (0, {V_MAX}]"
            )
        sweep_kp = finite_float(sweep_cfg.get("kp", 0.0), "sweep.kp")
        if not math.isclose(sweep_kp, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("velocity sweep requires sweep.kp = 0.0")

    return config


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def advance_tick(deadline: float, period: float) -> float:
    next_deadline = deadline + period
    now = time.perf_counter()
    if next_deadline <= now:
        return now + period
    return next_deadline


def build_can_id(command_type: int, motor_id: int, data_field: int = 0) -> int:
    return ((command_type & 0x1F) << 24) | ((data_field & 0xFFFF) << 8) | (motor_id & 0xFF)


def float_to_uint(value: float, low: float, high: float, bits: int = 16) -> int:
    value = clamp(value, low, high)
    return int((value - low) * ((1 << bits) - 1) / (high - low))


def uint_to_float(value: int, low: float, high: float, bits: int = 16) -> float:
    return float(value) * (high - low) / ((1 << bits) - 1) + low


def encode_mit(position: float, velocity: float, kp: float, kd: float, effort: float = 0.0) -> bytes:
    return bytes(
        [
            (float_to_uint(position, P_MIN, P_MAX) >> 8) & 0xFF,
            float_to_uint(position, P_MIN, P_MAX) & 0xFF,
            (float_to_uint(velocity, V_MIN, V_MAX) >> 8) & 0xFF,
            float_to_uint(velocity, V_MIN, V_MAX) & 0xFF,
            (float_to_uint(kp, KP_MIN, KP_MAX) >> 8) & 0xFF,
            float_to_uint(kp, KP_MIN, KP_MAX) & 0xFF,
            (float_to_uint(kd, KD_MIN, KD_MAX) >> 8) & 0xFF,
            float_to_uint(kd, KD_MIN, KD_MAX) & 0xFF,
        ]
    )


def send_frame(bus: can.BusABC, command_type: int, motor_id: int, payload: bytes, data_field: int = 0) -> None:
    bus.send(
        can.Message(
            arbitration_id=build_can_id(command_type, motor_id, data_field),
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


def parse_status(message: can.Message) -> tuple[int, float, float, float, float] | None:
    if not message.is_extended_id:
        return None
    can_id = message.arbitration_id & 0x1FFFFFFF
    if ((can_id >> 24) & 0x1F) != 0x02:
        return None
    motor_id = (can_id >> 8) & 0xFF
    if (can_id & 0xFF) != 0:
        return None
    data = bytes(message.data).ljust(8, b"\x00")
    pos = uint_to_float((data[0] << 8) | data[1], P_MIN, P_MAX)
    vel = uint_to_float((data[2] << 8) | data[3], V_MIN, V_MAX)
    torque = uint_to_float((data[4] << 8) | data[5], T_MIN, T_MAX)
    temperature = ((data[6] << 8) | data[7]) * 0.1
    return motor_id, pos, vel, torque, temperature


class JoyConfirm(Node):
    def __init__(self, button_index: int, node_name: str):
        super().__init__(node_name)
        self.button_index = button_index
        self.armed = False
        self.last_pressed = False
        self.event = False
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Joy, "/joy", self.callback, qos)

    def callback(self, message: Joy) -> None:
        pressed = self.button_index >= 0 and self.button_index < len(message.buttons) and message.buttons[self.button_index] == 1
        if not pressed:
            self.armed = True
        if self.armed and pressed and not self.last_pressed:
            self.event = True
            self.armed = False
        self.last_pressed = pressed

    def consume(self) -> bool:
        if not self.event:
            return False
        self.event = False
        return True


class DirectSweep:
    def __init__(self, config: dict[str, Any], config_path: Path, bus: can.BusABC, joy: JoyConfirm):
        self.config = config
        self.config_path = config_path
        self.bus = bus
        self.joy = joy
        self.ids = config["motor_ids"]
        self.swept_ids = config["swept_motor_ids"]
        self.hold_ids = config["hold_motor_ids"]
        self.swept_label = "/".join(str(motor_id) for motor_id in self.swept_ids)
        self.hold_label = "/".join(str(motor_id) for motor_id in self.hold_ids)
        self.sweep_base_policy = config["sweep_base_def_pos_policy_rad"]
        self.zero_reference_offset_policy = config["zero_reference_offset_policy_rad"]
        self.sweep_def_policy = [
            base + offset
            for base, offset in zip(
                self.sweep_base_policy,
                self.zero_reference_offset_policy,
            )
        ]
        self.flipped = config["flipped_motors"]
        self.sweep_def_raw = [
            (-value if self.flipped[i] else value)
            for i, value in enumerate(self.sweep_def_policy)
        ]

        self.can_cfg = config["can"]
        self.joy_cfg = config["joy"]
        self.return_cfg = config["return_to_def_pos"]
        self.hold_cfg = config["hold"]
        self.sweep_cfg = config["sweep"]
        self.control_mode = self.sweep_cfg["control_mode"]
        self.publish_rate = finite_float(self.sweep_cfg.get("publish_rate_hz", 200.0), "sweep.publish_rate_hz")
        self.record_rate = min(
            self.publish_rate,
            finite_float(self.sweep_cfg.get("record_rate_hz", self.publish_rate), "sweep.record_rate_hz"),
        )
        self.status: dict[int, tuple[float, float, float, float]] = {}
        self.return_start: dict[int, float] = {}
        self.latest_targets = list(self.sweep_def_raw)
        self.latest_velocity_targets = [0.0] * MOTOR_COUNT
        self.records: list[list[float]] = []
        self.enabled = False
        self.enabled_ids: list[int] = []
        self.control_started = False
        self.torque_nm = finite_float(self.sweep_cfg.get("torque_nm", 0.0), "sweep.torque_nm")
        if not T_MIN <= self.torque_nm <= T_MAX:
            raise ValueError(f"sweep.torque_nm must be in [{T_MIN}, {T_MAX}]")
        default_output = f"logs/sweep_motor{''.join(str(motor_id) for motor_id in self.swept_ids)}_{{timestamp}}.csv"
        self.output_path = self.make_output_path(str(self.sweep_cfg.get("output_file", default_output)))

    def make_output_path(self, value: str) -> Path:
        path = Path(value.replace("{timestamp}", time.strftime("%Y%m%d_%H%M%S")))
        if not path.is_absolute():
            path = self.config_path.parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def read_initial_status_once(self) -> None:
        print(
            "扫频基础姿态(policy): "
            + ", ".join(f"{value:+.5f}" for value in self.sweep_base_policy),
            flush=True,
        )
        print(
            "零点参考偏值(policy): "
            + ", ".join(
                f"{value:+.5f}" for value in self.zero_reference_offset_policy
            ),
            flush=True,
        )
        print(
            "最终扫频 def-pos(raw): "
            + ", ".join(f"{value:+.5f}" for value in self.sweep_def_raw),
            flush=True,
        )
        direction_terms = []
        for motor_id, policy_sign in zip(
            self.swept_ids,
            self.sweep_cfg["signs"],
        ):
            raw_sign = -policy_sign if self.flipped[motor_id - 1] else policy_sign
            direction_terms.append(
                f"motor{motor_id}: policy={policy_sign:+.0f}sin, raw={raw_sign:+.0f}sin"
            )
        print("扫频方向: " + "; ".join(direction_terms), flush=True)
        timeout = finite_float(self.can_cfg.get("initial_status_timeout_sec", 1.0), "can.initial_status_timeout_sec")
        for motor_id in self.ids:
            # Exactly one status request per motor; the configured startup check
            # decides whether missing feedback is fatal after this pass.
            send_frame(self.bus, 0x02, motor_id, bytes(8), int(self.can_cfg.get("master_id", 0)))
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() < deadline:
            message = self.bus.recv(timeout=0.005)
            if message is None:
                continue
            parsed = parse_status(message)
            if parsed is None:
                continue
            motor_id, pos, vel, torque, temperature = parsed
            if motor_id in self.ids:
                self.status[motor_id] = (pos, vel, torque, temperature)

        print(
            f"=== motor {self.swept_label} initial state (one 0x02 sample pass) ===",
            flush=True,
        )
        for motor_id in self.ids:
            state = self.status.get(motor_id)
            if state is None:
                print(f"motor_{motor_id}: no response", flush=True)
            else:
                pos, vel, torque, temperature = state
                print(
                    f"motor_{motor_id}: pos={pos:+.6f} rad vel={vel:+.6f} "
                    f"torque={torque:+.6f} temp={temperature:.1f} C",
                    flush=True,
                )
        if bool(self.can_cfg.get("require_initial_feedback", True)):
            missing = [motor_id for motor_id in self.ids if motor_id not in self.status]
            if missing:
                raise RuntimeError(f"启动反馈检查失败，未收到电机: {missing}")
        for motor_id in self.swept_ids:
            self.return_start[motor_id] = self.status.get(
                motor_id,
                (self.sweep_def_raw[motor_id - 1], 0.0, 0.0, 0.0),
            )[0]

    def enable_all_motors(self) -> None:
        if not bool(self.can_cfg.get("enable_motors", True)):
            return
        retries = int(self.can_cfg.get("enable_retries", 5))
        master_id = int(self.can_cfg.get("master_id", 0))
        for motor_id in self.ids:
            for _ in range(retries):
                send_frame(self.bus, 0x03, motor_id, bytes(8), master_id)
                time.sleep(0.01)
            # Match RS05MotorDriver::MotorUnlock before switching to MIT mode.
            time.sleep(0.50)
            send_frame(
                self.bus,
                0x12,
                motor_id,
                encode_set_param_u8(IDX_RUN_MODE, RUN_MODE_MIT),
                master_id,
            )
            time.sleep(0.10)
            self.enabled_ids.append(motor_id)
            self.enabled = True
        print("1~6 号电机已使能，未设置零点。", flush=True)

    def disable_all_motors(self) -> None:
        if not self.enabled_ids or not bool(self.can_cfg.get("disable_on_exit", True)):
            return
        master_id = int(self.can_cfg.get("master_id", 0))
        for motor_id in list(self.enabled_ids):
            try:
                send_frame(self.bus, 0x04, motor_id, bytes(8), master_id)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] motor_{motor_id} 禁用帧发送失败: {exc}", file=sys.stderr)
        self.enabled_ids.clear()
        self.enabled = False

    def gain_for(self, motor_id: int, kp: float, kd: float) -> tuple[float, float]:
        if motor_id in self.hold_ids:
            index = self.hold_ids.index(motor_id)
            return self.hold_cfg["kp"][index], self.hold_cfg["kd"][index]
        return kp, kd

    def make_targets_and_gains(
        self,
        phase: str,
        progress: float = 1.0,
        signal: float = 0.0,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        targets = list(self.sweep_def_raw)
        velocities = [0.0] * MOTOR_COUNT
        kp = [0.0] * MOTOR_COUNT
        kd = [0.0] * MOTOR_COUNT
        return_kp = finite_float(self.return_cfg.get("kp", 1.0), "return_to_def_pos.kp")
        return_kd = finite_float(self.return_cfg.get("kd", 0.1), "return_to_def_pos.kd")
        sweep_kp = finite_float(self.sweep_cfg.get("kp", 2.0), "sweep.kp")
        sweep_kd = finite_float(self.sweep_cfg.get("kd", 0.1), "sweep.kd")

        for motor_id in self.ids:
            index = motor_id - 1
            if motor_id in self.hold_ids:
                kp[index], kd[index] = self.gain_for(motor_id, return_kp, return_kd)
                continue
            if self.control_mode == "velocity":
                # Kp=0 makes the position payload inactive. Keep the initial
                # wheel position in the frame and control only velocity via Kd.
                targets[index] = self.return_start[motor_id]
                kp[index] = 0.0
                kd[index] = sweep_kd if phase == "sweep" else return_kd
                if phase == "sweep":
                    policy_velocity = (
                        signal
                        * self.sweep_cfg["signs"][
                            self.swept_ids.index(motor_id)
                        ]
                    )
                    velocities[index] = (
                        -policy_velocity
                        if self.flipped[index]
                        else policy_velocity
                    )
                continue
            if phase == "return":
                start = self.return_start[motor_id]
                targets[index] = start + (self.sweep_def_raw[index] - start) * progress
                kp[index], kd[index] = return_kp, return_kd
            elif phase == "sweep":
                policy_command = signal * self.sweep_cfg["signs"][self.swept_ids.index(motor_id)]
                targets[index] = self.sweep_def_raw[index] + (
                    -policy_command if self.flipped[index] else policy_command
                )
                kp[index], kd[index] = sweep_kp, sweep_kd
            else:
                kp[index], kd[index] = return_kp, return_kd
        return targets, velocities, kp, kd

    def send_control(
        self,
        targets: list[float],
        velocities: list[float],
        kp: list[float],
        kd: list[float],
    ) -> None:
        self.control_started = True
        # In MIT 0x01, the CAN-ID data field is the uint16 torque value.
        # Zero Nm is the midpoint of [-14, 14], i.e. 0x7fff, not integer 0.
        torque_uint = float_to_uint(self.torque_nm, T_MIN, T_MAX)
        for motor_id in self.ids:
            index = motor_id - 1
            send_frame(
                self.bus,
                0x01,
                motor_id,
                encode_mit(
                    targets[index],
                    velocities[index],
                    kp[index],
                    kd[index],
                ),
                torque_uint,
            )
        self.latest_targets = list(targets)
        self.latest_velocity_targets = list(velocities)

    def drain_feedback(self) -> None:
        # Feedback is recorded when naturally available, but never used for
        # online detection, stopping, or phase transitions.
        while True:
            message = self.bus.recv(timeout=0.0)
            if message is None:
                return
            parsed = parse_status(message)
            if parsed is None:
                continue
            motor_id, pos, vel, torque, temperature = parsed
            if motor_id in self.ids:
                self.status[motor_id] = (pos, vel, torque, temperature)

    def record(self, timestamp: float) -> None:
        row: list[float] = [timestamp]
        for motor_id in self.ids:
            index = motor_id - 1
            state = self.status.get(motor_id, (math.nan, math.nan, math.nan, math.nan))
            row.extend(
                [
                    self.latest_targets[index],
                    self.latest_velocity_targets[index],
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                ]
            )
        self.records.append(row)

    def wait_for_button(self, message: str) -> None:
        timeout = finite_float(self.joy_cfg.get("timeout_sec", 120.0), "joy.timeout_sec")
        print(message, flush=True)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.joy, timeout_sec=0.01)
            self.drain_feedback()
            if self.joy.consume():
                return
        raise TimeoutError("手柄确认超时")

    def hold_until_button(self, message: str, kp_phase: str) -> None:
        timeout = finite_float(self.joy_cfg.get("timeout_sec", 120.0), "joy.timeout_sec")
        print(message, flush=True)
        deadline = time.monotonic() + timeout
        period = 1.0 / self.publish_rate
        next_tick = time.perf_counter()
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.perf_counter()
            if now >= next_tick:
                targets, velocities, kp, kd = self.make_targets_and_gains(kp_phase)
                self.send_control(targets, velocities, kp, kd)
                next_tick = advance_tick(next_tick, period)
            self.drain_feedback()
            rclpy.spin_once(self.joy, timeout_sec=0.001)
            if self.joy.consume():
                return
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(min(sleep_for, 0.001))
        raise TimeoutError("手柄确认超时")

    def run(self) -> None:
        self.read_initial_status_once()
        self.enable_all_motors()
        if self.control_mode == "velocity":
            first_gate_message = (
                f"请确认安全，释放手柄 A 键后第一次按 A：{self.hold_label} "
                "以 P=1,D=0.1 保持扫频专用 def-pos，"
                f"{self.swept_label} 进入 Kp=0 的 0 rad/s D-only 保持。"
            )
        else:
            first_gate_message = (
                f"请确认安全，释放手柄 A 键后第一次按 A，开始 "
                f"{self.swept_label} 以 P=1,D=0.1 缓慢回扫频专用 def-pos。"
            )
        self.wait_for_button(first_gate_message)

        return_speed = finite_float(self.return_cfg.get("speed_rad_s", 0.1), "return_to_def_pos.speed_rad_s")
        min_duration = finite_float(self.return_cfg.get("min_duration_sec", 1.0), "return_to_def_pos.min_duration_sec")
        if self.control_mode == "velocity":
            duration = min_duration
        else:
            max_distance = max(
                abs(self.return_start[motor_id] - self.sweep_def_raw[motor_id - 1])
                for motor_id in self.swept_ids
            )
            duration = max(min_duration, max_distance / return_speed)
        period = 1.0 / self.publish_rate
        start = time.perf_counter()
        next_tick = start
        next_record = start
        if self.control_mode == "velocity":
            print(
                f"开始保持扫频专用姿态，{self.swept_label} 发送 0 rad/s；"
                f"持续 {duration:.2f}s，不检测实际到位。",
                flush=True,
            )
        else:
            print(
                f"开始回扫频专用 def-pos，轨迹时长 {duration:.2f}s；"
                "不检测实际到位。",
                flush=True,
            )
        while rclpy.ok() and time.perf_counter() - start < duration:
            now = time.perf_counter()
            if now >= next_tick:
                progress = min(max((now - start) / duration, 0.0), 1.0)
                targets, velocities, kp, kd = self.make_targets_and_gains(
                    "return",
                    progress,
                )
                self.send_control(targets, velocities, kp, kd)
                if now >= next_record:
                    self.record(now - start)
                    next_record += 1.0 / self.record_rate
                next_tick = advance_tick(next_tick, period)
            self.drain_feedback()
            rclpy.spin_once(self.joy, timeout_sec=0.001)
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(min(sleep_for, 0.001))

        targets, velocities, kp, kd = self.make_targets_and_gains("hold")
        self.send_control(targets, velocities, kp, kd)
        self.hold_until_button(
            "已发送到扫频专用 def-pos。人工确认后第二次按手柄 A 键，"
            f"进入 hold 阶段（hold 仅 {self.hold_label}）。",
            "hold",
        )
        self.hold_until_button(
            f"hold 阶段保持中。人工确认后第三次按手柄 A 键，"
            f"开始 {self.swept_label} 正式扫频。",
            "hold",
        )

        frequency_start = finite_float(self.sweep_cfg.get("start_frequency_hz", 0.1), "sweep.start_frequency_hz")
        frequency_end = finite_float(self.sweep_cfg.get("end_frequency_hz", 5.0), "sweep.end_frequency_hz")
        if self.control_mode == "velocity":
            amplitude = finite_float(
                self.sweep_cfg.get("velocity_amplitude_rad_s", 5.0),
                "sweep.velocity_amplitude_rad_s",
            )
            amplitude_unit = "rad/s"
        else:
            amplitude = finite_float(
                self.sweep_cfg.get("amplitude_rad", 0.15),
                "sweep.amplitude_rad",
            )
            amplitude_unit = "rad"
        sweep_duration = finite_float(self.sweep_cfg.get("duration_sec", 40.0), "sweep.duration_sec")
        start = time.perf_counter()
        next_tick = start
        next_record = start
        print(
            f"开始 {self.swept_label} 扫频: "
            f"{frequency_start:.3f}->{frequency_end:.3f}Hz, "
            f"amplitude={amplitude:.4f}{amplitude_unit}",
            flush=True,
        )
        while rclpy.ok() and time.perf_counter() - start < sweep_duration:
            now = time.perf_counter()
            if now >= next_tick:
                elapsed = now - start
                phase = frequency_start * elapsed + 0.5 * (frequency_end - frequency_start) * elapsed * elapsed / sweep_duration
                signal = amplitude * math.sin(2.0 * math.pi * phase)
                targets, velocities, kp, kd = self.make_targets_and_gains(
                    "sweep",
                    signal=signal,
                )
                self.send_control(targets, velocities, kp, kd)
                if now >= next_record:
                    self.record(elapsed)
                    next_record += 1.0 / self.record_rate
                next_tick = advance_tick(next_tick, period)
            self.drain_feedback()
            rclpy.spin_once(self.joy, timeout_sec=0.001)
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(min(sleep_for, 0.001))

        targets, velocities, kp, kd = self.make_targets_and_gains("hold")
        self.send_control(targets, velocities, kp, kd)
        self.save_csv()
        if self.control_mode == "velocity":
            completion_message = (
                "扫频完成，3/6 号目标速度已回到 0 rad/s；"
                "退出时将发送 0x04 禁用帧。"
            )
        else:
            completion_message = (
                "扫频完成，已回到扫频专用 def-pos；"
                "退出时将发送 0x04 禁用帧。"
            )
        print(completion_message, flush=True)

    def save_csv(self) -> None:
        header = ["timestamp"]
        for motor_id in self.ids:
            header.extend(
                [
                    f"motor_{motor_id}_target_raw",
                    f"motor_{motor_id}_target_vel_raw",
                    f"motor_{motor_id}_pos_raw",
                    f"motor_{motor_id}_vel_raw",
                    f"motor_{motor_id}_torque_raw",
                    f"motor_{motor_id}_temp_c",
                ]
            )
        with self.output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(self.records)
        print(f"CSV 已保存: {self.output_path} ({len(self.records)} rows)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct-CAN guided mirrored-motor sweep")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("sweep_motor14.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    bus = None
    joy = None
    controller = None
    try:
        config = load_config(config_path)
        if not rclpy.ok():
            rclpy.init()
        pair_name = "".join(str(motor_id) for motor_id in config["swept_motor_ids"])
        joy = JoyConfirm(
            int(config["joy"].get("confirm_button", 0)),
            f"sweep_motor{pair_name}_joy_confirm",
        )
        bus = can.Bus(
            interface="socketcan",
            channel=str(config["can"].get("channel", "can0")),
        )
        controller = DirectSweep(config, config_path, bus, joy)
        controller.run()
        return 0
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止脚本。", flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if controller is not None and bus is not None:
            # Leave motors at the configured sweep def-pos before shutdown.
            if controller.control_started:
                try:
                    targets, velocities, kp, kd = controller.make_targets_and_gains("hold")
                    for _ in range(max(1, int(controller.publish_rate * 0.2))):
                        controller.send_control(targets, velocities, kp, kd)
                        time.sleep(1.0 / controller.publish_rate)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] 退出保持扫频专用 def-pos 失败: {exc}", file=sys.stderr)
            # A failed hold frame must not prevent the 0x04 disable attempt.
            try:
                controller.disable_all_motors()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 退出禁用电机失败: {exc}", file=sys.stderr)
        if bus is not None:
            bus.shutdown()
        if joy is not None:
            joy.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
