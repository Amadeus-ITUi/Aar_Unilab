#!/usr/bin/env python3
"""WE11 fail-closed daily deployment supervisor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import rclpy
from esd_link_msgs.msg import LinkStatus, LowerState
from motors.msg import MotorRuntimeStatus
from motors.srv import SetOperationalState
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, UInt8
from std_srvs.srv import SetBool, Trigger

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent / "hardware"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rgb_led import RGBLed  # noqa: E402
from we11_state import (  # noqa: E402
    ButtonTracker,
    StartupMode,
    We11State,
    can_transition,
    toggle_startup_mode,
)


COLORS = {
    "white": (255, 255, 255),
    "blue": (0, 0, 255),
    "purple": (148, 0, 211),
    "cyan": (0, 255, 255),
    "green": (0, 255, 0),
    "yellow": (255, 180, 0),
    "magenta": (255, 0, 180),
    "orange": (255, 64, 0),
}

FAULT_COLORS = {
    "CAN": COLORS["blue"],
    "IMU": COLORS["purple"],
    "DEVICE": COLORS["cyan"],
    "WING": COLORS["yellow"],
    "ROS": COLORS["magenta"],
    "MAINTENANCE": COLORS["orange"],
}

SERVICE_DISCOVERY_TIMEOUT_SECONDS = 5.0
FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS = 5.0
# Six serial MotorInit() calls take about 5.1 seconds, followed by the configured
# 3-second ramp. Keep several seconds of scheduling/CAN margin above that budget.
STANDBY_TRANSITION_TIMEOUT_SECONDS = 15.0
# The service performs the configured 3-second release synchronously. Keep
# enough margin for CAN scheduling and the final MotorLock commands.
SOFT_DISARM_RESPONSE_TIMEOUT_SECONDS = 8.0
RGB_HANDOFF_READY_FD_ENV = "WE11_RGB_HANDOFF_READY_FD"
RGB_HANDOFF_RELEASE_FD_ENV = "WE11_RGB_HANDOFF_RELEASE_FD"
RGB_HANDOFF_TIMEOUT_SECONDS = 15.0


def complete_rgb_handoff() -> None:
    """Synchronize GPIO ownership with the boot-time lifecycle launcher."""
    ready_value = os.environ.pop(RGB_HANDOFF_READY_FD_ENV, None)
    release_value = os.environ.pop(RGB_HANDOFF_RELEASE_FD_ENV, None)
    if ready_value is None and release_value is None:
        return
    if ready_value is None or release_value is None:
        raise RuntimeError("incomplete RGB handoff environment")
    try:
        ready_fd = int(ready_value)
        release_fd = int(release_value)
    except ValueError as exc:
        raise RuntimeError("invalid RGB handoff file descriptor") from exc
    if ready_fd < 3 or release_fd < 3 or ready_fd == release_fd:
        raise RuntimeError("unsafe RGB handoff file descriptor")
    try:
        os.write(ready_fd, b"R")
        readable, _, _ = select.select(
            [release_fd], [], [], RGB_HANDOFF_TIMEOUT_SECONDS
        )
        if not readable or os.read(release_fd, 1) != b"R":
            raise RuntimeError("RGB handoff release was not acknowledged")
    finally:
        for fd in (ready_fd, release_fd):
            try:
                os.close(fd)
            except OSError:
                pass

def find_deploy_root() -> Path:
    configured = os.environ.get("WE11_DEPLOY_ROOT")
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "start_robot.sh").is_file() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("cannot locate Deploy root; set WE11_DEPLOY_ROOT")


@dataclass(slots=True)
class StartupFault(RuntimeError):
    category: str
    detail: int
    code: str
    reason: str
    hardware: bool = True

    def __str__(self) -> str:
        return f"{self.code}: {self.reason}"


class ManagedProcess:
    def __init__(self, name: str, command: list[str], log_file: Path) -> None:
        self.name = name
        self.command = command
        self.log_file = log_file
        self._stream = log_file.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdout=self._stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    def poll(self) -> int | None:
        return self.process.poll()

    def stop(self) -> None:
        if self.process.poll() is not None:
            self._stream.close()
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=4)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=2)
        finally:
            self._stream.close()


class SupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("we11_supervisor")
        self.lock = threading.Lock()
        self.buttons = ButtonTracker(exit_hold_seconds=0.25)
        self.joy_stamp = 0.0
        self.mode_stamp = 0.0
        self.policy_frames: deque[tuple[float, tuple[float, ...]]] = deque(maxlen=500)
        self.inference_mode = 0
        self.link_status: LinkStatus | None = None
        self.lower_state: LowerState | None = None

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Joy, "/joy", self._joy, qos_profile_sensor_data)
        self.create_subscription(
            LowerState, "/lower/state", self._lower_state, qos_profile_sensor_data
        )
        self.create_subscription(Float32MultiArray, "/policy/commands", self._policy, 10)
        self.create_subscription(UInt8, "/inference/runtime_mode", self._mode, latched)
        self.create_subscription(LinkStatus, "/lower/link_status", self._link_status, latched)

        self.motor_state_client = self.create_client(
            SetOperationalState, "/motors/set_operational_state"
        )
        self.soft_disarm_client = self.create_client(Trigger, "/motors/soft_disarm")
        self.wing_rc_client = self.create_client(SetBool, "/wing/enable_rc")
        self.policy_client = self.create_client(SetBool, "/inference/set_policy_enabled")

    @staticmethod
    def _finite(values: tuple[float, ...]) -> bool:
        return all(math.isfinite(value) for value in values)

    def _joy(self, msg: Joy) -> None:
        now = time.monotonic()
        self.buttons.update(tuple(msg.buttons), tuple(msg.axes), now)
        with self.lock:
            self.joy_stamp = now

    def _lower_state(self, msg: LowerState) -> None:
        with self.lock:
            self.lower_state = msg

    def _policy(self, msg: Float32MultiArray) -> None:
        now = time.monotonic()
        with self.lock:
            self.policy_frames.append((now, tuple(msg.data)))

    def _mode(self, msg: UInt8) -> None:
        with self.lock:
            self.inference_mode = int(msg.data)
            self.mode_stamp = time.monotonic()

    def _link_status(self, msg: LinkStatus) -> None:
        with self.lock:
            self.link_status = msg


class We11Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = find_deploy_root()
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        self.state = We11State.BOOT
        self.stop_requested = threading.Event()
        self.processes: list[ManagedProcess] = []
        self.node: SupervisorNode | None = None
        self.executor: MultiThreadedExecutor | None = None
        self.executor_thread: threading.Thread | None = None
        self.led: RGBLed | None = None
        self._cleaned = False

    def _signal(self, _signum: int, _frame: object) -> None:
        self.stop_requested.set()

    def _set_state(self, state: We11State) -> None:
        if not can_transition(self.state, state):
            raise RuntimeError(f"illegal WE11 state transition: {self.state.name} -> {state.name}")
        self.state = state
        print(f"[STATE] {state.name}", flush=True)
        if self.led is None:
            return
        if state == We11State.BOOT:
            self.led.breathe(COLORS["white"], brightness=0.5, period=2.0)
        elif state == We11State.LINK_CHECK:
            self.led.breathe(COLORS["blue"], brightness=0.5, period=2.0)
        elif state == We11State.SYSTEM_READY:
            self.led.breathe(COLORS["cyan"], brightness=0.5, period=2.0)
        elif state == We11State.WAIT_STANDBY:
            self.led.breathe(COLORS["cyan"], brightness=0.5, period=2.0)
        elif state == We11State.STANDBY_ENTERING:
            self.led.blink(COLORS["white"], brightness=0.5, interval=0.25)
        elif state == We11State.POLICY_ENTERING:
            self.led.blink(COLORS["green"], brightness=0.5, interval=0.25)
        elif state == We11State.STANDBY:
            self.led.solid(COLORS["white"], brightness=0.5)
        elif state == We11State.POLICY:
            self.led.solid(COLORS["green"], brightness=0.5)
        elif state == We11State.STOPPED:
            self.led.off()

    def _start_ros(self) -> None:
        # The supervisor owns SIGINT/SIGTERM so it can keep ROS alive long
        # enough to finish the disable transaction before shutting down rclpy.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = SupervisorNode()
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self._spin_ros, name="we11-ros-executor", daemon=True
        )
        self.executor_thread.start()

    def _spin_ros(self) -> None:
        assert self.executor is not None
        try:
            self.executor.spin()
        except Exception:
            # rclpy's SIGINT handler can invalidate the context before the
            # supervisor's orderly cleanup joins this thread.
            if rclpy.ok():
                raise

    def _start_process(self, name: str, command: list[str]) -> None:
        log_file = self.log_dir / f"we11_{name}_{self.stamp}.log"
        print(f"[START] {name}: {' '.join(command)} -> {log_file}", flush=True)
        try:
            self.processes.append(ManagedProcess(name, command, log_file))
        except Exception as exc:
            raise StartupFault("ROS", 1, f"PROCESS_START_{name.upper()}", str(exc)) from exc

    def _process(self, name: str) -> ManagedProcess | None:
        return next((process for process in self.processes if process.name == name), None)

    def _tick(self) -> None:
        if self.stop_requested.is_set() or (self.node and self.node.buttons.exit_requested):
            raise KeyboardInterrupt
        if self.led is not None:
            self.led.raise_if_failed()
        for process in self.processes:
            code = process.poll()
            if code is not None:
                raise StartupFault(
                    "ROS", 2, f"PROCESS_EXIT_{process.name.upper()}",
                    f"{process.name} exited with code {code}",
                    hardware=process.name in {"bridge", "motors", "wing"},
                )

    def _wait(self, predicate: Callable[[], bool], timeout: float, fault: StartupFault) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._tick()
            if predicate():
                return
            time.sleep(0.02)
        raise fault

    def _wait_button(self, index: int) -> None:
        assert self.node is not None
        self.node.buttons.clear(index)
        while True:
            self._tick()
            if self.node.buttons.consume(index):
                return
            time.sleep(0.02)

    def _wait_startup_confirmation(self) -> StartupMode:
        """Let X select the startup path and Y execute the selected path."""
        assert self.node is not None
        mode = StartupMode.STANDBY
        self.node.buttons.clear(2)
        self.node.buttons.clear(3)
        print(
            "[SELECT] 普通 Standby 启动（青色呼吸）；按 X 切换倒地自启，按 Y 执行",
            flush=True,
        )
        while True:
            self._tick()
            if self.node.buttons.consume(2):
                mode = toggle_startup_mode(mode)
                if mode == StartupMode.DIRECT_POLICY:
                    if self.led is not None:
                        self.led.breathe(COLORS["green"], brightness=0.5, period=2.0)
                    print(
                        "[SELECT] 倒地自启 Policy（绿色呼吸）；按 X 切回普通启动，按 Y 执行",
                        flush=True,
                    )
                else:
                    if self.led is not None:
                        self.led.breathe(COLORS["cyan"], brightness=0.5, period=2.0)
                    print(
                        "[SELECT] 普通 Standby 启动（青色呼吸）；按 X 切换倒地自启，按 Y 执行",
                        flush=True,
                    )
            if self.node.buttons.consume(3):
                return mode
            time.sleep(0.02)

    def _call(
        self,
        client: object,
        request: object,
        *,
        response_timeout: float,
        unavailable_fault: StartupFault,
        timeout_fault: StartupFault,
        availability_timeout: float = SERVICE_DISCOVERY_TIMEOUT_SECONDS,
    ) -> object:
        if not client.wait_for_service(timeout_sec=availability_timeout):
            raise unavailable_fault
        future = client.call_async(request)
        deadline = time.monotonic() + response_timeout
        while time.monotonic() < deadline:
            self._tick()
            if future.done():
                result = future.result()
                if result is None:
                    raise timeout_fault
                return result
            time.sleep(0.02)
        raise timeout_fault

    def _check_model(self) -> None:
        models = self.root / "src/inference/models"
        manifest_path = models / "lab_policy_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            item = manifest["artifacts"]["mnn"]
            model_path = models / item["file"]
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise StartupFault(
                "DEVICE", 3, "MODEL_MANIFEST_INVALID", str(exc), False
            ) from exc
        if digest != item["sha256"] or model_path.stat().st_size != item["size_bytes"]:
            raise StartupFault(
                "DEVICE", 3, "MODEL_HASH_MISMATCH", f"invalid model artifact: {model_path}", False
            )

    def _lower_state_error(self) -> str | None:
        assert self.node is not None
        with self.node.lock:
            state = self.node.lower_state
        if state is None:
            return "no LowerState"
        if state.session_id == 0 or state.schema_id != 1:
            return "invalid session/schema"
        if state.active_port_mask != 0x1FE or state.offline_port_mask != 0:
            return "P1-P8 mask/offline invalid"
        if state.fault_flags != 0 or state.control_state in (5, 6):
            return "lower controller fault"
        if state.imu_valid_mask & 0x07 != 0x07:
            return "IMU invalid"
        if sorted(state.port_id) != list(range(1, 9)):
            return "P1-P8 layout invalid"
        if any((valid & 0x03) != 0x03 for valid in state.valid_mask):
            return "P1-P8 feedback invalid"
        imu = state.imu
        values = (
            imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w,
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z,
            *state.position_rad, *state.velocity_rad_s, *state.effort_nm,
        )
        return None if self.node._finite(values) else "non-finite LowerState"

    def _lower_state_ready(self) -> bool:
        assert self.node is not None
        with self.node.lock:
            state = self.node.lower_state
        return self._lower_state_error() is None and state is not None and state.control_state == 2

    def _policy_zero_ready(self) -> bool:
        assert self.node is not None
        with self.node.lock:
            latest = self.node.policy_frames[-1][1] if self.node.policy_frames else ()
        if len(latest) != 6:
            return False
        return all(math.isfinite(value) and abs(value) <= 1.0e-6 for value in latest)

    def _link_ready(self) -> bool:
        """Validate session identity; timing fields remain diagnostics only."""
        assert self.node is not None
        with self.node.lock:
            status = self.node.link_status
        return bool(
            status is not None
            and status.session_valid
            and status.schema_id == 1
            and status.config_fingerprint != 0
            and status.active_port_mask == 0x1FE
            and status.offline_port_mask == 0
            and status.fault_flags == 0
        )

    def _runtime_ready_for_policy(self) -> tuple[bool, str]:
        assert self.node is not None
        lower_error = self._lower_state_error()
        if lower_error is not None:
            return False, lower_error
        with self.node.lock:
            state = self.node.lower_state
            latest_policy = self.node.policy_frames[-1][1] if self.node.policy_frames else ()
        if state is None or state.control_state not in (3, 4):
            return False, "lower controller is not enabled"
        if len(latest_policy) != 6 or any(not math.isfinite(value) for value in latest_policy):
            return False, "policy output invalid"
        return True, ""

    def _inference_command(self) -> list[str]:
        command = [
            "ros2", "launch", "inference", "lab_inference_node.launch.py",
            "joy_policy_gate_enabled:=false", "joy_policy_start_enabled:=false",
        ]
        if self.args.force_zero_policy:
            command.append("force_zero_policy_commands:=true")
        return command

    def _startup(self) -> None:
        complete_rgb_handoff()
        if not self.args.no_rgb:
            self.led = RGBLed(red_pin=22, green_pin=17, blue_pin=27, common_anode=True)
        else:
            print("[MAINTENANCE] supervisor RGB disabled; terminal state is authoritative", flush=True)
        self._set_state(We11State.BOOT)
        self._start_ros()

        self._set_state(We11State.LINK_CHECK)
        self._start_process(
            "bridge", ["ros2", "launch", "esd_link_bridge", "esd_link_bridge.launch.py"]
        )
        self._wait(
            self._link_ready,
            15.0,
            StartupFault(
                "CAN", 9, "ESD_LINK_CONTRACT_INVALID",
                "session/schema/fingerprint/P1-P8 link check failed",
            ),
        )
        self._wait(
            self._lower_state_ready,
            3.0,
            StartupFault("DEVICE", 4, "LOWER_FEEDBACK_INVALID", "P1-P8 feedback failed"),
        )

        self._set_state(We11State.SYSTEM_READY)
        self._check_model()
        joy_dev = os.environ.get("START_ROBOT_JOY_DEV", "/dev/input/js0")
        if not os.path.exists(joy_dev) or not os.access(joy_dev, os.R_OK):
            raise StartupFault("DEVICE", 1, "GAMEPAD_DEVICE_MISSING", joy_dev, False)
        self._start_process(
            "joy", ["ros2", "launch", "xbox", "xbox.launch.py", f"joy_dev:={joy_dev}"]
        )
        assert self.node is not None
        self._wait(
            lambda: time.monotonic() - self.node.joy_stamp < 0.5,
            8.0,
            StartupFault("DEVICE", 2, "JOY_TOPIC_TIMEOUT", "no fresh /joy", False),
        )
        if self.args.preflight_only:
            print("[PASS] ESD-Link preflight complete; P1-P8 remained DISABLED", flush=True)
            return

        if self.args.force_zero_policy:
            print(
                "[TEST] 强制零策略模式：进入 POLICY 后仍发布原生六维零命令",
                flush=True,
            )
        self._start_process("inference", self._inference_command())
        self._wait(
            self._policy_zero_ready,
            15.0,
            StartupFault(
                "ROS", 3, "STANDBY_OUTPUT_INVALID",
                "zero policy output not verified", False,
            ),
        )

        wing_request = SetBool.Request()
        wing_request.data = True
        wing_result = self._call(
            self.node.wing_rc_client,
            wing_request,
            response_timeout=FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS,
            unavailable_fault=StartupFault(
                "WING", 3, "WING_RC_SERVICE_UNAVAILABLE",
                "/wing/enable_rc was not available within 5 seconds",
            ),
            timeout_fault=StartupFault(
                "WING", 3, "WING_RC_RESPONSE_TIMEOUT",
                "/wing/enable_rc did not respond within 5 seconds",
            ),
        )
        if not wing_result.success:
            raise StartupFault("WING", 3, "WING_RC_ENABLE_REJECTED", wing_result.message)
        print("[READY] ESD-Link 自检通过；等待整体 P1-P8 Standby 确认", flush=True)
        self._set_state(We11State.WAIT_STANDBY)

    def _activate_selected_mode(self, startup_mode: StartupMode) -> None:
        """Enter the selected mode without routing direct Policy through Standby."""
        assert self.node is not None
        direct_policy = startup_mode == StartupMode.DIRECT_POLICY
        self._set_state(
            We11State.POLICY_ENTERING if direct_policy else We11State.STANDBY_ENTERING
        )
        request = SetOperationalState.Request()
        request.target_state = (
            SetOperationalState.Request.ACTIVE
            if direct_policy
            else SetOperationalState.Request.STANDBY
        )
        if direct_policy:
            print(
                "[ACTION] 倒地自启: P1-P8 整体使能，不发保持帧、不执行默认姿态斜坡；"
                "等待第一帧真实策略命令；"
                f"response timeout={STANDBY_TRANSITION_TIMEOUT_SECONDS:.1f}s",
                flush=True,
            )
        else:
            print(
                "[ACTION] ESD-Link: P1-P8 整体使能并进入 3 秒 Standby 姿态斜坡；"
                f"response timeout={STANDBY_TRANSITION_TIMEOUT_SECONDS:.1f}s",
                flush=True,
            )
        result = self._call(
            self.node.motor_state_client,
            request,
            response_timeout=STANDBY_TRANSITION_TIMEOUT_SECONDS,
            unavailable_fault=StartupFault(
                "CAN", 8,
                "DIRECT_POLICY_SERVICE_UNAVAILABLE"
                if direct_policy
                else "STANDBY_SERVICE_UNAVAILABLE",
                "/motors/set_operational_state was not available within 5 seconds",
            ),
            timeout_fault=StartupFault(
                "CAN", 8,
                "DIRECT_POLICY_TRANSITION_TIMEOUT"
                if direct_policy
                else "STANDBY_TRANSITION_TIMEOUT",
                "bridge did not confirm the selected full-device mode within 15 seconds",
            ),
        )
        expected_state = (
            MotorRuntimeStatus.ACTIVE if direct_policy else MotorRuntimeStatus.STANDBY
        )
        if not result.success or result.actual_state != expected_state:
            raise StartupFault(
                "CAN", 8,
                "DIRECT_POLICY_ACTIVATION_REJECTED"
                if direct_policy
                else "STANDBY_RAMP_REJECTED",
                result.message,
            )
        if direct_policy:
            gate_deadline = time.monotonic() + 1.0
            reason = "runtime inputs did not become ready"
            while time.monotonic() < gate_deadline:
                self._tick()
                ready, reason = self._runtime_ready_for_policy()
                if ready:
                    break
                time.sleep(0.02)
            else:
                raise StartupFault(
                    "DEVICE", 5, "DIRECT_POLICY_GATE_REJECTED", reason
                )
            policy_request = SetBool.Request()
            policy_request.data = True
            policy_result = self._call(
                self.node.policy_client,
                policy_request,
                response_timeout=FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS,
                unavailable_fault=StartupFault(
                    "ROS", 4, "INFERENCE_SERVICE_UNAVAILABLE",
                    "/inference/set_policy_enabled was not available within 5 seconds",
                ),
                timeout_fault=StartupFault(
                    "ROS", 4, "POLICY_SWITCH_RESPONSE_TIMEOUT",
                    "inference did not confirm direct Policy start within 5 seconds",
                ),
            )
            if not policy_result.success:
                raise StartupFault(
                    "ROS", 4, "DIRECT_POLICY_SWITCH_REJECTED",
                    policy_result.message,
                )
            self._wait(
                lambda: self.node is not None and self.node.inference_mode == 1
                and time.monotonic() - self.node.mode_stamp < 0.5,
                1.0,
                StartupFault(
                    "ROS", 4, "DIRECT_POLICY_MODE_TIMEOUT",
                    "inference service succeeded but fresh POLICY mode was not observed",
                ),
            )
            self._set_state(We11State.POLICY)
        else:
            self._set_state(We11State.STANDBY)

    def _hard_disarm_best_effort(self) -> None:
        """Request the existing immediate MotorLock path without masking a fault."""
        if self.node is None or not self.node.motor_state_client.service_is_ready():
            print("[WARN] immediate motor DISARMED service is unavailable", flush=True)
            return
        request = SetOperationalState.Request()
        request.target_state = SetOperationalState.Request.DISARMED
        try:
            future = self.node.motor_state_client.call_async(request)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and not future.done():
                time.sleep(0.02)
            if not future.done() or future.result() is None or not future.result().success:
                print("[WARN] immediate motor DISARMED fallback did not confirm", flush=True)
        except Exception as exc:
            print(f"[WARN] immediate motor DISARMED fallback failed: {exc}", flush=True)

    def _soft_disarm_to_wait(self) -> None:
        """Request the frozen lower controller's full-device safe disable."""
        assert self.node is not None
        print(
            "[ACTION] Y: 请求 P1-P8 整体安全失能并等待确认",
            flush=True,
        )
        try:
            result = self._call(
                self.node.soft_disarm_client,
                Trigger.Request(),
                response_timeout=SOFT_DISARM_RESPONSE_TIMEOUT_SECONDS,
                unavailable_fault=StartupFault(
                    "CAN", 8, "SOFT_DISARM_SERVICE_UNAVAILABLE",
                    "/motors/soft_disarm was not available within 5 seconds",
                ),
                timeout_fault=StartupFault(
                    "CAN", 8, "SOFT_DISARM_RESPONSE_TIMEOUT",
                    "motors did not finish the 3-second gradual release within 8 seconds",
                ),
            )
            if not result.success:
                raise StartupFault(
                    "CAN", 8, "SOFT_DISARM_REJECTED", result.message
                )
        except StartupFault:
            self._hard_disarm_best_effort()
            raise

        policy_request = SetBool.Request()
        policy_request.data = False
        policy_result = self._call(
            self.node.policy_client,
            policy_request,
            response_timeout=FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS,
            unavailable_fault=StartupFault(
                "ROS", 4, "INFERENCE_SERVICE_UNAVAILABLE",
                "/inference/set_policy_enabled was not available after soft disarm", False,
            ),
            timeout_fault=StartupFault(
                "ROS", 4, "INFERENCE_STANDBY_RESPONSE_TIMEOUT",
                "inference did not confirm Standby after soft disarm", False,
            ),
        )
        if not policy_result.success:
            raise StartupFault(
                "ROS", 4, "INFERENCE_STANDBY_REJECTED", policy_result.message, False
            )

        def motors_are_disarmed() -> bool:
            assert self.node is not None
            with self.node.lock:
                state = self.node.lower_state
            return state is not None and state.control_state == 2

        self._wait(
            motors_are_disarmed,
            1.0,
            StartupFault(
                "CAN", 8, "SOFT_DISARM_STATUS_TIMEOUT",
                "soft-disarm service succeeded but lower DISABLED was not observed",
            ),
        )
        self.node.buttons.clear(2)
        self.node.buttons.clear(3)
        self._set_state(We11State.WAIT_STANDBY)
        print(
            "[READY] P1-P8 已整体失能；重新使能前必须重新确认 Standby",
            flush=True,
        )

    def _runtime(self) -> None:
        assert self.node is not None
        policy_deadline: float | None = None
        while True:
            self._tick()
            ready, reason = self._runtime_ready_for_policy()
            if not ready:
                if "IMU" in reason:
                    raise StartupFault("IMU", 3, "RUNTIME_IMU_LOST", reason, False)
                if "motor" in reason:
                    raise StartupFault("CAN", 7, "RUNTIME_MOTOR_LOST", reason, True)
                if "wing" in reason:
                    raise StartupFault("WING", 4, "RUNTIME_WING_LOST", reason, True)
                if "policy" in reason:
                    raise StartupFault("ROS", 5, "RUNTIME_INFERENCE_LOST", reason, False)
                raise StartupFault("DEVICE", 5, "RUNTIME_DEVICE_LOST", reason, False)
            if policy_deadline is not None and time.monotonic() >= policy_deadline:
                print(
                    f"[SMOKE] {self.args.policy_smoke_seconds:.1f}s POLICY window complete; "
                    "returning to STANDBY",
                    flush=True,
                )
                self._leave_policy_to_standby()
                policy_deadline = None
                continue
            if self.node.buttons.consume(3):
                self._soft_disarm_to_wait()
                return
            if self.node.buttons.consume(2):
                entering_policy = self.state != We11State.POLICY
                if entering_policy:
                    ready, reason = self._runtime_ready_for_policy()
                    if not ready:
                        print(f"[REJECT] Policy: {reason}", flush=True)
                        time.sleep(0.1)
                        continue
                    request = SetBool.Request()
                    request.data = True
                    result = self._call(
                        self.node.policy_client,
                        request,
                        response_timeout=FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS,
                        unavailable_fault=StartupFault(
                            "ROS", 4, "INFERENCE_SERVICE_UNAVAILABLE",
                            "/inference/set_policy_enabled was not available within 5 seconds", False,
                        ),
                        timeout_fault=StartupFault(
                            "ROS", 4, "POLICY_SWITCH_RESPONSE_TIMEOUT",
                            "inference did not confirm the policy switch within 5 seconds", False,
                        ),
                    )
                    if not result.success:
                        print(f"[REJECT] Policy: {result.message}", flush=True)
                        continue
                    mode_request = SetOperationalState.Request()
                    mode_request.target_state = SetOperationalState.Request.ACTIVE
                    mode_result = self._call(
                        self.node.motor_state_client,
                        mode_request,
                        response_timeout=FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS,
                        unavailable_fault=StartupFault(
                            "CAN", 8, "POLICY_SERVICE_UNAVAILABLE",
                            "/motors/set_operational_state unavailable",
                        ),
                        timeout_fault=StartupFault(
                            "CAN", 8, "POLICY_MODE_TIMEOUT",
                            "bridge did not select POLICY within 5 seconds",
                        ),
                    )
                    if not mode_result.success or mode_result.actual_state != MotorRuntimeStatus.ACTIVE:
                        raise StartupFault("CAN", 8, "POLICY_MODE_REJECTED", mode_result.message)
                    self._set_state(We11State.POLICY)
                    smoke_seconds = getattr(self.args, "policy_smoke_seconds", 0.0)
                    if smoke_seconds > 0.0:
                        policy_deadline = time.monotonic() + smoke_seconds
                        print(
                            f"[SMOKE] POLICY will automatically return to STANDBY "
                            f"after {smoke_seconds:.1f}s",
                            flush=True,
                        )
                else:
                    self._leave_policy_to_standby()
                    policy_deadline = None
            time.sleep(0.02)

    def _leave_policy_to_standby(self) -> None:
        """Restore bridge hold before stopping inference policy publication."""
        assert self.node is not None
        mode_request = SetOperationalState.Request()
        mode_request.target_state = SetOperationalState.Request.STANDBY
        mode_result = self._call(
            self.node.motor_state_client,
            mode_request,
            response_timeout=FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS,
            unavailable_fault=StartupFault(
                "CAN", 8, "STANDBY_SERVICE_UNAVAILABLE",
                "/motors/set_operational_state unavailable",
            ),
            timeout_fault=StartupFault(
                "CAN", 8, "STANDBY_MODE_TIMEOUT",
                "bridge did not return to STANDBY",
            ),
        )
        if not mode_result.success or mode_result.actual_state != MotorRuntimeStatus.STANDBY:
            raise StartupFault("CAN", 8, "STANDBY_MODE_REJECTED", mode_result.message)
        request = SetBool.Request()
        request.data = False
        result = self._call(
            self.node.policy_client,
            request,
            response_timeout=FAST_SERVICE_RESPONSE_TIMEOUT_SECONDS,
            unavailable_fault=StartupFault(
                "ROS", 4, "INFERENCE_SERVICE_UNAVAILABLE",
                "/inference/set_policy_enabled was not available within 5 seconds", False,
            ),
            timeout_fault=StartupFault(
                "ROS", 4, "POLICY_SWITCH_RESPONSE_TIMEOUT",
                "inference did not confirm the policy switch within 5 seconds", False,
            ),
        )
        if not result.success:
            raise StartupFault("ROS", 4, "POLICY_SWITCH_REJECTED", result.message, False)
        self._set_state(We11State.STANDBY)

    def _safe_service_shutdown(self) -> None:
        if self.node is None:
            return
        try:
            request = SetBool.Request()
            request.data = False
            futures = []
            if self.node.policy_client.service_is_ready():
                futures.append(self.node.policy_client.call_async(request))
            if self.node.wing_rc_client.service_is_ready():
                futures.append(self.node.wing_rc_client.call_async(request))
            motor_request = SetOperationalState.Request()
            motor_request.target_state = SetOperationalState.Request.DISARMED
            motor_future = None
            if self.node.motor_state_client.service_is_ready():
                motor_future = self.node.motor_state_client.call_async(motor_request)
                futures.append(motor_future)

            deadline = time.monotonic() + SOFT_DISARM_RESPONSE_TIMEOUT_SECONDS
            while futures and time.monotonic() < deadline:
                if all(future.done() for future in futures):
                    break
                time.sleep(0.02)
            if motor_future is not None and not motor_future.done():
                print(
                    "[WARN] normal shutdown disable service did not finish before timeout",
                    file=sys.stderr,
                    flush=True,
                )
                return

            state_deadline = time.monotonic() + 1.0
            while time.monotonic() < state_deadline:
                with self.node.lock:
                    state = self.node.lower_state
                if state is not None and state.control_state == 2:
                    print("[STOP] lower controller confirmed DISABLED", flush=True)
                    return
                time.sleep(0.02)
            print(
                "[WARN] normal shutdown did not observe lower DISABLED",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] service shutdown failed: {exc}", file=sys.stderr, flush=True)

    def _enter_fault_standby(self) -> None:
        """Zero inference and request full-device disable on any fault."""
        if self.node is None:
            return
        try:
            request = SetBool.Request()
            request.data = False
            futures = []
            if self.node.policy_client.service_is_ready():
                futures.append(self.node.policy_client.call_async(request))
            if self.node.wing_rc_client.service_is_ready():
                futures.append(self.node.wing_rc_client.call_async(request))
            disable = SetOperationalState.Request()
            disable.target_state = SetOperationalState.Request.DISARMED
            if self.node.motor_state_client.service_is_ready():
                futures.append(self.node.motor_state_client.call_async(disable))
            deadline = time.monotonic() + 1.0
            while futures and time.monotonic() < deadline:
                if all(future.done() for future in futures):
                    break
                time.sleep(0.02)
        except Exception as exc:
            print(f"[WARN] fault-standby request failed: {exc}", file=sys.stderr, flush=True)

    def cleanup(self, *, keep_led: bool = False) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if rclpy.ok():
            self._safe_service_shutdown()
        for process in reversed(self.processes):
            try:
                process.stop()
            except Exception as exc:
                print(f"[WARN] failed to stop {process.name}: {exc}", file=sys.stderr)
        disable_tool = (
            self.root / "install/esd_link_bridge/lib/esd_link_bridge/"
            "esd_link_emergency_disable"
        )
        # Preflight never enables the lower controller. Avoid reopening the
        # serial port after a clean read-only bridge shutdown merely to send a
        # redundant disable command (some USB reconnects report status=65535).
        if disable_tool.exists() and not self.args.preflight_only:
            subprocess.run(
                [str(disable_tool)],
                check=False, timeout=5,
            )
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2.0)
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.executor_thread is not None:
            self.executor_thread.join(timeout=2.0)
        if self.led is not None and not keep_led:
            self.led.close()

    def _fault(self, fault: StartupFault) -> int:
        if can_transition(self.state, We11State.FAULT):
            self.state = We11State.FAULT
        print(f"[FAULT] category={fault.category} detail={fault.detail} {fault}",
              file=sys.stderr, flush=True)
        if fault.hardware:
            self.cleanup(keep_led=True)
        else:
            self._enter_fault_standby()
        if self.led is not None and self.led.healthy:
            self.led.fault_pattern(FAULT_COLORS[fault.category], fault.detail, brightness=0.5)
        if self.args.no_fault_latch:
            if not self._cleaned:
                self.cleanup(keep_led=False)
            elif self.led is not None:
                self.led.close()
            return 1
        while not self.stop_requested.wait(0.05):
            if self.node is not None and self.node.buttons.exit_requested:
                print("[GAMEPAD] D-pad-up+A accepted while fault is latched", flush=True)
                break
            if self.led is not None and not self.led.healthy:
                break
        if not self._cleaned:
            self.cleanup(keep_led=False)
        elif self.led is not None:
            self.led.close()
        return 1

    def run(self) -> int:
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)
        try:
            self._startup()
            if not self.args.preflight_only:
                while True:
                    startup_mode = self._wait_startup_confirmation()
                    self._activate_selected_mode(startup_mode)
                    self._runtime()
            self._set_state(We11State.STOPPED)
            self.cleanup()
            return 0
        except KeyboardInterrupt:
            print("[STOP] operator requested safe shutdown", flush=True)
            self._set_state(We11State.STOPPED)
            self.cleanup()
            return 0
        except StartupFault as fault:
            return self._fault(fault)
        except Exception as exc:
            fault = StartupFault("ROS", 9, "UNEXPECTED_SUPERVISOR_ERROR", repr(exc))
            return self._fault(fault)


def main() -> int:
    parser = argparse.ArgumentParser(description="WE11 daily deployment state supervisor")
    parser.add_argument(
        "--hardware-backend", "--hardware_backend", choices=("esd_link",),
        default="esd_link",
        help="hardware transport backend (frozen production default: esd_link)",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="stop after automatic ESD-Link/P1-P8/IMU checks; never enable a motor",
    )
    parser.add_argument(
        "--force-zero-policy", action="store_true",
        help=("bench test: when POLICY is selected, publish native six-axis zero "
              "commands instead of MNN actions"),
    )
    parser.add_argument(
        "--policy-smoke-seconds", type=float, default=0.0,
        help=("test-only: after entering real POLICY, automatically return to "
              "STANDBY after this many seconds; 0 disables the limit"),
    )
    parser.add_argument(
        "--no-rgb", action="store_true",
        help="maintenance mode: do not access RGB GPIO owned by the lifecycle launcher",
    )
    parser.add_argument(
        "--no-fault-latch", action="store_true",
        help="test-only: return nonzero after showing the fault instead of waiting forever",
    )
    return We11Supervisor(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
