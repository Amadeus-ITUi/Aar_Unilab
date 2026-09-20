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
from motors.msg import MotorRuntimeStatus, WingRuntimeStatus
from motors.srv import SetOperationalState
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState, Joy
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
    sample_rate_ok,
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
        self.imu_samples: deque[tuple[float, bool]] = deque(maxlen=2000)
        self.joy_stamp = 0.0
        self.joint_stamp = 0.0
        self.wing_angle_stamp = 0.0
        self.motor_status_stamp = 0.0
        self.wing_status_stamp = 0.0
        self.policy_stamp = 0.0
        self.mode_stamp = 0.0
        self.joint_state: JointState | None = None
        self.motor_status: MotorRuntimeStatus | None = None
        self.wing_status: WingRuntimeStatus | None = None
        self.policy_frames: deque[tuple[float, tuple[float, ...]]] = deque(maxlen=500)
        self.inference_mode = 0

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Imu, "/IMU_data", self._imu, qos_profile_sensor_data)
        self.create_subscription(Joy, "/joy", self._joy, qos_profile_sensor_data)
        self.create_subscription(
            JointState, "/policy/joint_states", self._joint, qos_profile_sensor_data
        )
        self.create_subscription(
            JointState, "/policy/wing_angles", self._wing_angles, qos_profile_sensor_data
        )
        self.create_subscription(
            MotorRuntimeStatus, "/motors/runtime_status", self._motor_status, latched
        )
        self.create_subscription(
            WingRuntimeStatus, "/wing/runtime_status", self._wing_status, latched
        )
        self.create_subscription(Float32MultiArray, "/policy/commands", self._policy, 10)
        self.create_subscription(UInt8, "/inference/runtime_mode", self._mode, latched)

        self.motor_state_client = self.create_client(
            SetOperationalState, "/motors/set_operational_state"
        )
        self.soft_disarm_client = self.create_client(Trigger, "/motors/soft_disarm")
        self.wing_rc_client = self.create_client(SetBool, "/wing/enable_rc")
        self.policy_client = self.create_client(SetBool, "/inference/set_policy_enabled")

    @staticmethod
    def _finite(values: tuple[float, ...]) -> bool:
        return all(math.isfinite(value) for value in values)

    def _imu(self, msg: Imu) -> None:
        values = (
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
        )
        with self.lock:
            self.imu_samples.append((time.monotonic(), self._finite(values)))

    def _joy(self, msg: Joy) -> None:
        now = time.monotonic()
        self.buttons.update(tuple(msg.buttons), tuple(msg.axes), now)
        with self.lock:
            self.joy_stamp = now

    def _joint(self, msg: JointState) -> None:
        with self.lock:
            self.joint_state = msg
            self.joint_stamp = time.monotonic()

    def _wing_angles(self, _msg: JointState) -> None:
        with self.lock:
            self.wing_angle_stamp = time.monotonic()

    def _motor_status(self, msg: MotorRuntimeStatus) -> None:
        with self.lock:
            self.motor_status = msg
            self.motor_status_stamp = time.monotonic()

    def _wing_status(self, msg: WingRuntimeStatus) -> None:
        with self.lock:
            self.wing_status = msg
            self.wing_status_stamp = time.monotonic()

    def _policy(self, msg: Float32MultiArray) -> None:
        now = time.monotonic()
        with self.lock:
            self.policy_stamp = now
            self.policy_frames.append((now, tuple(msg.data)))

    def _mode(self, msg: UInt8) -> None:
        with self.lock:
            self.inference_mode = int(msg.data)
            self.mode_stamp = time.monotonic()


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
        elif state == We11State.CAN_CHECK:
            self.led.breathe(COLORS["blue"], brightness=0.5, period=2.0)
        elif state == We11State.IMU_CHECK:
            self.led.breathe(COLORS["purple"], brightness=0.5, period=2.0)
        elif state == We11State.DEVICE_CHECK:
            self.led.breathe(COLORS["cyan"], brightness=0.5, period=2.0)
        elif state == We11State.WING_POSITIONING:
            self.led.blink(COLORS["yellow"], brightness=0.5, interval=0.3)
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
        rclpy.init()
        self.node = SupervisorNode()
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self.executor.spin, name="we11-ros-executor", daemon=True
        )
        self.executor_thread.start()

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
                    hardware=process.name in {"motors", "wing"},
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
                    assert self.led is not None
                    self.led.breathe(COLORS["green"], brightness=0.5, period=2.0)
                    print(
                        "[SELECT] 倒地自启 Policy（绿色呼吸）；按 X 切回普通启动，按 Y 执行",
                        flush=True,
                    )
                else:
                    assert self.led is not None
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

    def _check_can(self) -> None:
        result = subprocess.run(
            ["ip", "link", "show", "can0"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0 or "UP" not in result.stdout.splitlines()[0]:
            raise StartupFault("CAN", 9, "CAN0_NOT_READY", "can0 is missing or not UP")
        tool = self.root / "src/deploy_tools/scripts/can/read_motor_status.py"
        for motor_id in range(1, 9):
            command = [
                sys.executable, str(tool), "--channel", "can0", "--motor-id", str(motor_id),
                "--request", "--timeout", "1.5", "--requests", "3",
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=7, check=False
                )
            except subprocess.TimeoutExpired as exc:
                raise StartupFault(
                    "CAN", motor_id, f"MOTOR_{motor_id}_CAN_TIMEOUT", str(exc)
                ) from exc
            if completed.returncode != 0:
                reason = (completed.stderr or completed.stdout).strip()[-1000:]
                raise StartupFault(
                    "CAN", motor_id, f"MOTOR_{motor_id}_NO_FEEDBACK", reason
                )
            print(f"[CHECK] can0 motor {motor_id}: OK", flush=True)

    def _check_imu(self) -> None:
        assert self.node is not None
        startup_started = time.monotonic()

        def first_sample_received() -> bool:
            with self.node.lock:
                return any(stamp >= startup_started for stamp, _ in self.node.imu_samples)

        self._wait(
            first_sample_received,
            15.0,
            StartupFault(
                "IMU", 1, "IMU_NO_DATA",
                "no IMU messages within 15 seconds of launch", False,
            ),
        )

        started = time.monotonic()
        self._wait(
            lambda: time.monotonic() - started >= 1.0,
            1.3,
            StartupFault("IMU", 1, "IMU_NO_DATA", "IMU window did not complete", False),
        )
        ended = time.monotonic()
        with self.node.lock:
            samples = [(stamp, valid) for stamp, valid in self.node.imu_samples
                       if started <= stamp <= ended]
        if not samples:
            raise StartupFault("IMU", 1, "IMU_NO_DATA", "no IMU messages", False)
        if not all(valid for _, valid in samples):
            raise StartupFault("IMU", 3, "IMU_INVALID_DATA", "non-finite IMU value", False)
        passed, rate, largest_gap = sample_rate_ok([stamp for stamp, _ in samples])
        print(
            f"[CHECK] IMU window 1/1: {rate:.2f} Hz, max_gap={largest_gap:.4f}s",
            flush=True,
        )
        if not passed:
            raise StartupFault(
                "IMU", 2, "IMU_RATE_LOW",
                f"window 1: {rate:.2f} Hz, max gap {largest_gap:.4f}s",
                False,
            )

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

    def _joint_ready(self) -> bool:
        assert self.node is not None
        with self.node.lock:
            status = self.node.motor_status
            joint = self.node.joint_state
            fresh = time.monotonic() - self.node.motor_status_stamp < 0.5
        if status is None or joint is None or not fresh:
            return False
        expected_names = [f"motor_{index}" for index in range(1, 7)]
        vectors = (
            tuple(status.position), tuple(status.velocity), tuple(status.effort),
            tuple(joint.position), tuple(joint.velocity), tuple(joint.effort),
        )
        return (
            status.state == MotorRuntimeStatus.DISARMED
            and list(status.motor_ids) == [1, 2, 3, 4, 5, 6]
            and len(status.online) == 6 and all(status.online)
            and list(joint.name) == expected_names
            and all(len(vector) == 6 for vector in vectors)
            and all(math.isfinite(value) for vector in vectors for value in vector)
        )

    def _policy_zero_ready(self) -> bool:
        assert self.node is not None
        cutoff = time.monotonic() - 0.6
        with self.node.lock:
            frames = [(stamp, data) for stamp, data in self.node.policy_frames if stamp >= cutoff]
        if len(frames) < 20:
            return False
        if any(
            len(data) != 6
            or any(not math.isfinite(value) or abs(value) > 1.0e-6 for value in data)
            for _, data in frames
        ):
            return False
        passed, rate, _ = sample_rate_ok(
            [stamp for stamp, _ in frames], minimum_hz=40.0, maximum_gap=0.1
        )
        return passed and rate < 70.0

    def _wing_positioning_ready(self, started: float) -> bool:
        assert self.node is not None
        with self.node.lock:
            status = self.node.wing_status
            stamp = self.node.wing_status_stamp
        if status is None or stamp < started or time.monotonic() - stamp >= 0.5:
            return False
        return (
            status.state == WingRuntimeStatus.POSITION_HOLD
            and status.feedback_ok
            and list(status.motor_ids) == [7, 8]
            and len(status.position) == 2
            and len(status.velocity) == 2
            and all(math.isfinite(value) for value in (*status.position, *status.velocity))
        )

    def _runtime_ready_for_policy(self) -> tuple[bool, str]:
        assert self.node is not None
        now = time.monotonic()
        with self.node.lock:
            motor = self.node.motor_status
            wing = self.node.wing_status
            checks = {
                "IMU stale": (
                    not self.node.imu_samples
                    or now - self.node.imu_samples[-1][0] > 0.2
                    or not self.node.imu_samples[-1][1]
                ),
                "Joy stale": now - self.node.joy_stamp > 0.5,
                "motor status stale": now - self.node.motor_status_stamp > 0.5,
                "wing status stale": now - self.node.wing_status_stamp > 0.5,
                "joint input stale": now - self.node.joint_stamp > 0.2,
                "wing angle input stale": now - self.node.wing_angle_stamp > 0.2,
                "policy output stale": now - self.node.policy_stamp > 0.2,
            }
            latest_policy = self.node.policy_frames[-1][1] if self.node.policy_frames else ()
        for reason, failed in checks.items():
            if failed:
                return False, reason
        if (
            motor is None
            or motor.state not in (MotorRuntimeStatus.STANDBY, MotorRuntimeStatus.ACTIVE)
            or not all(motor.online)
        ):
            return False, "motor health rejected policy"
        if any(
            len(values) != 6 or any(not math.isfinite(value) for value in values)
            for values in (motor.position, motor.velocity, motor.effort)
        ):
            return False, "motor data rejected policy"
        if wing is None or wing.state != WingRuntimeStatus.RC_CONTROL or not wing.feedback_ok:
            return False, "wing health rejected policy"
        if (
            len(wing.position) != 2 or len(wing.velocity) != 2 or
            any(not math.isfinite(value) for value in (*wing.position, *wing.velocity))
        ):
            return False, "wing data rejected policy"
        if len(latest_policy) != 6 or any(not math.isfinite(value) for value in latest_policy):
            return False, "policy output invalid"
        return True, ""

    def _startup(self) -> None:
        complete_rgb_handoff()
        self.led = RGBLed(red_pin=22, green_pin=17, blue_pin=27, common_anode=True)
        self._set_state(We11State.BOOT)
        self._start_ros()

        self._set_state(We11State.CAN_CHECK)
        self._check_can()

        self._set_state(We11State.IMU_CHECK)
        self._start_process("imu", ["ros2", "launch", "hipnuc_imu", "imu_node.launch.py"])
        self._check_imu()

        self._set_state(We11State.DEVICE_CHECK)
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
            self._check_model()
        self._start_process("motors", ["ros2", "launch", "motors", "motors_node.launch.py"])
        self._wait(
            self._joint_ready,
            20.0,
            StartupFault("DEVICE", 4, "LEG_FEEDBACK_INVALID", "1-6 read-only feedback failed"),
        )
        if self.args.preflight_only:
            print("[PASS] WE11 preflight complete; motors remained DISARMED", flush=True)
            return

        self._set_state(We11State.WING_POSITIONING)
        wing_positioning_started = time.monotonic()
        self._start_process("wing", ["ros2", "launch", "motors", "wing_motor_node.launch.py"])
        self._wait(
            lambda: self._wing_positioning_ready(wing_positioning_started),
            45.0,
            StartupFault(
                "WING", 1, "WING_POSITIONING_TIMEOUT",
                "wing node did not finish the +90/-90 degree startup target command",
            ),
        )

        self._start_process(
            "inference",
            ["ros2", "launch", "inference", "lab_inference_node.launch.py",
             "joy_policy_gate_enabled:=false", "joy_policy_start_enabled:=false"],
        )
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
        print("[READY] 等待态已开放右摇杆机翼控制", flush=True)
        self._set_state(We11State.WAIT_STANDBY)

    def _activate_selected_mode(self, startup_mode: StartupMode) -> None:
        """Initialize the legs for the operator-selected normal or direct start."""
        assert self.node is not None
        self._set_state(
            We11State.POLICY_ENTERING
            if startup_mode == StartupMode.DIRECT_POLICY
            else We11State.STANDBY_ENTERING
        )
        request = SetOperationalState.Request()
        if startup_mode == StartupMode.DIRECT_POLICY:
            request.target_state = SetOperationalState.Request.ACTIVE
            print(
                "[ACTION] 倒地自启: 串行初始化 1-6 号电机，不执行默认姿态斜坡；"
                f"response timeout={STANDBY_TRANSITION_TIMEOUT_SECONDS:.1f}s",
                flush=True,
            )
        else:
            request.target_state = SetOperationalState.Request.STANDBY
            print(
                "[ACTION] Standby: sequential motor initialization (~5.1s) + "
                f"3.0s ramp; response timeout={STANDBY_TRANSITION_TIMEOUT_SECONDS:.1f}s",
                flush=True,
            )
        transition_name = (
            "倒地自启" if startup_mode == StartupMode.DIRECT_POLICY else "Standby"
        )
        result = self._call(
            self.node.motor_state_client,
            request,
            response_timeout=STANDBY_TRANSITION_TIMEOUT_SECONDS,
            unavailable_fault=StartupFault(
                "CAN", 8,
                "DIRECT_POLICY_SERVICE_UNAVAILABLE"
                if startup_mode == StartupMode.DIRECT_POLICY
                else "STANDBY_SERVICE_UNAVAILABLE",
                "/motors/set_operational_state was not available within 5 seconds",
            ),
            timeout_fault=StartupFault(
                "CAN", 8,
                "DIRECT_POLICY_TRANSITION_TIMEOUT"
                if startup_mode == StartupMode.DIRECT_POLICY
                else "STANDBY_TRANSITION_TIMEOUT",
                f"motor node did not finish the {transition_name} transition within 15 seconds",
            ),
        )
        expected_motor_state = (
            MotorRuntimeStatus.ACTIVE
            if startup_mode == StartupMode.DIRECT_POLICY
            else MotorRuntimeStatus.STANDBY
        )
        if not result.success or result.actual_state != expected_motor_state:
            fault_code = (
                "DIRECT_POLICY_ACTIVATION_REJECTED"
                if startup_mode == StartupMode.DIRECT_POLICY
                else "STANDBY_RAMP_REJECTED"
            )
            raise StartupFault("CAN", 8, fault_code, result.message)
        if startup_mode == StartupMode.DIRECT_POLICY:
            gate_deadline = time.monotonic() + 1.0
            reason = "runtime inputs did not become ready"
            while time.monotonic() < gate_deadline:
                self._tick()
                ready, reason = self._runtime_ready_for_policy()
                if ready:
                    break
                time.sleep(0.02)
            else:
                # Legs are already ACTIVE without a holding posture here. Any
                # rejection before Policy owns the command stream must fully
                # disarm instead of entering the normal fault-Standby path.
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
        """Freeze the last leg command, release force over three seconds, then wait."""
        assert self.node is not None
        print(
            "[ACTION] Y: 保持末帧目标位置，3 秒线性卸载 1-6 号的速度/力矩/Kp/Kd",
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
                status = self.node.motor_status
                stamp = self.node.motor_status_stamp
            return (
                status is not None
                and status.state == MotorRuntimeStatus.DISARMED
                and time.monotonic() - stamp < 0.5
            )

        self._wait(
            motors_are_disarmed,
            1.0,
            StartupFault(
                "CAN", 8, "SOFT_DISARM_STATUS_TIMEOUT",
                "soft-disarm service succeeded but fresh DISARMED status was not observed",
            ),
        )
        self.node.buttons.clear(2)
        self.node.buttons.clear(3)
        self._set_state(We11State.WAIT_STANDBY)
        print(
            "[READY] 1-6 号已无阻尼失能；右摇杆机翼控制保持开放",
            flush=True,
        )

    def _runtime(self) -> None:
        assert self.node is not None
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
            if self.node.buttons.consume(3):
                self._soft_disarm_to_wait()
                return
            if self.node.buttons.consume(2):
                request = SetBool.Request()
                request.data = self.state != We11State.POLICY
                if request.data:
                    ready, reason = self._runtime_ready_for_policy()
                    if not ready:
                        print(f"[REJECT] Policy: {reason}", flush=True)
                        time.sleep(0.1)
                        continue
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
                if result.success:
                    self._set_state(We11State.POLICY if request.data else We11State.STANDBY)
                else:
                    print(f"[REJECT] Policy: {result.message}", flush=True)
            time.sleep(0.02)

    def _safe_service_shutdown(self) -> None:
        if self.node is None:
            return
        try:
            request = SetBool.Request()
            request.data = False
            if self.node.policy_client.service_is_ready():
                self.node.policy_client.call_async(request)
            if self.node.wing_rc_client.service_is_ready():
                self.node.wing_rc_client.call_async(request)
            motor_request = SetOperationalState.Request()
            motor_request.target_state = SetOperationalState.Request.DISARMED
            if self.node.motor_state_client.service_is_ready():
                self.node.motor_state_client.call_async(motor_request)
            time.sleep(0.3)
        except Exception as exc:
            print(f"[WARN] service shutdown failed: {exc}", file=sys.stderr, flush=True)

    def _enter_fault_standby(self) -> None:
        """Zero inference, hold the wings in place, and keep healthy legs in Standby."""
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
        self._safe_service_shutdown()
        for process in reversed(self.processes):
            try:
                process.stop()
            except Exception as exc:
                print(f"[WARN] failed to stop {process.name}: {exc}", file=sys.stderr)
        disable_tool = self.root / "src/deploy_tools/scripts/control/emergency_disable_motors.py"
        if Path("/sys/class/net/can0").exists() and disable_tool.exists():
            subprocess.run(
                [sys.executable, str(disable_tool), "--channel", "can0", "--motor-ids",
                 "1,2,3,4,5,6,7,8", "--retries", "10", "--interval", "0.01"],
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
        "--preflight-only", action="store_true",
        help="stop after automatic CAN/IMU/device checks; never enable a motor",
    )
    parser.add_argument(
        "--no-fault-latch", action="store_true",
        help="test-only: return nonzero after showing the fault instead of waiting forever",
    )
    return We11Supervisor(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
