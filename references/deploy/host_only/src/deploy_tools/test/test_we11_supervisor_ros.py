#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import rclpy
import pytest
from motors.msg import MotorRuntimeStatus, WingRuntimeStatus
from sensor_msgs.msg import Imu, JointState, Joy
from std_msgs.msg import Float32MultiArray

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT / "control"))
sys.path.insert(0, str(SCRIPT_ROOT / "hardware"))

from we11_supervisor import (  # noqa: E402
    SOFT_DISARM_RESPONSE_TIMEOUT_SECONDS,
    STANDBY_TRANSITION_TIMEOUT_SECONDS,
    StartupFault,
    SupervisorNode,
    We11Supervisor,
    complete_rgb_handoff,
)
from we11_state import We11State  # noqa: E402


def make_joint_state() -> JointState:
    msg = JointState()
    msg.name = [f"motor_{index}" for index in range(1, 7)]
    msg.position = [0.0] * 6
    msg.velocity = [0.0] * 6
    msg.effort = [0.0] * 6
    return msg


def make_motor_status(state: int) -> MotorRuntimeStatus:
    msg = MotorRuntimeStatus()
    msg.state = state
    msg.motor_ids = [1, 2, 3, 4, 5, 6]
    msg.online = [True] * 6
    msg.position = [0.0] * 6
    msg.velocity = [0.0] * 6
    msg.effort = [0.0] * 6
    return msg


def make_wing_status(state: int = WingRuntimeStatus.RC_CONTROL) -> WingRuntimeStatus:
    msg = WingRuntimeStatus()
    msg.state = state
    msg.motor_ids = [7, 8]
    msg.position = [0.0, 0.0]
    msg.velocity = [0.0, 0.0]
    msg.feedback_ok = True
    return msg


def test_fake_ros_inputs_cover_device_and_policy_readiness() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        supervisor = We11Supervisor.__new__(We11Supervisor)
        supervisor.node = node

        joy = Joy()
        joy.buttons = [0] * 11
        joy.axes = [0.0] * 8
        node._joy(joy)
        node._imu(Imu())
        node._joint(make_joint_state())
        node._motor_status(make_motor_status(MotorRuntimeStatus.DISARMED))
        assert supervisor._joint_ready()

        node._motor_status(make_motor_status(MotorRuntimeStatus.STANDBY))
        node._wing_status(make_wing_status())
        wing_angles = JointState()
        wing_angles.name = ["motor_7", "motor_8"]
        wing_angles.position = [0.0, 0.0]
        wing_angles.velocity = [0.0, 0.0]
        node._wing_angles(wing_angles)
        node._policy(Float32MultiArray(data=[0.0] * 6))

        now = time.monotonic()
        with node.lock:
            node.policy_frames.clear()
            for index in range(31):
                node.policy_frames.append((now - 0.60 + index * 0.02, (0.0,) * 6))
            node.policy_stamp = now
        assert supervisor._policy_zero_ready()
        ready, reason = supervisor._runtime_ready_for_policy()
        assert ready, reason

        node._motor_status(make_motor_status(MotorRuntimeStatus.ACTIVE))
        ready, reason = supervisor._runtime_ready_for_policy()
        assert ready, reason
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_fake_ros_invalid_and_stale_inputs_are_rejected() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        supervisor = We11Supervisor.__new__(We11Supervisor)
        supervisor.node = node
        node._joint(make_joint_state())
        bad = make_motor_status(MotorRuntimeStatus.DISARMED)
        bad.position[2] = math.nan
        node._motor_status(bad)
        assert not supervisor._joint_ready()

        node._motor_status(make_motor_status(MotorRuntimeStatus.STANDBY))
        node._wing_status(make_wing_status())
        node._policy(Float32MultiArray(data=[0.0] * 6))
        ready, reason = supervisor._runtime_ready_for_policy()
        assert not ready
        assert reason in {"IMU stale", "Joy stale", "wing angle input stale"}
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_wing_positioning_requires_fresh_finite_position_hold_status() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        supervisor = We11Supervisor.__new__(We11Supervisor)
        supervisor.node = node
        started = time.monotonic()

        node._wing_status(make_wing_status(WingRuntimeStatus.POSITIONING))
        assert not supervisor._wing_positioning_ready(started)

        hold = make_wing_status(WingRuntimeStatus.POSITION_HOLD)
        hold.position = [math.pi / 2.0, -math.pi / 2.0]
        node._wing_status(hold)
        assert supervisor._wing_positioning_ready(started)

        hold.velocity[1] = math.nan
        node._wing_status(hold)
        assert not supervisor._wing_positioning_ready(started)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_leg_startup_does_not_reject_joint_position() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        supervisor = We11Supervisor.__new__(We11Supervisor)
        supervisor.node = node
        node._motor_status(make_motor_status(MotorRuntimeStatus.DISARMED))

        joint = make_joint_state()
        joint.position = [100.0, -100.0, 50.0, -50.0, 25.0, -25.0]
        node._joint(joint)
        assert supervisor._joint_ready()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_imu_check_waits_for_first_sample_before_rate_windows() -> None:
    class FakeNode:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.imu_samples: deque[tuple[float, bool]] = deque()

    class RateWindowStarted(Exception):
        pass

    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.node = FakeNode()
    waits: list[tuple[float, StartupFault]] = []

    def fake_wait(predicate, timeout: float, fault: StartupFault) -> None:
        waits.append((timeout, fault))
        if len(waits) == 1:
            assert not predicate()
            with supervisor.node.lock:
                supervisor.node.imu_samples.append((time.monotonic(), True))
            assert predicate()
            return
        raise RateWindowStarted

    supervisor._wait = fake_wait

    try:
        supervisor._check_imu()
    except RateWindowStarted:
        pass
    else:
        raise AssertionError("rate window was not started")

    assert len(waits) == 2
    assert waits[0][0] == 15.0
    assert waits[0][1].code == "IMU_NO_DATA"
    assert waits[1][0] == 1.3


def test_service_unavailable_and_response_timeout_are_distinct() -> None:
    class PendingFuture:
        @staticmethod
        def done() -> bool:
            return False

    class FakeClient:
        def __init__(self, available: bool) -> None:
            self.available = available

        def wait_for_service(self, timeout_sec: float) -> bool:
            assert timeout_sec > 0.0
            return self.available

        @staticmethod
        def call_async(_request: object) -> PendingFuture:
            return PendingFuture()

    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor._tick = lambda: None
    unavailable = StartupFault("CAN", 8, "SERVICE_UNAVAILABLE", "missing")
    timed_out = StartupFault("CAN", 8, "RESPONSE_TIMEOUT", "late")

    with pytest.raises(StartupFault) as unavailable_result:
        supervisor._call(
            FakeClient(False), object(), response_timeout=0.01,
            unavailable_fault=unavailable, timeout_fault=timed_out,
            availability_timeout=0.01,
        )
    assert unavailable_result.value.code == "SERVICE_UNAVAILABLE"

    with pytest.raises(StartupFault) as timeout_result:
        supervisor._call(
            FakeClient(True), object(), response_timeout=0.01,
            unavailable_fault=unavailable, timeout_fault=timed_out,
            availability_timeout=0.01,
        )
    assert timeout_result.value.code == "RESPONSE_TIMEOUT"


def test_standby_timeout_has_margin_above_known_transition_budget() -> None:
    serial_motor_init_budget = 6 * (5 * 0.01 + 0.5 + 3 * 0.1)
    configured_ramp_budget = 3.0
    assert STANDBY_TRANSITION_TIMEOUT_SECONDS >= (
        serial_motor_init_budget + configured_ramp_budget + 5.0
    )


def test_soft_disarm_timeout_has_margin_above_release_duration() -> None:
    assert SOFT_DISARM_RESPONSE_TIMEOUT_SECONDS >= 3.0 + 5.0


def test_soft_disarm_freezes_motors_before_disabling_inference() -> None:
    calls: list[tuple[object, object]] = []
    soft_client = object()
    policy_client = object()
    cleared: list[int] = []
    node = SimpleNamespace(
        soft_disarm_client=soft_client,
        policy_client=policy_client,
        lock=threading.Lock(),
        motor_status=make_motor_status(MotorRuntimeStatus.STANDBY),
        motor_status_stamp=time.monotonic(),
        buttons=SimpleNamespace(clear=lambda index: cleared.append(index)),
    )
    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.node = node
    supervisor.state = We11State.POLICY
    supervisor.led = None

    def fake_call(client, request, **_kwargs):
        calls.append((client, request))
        if client is soft_client:
            node.motor_status = make_motor_status(MotorRuntimeStatus.DISARMED)
            node.motor_status_stamp = time.monotonic()
        return SimpleNamespace(success=True, message="ok")

    supervisor._call = fake_call
    supervisor._wait = lambda predicate, _timeout, _fault: assert_predicate(predicate)

    supervisor._soft_disarm_to_wait()

    assert [client for client, _ in calls] == [soft_client, policy_client]
    assert calls[1][1].data is False
    assert supervisor.state == We11State.WAIT_STANDBY
    assert cleared == [2, 3]


def test_soft_disarm_rejection_requests_immediate_hard_disarm() -> None:
    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.node = SimpleNamespace(soft_disarm_client=object())
    supervisor.state = We11State.STANDBY
    fallback_calls: list[bool] = []
    supervisor._call = lambda *_args, **_kwargs: SimpleNamespace(
        success=False, message="rejected"
    )
    supervisor._hard_disarm_best_effort = lambda: fallback_calls.append(True)

    with pytest.raises(StartupFault) as result:
        supervisor._soft_disarm_to_wait()

    assert result.value.code == "SOFT_DISARM_REJECTED"
    assert fallback_calls == [True]


def assert_predicate(predicate) -> None:
    assert predicate()


def test_rgb_handoff_acknowledges_launcher_before_gpio_acquisition() -> None:
    ready_read_fd, ready_write_fd = os.pipe()
    release_read_fd, release_write_fd = os.pipe()
    try:
        os.environ["WE11_RGB_HANDOFF_READY_FD"] = str(ready_write_fd)
        os.environ["WE11_RGB_HANDOFF_RELEASE_FD"] = str(release_read_fd)
        os.write(release_write_fd, b"R")

        complete_rgb_handoff()

        assert os.read(ready_read_fd, 1) == b"R"
        assert "WE11_RGB_HANDOFF_READY_FD" not in os.environ
        assert "WE11_RGB_HANDOFF_RELEASE_FD" not in os.environ
        ready_write_fd = -1
        release_read_fd = -1
    finally:
        os.environ.pop("WE11_RGB_HANDOFF_READY_FD", None)
        os.environ.pop("WE11_RGB_HANDOFF_RELEASE_FD", None)
        for fd in (ready_read_fd, ready_write_fd, release_read_fd, release_write_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
