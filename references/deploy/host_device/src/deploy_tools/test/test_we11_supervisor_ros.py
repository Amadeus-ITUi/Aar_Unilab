#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import rclpy
import pytest
from esd_link_msgs.msg import LinkStatus, LowerState
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
from we11_state import StartupMode, We11State  # noqa: E402


def make_lower_state(control_state: int = 2) -> LowerState:
    msg = LowerState()
    msg.session_id = 1
    msg.schema_id = 1
    msg.control_state = control_state
    msg.imu_valid_mask = 0x07
    msg.imu.orientation.w = 1.0
    msg.port_id = list(range(1, 9))
    msg.valid_mask = [0x03] * 8
    msg.position_rad = [0.0] * 8
    msg.velocity_rad_s = [0.0] * 8
    msg.effort_nm = [0.0] * 8
    msg.active_port_mask = 0x1FE
    return msg


def test_fake_ros_inputs_cover_device_and_policy_readiness() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        supervisor = We11Supervisor.__new__(We11Supervisor)
        supervisor.node = node

        node._lower_state(make_lower_state())
        assert supervisor._lower_state_ready()

        node._lower_state(make_lower_state(control_state=3))
        node._policy(Float32MultiArray(data=[0.0] * 6))
        assert supervisor._policy_zero_ready()
        ready, reason = supervisor._runtime_ready_for_policy()
        assert ready, reason

        node._lower_state(make_lower_state(control_state=4))
        ready, reason = supervisor._runtime_ready_for_policy()
        assert ready, reason
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_force_zero_policy_launch_argument_is_forwarded() -> None:
    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.args = SimpleNamespace(force_zero_policy=True)
    assert supervisor._inference_command()[-1] == "force_zero_policy_commands:=true"

    supervisor.args = SimpleNamespace(force_zero_policy=False)
    assert "force_zero_policy_commands:=true" not in supervisor._inference_command()


def test_invalid_native_lower_state_is_rejected() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        supervisor = We11Supervisor.__new__(We11Supervisor)
        supervisor.node = node
        bad = make_lower_state()
        bad.position_rad[2] = math.nan
        node._lower_state(bad)
        assert not supervisor._lower_state_ready()

        bad = make_lower_state(control_state=4)
        bad.offline_port_mask = 1 << 3
        node._lower_state(bad)
        node._policy(Float32MultiArray(data=[0.0] * 6))
        ready, reason = supervisor._runtime_ready_for_policy()
        assert not ready
        assert reason == "P1-P8 mask/offline invalid"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_startup_does_not_reject_finite_joint_position() -> None:
    rclpy.init()
    node = SupervisorNode()
    try:
        supervisor = We11Supervisor.__new__(We11Supervisor)
        supervisor.node = node
        state = make_lower_state()
        state.position_rad = [100.0, -100.0, 50.0, -50.0, 25.0, -25.0, 2.0, -0.7]
        node._lower_state(state)
        assert supervisor._lower_state_ready()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_link_timing_diagnostics_do_not_gate_session_readiness() -> None:
    node = SimpleNamespace(lock=threading.Lock(), link_status=LinkStatus())
    node.link_status.session_valid = True
    node.link_status.schema_id = 1
    node.link_status.config_fingerprint = 0x6FC12897
    node.link_status.active_port_mask = 0x1FE
    node.link_status.state_rate_hz = 1.0
    node.link_status.latest_state_age_ms = 5000.0
    node.link_status.maximum_interarrival_ms = 5000.0
    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.node = node
    assert supervisor._link_ready()


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


def test_direct_policy_activation_never_requests_standby() -> None:
    calls: list[tuple[object, object]] = []
    motor_client = object()
    policy_client = object()
    node = SimpleNamespace(
        motor_state_client=motor_client,
        policy_client=policy_client,
        lock=threading.Lock(),
        lower_state=make_lower_state(control_state=2),
        policy_frames=[(time.monotonic(), (0.0,) * 6)],
        inference_mode=0,
        mode_stamp=0.0,
        _finite=lambda values: all(math.isfinite(value) for value in values),
    )
    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.node = node
    supervisor.state = We11State.WAIT_STANDBY
    supervisor.led = None
    supervisor._tick = lambda: None

    def fake_call(client, request, **_kwargs):
        calls.append((client, request))
        if client is motor_client:
            node.lower_state = make_lower_state(control_state=3)
            return SimpleNamespace(success=True, actual_state=2, message="active")
        node.inference_mode = 1
        node.mode_stamp = time.monotonic()
        return SimpleNamespace(success=True, message="policy enabled")

    supervisor._call = fake_call
    supervisor._wait = lambda predicate, _timeout, _fault: assert_predicate(predicate)

    supervisor._activate_selected_mode(StartupMode.DIRECT_POLICY)

    assert [client for client, _ in calls] == [motor_client, policy_client]
    assert calls[0][1].target_state == 2
    assert calls[1][1].data is True
    assert supervisor.state == We11State.POLICY


def test_policy_exit_restores_bridge_standby_before_stopping_inference() -> None:
    calls: list[tuple[object, object]] = []
    motor_client = object()
    policy_client = object()
    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.node = SimpleNamespace(
        motor_state_client=motor_client,
        policy_client=policy_client,
    )
    supervisor.state = We11State.POLICY
    supervisor.led = None

    def fake_call(client, request, **_kwargs):
        calls.append((client, request))
        if client is motor_client:
            return SimpleNamespace(success=True, actual_state=1, message="standby")
        return SimpleNamespace(success=True, message="policy disabled")

    supervisor._call = fake_call
    supervisor._leave_policy_to_standby()

    assert [client for client, _ in calls] == [motor_client, policy_client]
    assert calls[0][1].target_state == 1
    assert calls[1][1].data is False
    assert supervisor.state == We11State.STANDBY


def test_soft_disarm_freezes_motors_before_disabling_inference() -> None:
    calls: list[tuple[object, object]] = []
    soft_client = object()
    policy_client = object()
    cleared: list[int] = []
    node = SimpleNamespace(
        soft_disarm_client=soft_client,
        policy_client=policy_client,
        lock=threading.Lock(),
        lower_state=make_lower_state(control_state=4),
        buttons=SimpleNamespace(clear=lambda index: cleared.append(index)),
    )
    supervisor = We11Supervisor.__new__(We11Supervisor)
    supervisor.node = node
    supervisor.state = We11State.POLICY
    supervisor.led = None

    def fake_call(client, request, **_kwargs):
        calls.append((client, request))
        if client is soft_client:
            node.lower_state = make_lower_state(control_state=2)
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
