#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys

from esd_link_msgs.msg import LinkStatus, LowerState


DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"
sys.path.insert(0, str(DIAGNOSTICS_DIR))

from esd_policy_imu_monitor import (  # noqa: E402
    final_policy_imu,
    inference_compatible_error,
    read_only_link_error,
)


def valid_state() -> LowerState:
    state = LowerState()
    state.schema_id = 1
    state.active_port_mask = 0x1FE
    state.offline_port_mask = 0
    state.fault_flags = 0
    state.imu_valid_mask = 0x07
    state.control_state = 2
    state.port_id = list(range(1, 9))
    state.valid_mask = [0x03] * 8
    state.position_rad = [0.0] * 8
    state.velocity_rad_s = [0.0] * 8
    state.imu.angular_velocity.x = 1.25
    state.imu.angular_velocity.y = -2.5
    state.imu.angular_velocity.z = 3.75
    state.imu.angular_velocity_covariance[0] = -0.1
    state.imu.angular_velocity_covariance[1] = 0.2
    state.imu.angular_velocity_covariance[2] = -0.97
    return state


def test_monitor_returns_exact_native_inference_values_without_another_flip() -> None:
    angular, gravity = final_policy_imu(valid_state())
    assert angular == (1.25, -2.5, 3.75)
    assert gravity == (-0.1, 0.2, -0.97)


def test_valid_lower_state_matches_native_inference_gates() -> None:
    assert inference_compatible_error(valid_state()) is None


def test_invalid_imu_or_port_feedback_is_rejected() -> None:
    state = valid_state()
    state.imu_valid_mask = 0
    assert "imu_valid_mask" in str(inference_compatible_error(state))

    state = valid_state()
    state.valid_mask[4] = 0
    assert "valid mask" in str(inference_compatible_error(state))

    state = valid_state()
    state.position_rad[2] = math.nan
    assert "non-finite actuator" in str(inference_compatible_error(state))


def test_non_finite_final_policy_imu_is_rejected() -> None:
    state = valid_state()
    state.imu.angular_velocity.z = math.nan
    assert "non-finite final policy IMU" in str(inference_compatible_error(state))


def test_offline_ports_are_reported_before_first_valid_sample() -> None:
    status = LinkStatus()
    status.session_valid = True
    status.control_mode = 5
    status.device_control_state = 2
    status.fault_flags = 0x08
    status.offline_port_mask = 0x0E
    error = read_only_link_error(status)
    assert error is not None
    assert "fault=0x8" in error
    assert "offline=0xe" in error
