#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import rclpy
from esd_link_msgs.msg import LinkStatus, LowerState


CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from esd_control_timeouts import CONTROL_MODE_SERVICE_TIMEOUT_SECONDS  # noqa: E402
from esd_pair_sweep import (  # noqa: E402
    apply_sweep_overrides,
    gain_filename_token,
    load_config,
    SweepNode,
)


class CapturePublisher:
    def __init__(self) -> None:
        self.count = 0
        self.last_source_sequence = 0
        self.source_sequences: list[int] = []

    def publish(self, message) -> None:
        self.count += 1
        self.last_source_sequence = message.source_state_sample_seq
        self.source_sequences.append(message.source_state_sample_seq)


def test_mode_service_timeout_covers_bridge_disable_windows() -> None:
    assert CONTROL_MODE_SERVICE_TIMEOUT_SECONDS > 5.0 + 6.5


def test_wheel_sweep_rejects_amplitude_above_restored_limit(
    tmp_path: Path,
) -> None:
    config = tmp_path / "wheel.yaml"
    config.write_text(
        "swept_motor_ids: [3, 6]\n"
        "sweep:\n"
        "  publish_rate_hz: 200.0\n"
        "  velocity_amplitude_rad_s: 5.01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be in"):
        load_config(config)


def test_identification_overrides_preserve_base_config() -> None:
    base = {
        "swept_motor_ids": [1, 4],
        "sweep": {"kp": 2.0, "kd": 0.1, "amplitude_rad": 0.2},
    }
    result = apply_sweep_overrides(
        base, sweep_kp=4.0, sweep_kd=0.2,
        output_file="logs/motor14_{timestamp}_kp4_kd0p2.csv")

    assert base["sweep"]["kp"] == 2.0
    assert base["sweep"]["kd"] == 0.1
    assert result["sweep"]["kp"] == 4.0
    assert result["sweep"]["kd"] == 0.2
    assert result["sweep"]["amplitude_rad"] == 0.2
    assert result["sweep"]["output_file"].endswith("_kp4_kd0p2.csv")


def test_wheel_identification_override_stays_d_only() -> None:
    base = {"swept_motor_ids": [3, 6], "sweep": {"kp": 0.0, "kd": 0.1}}
    result = apply_sweep_overrides(base, sweep_kd=0.05)
    assert result["sweep"]["kd"] == 0.05
    assert result["sweep"]["output_file"] == (
        "logs/motor36_{timestamp}_kp0_kd0p05.csv")
    with pytest.raises(ValueError, match="requires Kp=0"):
        apply_sweep_overrides(base, sweep_kp=0.1)


@pytest.mark.parametrize(
    ("value", "token"), [(0.0, "0"), (0.05, "0p05"), (0.1, "0p1"), (4.0, "4")])
def test_gain_filename_token(value: float, token: str) -> None:
    assert gain_filename_token(value) == token


def test_state_driven_stream_never_reuses_a_source_sequence() -> None:
    rclpy.init()
    node = SweepNode({}, Path("sweep_motor14.yaml"))
    original_publisher = node.publisher
    capture = CapturePublisher()
    node.publisher = capture
    state = LowerState()
    state.schema_id = 1
    state.active_port_mask = 0x1FE
    state.control_state = 4
    state.imu_valid_mask = 0x07
    state.port_id = list(range(1, 9))
    state.valid_mask = [0x03] * 8
    state.state_sample_seq = 1234
    try:
        node.start_streaming(200.0)
        node._state(state)
        node._state(state)
        assert capture.source_sequences == [1234]

        state = copy.deepcopy(state)
        state.state_sample_seq = 1235
        node.next_tx_time = 0.0
        node._state(state)
        assert capture.source_sequences == [1234, 1235]
        assert node.stream_error is None
    finally:
        node.stop_streaming()
        node.publisher = original_publisher
        node.destroy_node()
        rclpy.shutdown()


def test_explicit_bridge_fault_stops_maintenance_stream() -> None:
    rclpy.init()
    node = SweepNode({}, Path("sweep_motor14.yaml"))
    try:
        state = LowerState()
        state.schema_id = 1
        state.active_port_mask = 0x1FE
        state.control_state = 2
        state.imu_valid_mask = 0x07
        state.port_id = list(range(1, 9))
        state.valid_mask = [0x03] * 8
        node._state(state)
        assert node.state_health_error() is None

        status = LinkStatus()
        status.link_state = LinkStatus.FAULT
        status.control_mode = 5
        node._link_status(status)
        assert node.state_health_error() == "bridge control mode FAULT"
    finally:
        node.destroy_node()
        rclpy.shutdown()
