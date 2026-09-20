#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"
sys.path.insert(0, str(DIAGNOSTICS_DIR))

from esd_manual_range_recorder import (  # noqa: E402
    RangeStatistics,
    counter_delta,
    dashboard_text,
    lower_position,
    lower_range,
)


def test_range_statistics_tracks_all_eight_ports() -> None:
    stats = RangeStatistics()
    stats.update([float(i) for i in range(8)], [0.1] * 8, [-0.2] * 8)
    stats.update([float(i) - 0.5 for i in range(8)], [-0.3] * 8, [0.4] * 8)
    assert stats.samples == 2
    assert stats.minimum[7] == pytest.approx(6.5)
    assert stats.maximum[7] == pytest.approx(7.0)
    assert stats.max_abs_velocity == pytest.approx([0.3] * 8)
    assert stats.max_abs_effort == pytest.approx([0.4] * 8)


def test_range_statistics_rejects_bad_layout() -> None:
    with pytest.raises(ValueError, match="eight ports"):
        RangeStatistics().update([0.0] * 7, [0.0] * 8, [0.0] * 8)


def test_range_statistics_rejects_non_finite_feedback() -> None:
    position = [0.0] * 8
    position[3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        RangeStatistics().update(position, [0.0] * 8, [0.0] * 8)


def test_counter_delta_handles_new_session_reset() -> None:
    assert counter_delta(10, 14) == 4
    assert counter_delta(10, 2) == 2


def test_lower_encoder_coordinates_match_bridge_inverse_contract() -> None:
    assert lower_position(1, -0.29) == pytest.approx(1.21020)
    assert lower_position(2, 0.30) == pytest.approx(1.28338)
    assert lower_position(7, 1.53) == pytest.approx(-1.53)
    assert lower_range(1, -0.98, 0.93) == pytest.approx((-0.0098, 1.9002))


def test_dashboard_contains_current_and_historical_values() -> None:
    stats = RangeStatistics()
    stats.update([0.0] * 8, [0.1] * 8, [0.2] * 8)
    stats.update([0.5] * 8, [-0.1] * 8, [-0.2] * 8)
    node = SimpleNamespace(
        last_state=SimpleNamespace(
            position_rad=[0.25] * 8,
            velocity_rad_s=[0.01] * 8,
        ),
        statistics=stats,
        link_end=SimpleNamespace(
            control_mode=0,
            device_control_state=2,
            state_rate_hz=250.0,
            latest_state_age_ms=2.0,
            sequence_gaps=0,
            cobs_errors=0,
            crc_errors=0,
            last_command_sequence=0,
            last_applied_command_sequence=0,
        ),
        recorder_gaps=0,
        error="",
    )
    text = dashboard_text(node, 12.5, 120.0)
    assert "纯只读" in text
    assert "P1" in text
    assert "P8" in text
    assert "轮子" in text
    assert "+0.250000" in text
