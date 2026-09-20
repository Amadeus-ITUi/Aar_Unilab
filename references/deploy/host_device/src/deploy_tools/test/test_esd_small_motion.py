#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pytest


CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from esd_small_motion import (  # noqa: E402
    position_targets,
    verify_motion_state,
    verify_response,
)


def test_position_targets_are_symmetric_and_inside_limits() -> None:
    assert position_targets(1, -0.29, 0.03) == pytest.approx((-0.26, -0.32))
    assert position_targets(7, 1.53, 0.03) == pytest.approx((1.56, 1.50))


def test_position_target_outside_limit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="软件限位"):
        position_targets(8, -1.99, 0.03)
    with pytest.raises(RuntimeError, match="软件限位"):
        position_targets(2, 0.23, 0.03)


def test_motion_envelope_accepts_selected_port_excursion() -> None:
    initial = [0.0] * 8
    current = [0.0] * 8
    current[0] = 0.03
    verify_motion_state(initial, current, [0.1] * 8, 1, 0.06, 2.0)


def test_unselected_port_motion_is_rejected() -> None:
    initial = [0.0] * 8
    current = [0.0] * 8
    current[6] = 0.031
    with pytest.raises(RuntimeError, match="P7"):
        verify_motion_state(initial, current, [0.0] * 8, 1, 0.06, 2.0)


def test_both_direction_responses_are_required() -> None:
    verify_response(1, 0.01, -0.01, 0.003)
    with pytest.raises(RuntimeError, match="反向反馈"):
        verify_response(1, 0.01, -0.001, 0.003)
