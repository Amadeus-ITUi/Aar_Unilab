#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pytest


CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from esd_hold_current import verify_hold_state  # noqa: E402


def test_stable_hold_state_is_accepted() -> None:
    initial = [0.0] * 8
    current = [0.02, -0.02, 8.0, 0.01, -0.01, -8.0, 0.03, -0.03]
    verify_hold_state(initial, current, [0.1] * 8, 0.08, 2.0)


def test_leg_or_wing_position_deviation_is_rejected() -> None:
    initial = [0.0] * 8
    current = [0.0] * 8
    current[6] = 0.081
    with pytest.raises(RuntimeError, match="P7"):
        verify_hold_state(initial, current, [0.0] * 8, 0.08, 2.0)


def test_any_port_velocity_excess_is_rejected() -> None:
    velocities = [0.0] * 8
    velocities[2] = 2.01
    with pytest.raises(RuntimeError, match="P3"):
        verify_hold_state([0.0] * 8, [0.0] * 8, velocities, 0.08, 2.0)
