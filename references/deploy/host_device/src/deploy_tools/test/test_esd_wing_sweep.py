#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from esd_wing_sweep import build_recovery_targets, WING_PORTS  # noqa: E402


def test_wing_recovery_uses_port_ids_and_holds_p1_through_p6() -> None:
    initial = [0.1, -0.2, 3.0, 0.4, -0.5, -6.0, 1.5, -1.5]
    centers = [0.35, -0.35]

    recovery = build_recovery_targets(initial, centers)

    assert WING_PORTS == (7, 8)
    assert recovery[:6] == initial[:6]
    assert recovery[6:] == centers
