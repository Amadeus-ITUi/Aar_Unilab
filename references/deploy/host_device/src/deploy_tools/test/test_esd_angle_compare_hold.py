#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest
from esd_link_msgs.msg import LowerState


CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from esd_angle_compare_hold import (  # noqa: E402
    load_gains,
    make_command,
    policy_target,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("-0.1", -0.1), ("0", 0.0), ("+0.1", 0.1)],
)
def test_only_three_comparison_targets_are_accepted(value: str, expected: float) -> None:
    assert policy_target(value) == expected


@pytest.mark.parametrize("value", ["-0.2", "0.01", "0.2", "nan", "inf"])
def test_other_targets_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        policy_target(value)


def test_full_command_targets_four_joints_and_holds_other_positions() -> None:
    state = LowerState()
    state.port_id = list(range(1, 9))
    state.position_rad = [float(port) for port in range(1, 9)]
    kp = [2.0, 8.0, 0.0, 2.0, 8.0, 0.0, 10.0, 10.0]
    kd = [0.1, 0.8, 0.2, 0.1, 0.8, 0.2, 0.5, 0.5]

    command = make_command(state, 0.1, kp, kd)

    assert list(command.position_rad) == pytest.approx(
        [0.1, 0.1, 3.0, 0.1, 0.1, 6.0, 7.0, 8.0])
    assert list(command.velocity_rad_s) == pytest.approx([0.0] * 8)
    assert list(command.kp) == pytest.approx(kp)
    assert list(command.kd) == pytest.approx(kd)
    assert list(command.effort_nm) == pytest.approx([0.0] * 8)


def test_gains_are_loaded_from_bridge_yaml() -> None:
    config = (
        Path(__file__).resolve().parents[2]
        / "esd_link_bridge/config/esd_link_bridge.yaml"
    )
    kp, kd = load_gains(config)
    assert kp == [2.0, 8.0, 0.0, 2.0, 8.0, 0.0, 10.0, 10.0]
    assert kd == [0.1, 0.8, 0.2, 0.1, 0.8, 0.2, 0.5, 0.5]
