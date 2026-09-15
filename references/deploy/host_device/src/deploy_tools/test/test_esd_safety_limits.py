#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from esd_safety_limits import (  # noqa: E402
    clamp_position_target,
    CONFIRMED_HARD_LIMITS,
    POSITION_LIMITS,
    pose_limit_error,
)


def test_command_target_clamps_inward_without_rejecting_current_feedback() -> None:
    assert clamp_position_target(2, 0.31) == pytest.approx(0.24)
    assert clamp_position_target(5, 0.30) == pytest.approx(0.22)
    assert clamp_position_target(8, 0.77) == pytest.approx(0.70)
    assert clamp_position_target(1, 0.10, margin=0.03) == pytest.approx(0.10)
    assert clamp_position_target(1, 0.83, margin=0.03) == pytest.approx(0.81)


def test_software_limits_retain_confirmed_mechanical_margin() -> None:
    for port, (soft_minimum, soft_maximum) in POSITION_LIMITS.items():
        hard_minimum, hard_maximum = CONFIRMED_HARD_LIMITS[port]
        assert soft_minimum - hard_minimum >= 0.075 - 1e-6
        assert hard_maximum - soft_maximum >= 0.075 - 1e-6


def test_full_pose_rejects_any_position_joint_outside_limit() -> None:
    safe = [0.0] * 8
    assert pose_limit_error(safe) is None
    unsafe = list(safe)
    unsafe[4] = 0.221
    assert "P5" in pose_limit_error(unsafe)


def test_python_limits_match_bridge_runtime_yaml() -> None:
    source_root = Path(__file__).resolve().parents[2]
    config_path = source_root / "esd_link_bridge" / "config" / "esd_link_bridge.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parameters = config["esd_link_bridge_node"]["ros__parameters"]
    leg_limits = parameters["joint_position_limits"]
    wing_limits = parameters["wing_position_limits"]
    for port in (1, 2, 4, 5):
        index = port - 1
        assert tuple(leg_limits[2 * index:2 * index + 2]) == POSITION_LIMITS[port]
    for port in (7, 8):
        index = port - 7
        assert tuple(wing_limits[2 * index:2 * index + 2]) == POSITION_LIMITS[port]


def test_pair_sweep_configs_remain_inside_software_limits() -> None:
    deploy_root = Path(__file__).resolve().parents[3]
    for pair_name in ("14", "25", "36"):
        config_path = (
            deploy_root / "scripts" / "sweeps" / f"sweep_motor{pair_name}"
            / f"sweep_motor{pair_name}.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        target = [
            base + offset
            for base, offset in zip(
                config["sweep_base_def_pos_policy_rad"],
                config["zero_reference_offset_policy_rad"],
            )
        ]
        assert pose_limit_error(target + [0.0, 0.0]) is None
        sweep = config["sweep"]
        if sweep["control_mode"] != "position":
            continue
        amplitude = sweep["amplitude_rad"]
        for port in config["swept_motor_ids"]:
            minimum, maximum = POSITION_LIMITS[port]
            assert target[port - 1] - amplitude >= minimum
            assert target[port - 1] + amplitude <= maximum
