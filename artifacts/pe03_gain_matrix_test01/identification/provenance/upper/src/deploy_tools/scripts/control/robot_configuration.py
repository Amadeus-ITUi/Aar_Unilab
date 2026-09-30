#!/usr/bin/env python3
"""Pure parsing for installed ESD-Link Robot Profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Actuator:
    port_id: int
    joint_name: str
    control: str
    direction: float
    mechanical_zero_rad: float
    continuous: bool
    default_position_rad: float
    kp: float
    kd: float
    command_velocity_max_rad_s: float
    device_velocity_max_rad_s: float
    position_limits_rad: tuple[float, float] | None


@dataclass(frozen=True)
class RobotCatalog:
    robot_id: str
    profile_id: str
    schema_id: int
    lower_config_fingerprint: int
    control_allowed: bool
    calibrated: bool
    actuators: tuple[Actuator, ...]
    robot_profile_path: Path

    @property
    def active_port_mask(self) -> int:
        return sum(1 << item.port_id for item in self.actuators)

    @property
    def actuator_by_name(self) -> dict[str, Actuator]:
        return {item.joint_name: item for item in self.actuators}

    @property
    def actuator_by_port(self) -> dict[int, Actuator]:
        return {item.port_id: item for item in self.actuators}


def mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def document(path: Path, name: str) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {name} {path}: {exc}") from exc
    return mapping(value, name)


def finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = finite(value, name)
    if result < 0.0 or (not allow_zero and result == 0.0):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return result


def _load_actuators(robot: dict) -> tuple[Actuator, ...]:
    source = robot.get("actuators")
    if not isinstance(source, list) or not source:
        raise ValueError("robot.actuators must be a non-empty sequence")
    result: list[Actuator] = []
    ports: set[int] = set()
    names: set[str] = set()
    for index, value in enumerate(source):
        item = mapping(value, f"robot.actuators[{index}]")
        try:
            port_id = int(item["port_id"])
            joint_name = str(item["joint_name"])
            control = str(item["control"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid robot.actuators[{index}] identity") from exc
        if port_id < 1 or port_id > 31 or port_id in ports:
            raise ValueError(f"invalid or duplicate actuator port_id {port_id}")
        if not joint_name or joint_name in names:
            raise ValueError(f"invalid or duplicate actuator joint_name {joint_name!r}")
        if control not in ("position", "velocity"):
            raise ValueError(f"unsupported control mode for {joint_name}: {control}")
        direction = finite(item.get("direction"), f"{joint_name}.direction")
        if not math.isclose(abs(direction), 1.0, abs_tol=1e-9):
            raise ValueError(f"{joint_name}.direction must be +1 or -1")
        continuous = item.get("continuous", False)
        if not isinstance(continuous, bool):
            raise ValueError(f"{joint_name}.continuous must be boolean")
        limits_value = item.get("position_limits_rad")
        limits: tuple[float, float] | None = None
        if limits_value is not None:
            if not isinstance(limits_value, list) or len(limits_value) != 2:
                raise ValueError(f"{joint_name}.position_limits_rad must contain two values")
            low = finite(limits_value[0], f"{joint_name}.position limit")
            high = finite(limits_value[1], f"{joint_name}.position limit")
            if low >= high:
                raise ValueError(f"{joint_name}.position_limits_rad is not increasing")
            limits = (low, high)
        actuator = Actuator(
            port_id=port_id,
            joint_name=joint_name,
            control=control,
            direction=direction,
            mechanical_zero_rad=finite(
                item.get("mechanical_zero_rad", 0.0),
                f"{joint_name}.mechanical_zero_rad",
            ),
            continuous=continuous,
            default_position_rad=finite(
                item.get("default_position_rad", 0.0),
                f"{joint_name}.default_position_rad",
            ),
            kp=positive(item.get("kp", 0.0), f"{joint_name}.kp", allow_zero=True),
            kd=positive(item.get("kd", 0.0), f"{joint_name}.kd", allow_zero=True),
            command_velocity_max_rad_s=positive(
                item["command_velocity_max_rad_s"],
                f"{joint_name}.command_velocity_max_rad_s",
            ),
            device_velocity_max_rad_s=positive(
                item["device_velocity_max_rad_s"],
                f"{joint_name}.device_velocity_max_rad_s",
            ),
            position_limits_rad=limits,
        )
        if control == "position" and limits is None:
            raise ValueError(f"position actuator {joint_name} has no position limits")
        ports.add(port_id)
        names.add(joint_name)
        result.append(actuator)
    return tuple(sorted(result, key=lambda item: item.port_id))


def load_robot_catalog(robot_profile_path: Path) -> RobotCatalog:
    robot_path = robot_profile_path.resolve(strict=True)
    robot = document(robot_path, "robot profile")
    if robot.get("format_version") != 1:
        raise ValueError("robot format_version must be 1")
    robot_id = str(robot.get("robot_id", ""))
    profile_id = str(robot.get("profile_id", ""))
    if not robot_id or not profile_id:
        raise ValueError("robot.robot_id and robot.profile_id must not be empty")
    try:
        schema_id = int(robot.get("schema_id", 0))
        fingerprint = int(robot.get("lower_config_fingerprint", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "robot schema_id and lower_config_fingerprint must be integers") from exc
    if schema_id <= 0 or fingerprint <= 0:
        raise ValueError("robot schema_id and lower_config_fingerprint must be positive")
    control_allowed = robot.get("control_allowed")
    calibrated = robot.get("calibrated")
    if not isinstance(control_allowed, bool) or not isinstance(calibrated, bool):
        raise ValueError("robot control_allowed and calibrated must be boolean")
    return RobotCatalog(
        robot_id=robot_id,
        profile_id=profile_id,
        schema_id=schema_id,
        lower_config_fingerprint=fingerprint,
        control_allowed=control_allowed,
        calibrated=calibrated,
        actuators=_load_actuators(robot),
        robot_profile_path=robot_path,
    )
