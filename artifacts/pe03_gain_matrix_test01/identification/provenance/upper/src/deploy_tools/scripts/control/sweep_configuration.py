#!/usr/bin/env python3
"""Pure configuration parsing for profile-driven ESD-Link frequency sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml

from robot_configuration import Actuator, load_robot_catalog


@dataclass(frozen=True)
class SweepTarget:
    joint_name: str
    port_id: int
    phase_sign: float
    amplitude: float


@dataclass(frozen=True)
class SweepGroup:
    group_id: str
    enabled: bool
    mode: str
    targets: tuple[SweepTarget, ...]
    preparation_pose_rad: dict[str, float]
    start_frequency_hz: float
    end_frequency_hz: float
    duration_sec: float
    kp: float
    kd: float


@dataclass(frozen=True)
class SweepCatalog:
    robot_id: str
    profile_id: str
    schema_id: int
    lower_config_fingerprint: int
    actuators: tuple[Actuator, ...]
    defaults: dict[str, float | int | None]
    groups: dict[str, SweepGroup]
    robot_profile_path: Path
    sweep_profile_path: Path

    @property
    def active_port_mask(self) -> int:
        return sum(1 << item.port_id for item in self.actuators)

    @property
    def actuator_by_name(self) -> dict[str, Actuator]:
        return {item.joint_name: item for item in self.actuators}


DEFAULT_FIELDS: dict[str, tuple[float, float, float]] = {
    "publish_rate_hz": (200.0, 0.0, 500.0),
    "record_rate_hz": (200.0, 0.0, 500.0),
    "state_timeout_sec": (10.0, 0.0, math.inf),
    "state_freshness_sec": (0.5, 0.0, math.inf),
    "return_speed_rad_s": (0.1, 0.0, math.inf),
    "return_min_duration_sec": (1.0, 0.0, math.inf),
    "settle_duration_sec": (0.25, 0.0, math.inf),
}

MIT_KP_MAX = 500.0
MIT_KD_MAX = 5.0


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _document(path: Path, name: str) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {name} {path}: {exc}") from exc
    return _mapping(value, name)


def finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = finite(value, name)
    if result < 0.0 or (not allow_zero and result == 0.0):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return result


def _load_defaults(sweep: dict) -> dict[str, float | int | None]:
    source = _mapping(sweep.get("defaults", {}), "sweep.defaults")
    result: dict[str, float | int | None] = {}
    for field, (fallback, minimum, maximum) in DEFAULT_FIELDS.items():
        value = finite(source.get(field, fallback), f"defaults.{field}")
        if value <= minimum or value > maximum:
            raise ValueError(f"defaults.{field} must be in ({minimum}, {maximum}]")
        result[field] = value
    if float(result["record_rate_hz"]) > float(result["publish_rate_hz"]):
        raise ValueError("defaults.record_rate_hz cannot exceed publish_rate_hz")
    has_hold_kp = "hold_kp" in source
    has_hold_kd = "hold_kd" in source
    if has_hold_kp != has_hold_kd:
        raise ValueError("defaults.hold_kp and hold_kd must be configured together")
    if has_hold_kp:
        hold_kp = _positive(
            source["hold_kp"], "defaults.hold_kp", allow_zero=True)
        hold_kd = _positive(source["hold_kd"], "defaults.hold_kd")
        if hold_kp > MIT_KP_MAX or hold_kd > MIT_KD_MAX:
            raise ValueError(
                f"hold gains exceed MIT limits Kp<={MIT_KP_MAX:g}, Kd<={MIT_KD_MAX:g}")
        result["hold_kp"] = hold_kp
        result["hold_kd"] = hold_kd
    else:
        result["hold_kp"] = None
        result["hold_kd"] = None
    return result


def _load_poses(sweep: dict, actuators: dict[str, Actuator]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    source = _mapping(sweep.get("poses", {}), "sweep.poses")
    for pose_name, pose_value in source.items():
        if not isinstance(pose_name, str) or not pose_name:
            raise ValueError("pose names must be non-empty strings")
        result[pose_name] = _validate_pose(
            _mapping(pose_value, f"poses.{pose_name}"), actuators, f"poses.{pose_name}")
    return result


def _validate_pose(
    source: dict, actuators: dict[str, Actuator], description: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_name, raw_position in source.items():
        name = str(raw_name)
        actuator = actuators.get(name)
        if actuator is None:
            raise ValueError(f"{description} references unknown joint {name!r}")
        if actuator.control != "position" or actuator.position_limits_rad is None:
            raise ValueError(f"{description} cannot position velocity joint {name!r}")
        if raw_position == "robot_default":
            position = actuator.default_position_rad
        elif raw_position == "limit_midpoint":
            low, high = actuator.position_limits_rad
            position = (low + high) / 2.0
        else:
            position = finite(raw_position, f"{description}.{name}")
        low, high = actuator.position_limits_rad
        if position < low or position > high:
            raise ValueError(
                f"{description}.{name}={position} exceeds robot limits [{low}, {high}]")
        result[name] = position
    return result


def _load_groups(
    sweep: dict,
    actuators: dict[str, Actuator],
    poses: dict[str, dict[str, float]],
    defaults: dict[str, float | int | None],
) -> dict[str, SweepGroup]:
    source = _mapping(sweep.get("groups"), "sweep.groups")
    if not source:
        raise ValueError("sweep.groups must not be empty")
    result: dict[str, SweepGroup] = {}
    target_owner: dict[str, str] = {}
    publish_rate = float(defaults["publish_rate_hz"])
    for raw_group_id, raw_group in source.items():
        group_id = str(raw_group_id)
        if not group_id or group_id in result:
            raise ValueError("sweep group ids must be unique non-empty strings")
        group = _mapping(raw_group, f"groups.{group_id}")
        enabled = group.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"groups.{group_id}.enabled must be boolean")
        mode = str(group.get("mode", ""))
        if mode not in ("position", "velocity"):
            raise ValueError(f"groups.{group_id}.mode must be position or velocity")
        pose_name = group.get("preparation_pose")
        inline_pose = group.get("preparation_pose_rad")
        if pose_name is not None and inline_pose is not None:
            raise ValueError(f"groups.{group_id} cannot define both preparation pose forms")
        if pose_name is not None:
            if not isinstance(pose_name, str) or pose_name not in poses:
                raise ValueError(f"groups.{group_id}.preparation_pose is unknown")
            pose = dict(poses[pose_name])
        else:
            pose = _validate_pose(
                _mapping(inline_pose or {}, f"groups.{group_id}.preparation_pose_rad"),
                actuators,
                f"groups.{group_id}.preparation_pose_rad",
            )
        targets_value = group.get("targets")
        if not isinstance(targets_value, list) or not targets_value:
            raise ValueError(f"groups.{group_id}.targets must be a non-empty sequence")
        targets: list[SweepTarget] = []
        local_names: set[str] = set()
        for index, raw_target in enumerate(targets_value):
            target = _mapping(raw_target, f"groups.{group_id}.targets[{index}]")
            name = str(target.get("joint_name", ""))
            actuator = actuators.get(name)
            if actuator is None:
                raise ValueError(f"groups.{group_id} references unknown joint {name!r}")
            if name in local_names:
                raise ValueError(f"groups.{group_id} repeats joint {name!r}")
            if name in target_owner:
                raise ValueError(
                    f"joint {name!r} belongs to both {target_owner[name]!r} and {group_id!r}")
            if actuator.control != mode:
                raise ValueError(
                    f"groups.{group_id} mode {mode} does not match {name} control "
                    f"{actuator.control}"
                )
            sign = finite(target.get("phase_sign"), f"groups.{group_id}.{name}.phase_sign")
            if not math.isclose(abs(sign), 1.0, abs_tol=1e-9):
                raise ValueError(f"groups.{group_id}.{name}.phase_sign must be +1 or -1")
            has_amplitude = "amplitude" in target
            has_limit_margin = "limit_margin_rad" in target
            if has_amplitude == has_limit_margin:
                raise ValueError(
                    f"groups.{group_id}.{name} must define exactly one of "
                    "amplitude or limit_margin_rad"
                )
            if has_limit_margin:
                if mode != "position" or actuator.position_limits_rad is None:
                    raise ValueError(
                        f"groups.{group_id}.{name}.limit_margin_rad requires "
                        "a position joint with limits"
                    )
                margin = _positive(
                    target["limit_margin_rad"],
                    f"groups.{group_id}.{name}.limit_margin_rad",
                )
                low, high = actuator.position_limits_rad
                midpoint = (low + high) / 2.0
                center = pose.get(name)
                if center is None or not math.isclose(center, midpoint, abs_tol=1e-9):
                    raise ValueError(
                        f"groups.{group_id}.{name}.limit_margin_rad requires "
                        "the preparation pose at the limit midpoint"
                    )
                amplitude = (high - low) / 2.0 - margin
                if amplitude <= 0.0:
                    raise ValueError(
                        f"groups.{group_id}.{name}.limit_margin_rad leaves "
                        "no positive sweep amplitude"
                    )
            else:
                amplitude = _positive(
                    target["amplitude"], f"groups.{group_id}.{name}.amplitude")
            if mode == "position":
                if name not in pose:
                    raise ValueError(
                        f"groups.{group_id} position target {name!r} is absent "
                        "from preparation pose"
                    )
                assert actuator.position_limits_rad is not None
                low, high = actuator.position_limits_rad
                center = pose[name]
                if center - amplitude < low or center + amplitude > high:
                    raise ValueError(
                        f"groups.{group_id}.{name} sweep [{center - amplitude}, "
                        f"{center + amplitude}] exceeds robot limits [{low}, {high}]")
            else:
                maximum = min(
                    actuator.command_velocity_max_rad_s,
                    actuator.device_velocity_max_rad_s,
                )
                if amplitude > maximum:
                    raise ValueError(
                        f"groups.{group_id}.{name} amplitude {amplitude} "
                        f"exceeds velocity {maximum}"
                    )
            targets.append(SweepTarget(name, actuator.port_id, sign, amplitude))
            local_names.add(name)
            target_owner[name] = group_id
        start = _positive(
            group.get("start_frequency_hz"), f"groups.{group_id}.start_frequency_hz")
        end = _positive(
            group.get("end_frequency_hz"), f"groups.{group_id}.end_frequency_hz")
        if end < start or end >= publish_rate / 2.0:
            raise ValueError(
                f"groups.{group_id} frequencies must satisfy 0 < start <= end < publish_rate/2")
        duration = _positive(group.get("duration_sec"), f"groups.{group_id}.duration_sec")
        kp = _positive(group.get("kp"), f"groups.{group_id}.kp", allow_zero=True)
        kd = _positive(group.get("kd"), f"groups.{group_id}.kd")
        if kp > MIT_KP_MAX or kd > MIT_KD_MAX:
            raise ValueError(
                f"groups.{group_id} gains exceed MIT limits "
                f"Kp<={MIT_KP_MAX:g}, Kd<={MIT_KD_MAX:g}")
        if mode == "velocity" and not math.isclose(kp, 0.0, abs_tol=1e-9):
            raise ValueError(f"groups.{group_id} velocity sweep requires kp=0")
        result[group_id] = SweepGroup(
            group_id=group_id,
            enabled=enabled,
            mode=mode,
            targets=tuple(targets),
            preparation_pose_rad=pose,
            start_frequency_hz=start,
            end_frequency_hz=end,
            duration_sec=duration,
            kp=kp,
            kd=kd,
        )
    return result


def load_sweep_catalog(robot_profile_path: Path, sweep_profile_path: Path) -> SweepCatalog:
    robot_catalog = load_robot_catalog(robot_profile_path)
    sweep_path = sweep_profile_path.resolve(strict=True)
    sweep = _document(sweep_path, "sweep profile")
    if sweep.get("format_version") != 1:
        raise ValueError("sweep format_version must be 1")
    if sweep.get("robot_id") != robot_catalog.robot_id:
        raise ValueError("sweep.robot_id does not match robot profile")
    if not robot_catalog.control_allowed or not robot_catalog.calibrated:
        raise ValueError("robot profile is not authorized and calibrated for control")
    actuator_tuple = robot_catalog.actuators
    actuators = {item.joint_name: item for item in actuator_tuple}
    defaults = _load_defaults(sweep)
    poses = _load_poses(sweep, actuators)
    groups = _load_groups(sweep, actuators, poses, defaults)
    return SweepCatalog(
        robot_id=robot_catalog.robot_id,
        profile_id=robot_catalog.profile_id,
        schema_id=robot_catalog.schema_id,
        lower_config_fingerprint=robot_catalog.lower_config_fingerprint,
        actuators=actuator_tuple,
        defaults=defaults,
        groups=groups,
        robot_profile_path=robot_catalog.robot_profile_path,
        sweep_profile_path=sweep_path,
    )


def select_group(
    catalog: SweepCatalog,
    *,
    group_id: str | None = None,
    joint_names: list[str] | None = None,
) -> tuple[SweepGroup, tuple[SweepTarget, ...]]:
    requested = tuple(joint_names or ())
    if (group_id is None) == (not requested):
        raise ValueError("select exactly one --group or one or more --joint values")
    if group_id is not None:
        group = catalog.groups.get(group_id)
        if group is None:
            raise ValueError(f"unknown sweep group {group_id!r}")
        selected = group.targets
    else:
        if len(set(requested)) != len(requested):
            raise ValueError("--joint values must not be repeated")
        owners = {
            candidate.group_id
            for candidate in catalog.groups.values()
            if any(target.joint_name in requested for target in candidate.targets)
        }
        known = {
            target.joint_name
            for candidate in catalog.groups.values()
            for target in candidate.targets
        }
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise ValueError("unknown sweep joint(s): " + ", ".join(unknown))
        if len(owners) != 1:
            raise ValueError("all --joint values must belong to one sweep group")
        group = catalog.groups[next(iter(owners))]
        requested_set = set(requested)
        selected = tuple(
            target for target in group.targets if target.joint_name in requested_set)
    if not group.enabled:
        raise ValueError(f"sweep group {group.group_id!r} is disabled")
    return group, selected


def apply_gain_overrides(
    group: SweepGroup,
    *,
    sweep_kp: float | None = None,
    sweep_kd: float | None = None,
) -> SweepGroup:
    kp = group.kp if sweep_kp is None else _positive(
        sweep_kp, "sweep kp", allow_zero=True)
    kd = group.kd if sweep_kd is None else _positive(sweep_kd, "sweep kd")
    if group.mode == "velocity" and not math.isclose(kp, 0.0, abs_tol=1e-9):
        raise ValueError("velocity sweep requires Kp=0")
    return SweepGroup(
        group_id=group.group_id,
        enabled=group.enabled,
        mode=group.mode,
        targets=group.targets,
        preparation_pose_rad=dict(group.preparation_pose_rad),
        start_frequency_hz=group.start_frequency_hz,
        end_frequency_hz=group.end_frequency_hz,
        duration_sec=group.duration_sec,
        kp=kp,
        kd=kd,
    )
