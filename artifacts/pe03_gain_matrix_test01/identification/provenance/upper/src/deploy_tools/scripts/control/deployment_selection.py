#!/usr/bin/env python3
"""Resolve one active robot-policy deployment without robot-name whitelists."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DeploymentSelection:
    deployment_id: str
    deployment_path: Path
    robot_profile_path: Path
    sweep_profile_path: Path
    deployment: dict
    launcher_start_button: int
    launcher_stop_button: int
    launcher_dpad_axis: int
    launcher_dpad_threshold: int
    enable_height_command: bool


def _document(path: Path, description: str) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid {description} {path}: root must be a mapping")
    return value


def load_active_deployment(
    root: Path, *, deployment_id: str | None = None
) -> DeploymentSelection:
    selector_path = root / "config/active_deployment.yaml"
    selector = _document(selector_path, "active deployment selector")
    try:
        if selector["format_version"] != 1:
            raise ValueError("format_version must be 1")
        selected = deployment_id or str(selector["active_deployment"])
        if selected.count("/") != 1 or any(not part for part in selected.split("/")):
            raise ValueError("active_deployment must be <robot_id>/<contract_id>")
        launcher = selector["launcher"]
        start_button = int(launcher["start_button"])
        stop_button = int(launcher["stop_button"])
        dpad_axis = int(launcher["dpad_axis"])
        dpad_threshold = int(launcher["dpad_threshold"])
        if (
            min(start_button, stop_button, dpad_axis) < 0
            or start_button == stop_button
            or dpad_threshold <= 0
        ):
            raise ValueError("launcher mapping is invalid")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid active deployment selector {selector_path}: {exc}") from exc

    index_path = root / "install/robot_packages/deployment_index.yaml"
    index = _document(index_path, "installed deployment index")
    try:
        if index["format_version"] != 1:
            raise ValueError("format_version must be 1")
        item = index["deployments"][selected]
        deployment_path = Path(item["deployment_path"])
        if not deployment_path.is_absolute() or not deployment_path.is_file():
            raise ValueError("deployment_path must name an installed file")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"deployment {selected!r} is not installed in {index_path}; "
            f"run ./tools/build_robot_package.sh {selected}: {exc}"
        ) from exc

    deployment = _document(deployment_path, "deployment contract")
    try:
        robot_id, contract_id = selected.split("/", 1)
        if (
            deployment["format_version"] != 1
            or deployment["deployment_id"] != selected
            or deployment["robot_id"] != robot_id
            or deployment["contract_id"] != contract_id
        ):
            raise ValueError("deployment identity does not match selector/index")
        profile_path = deployment_path.parent / str(deployment["robot_profile"])
        if not profile_path.is_file():
            raise ValueError(f"Robot Profile does not exist: {profile_path}")
        sweep_path = deployment_path.parent / str(deployment["sweep_profile"])
        if not sweep_path.is_file():
            raise ValueError(f"Sweep Profile does not exist: {sweep_path}")
        command_sources = deployment.get("command_sources", {})
        xbox = command_sources.get("xbox", {})
        enable_height = xbox.get("enable_height_command", False)
        if not isinstance(enable_height, bool):
            raise ValueError("command_sources.xbox.enable_height_command must be boolean")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid deployment contract {deployment_path}: {exc}") from exc

    return DeploymentSelection(
        deployment_id=selected,
        deployment_path=deployment_path.absolute(),
        robot_profile_path=profile_path.absolute(),
        sweep_profile_path=sweep_path.absolute(),
        deployment=deployment,
        launcher_start_button=start_button,
        launcher_stop_button=stop_button,
        launcher_dpad_axis=dpad_axis,
        launcher_dpad_threshold=dpad_threshold,
        enable_height_command=enable_height,
    )
