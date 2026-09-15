"""Versioned, robot-neutral inference release contract."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA_ID = "aar-unilab.actor.v1"
SUPPORTED_SCHEMA_IDS = frozenset({SCHEMA_ID})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_artifact(value: object, field: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be release-relative, got {value!r}")
    return path.as_posix()


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema")
    if schema not in SUPPORTED_SCHEMA_IDS:
        raise ValueError(f"unsupported release schema: {schema!r}")
    for key in ("robot", "task", "policy", "control", "artifacts"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"manifest.{key} must be an object")
    robot = payload["robot"]
    policy = payload["policy"]
    if not robot.get("id") or not isinstance(robot.get("joint_order"), list):
        raise ValueError("manifest.robot requires id and joint_order")
    for direction in ("inputs", "outputs"):
        tensors = policy.get(direction)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f"manifest.policy.{direction} must be a non-empty list")
        for tensor in tensors:
            if not isinstance(tensor, dict) or not tensor.get("name"):
                raise ValueError(f"invalid tensor in manifest.policy.{direction}")
            if not isinstance(tensor.get("shape"), list):
                raise ValueError(f"tensor {tensor.get('name')!r} has no shape list")
    for key, value in payload["artifacts"].items():
        if key.endswith("_path"):
            _relative_artifact(value, f"artifacts.{key}")
    return payload


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("deployment manifest must contain a JSON object")
    return validate_manifest(payload)


def write_sha256sums(release_dir: Path, paths: Iterable[Path]) -> Path:
    lines = [f"{sha256_file(path)}  {path.relative_to(release_dir).as_posix()}" for path in paths]
    target = release_dir / "SHA256SUMS"
    target.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    return target


def create_release(
    destination: str | Path,
    *,
    onnx: str | Path,
    runtime_config: str | Path,
    manifest: dict[str, Any],
    robot_files: Iterable[str | Path] = (),
    golden_inputs: Mapping[str, np.ndarray] | None = None,
    golden_outputs: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Materialize a self-contained release without embedding source paths."""
    release_dir = Path(destination)
    release_dir.mkdir(parents=True, exist_ok=False)
    policy_target = release_dir / "policy.onnx"
    config_target = release_dir / "runtime_config.yaml"
    shutil.copy2(onnx, policy_target)
    shutil.copy2(runtime_config, config_target)
    copied = [policy_target, config_target]
    robot_dir = release_dir / "robot"
    for source_value in robot_files:
        source = Path(source_value)
        target = robot_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    for directory_name, tensors in (
        ("golden_inputs", golden_inputs or {}),
        ("golden_outputs", golden_outputs or {}),
    ):
        directory = release_dir / directory_name
        for name, value in tensors.items():
            if Path(name).name != name:
                raise ValueError(f"golden tensor name must be a filename stem: {name!r}")
            target = directory / f"{name}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, np.asarray(value), allow_pickle=False)
            copied.append(target)
    manifest = dict(manifest)
    manifest["artifacts"] = dict(manifest.get("artifacts", {}))
    manifest["artifacts"].update(
        {"policy_path": "policy.onnx", "runtime_config_path": "runtime_config.yaml"}
    )
    validate_manifest(manifest)
    manifest_target = release_dir / "deployment_manifest.json"
    manifest_target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copied.append(manifest_target)
    write_sha256sums(release_dir, copied)
    return release_dir
