"""Cold-path asset, observation and checkpoint contracts owned by PE05."""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe05.config import ROOT, validate_config


def observation_layout() -> dict[str, Any]:
    return {
        "frame": [
            ["body_angular_velocity", 3],
            ["projected_gravity", 3],
            ["relative_joint_position", 6],
            ["joint_velocity", 6],
            ["applied_target_action", 6],
            ["phase_sin_cos", 2],
            ["gait", 4],
        ],
        "history": [10, 30, "oldest-first"],
        "command": ["vx", "vy", "yaw_rate"],
        "critic": [
            "body_linear_velocity",
            "current_clean_frame",
            "command",
            "detached_velocity_estimate",
        ],
        "network_inputs": {"encoder": 300, "actor": 36, "critic": 39},
    }


def asset_fingerprint(config: DictConfig) -> str:
    scene = repository_path(str(config.env.model_path), ROOT)
    folder = scene.parent
    files = [folder / "pe05.xml", scene, folder / "provenance.json"]
    for directory in ("collision_meshes", "runtime_meshes"):
        files.extend(sorted(p for p in (folder / directory).rglob("*") if p.is_file()))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(folder)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resume_contract(config: DictConfig) -> dict[str, Any]:
    value = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    for key in ("mode", "checkpoint", "play", "training"):
        value.pop(key, None)
    value["algo"].pop("max_iterations")
    value["algo"].pop("save_interval")
    return value


def validate_checkpoint(
    payload: dict[str, Any], config: DictConfig | None = None, asset_hash: str | None = None
) -> DictConfig:
    if payload.get("schema") != "pe05.training.v1" or payload.get("robot_id") != "pe05":
        raise ValueError("PE05 requires a pe05.training.v1 checkpoint with robot_id=pe05")
    saved = OmegaConf.create(payload["training_config"])
    validate_config(saved)
    if payload.get("observation_layout") != observation_layout():
        raise ValueError("PE05 checkpoint observation layout changed")
    if payload.get("training_contract") != resume_contract(saved):
        raise ValueError("PE05 checkpoint training contract is inconsistent")
    if config is not None and resume_contract(config) != resume_contract(saved):
        raise ValueError("resume changes training contract; start a new run")
    if payload.get("asset_sha256") != (asset_hash or asset_fingerprint(config or saved)):
        raise ValueError("PE05 assets changed since checkpoint")
    return saved


def resolve_checkpoint(value: str | int | Path, log_root: str | Path) -> Path:
    """Select the newest timestamped run, then its highest numeric saved iteration."""
    if str(value) != "-1":
        return Path(str(value)).expanduser()
    root = Path(log_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"PE05 training log directory does not exist: {root}")
    runs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        for fmt in ("%Y-%m-%d_%H-%M-%S_%f_mujoco", "%Y-%m-%d_%H-%M-%S_mujoco"):
            try:
                timestamp = datetime.strptime(path.name, fmt)
            except ValueError:
                continue
            runs.append((timestamp, path.name, path))
            break
    if not runs:
        raise FileNotFoundError(f"No timestamped PE05 training runs found in {root}")
    latest = max(runs)[2]
    models = []
    for path in latest.iterdir():
        match = re.fullmatch(r"model_([0-9]+)\.pt", path.name)
        if match is not None and path.is_file():
            models.append((int(match[1]), path.name, path))
    if not models:
        raise FileNotFoundError(
            f"Latest PE05 run has no saved model_<iteration>.pt yet: {latest}; "
            "wait for a checkpoint or specify an explicit checkpoint path"
        )
    return max(models)[2]
