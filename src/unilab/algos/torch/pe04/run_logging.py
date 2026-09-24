"""Descriptive TensorBoard run aliases without relocating training artifacts."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def run_start_time(run_dir: Path) -> datetime | None:
    """Read the training start time, not a checkpoint/event modification time."""
    for fmt in ("%Y-%m-%d_%H-%M-%S_%f_mujoco", "%Y-%m-%d_%H-%M-%S_mujoco"):
        try:
            return datetime.strptime(run_dir.name, fmt)
        except ValueError:
            pass
    return None


def _component(value: str) -> str:
    component = re.sub(r"[^\w.-]+", "_", value).strip("._-")
    if not component:
        raise ValueError("TensorBoard name components must contain letters or digits")
    return component


def index_tensorboard_run(
    run_dir: Path, config: DictConfig, *, started_at: datetime | None = None
) -> Path:
    """Publish a relative link and persist the identity for repeat indexing/resume."""
    run_dir = run_dir.resolve()
    events = run_dir / "tensorboard"
    if not events.is_dir():
        raise FileNotFoundError(f"TensorBoard event directory does not exist: {events}")
    experiment = config.training.get("experiment_name")
    if experiment is None:
        # Older checkpoints did not record the Hydra experiment selection explicitly.
        legacy_root = Path(str(config.training.get("log_root", run_dir.parent))).name
        experiment = {"pe04_walking": "walking", "pe04_standing": "standing"}.get(
            legacy_root, "baseline"
        )
    identity = {
        "robot": str(config.robot),
        "task": str(config.task_id),
        "simulator": str(config.simulator),
        "experiment": str(experiment),
        "observation": str(config.observation),
    }
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema") != "pe04.tensorboard-run.v1" or any(
            metadata.get(key) != value for key, value in identity.items()
        ):
            raise ValueError(f"existing TensorBoard run identity disagrees with config: {run_dir}")
        timestamp = datetime.fromisoformat(metadata["started_at"])
    else:
        timestamp = run_start_time(run_dir) or started_at or datetime.now()
    name = "__".join(
        (
            _component(identity["robot"].upper()),
            _component(identity["task"]) + "-" + _component(identity["simulator"]),
            _component(identity["experiment"]),
            timestamp.strftime("%Y-%m-%d_%H-%M-%S_%f"),
        )
    )
    index = run_dir.parent / "tensorboard_runs"
    index.mkdir(exist_ok=True)
    link = index / name
    if link.is_symlink() or link.exists():
        if not link.is_symlink() or link.resolve() != events:
            raise FileExistsError(f"TensorBoard run name already belongs to another path: {link}")
    else:
        link.symlink_to(os.path.relpath(events, index), target_is_directory=True)
    if not metadata_path.exists():
        metadata = {
            "schema": "pe04.tensorboard-run.v1",
            **identity,
            "started_at": timestamp.isoformat(timespec="microseconds"),
            "run_name": name,
            "event_path": "tensorboard",
        }
        with metadata_path.open("x") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return link


def index_existing_runs(log_root: Path) -> list[Path]:
    """Index direct timestamped runs; validation subtrees are separate by construction."""
    links = []
    for run in sorted(log_root.iterdir()):
        if not run.is_dir() or run_start_time(run) is None:
            continue
        config_path = run / "training_config.yaml"
        if not config_path.is_file() or not any(
            (run / "tensorboard").glob("events.out.tfevents.*")
        ):
            continue
        config = OmegaConf.load(config_path)
        if config.robot != "pe04" or config.observation != "pe04_tron1_v1":
            continue
        links.append(index_tensorboard_run(run, config))
    return links
