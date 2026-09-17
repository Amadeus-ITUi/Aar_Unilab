#!/usr/bin/env python3
"""Rebuild completed PE02 runs' TensorBoard groups, keeping an event archive."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from torch.utils.tensorboard import SummaryWriter

from unilab.algos.torch.pe02.tensorboard import tensorboard_metrics


def _signature(paths: list[Path]) -> list[tuple[str, int, int]]:
    return [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def rebuild(run_dir: Path) -> tuple[Path, int]:
    """Replace events only after validating the source and staging the new file."""
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "training_config.yaml"
    event_dir = run_dir / "tensorboard"
    event_files = sorted(event_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise ValueError(f"no existing TensorBoard events in {event_dir}")
    source_files = [metrics_path, config_path, *sorted(event_dir.rglob("*"))]
    source_files = [path for path in source_files if path.is_file()]
    signature = _signature(source_files)
    config = OmegaConf.load(config_path)
    if config.robot != "pe02" or config.observation != "pe02_v2":
        raise ValueError("only formal PE02 runs can be rebuilt")
    policy_dt = 1.0 / float(config.control.policy_hz)
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("no training metrics to rebuild")
    wall_times: dict[tuple[int, str], float] = {}
    for path in event_files:
        for record in RawEventFileLoader(str(path)).Load():
            event = Event.FromString(record)
            for value in event.summary.value:
                for tag in tensorboard_metrics({value.tag: 0.0}, policy_dt=policy_dt):
                    wall_times[event.step, tag] = event.wall_time
    for row in rows:
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("training metrics contain non-finite values")
        if any(
            (int(row["iteration"]), tag) not in wall_times
            for tag in tensorboard_metrics(row, policy_dt=policy_dt)
        ):
            raise ValueError("missing original event timestamp; stop training before rebuilding")

    # Stage beside the log root so TensorBoard cannot see partially rebuilt runs.
    with tempfile.TemporaryDirectory(prefix=".pe02-tensorboard-", dir=run_dir.parent.parent) as tmp:
        staging = Path(tmp)
        rebuilt = staging / "rebuilt"
        with SummaryWriter(str(rebuilt)) as writer:
            for row in rows:
                iteration = int(row["iteration"])
                for tag, value in tensorboard_metrics(row, policy_dt=policy_dt).items():
                    writer.add_scalar(tag, value, iteration, walltime=wall_times[iteration, tag])
        backup = run_dir / (
            "tensorboard_before_we11_style_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".zip"
        )
        with zipfile.ZipFile(staging / "backup.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in source_files[2:]:
                archive.write(path, path.relative_to(run_dir))
        if (
            _signature(source_files) != signature
            or sorted(event_dir.glob("events.out.tfevents.*")) != event_files
        ):
            raise RuntimeError("run changed during rebuild; stop training and retry")
        (staging / "backup.zip").rename(backup)
        original = staging / "original"
        event_dir.rename(original)
        try:
            rebuilt.rename(event_dir)
        except BaseException:
            original.rename(event_dir)
            raise
    return backup, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="+", help="completed PE02 run directories")
    args = parser.parse_args()
    for run_dir in args.run_dir:
        backup, count = rebuild(run_dir)
        print(f"{run_dir}: rebuilt {count} iterations; original events: {backup}")


if __name__ == "__main__":
    main()
