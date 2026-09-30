#!/usr/bin/env python3
"""Mirror existing/live PE05 scalar events into PE03-style groups without editing source logs."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from torch.utils.tensorboard import SummaryWriter

from unilab.algos.torch.pe05.run_logging import index_tensorboard_run
from unilab.algos.torch.pe05.tensorboard import tensorboard_metrics


class EventMirror:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        config = OmegaConf.load(self.run_dir / "training_config.yaml")
        if config.robot != "pe05" or config.observation != "pe05_v1":
            raise ValueError("only PE05 runs can be mirrored")
        self.policy_dt = 1 / float(config.control.policy_hz)
        self.source = self.run_dir / "tensorboard"
        if not any(self.source.glob("events.out.tfevents.*")):
            raise ValueError(f"no source TensorBoard events: {self.source}")
        self.destination = self.run_dir / "tensorboard_pe03_style"
        # A fresh replay at step zero hides any previous mirror scalars. Originals are untouched.
        self.writer = SummaryWriter(str(self.destination), purge_step=0, flush_secs=5)
        self.loaders = {}
        try:
            self.link = index_tensorboard_run(
                self.run_dir, config, event_subdir=self.destination.name
            )
        except Exception:
            self.writer.close()
            raise

    def sync(self) -> int:
        count = 0
        for path in sorted(self.source.glob("events.out.tfevents.*")):
            if path not in self.loaders:
                self.loaders[path] = RawEventFileLoader(str(path))
            for record in self.loaders[path].Load():
                event = Event.FromString(record)
                if event.HasField("session_log"):
                    self.writer.file_writer.add_event(event)
                for value in event.summary.value:
                    if not value.HasField("simple_value"):
                        continue
                    for tag, scalar in tensorboard_metrics(
                        {value.tag: value.simple_value}, policy_dt=self.policy_dt
                    ).items():
                        self.writer.add_scalar(tag, scalar, event.step, walltime=event.wall_time)
                        count += 1
        self.writer.flush()
        return count

    def close(self):
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+", type=Path)
    parser.add_argument(
        "--watch", action="store_true", help="sync appended events every five seconds"
    )
    parser.add_argument("--until-pid", type=int, help="exit after this training process finishes")
    args = parser.parse_args()
    mirrors = []
    try:
        for run in args.run_dir:
            mirror = EventMirror(run)
            mirrors.append(mirror)
            print(f"{mirror.link}: synced {mirror.sync()} scalars", flush=True)
        while args.watch:
            if args.until_pid is not None:
                try:
                    os.kill(args.until_pid, 0)
                except ProcessLookupError:
                    break
            time.sleep(5)
            for mirror in mirrors:
                mirror.sync()
    finally:
        for mirror in mirrors:
            try:
                mirror.sync()
            finally:
                mirror.close()


if __name__ == "__main__":
    main()
