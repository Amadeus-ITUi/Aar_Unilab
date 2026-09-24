#!/usr/bin/env python3
"""Measure PE04 rollout/update latency without writing training checkpoints."""

from __future__ import annotations

import argparse
import cProfile
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import torch

from unilab.algos.torch.pe04.runner import PE04Runner
from unilab.envs.locomotion.pe04.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--legacy-telemetry", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations positive")
    cfg = load_config(
        [
            *args.overrides,
            "training.logger=none",
            "training.evaluation_interval=0",
            "training.export=false",
        ]
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="pe04-benchmark-") as directory:
        runner = PE04Runner(cfg, Path(directory))
        try:
            if args.legacy_telemetry:
                runner.env.backend.fused_telemetry = False

            def synchronize():
                if runner.device.type == "cuda":
                    torch.cuda.synchronize(runner.device)

            for iteration in range(args.warmup + args.iterations):
                synchronize()
                start = time.perf_counter()
                rollout, _ = runner.collect()
                synchronize()
                collected = time.perf_counter()
                runner.algorithm.update(rollout)
                synchronize()
                finished = time.perf_counter()
                row = dict(
                    iteration=iteration,
                    warmup=iteration < args.warmup,
                    collection_seconds=collected - start,
                    update_seconds=finished - collected,
                    total_seconds=finished - start,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
            measured = rows[args.warmup :]
            report = dict(
                schema="pe04.benchmark.v1",
                num_envs=cfg.algo.num_envs,
                rollout_steps=cfg.algo.num_steps_per_env,
                epochs=cfg.algo.num_learning_epochs,
                minibatches=cfg.algo.num_mini_batches,
                mujoco_threads=cfg.training.mujoco_threads,
                matmul_precision=cfg.training.matmul_precision,
                update_precision=runner.algorithm.update_precision,
                fused_telemetry=runner.env.backend.fused_telemetry,
                device=str(runner.device),
                cpu=platform.processor(),
                gpu=torch.cuda.get_device_name(runner.device)
                if runner.device.type == "cuda"
                else None,
                mean={
                    key: statistics.mean(row[key] for row in measured)
                    for key in ("collection_seconds", "update_seconds", "total_seconds")
                },
                median_total_seconds=statistics.median(row["total_seconds"] for row in measured),
                rows=rows,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["mean"]), flush=True)
            if args.profile:
                profiler = cProfile.Profile()
                profiler.enable()
                rollout, _ = runner.collect()
                runner.algorithm.update(rollout)
                synchronize()
                profiler.disable()
                args.profile.parent.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(str(args.profile))
        finally:
            runner.close()


if __name__ == "__main__":
    main()
