#!/usr/bin/env python3
"""Benchmark PE03's complete rollout and PPO/encoder update, without export I/O."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path


def memory() -> dict[str, int]:
    return {
        key: int(value.split()[0]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        for key, _, value in [line.partition(":")]
        if key in {"VmRSS", "VmHWM", "VmSwap"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--experiment", choices=("standing", "walking", "gait_fixed"), default="walking"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--matmul-precision", choices=("highest", "high"))
    parser.add_argument(
        "--cache-update-inputs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--native-pd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--robustness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use --no-robustness for a nominal gait throughput comparison",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.envs, args.threads, args.iterations, args.steps) <= 0 or args.warmup < 0:
        parser.error("positive envs/threads/iterations/steps and nonnegative warmup required")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.robustness and args.experiment != "gait_fixed":
        parser.error("--no-robustness is currently a gait_fixed comparison")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"

    import numpy as np
    import torch

    from unilab.algos.torch.pe03.runner import PE03Runner
    from unilab.envs.locomotion.pe03.config import load_config

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    overrides = [
        f"+experiment={args.experiment}",
        f"algo.num_envs={args.envs}",
        f"algo.num_steps_per_env={args.steps}",
        f"training.mujoco_threads={args.threads}",
        f"training.seed={args.seed}",
        f"training.device={device}",
        f"training.cache_update_inputs={str(args.cache_update_inputs).lower()}",
        "training.logger=none",
        "training.export=false",
        "training.evaluation_interval=0",
        f"training.native_pd={str(args.native_pd).lower()}",
    ]
    if args.model:
        overrides.append(f"env.model_path={args.model}")
    if not args.robustness:
        overrides.extend(
            [
                "noise.enabled=false",
                "domain_rand.enabled=false",
                "domain_rand.push_enabled=false",
                "domain_rand.delay_ms_range=[0,0]",
                "env.reset_randomization.enabled=false",
            ]
        )
    if args.matmul_precision is not None:
        overrides.append(f"training.matmul_precision={args.matmul_precision}")
    config = load_config(overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner = PE03Runner(config, args.output.with_suffix(""))
    records = []
    model = runner.env.backend.model
    metadata = {
        "experiment": args.experiment,
        "task": str(config.task_id),
        "observation": str(config.observation),
        "steps_per_env": int(config.algo.num_steps_per_env),
        "torch_version": torch.__version__,
        "update_matmul_precision": str(config.training.matmul_precision),
        "cache_update_inputs": bool(config.training.cache_update_inputs),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_sha256": {
            cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            for cls in (type(runner), type(runner.policy), type(runner.algorithm), type(runner.env))
        },
        "model_buffer_bytes": int(model.nbuffer),
        "model_sha256": runner.asset_hash,
        "native_pd": runner.env.backend.native_pd,
        "robustness": {
            "noise": bool(config.noise.enabled),
            "domain_rand": bool(config.domain_rand.enabled),
            "reset": bool(config.env.get("reset_randomization", {}).get("enabled", False)),
        },
        "collision_geoms": int(((model.geom_bodyid != 0) & (model.geom_contype != 0)).sum()),
        "mesh_vertices": model.nmeshvert,
        "mesh_faces": model.nmeshface,
        "physics_hz": int(config.control.physics_hz),
        "motor_hz": int(config.control.motor_hz),
        "policy_hz": int(config.control.policy_hz),
        "device": str(runner.device),
        "ppo_epochs": int(config.algo.num_learning_epochs),
        "minibatches": int(config.algo.num_mini_batches),
    }
    try:
        with args.output.with_suffix(".jsonl").open("w") as stream:
            for iteration in range(args.warmup + args.iterations):
                synchronize()
                started = time.perf_counter()
                rollout, _ = runner.collect()
                synchronize()
                collected = time.perf_counter()
                metrics = runner.algorithm.update(rollout)
                synchronize()
                ended = time.perf_counter()
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("non-finite update metrics")
                record = {
                    "iteration": iteration,
                    "collection_seconds": collected - started,
                    "update_seconds": ended - collected,
                    "total_seconds": ended - started,
                    "samples_per_second": args.envs * args.steps / (ended - started),
                    "memory": memory(),
                }
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)
        measured = records[args.warmup :]
        summary = {
            "samples_per_second": args.envs
            * args.steps
            * len(measured)
            / sum(row["total_seconds"] for row in measured),
            **{
                key: float(np.mean([row[key] for row in measured]))
                for key in ("collection_seconds", "update_seconds", "total_seconds")
            },
            "peak_rss_bytes": max(row["memory"]["VmHWM"] for row in records),
            "peak_swap_bytes": max(row["memory"]["VmSwap"] for row in records),
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
            ),
            "ppo_updates": runner.algorithm.updates,
            "encoder_updates": runner.algorithm.encoder_updates,
        }
        args.output.write_text(
            json.dumps(
                {
                    "envs": args.envs,
                    "threads": args.threads,
                    "seed": args.seed,
                    "warmup": args.warmup,
                    "metadata": metadata,
                    "records": records,
                    "summary": summary,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(summary), flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
