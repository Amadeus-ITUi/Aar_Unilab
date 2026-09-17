#!/usr/bin/env python3
"""Measure the current WE11 training stack in isolated, sequential processes.

Physics threads vary; Torch/BLAS CPU threads stay at one. CUDA is synchronized
at rollout/update boundaries. Checkpoint/export I/O is excluded from throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def memory() -> dict[str, int]:
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM", "VmSwap"}:
            status[key] = int(value.split()[0]) * 1024
    status["ru_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return status


def summarize(directory: Path) -> dict:
    """Aggregate repetitions and extrapolate host RSS with a fixed + per-env model."""
    runs = [json.loads(path.read_text()) for path in sorted(directory.glob("e*_t*_r*.json"))]
    if not runs:
        raise ValueError("no completed matrix runs")
    groups = {}
    for environments in (2048, 4096):
        for threads in (16, 24, 32):
            selected = [
                run for run in runs if run["envs"] == environments and run["threads"] == threads
            ]
            if not selected:
                continue
            records = [record for run in selected for record in run["records"][run["warmup"] :]]
            groups[f"{environments}x{threads}"] = {
                "environments": environments,
                "threads": threads,
                "repeats": len(selected),
                "samples_per_second": environments
                * 24
                * len(records)
                / sum(record["total_seconds"] for record in records),
                "collection_seconds": statistics.mean(
                    record["collection_seconds"] for record in records
                ),
                "update_seconds": statistics.mean(record["update_seconds"] for record in records),
                "total_seconds": statistics.mean(record["total_seconds"] for record in records),
                "repeat_fps": [run["summary"]["samples_per_second"] for run in selected],
                "rss_gib": statistics.mean(run["summary"]["peak_rss_bytes"] for run in selected)
                / 1024**3,
                "cuda_reserved_gib": statistics.mean(
                    run["summary"]["cuda_peak_reserved_bytes"] for run in selected
                )
                / 1024**3,
                "swap_gib": max(run["summary"]["peak_swap_bytes"] for run in selected) / 1024**3,
            }
    projections = {}
    for threads in (16, 24, 32):
        if f"2048x{threads}" in groups and f"4096x{threads}" in groups:
            small, large = groups[f"2048x{threads}"], groups[f"4096x{threads}"]
            projections[threads] = {
                "rss_8192_gib": 3 * large["rss_gib"] - 2 * small["rss_gib"],
                "cuda_reserved_8192_gib": 3 * large["cuda_reserved_gib"]
                - 2 * small["cuda_reserved_gib"],
            }
    result = {
        "task": runs[0]["task"],
        "groups": groups,
        "projections_8192": projections,
        "best": max(groups, key=lambda key: groups[key]["samples_per_second"]),
        "projection_formula": "M(8192) = 3*M(4096) - 2*M(2048), fixed overhead + linear per-env memory",
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_matrix(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = [(2048, 16), (4096, 24), (2048, 32), (4096, 16), (2048, 24), (4096, 32)]
    process_env = dict(os.environ)
    process_env.pop("PYTHONPATH", None)
    process_env.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1"
    )
    for repeat in range(1, args.repeats + 1):
        for environments, threads in pairs if repeat % 2 else reversed(pairs):
            stem = f"e{environments}_t{threads}_r{repeat}"
            target = args.output / f"{stem}.json"
            if target.exists():
                raise FileExistsError(f"use an empty output directory: {target}")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--envs",
                str(environments),
                "--threads",
                str(threads),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--seed",
                str(args.seed),
                "--task",
                args.task,
                "--device",
                args.device,
                "--output",
                str(target),
            ]
            print(f"START {stem}", flush=True)
            with (args.output / f"{stem}.log").open("w") as stream:
                subprocess.run(
                    command,
                    env=process_env,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(f"FINISHED {stem}", flush=True)
    print(json.dumps(summarize(args.output), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task", default="dr002_joystick_flat_we11/mujoco")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.matrix:
        run_matrix(args)
        return
    if not args.envs or not args.threads:
        parser.error("--envs and --threads are required for a single run")
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        raise FileExistsError(f"use a new output path: {args.output}")
    os.environ["UNILAB_MUJOCO_NTHREADS"] = str(args.threads)

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner
    from scripts.train_rsl_rl import (
        _algo_config_dict,
        _resolve_ppo_wrapper_cls,
        apply_ppo_runtime_flags,
        build_ppo_env_cfg_override,
    )

    from unilab.training import apply_configured_training_seed, create_env, ensure_registries
    from unilab.training.rsl_rl import normalize_ppo_train_cfg
    from unilab.utils.nan_guard import NanGuard, NanGuardCfg

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.device(args.device).type != "cuda":
        raise ValueError("this benchmark fixes PPO on CUDA and varies MuJoCo CPU threads")
    torch.cuda.set_device(torch.device(args.device))
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/ppo")):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task={args.task}",
                f"algo.num_envs={args.envs}",
                f"algo.seed={args.seed}",
                "training.no_play=true",
            ],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output.with_suffix(".yaml"), resolve=True)
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stages = {"imports": memory()}
    started = time.perf_counter()
    env = create_env(cfg, num_envs=args.envs, env_cfg_override=build_ppo_env_cfg_override(cfg))
    try:
        stages["environment"] = memory()
        guard = cfg.training.nan_guard
        if guard.enabled:
            env.set_nan_guard(
                NanGuard(
                    NanGuardCfg(
                        enabled=True,
                        buffer_size=int(guard.buffer_size),
                        max_envs_to_dump=int(guard.max_envs_to_dump),
                        output_dir=str(args.output.parent / "nan_guard"),
                    ),
                    num_envs=env.num_envs,
                    supports_state_playback=env.play_capabilities.supports_physics_state_playback,
                )
            )
        raw = _algo_config_dict(cfg)
        wrapped = _resolve_ppo_wrapper_cls(raw)(env, device=args.device)
        train_cfg = normalize_ppo_train_cfg(raw)
        apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=True)
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=args.device)
        stages["runner"] = memory()
        backend = env._backend
        model = backend._model
        metadata = {
            "model_buffer_bytes": int(model.nbuffer),
            "model_variants": len(backend._model_variants),
            "pool": type(backend._pool).__name__,
            "physics_threads": int(backend._n_threads),
            "torch_cpu_threads": torch.get_num_threads(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(),
            "physics_hz": 1 / float(cfg.env.sim_dt),
            "policy_hz": 1 / float(cfg.env.ctrl_dt),
            "rollout_steps": int(cfg.algo.num_steps_per_env),
            "ppo_epochs": int(cfg.algo.algorithm.num_learning_epochs),
            "minibatches": int(cfg.algo.algorithm.num_mini_batches),
            "initialization_seconds": time.perf_counter() - started,
        }
        records = []
        timing = {}
        original_act = runner.alg.act
        original_returns = runner.alg.compute_returns

        def act(*values, **kwargs):
            if "start" not in timing:
                torch.cuda.synchronize()
                timing["start"] = time.perf_counter()
            return original_act(*values, **kwargs)

        def compute_returns(*values, **kwargs):
            torch.cuda.synchronize()
            timing["collected"] = time.perf_counter()
            return original_returns(*values, **kwargs)

        def log(**values):
            torch.cuda.synchronize()
            ended = time.perf_counter()
            record = {
                "iteration": values["it"],
                "collection_seconds": timing["collected"] - timing["start"],
                "update_seconds": ended - timing["collected"],
                "total_seconds": ended - timing["start"],
                "samples_per_second": args.envs
                * int(cfg.algo.num_steps_per_env)
                / (ended - timing["start"]),
                "memory": memory(),
                "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
            }
            records.append(record)
            timing.clear()
            with args.output.with_suffix(".jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"BENCH envs={args.envs} threads={args.threads} iteration={values['it']} "
                f"fps={record['samples_per_second']:.0f} rss_gib={record['memory']['VmRSS'] / 1024**3:.3f}",
                flush=True,
            )

        runner.alg.act = act
        runner.alg.compute_returns = compute_returns
        runner.logger.log = log
        runner.learn(args.warmup + args.iterations, init_at_random_ep_len=True)
        measured = records[args.warmup :]
        result = {
            "task": args.task,
            "envs": args.envs,
            "threads": args.threads,
            "seed": args.seed,
            "warmup": args.warmup,
            "measured_iterations": args.iterations,
            "metadata": metadata,
            "memory_stages": stages,
            "records": records,
            "summary": {
                "samples_per_second": args.envs
                * int(cfg.algo.num_steps_per_env)
                * len(measured)
                / sum(r["total_seconds"] for r in measured),
                "median_samples_per_second": statistics.median(
                    r["samples_per_second"] for r in measured
                ),
                "collection_seconds": statistics.mean(r["collection_seconds"] for r in measured),
                "update_seconds": statistics.mean(r["update_seconds"] for r in measured),
                "total_seconds": statistics.mean(r["total_seconds"] for r in measured),
                "peak_rss_bytes": max(r["memory"]["VmHWM"] for r in records),
                "final_rss_bytes": records[-1]["memory"]["VmRSS"],
                "peak_swap_bytes": max(r["memory"]["VmSwap"] for r in records),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
