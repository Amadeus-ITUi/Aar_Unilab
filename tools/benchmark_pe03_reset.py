#!/usr/bin/env python3
"""Compare exact reset geometry with cached identical corrections (same trajectories)."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np

    from unilab.envs.locomotion.pe03.config import load_config
    from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv

    results = {}
    for episode in (20.0, 0.1):
        config = load_config(
            [
                "+experiment=gait_fixed",
                f"algo.num_envs={args.envs}",
                "training.mujoco_threads=4",
                f"env.episode_length_s={episode}",
            ]
        )
        env = PE03GaitEnv(config)
        try:
            b = env.backend
            geometry = b.align_reset_soles
            actions = np.zeros((args.envs, 6))
            for _ in range(20):
                env.step(actions)
            initial = env.snapshot()
            records = []

            def record(qpos, clearance):
                correction = geometry(qpos, clearance)
                records.append(correction.copy())
                return correction

            b.align_reset_soles = record
            for _ in range(args.steps):
                env.step(actions)
            expected = env.snapshot()
            durations = {"exact": [], "cached": []}
            for repeat in range(args.repeats):
                for mode in ("exact", "cached") if repeat % 2 == 0 else ("cached", "exact"):
                    env.restore(initial)
                    corrections = iter(records)

                    def cached(qpos, clearance):
                        correction = next(corrections)
                        qpos[:, 2] += correction
                        return correction

                    b.align_reset_soles = geometry if mode == "exact" else cached
                    started = perf_counter()
                    for _ in range(args.steps):
                        env.step(actions)
                    durations[mode].append(perf_counter() - started)
                    np.testing.assert_array_equal(env.backend.state, expected["backend"]["state"])
                    np.testing.assert_array_equal(env.history, expected["arrays"]["history"])
            exact, cached_time = (float(np.median(durations[key])) for key in ("exact", "cached"))
            results[str(episode)] = dict(
                envs=args.envs,
                steps=args.steps,
                durations_s=durations,
                reset_batches=len(records),
                reset_environments=sum(len(r) for r in records),
                exact_samples_per_second=args.envs * args.steps / exact,
                cached_samples_per_second=args.envs * args.steps / cached_time,
                throughput_loss_fraction=1 - cached_time / exact,
                identical_trajectory=True,
            )
            if episode == 0.1:
                micro = {}
                for count in (1, 64, 256, 4096):
                    pose = np.tile(env.home, (count, 1))
                    times = []
                    for _ in range(50):
                        start = perf_counter()
                        b.reset_geometry.heights(pose)
                        times.append((perf_counter() - start) * 1000)
                    micro[count] = dict(
                        median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))
                    )
                results["geometry_only"] = micro
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
