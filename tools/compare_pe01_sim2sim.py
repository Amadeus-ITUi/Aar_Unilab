#!/usr/bin/env python3
"""Compare Python and C++ PE01 ONNX closed-loop trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from unilab.envs.locomotion.pe01 import PE01Env


def python_trajectory(release: Path, steps: int) -> list[dict[str, float]]:
    manifest = json.loads((release / "deployment_manifest.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(release / "policy.onnx"), providers=["CPUExecutionProvider"])
    env = PE01Env(release / manifest["artifacts"]["scene_path"])
    observation = env.reset()
    command = np.load(release / "golden_inputs/command.npy", allow_pickle=False)
    result: list[dict[str, float]] = []
    for _ in range(steps):
        history = observation.actor.reshape(1, -1)
        frame = history[:, -30:]
        action = session.run(
            ["action"],
            {
                "observation_history": history,
                "observation": frame,
                "command": command,
            },
        )[0][0]
        observation, _, _, info = env.step(action)
        row = {"base_height": info["base_height"]}
        row.update({f"action_{index}": float(value) for index, value in enumerate(action)})
        result.append(row)
    return result


def cpp_trajectory(binary: Path, release: Path, steps: int, output: Path) -> list[dict[str, float]]:
    subprocess.run(
        [str(binary), str(release), "--steps", str(steps), "--telemetry", str(output)],
        check=True,
    )
    with (output / "sim2sim.csv").open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items() if key != "time_seconds"}
            for row in csv.DictReader(stream)
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    binary = args.binary.resolve()
    release = args.release.resolve()
    cache = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/ssd/conda/cache/aar_unilab-native"))
    cache.mkdir(parents=True, exist_ok=True)
    for steps in args.steps:
        expected = python_trajectory(release, steps)
        with tempfile.TemporaryDirectory(prefix=f"pe01-cpp-{steps}-", dir=cache) as directory:
            actual = cpp_trajectory(binary, release, steps, Path(directory))
        if len(expected) != len(actual):
            raise RuntimeError(f"trajectory length differs at {steps} steps")
        maximum_error = 0.0
        for expected_row, actual_row in zip(expected, actual, strict=True):
            if expected_row.keys() != actual_row.keys():
                raise RuntimeError("telemetry columns differ")
            maximum_error = max(
                maximum_error,
                *(abs(expected_row[key] - actual_row[key]) for key in expected_row),
            )
        if maximum_error > args.atol:
            raise RuntimeError(
                f"PE01 Python/C++ {steps}-step trajectory mismatch: "
                f"max_error={maximum_error:.9g} > {args.atol:.9g}"
            )
        print(f"PE01 Python/C++ {steps}-step trajectory: max_error={maximum_error:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
