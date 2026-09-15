#!/usr/bin/env python3
"""Build a self-contained WE11 Getup Sim2Sim release from the preserved baseline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort

from unilab.release_contract import validate_manifest, write_sha256sums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(root / "models/we11/getup/policy.onnx", destination / "policy.onnx")
    shutil.copytree(
        root / "src/unilab/assets/robots/dr002/we11",
        destination / "robot",
        ignore=shutil.ignore_patterns("*.npz", "*.json", "README.md", "urdf"),
    )
    # The training backend overrides the source XML's authoring-time 1 ms step
    # with cfg.sim_dt=2.5 ms. Materialize that resolved value in the release.
    model_path = destination / "robot/we11.xml"
    model_xml = model_path.read_text(encoding="utf-8")
    if 'timestep="0.001"' not in model_xml:
        raise RuntimeError("unexpected WE11 source timestep")
    model_path.write_text(model_xml.replace('timestep="0.001"', 'timestep="0.0025"', 1), encoding="utf-8")
    (destination / "runtime_config.yaml").write_text(
        "robot: we11\ntask: getup\nsimulator: mujoco\nphysics_hz: 400\n"
        "motor_hz: 200\npolicy_hz: 50\nobservation: we11_v2_145\n",
        encoding="utf-8",
    )
    golden_obs = np.zeros((1, 145), dtype=np.float32)
    session = ort.InferenceSession(
        str(destination / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    golden_action = session.run(["act"], {"obs": golden_obs})[0]
    (destination / "golden_inputs").mkdir()
    (destination / "golden_outputs").mkdir()
    np.save(destination / "golden_inputs/obs.npy", golden_obs, allow_pickle=False)
    np.save(destination / "golden_outputs/act.npy", golden_action, allow_pickle=False)
    manifest = validate_manifest(
        {
            "schema": "aar-unilab.actor.v1",
            "robot": {
                "id": "we11",
                "asset_version": "wheel-leg-training-baseline-20260915",
                "joint_order": [
                    "left_thigh_joint",
                    "left_calf_joint",
                    "left_foot_joint",
                    "right_thigh_joint",
                    "right_calf_joint",
                    "right_foot_joint",
                ],
            },
            "task": {"id": "getup"},
            "policy": {
                "observation_builder": "we11_v2_145",
                "inputs": [{"name": "obs", "dtype": "float32", "shape": [1, 145]}],
                "outputs": [{"name": "act", "dtype": "float32", "shape": [1, 6]}],
                "history": {
                    "length": 5,
                    "layout": "term-major",
                    "term_sizes": [3, 3, 4, 6, 6, 2, 2, 3],
                },
                "hidden_state": [],
            },
            "control": {
                "physics_hz": 400,
                "motor_hz": 200,
                "policy_hz": 50,
                "action_scale": 1.0,
                "action_clip": 100.0,
                "command_delay_steps": {"minimum": 2, "maximum": 8},
                "control_adapter": "we11_mixed_pd_v1",
            },
            "artifacts": {
                "policy_path": "policy.onnx",
                "runtime_config_path": "runtime_config.yaml",
                "scene_path": "robot/scene_flat_we11.xml",
            },
        }
    )
    manifest_path = destination / "deployment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    write_sha256sums(destination, files)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
