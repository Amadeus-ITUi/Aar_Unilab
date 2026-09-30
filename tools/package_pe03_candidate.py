"""Prepare a PE03 MNN candidate and its matching hardware policy profile.

Writes a staging directory only: does not install, select, or enable a policy.
The generated policy.yaml must accompany the weights; an old profile may use
different action limits even when its tensor dimensions match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.release / "deployment_manifest.json").read_text())
    runtime = json.loads((args.release / manifest["artifacts"]["pe03_runtime_path"]).read_text())
    config = yaml.safe_load((args.release / "runtime_config.yaml").read_text())
    if manifest["robot"]["id"] != "pe03" or config["observation"] != "pe03_v4":
        raise ValueError("requires a PE03 v4 release")
    if not runtime["schema"].endswith(".identified.v1") or runtime["position_difference"]:
        raise ValueError("requires the identified actuator/encoder-velocity contract")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.release / "policy.onnx", args.output / "policy.onnx")
    shutil.copy2(args.release / "runtime_config.yaml", args.output / "training_config.yaml")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        str(args.converter.resolve().parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    subprocess.run(
        [
            str(args.converter.resolve()),
            "-f",
            "ONNX",
            "--modelFile",
            str((args.output / "policy.onnx").resolve()),
            "--MNNModel",
            str((args.output / "policy.mnn").resolve()),
            "--bizCode",
            "pe03_identified",
        ],
        env=env,
        check=True,
    )
    n = runtime["normalization"]
    if [n[k] for k in ("lin_vel", "ang_vel", "dof_pos")] != [1, 1, 1]:
        raise ValueError("current hardware adapter assumes unit linear/angular/position scales")
    limits = np.array(runtime["joint_target_limits"])
    profile = dict(
        format_version=1,
        contract_id="gait_v4",
        contract_version=1,
        legacy_profile_id="point_foot",
        joint_order=manifest["robot"]["joint_order"],
        default_position_rad=runtime["default_joint_position"],
        kp=runtime["kp"],
        kd=runtime["kd"],
        training_position_min_rad=limits[:, 0].tolist(),
        training_position_max_rad=limits[:, 1].tolist(),
        observation=dict(
            frame_size=38,
            history_length=30,
            history_layout="frame_major_oldest_to_newest",
            velocity_scale=n["dof_vel"],
            clip=n["clip_observations"],
        ),
        action=dict(
            output_tensor="action",
            clip=config["control"]["action_clip"],
            scale=config["control"]["action_scale"],
            user_torque_limit_nm=runtime["user_torque_limit"],
            torque_envelope_mean_kp=float(np.mean(runtime["kp"])),
            torque_envelope_mean_kd=float(np.mean(runtime["kd"])),
        ),
        command=dict(
            input_tensor="command",
            scale=[
                n[k]
                for k in (
                    "lin_vel",
                    "lin_vel",
                    "ang_vel",
                    "frequency",
                    "support_fraction",
                    "clearance",
                )
            ],
            enable_lateral=True,
            gait_frequency_hz=runtime["gait"][0],
            support_fraction=runtime["gait"][1],
            clearance_m=runtime["gait"][2],
        ),
        runtime=dict(
            observation_tensor="observation",
            history_tensor="observation_history",
            reset_phase=runtime["reset_phase"],
            policy_dt=1 / config["control"]["policy_hz"],
        ),
    )
    (args.output / "policy.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    # No artificial delay is emitted into the hardware policy profile.
    # Delay and passive dynamics belong exclusively to the simulation model.
    session = ort.InferenceSession(
        str(args.output / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(3009)
    history = rng.normal(0, 0.35, (64, 1140)).astype(np.float32)
    history[0] = 0
    frame = history[:, -38:].copy()
    command = np.tile(np.load(args.release / "golden_inputs/command.npy"), (64, 1))
    command[1:, :3] = rng.uniform(-0.5, 0.5, (63, 3))
    expected = np.concatenate(
        [
            session.run(
                ["action"],
                dict(
                    observation_history=history[i : i + 1],
                    observation=frame[i : i + 1],
                    command=command[i : i + 1],
                ),
            )[0]
            for i in range(64)
        ]
    )
    np.savez_compressed(
        args.output / "parity_cases.npz",
        observation_history=history,
        observation=frame,
        command=command,
        expected=expected,
    )
    metadata = dict(
        status="candidate_not_activated",
        release=str(args.release.resolve()),
        converter_sha256=sha256(args.converter),
        control_model=runtime["actuator_model"],
        hardware_delay_injected=False,
        matching_policy_profile_required=True,
        note="Staging only. The supplied profile differs from legacy action limits; do not select weights under an unmatched profile.",
    )
    metadata["sha256"] = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "candidate_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
