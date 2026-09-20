#!/usr/bin/env python3
"""Import the July-30 WE11 wheel velocity chirps for MuJoCo PACE fitting.

Each output Kd directory contains two independent sources.  Motor 3 and motor
6 are converted from raw encoder coordinates to the common WE11/MuJoCo joint
coordinates, but remain separate so a replay can release exactly one wheel and
equality-lock the other five joints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from import_real_sweep_csv_for_pd_fit import (
    make_out_dir,
    read_csv,
    resample_uniform,
    sha256_file,
    slice_after_last_timestamp_reset,
)
from mujoco_dr002_common import JOINT_NAMES, resolve_repo_path, save_chirp_data

WHEEL_FILE_RE = re.compile(r"motor36_(?P<stamp>\d{8}_\d{6})_kp0_kd(?P<kd>0p05|0p1|0p2)\.csv$")
EXPECTED_KD = (0.05, 0.1, 0.2)
WHEEL_MOTORS = (3, 6)
MOTOR_TO_JOINT = {3: 2, 6: 5}
# Deployment flips motor 3 and leaves motor 6 unflipped.  The raw commands are
# mirrored, so both become the same signed command in the policy/MuJoCo frame.
RAW_TO_MUJOCO_SIGN = {3: -1.0, 6: 1.0}
DEFAULT_ANGLES = np.asarray([0.8, -1.6, 0.0, 0.8, -1.6, 0.0], dtype=np.float64)
DEFAULT_RESAMPLE_DT_S = 0.005
EXPECTED_COMMAND_AMPLITUDE_RAD_S = 5.0
COMMAND_AMPLITUDE_TOLERANCE_RAD_S = 1.0e-3


def parse_kd(path: Path) -> float:
    match = WHEEL_FILE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unsupported WE11 wheel sweep filename: {path.name}")
    return float(match.group("kd").replace("p", "."))


def discover_sources(csv_dir: Path) -> dict[float, Path]:
    selected: dict[float, Path] = {}
    for path in sorted(csv_dir.glob("motor36_*.csv")):
        if WHEEL_FILE_RE.fullmatch(path.name) is None:
            continue
        kd = parse_kd(path)
        if kd in selected:
            raise ValueError(f"duplicate Kd={kd:g} wheel sweeps: {selected[kd]} and {path}")
        selected[kd] = path
    missing = [kd for kd in EXPECTED_KD if kd not in selected]
    if missing:
        raise FileNotFoundError(f"missing WE11 wheel sweeps for Kd={missing} under {csv_dir}")
    return selected


def write_trace_csv(
    path: Path,
    time: np.ndarray,
    command: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    torque_raw: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "target_velocity_mujoco_rad_s",
                "position_mujoco_relative_rad",
                "velocity_mujoco_rad_s",
                "torque_raw_nm",
            ]
        )
        writer.writerows(zip(time, command, position, velocity, torque_raw, strict=True))


def import_one_kd(
    source_csv: Path,
    out_dir: Path,
    *,
    kd: float,
    default_angles: np.ndarray,
    resample_dt: float,
) -> dict[str, Any]:
    raw = read_csv(source_csv)
    raw_time, segment, segment_meta = slice_after_last_timestamp_reset(raw)
    required = [
        f"motor_{motor_id}_{suffix}"
        for motor_id in WHEEL_MOTORS
        for suffix in ("target_vel_raw", "pos_raw", "vel_raw", "torque_raw")
    ]
    missing = [name for name in required if name not in segment]
    if missing:
        raise KeyError(f"{source_csv.name}: missing columns {missing}")

    traces: dict[str, np.ndarray] = {}
    command_names: set[str] = set()
    for motor_id in WHEEL_MOTORS:
        for suffix in ("target_vel_raw", "pos_raw", "vel_raw", "torque_raw"):
            name = f"motor_{motor_id}_{suffix}"
            values = np.asarray(segment[name], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{source_csv.name}: {name} contains non-finite values")
            traces[name] = values
        command_names.add(f"motor_{motor_id}_target_vel_raw")

    time, sampled = resample_uniform(
        raw_time,
        traces,
        resample_dt,
        zero_order_hold_traces=command_names,
    )
    transformed_commands: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for motor_id in WHEEL_MOTORS:
        joint_id = MOTOR_TO_JOINT[motor_id]
        sign = RAW_TO_MUJOCO_SIGN[motor_id]
        raw_command = sampled[f"motor_{motor_id}_target_vel_raw"]
        raw_position = sampled[f"motor_{motor_id}_pos_raw"]
        raw_velocity = sampled[f"motor_{motor_id}_vel_raw"]
        raw_torque = sampled[f"motor_{motor_id}_torque_raw"]

        command = sign * raw_command
        position = sign * (raw_position - raw_position[0])
        velocity = sign * raw_velocity
        command_min = float(np.min(command))
        command_max = float(np.max(command))
        if (
            abs(command_min + EXPECTED_COMMAND_AMPLITUDE_RAD_S) > COMMAND_AMPLITUDE_TOLERANCE_RAD_S
            or abs(command_max - EXPECTED_COMMAND_AMPLITUDE_RAD_S)
            > COMMAND_AMPLITUDE_TOLERANCE_RAD_S
        ):
            raise ValueError(
                f"{source_csv.name}: motor {motor_id} command range "
                f"[{command_min:.9g},{command_max:.9g}] is not ±5 rad/s"
            )

        dof_pos = np.tile(default_angles, (time.size, 1))
        des_dof_pos = dof_pos.copy()
        dof_vel = np.zeros_like(dof_pos)
        des_dof_vel = np.zeros_like(dof_pos)
        dof_pos[:, joint_id] = position
        dof_vel[:, joint_id] = velocity
        des_dof_vel[:, joint_id] = command
        label = "left" if motor_id == 3 else "right"
        stem = f"we11_{label}_wheel_kd{str(kd).replace('.', 'p')}_velocity_chirp"
        pt_path = save_chirp_data(
            out_dir / f"{stem}.pt",
            time,
            dof_pos,
            des_dof_pos,
            dof_vel=dof_vel,
            des_dof_vel=des_dof_vel,
        )
        trace_csv = out_dir / f"{stem}.csv"
        write_trace_csv(trace_csv, time, command, position, velocity, raw_torque)
        transformed_commands.append(command)
        sources.append(
            {
                "motor_id": motor_id,
                "joint_id": joint_id,
                "joint": JOINT_NAMES[joint_id],
                "active_joint_ids": [joint_id],
                "active_joints": [JOINT_NAMES[joint_id]],
                "target_type": "velocity",
                "kp": 0.0,
                "kd": float(kd),
                "duration_s": float(time[-1]),
                "command_min_rad_s": command_min,
                "command_max_rad_s": command_max,
                "raw_to_mujoco_sign": sign,
                "source_csv": str(source_csv.resolve()),
                "source_csv_sha256": sha256_file(source_csv),
                "pt": pt_path.name,
                "trace_csv": trace_csv.name,
                "trace_csv_sha256": sha256_file(trace_csv),
                "formal_segment": segment_meta,
            }
        )

    paired_command_error = float(np.max(np.abs(transformed_commands[0] - transformed_commands[1])))
    if paired_command_error > 1.0e-9:
        raise ValueError(
            f"{source_csv.name}: converted left/right commands differ by "
            f"{paired_command_error:.9g} rad/s"
        )
    return {
        "format": "we11_real_wheel_velocity_sweep_v1",
        "control_mode": "mixed",
        "controller": "MIT_D_only_velocity",
        "kp": 0.0,
        "kd": float(kd),
        "joint_names": list(JOINT_NAMES),
        "default_angles": default_angles.tolist(),
        "sample_rate_hz": float(1.0 / resample_dt),
        "resample_dt_s": float(resample_dt),
        "resampling": {
            "enabled": True,
            "target_dt_s": float(resample_dt),
            "target_sample_rate_hz": float(1.0 / resample_dt),
            "time_grid": "integer_tick_divided_by_sample_rate_hz_float64",
            "methods": {
                "command_velocity": "zero_order_hold_previous_sample",
                "measured_position": "linear_interpolation",
                "measured_velocity": "linear_interpolation",
                "measured_torque": "linear_interpolation",
            },
        },
        "chirp": {
            "type": "linear",
            "start_hz": 0.1,
            "end_hz": 5.0,
            "duration_s": 40.0,
            "command_amplitude_rad_s": EXPECTED_COMMAND_AMPLITUDE_RAD_S,
        },
        "coordinate_contract": {
            "joint_order": list(JOINT_NAMES),
            "raw_to_mujoco_velocity_sign": [None, None, -1.0, None, None, 1.0],
            "motor3_flipped": True,
            "motor6_flipped": False,
            "wheel_position_origin": "first_formal_sweep_sample",
        },
        "fixture_contract": {
            "fit_replay": "one_target_wheel_free_other_five_joints_equality_locked",
            "fixed_base": True,
        },
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": sha256_file(source_csv),
        "paired_command_max_abs_error_rad_s": paired_command_error,
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory produced by the Deploy motor-3/6 acquisition script.",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/we11_real_wheel_sweep_20260730_import",
    )
    parser.add_argument("--run-name", default="we11_wheel_kd005_01_02_velocity_pm5")
    parser.add_argument("--resample-dt", type=float, default=DEFAULT_RESAMPLE_DT_S)
    parser.add_argument("--default-angles", type=float, nargs=6, default=DEFAULT_ANGLES.tolist())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir}")
    if not math.isclose(args.resample_dt, DEFAULT_RESAMPLE_DT_S, abs_tol=1.0e-12):
        raise ValueError("WE11 wheel import requires 200 Hz output: --resample-dt 0.005")
    default_angles = np.asarray(args.default_angles, dtype=np.float64)
    if default_angles.shape != (6,) or not np.all(np.isfinite(default_angles)):
        raise ValueError("--default-angles must contain six finite values")

    selected = discover_sources(csv_dir)
    out_dir = make_out_dir(args.out_root, args.run_name)
    group_entries: list[dict[str, Any]] = []
    for kd in EXPECTED_KD:
        group_name = f"wheel_kd{str(kd).replace('.', 'p')}"
        group_dir = out_dir / group_name
        group_dir.mkdir()
        manifest = import_one_kd(
            selected[kd],
            group_dir,
            kd=kd,
            default_angles=default_angles,
            resample_dt=args.resample_dt,
        )
        manifest_path = group_dir / "chirp_source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        group_entries.append(
            {
                "kd": kd,
                "directory": group_name,
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        print(
            f"[DONE] Kd={kd:g}: {group_dir} (left/right imported separately)",
            flush=True,
        )

    combined_dir = out_dir / "combined_kd0p05_0p1_0p2"
    combined_dir.mkdir()
    combined_sources: list[dict[str, Any]] = []
    for entry in group_entries:
        group_dir = out_dir / str(entry["directory"])
        group_manifest = json.loads(
            (group_dir / "chirp_source_manifest.json").read_text(encoding="utf-8")
        )
        for source in group_manifest["sources"]:
            combined_sources.append(
                {
                    **source,
                    "pt": str(Path("..") / group_dir.name / str(source["pt"])),
                    "trace_csv": str(Path("..") / group_dir.name / str(source["trace_csv"])),
                }
            )
    combined_manifest = {
        "format": "we11_real_wheel_velocity_sweep_combined_v1",
        "control_mode": "mixed",
        "controller": "MIT_D_only_velocity_with_per_source_gain",
        "joint_names": list(JOINT_NAMES),
        "default_angles": default_angles.tolist(),
        "sample_rate_hz": float(1.0 / args.resample_dt),
        "resample_dt_s": float(args.resample_dt),
        "chirp": {
            "type": "linear",
            "start_hz": 0.1,
            "end_hz": 5.0,
            "duration_s": 40.0,
            "command_amplitude_rad_s": EXPECTED_COMMAND_AMPLITUDE_RAD_S,
        },
        "fixture_contract": {
            "fit_replay": "one_target_wheel_free_other_five_joints_equality_locked",
            "fixed_base": True,
        },
        "sources": combined_sources,
    }
    combined_manifest_path = combined_dir / "chirp_source_manifest.json"
    combined_manifest_path.write_text(
        json.dumps(combined_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    index = {
        "format": "we11_real_wheel_velocity_sweep_index_v1",
        "source_dir": str(csv_dir),
        "output_dir": str(out_dir),
        "groups": group_entries,
        "combined": {
            "directory": combined_dir.name,
            "manifest": combined_manifest_path.name,
            "manifest_sha256": sha256_file(combined_manifest_path),
            "semantics": "shared dynamics and delay; per-source Kd retained",
        },
    }
    index_path = out_dir / "wheel_sweep_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] index: {index_path}", flush=True)
    print(f"[DONE] combined fit manifest: {combined_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
