#!/usr/bin/env python3
"""Import current ESD-Link WE11 paired sweeps for MuJoCo PACE fitting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from import_real_sweep_csv_for_pd_fit import (
    make_out_dir,
    resample_uniform,
    resampling_metadata,
    sha256_file,
    write_truth_csv,
)
from mujoco_dr002_common import DEFAULT_ANGLES, JOINT_NAMES, resolve_repo_path, save_chirp_data

RESAMPLE_DT_S = 0.005
CHIRP_DURATION_S = 40.0
CHIRP_START_HZ = 0.1
CHIRP_END_HZ = 5.0
POST_HOLD_MIN_S = 0.15
POSITION_CENTER_EPS_RAD = 1.0e-5
WHEEL_WRAP_PERIOD_RAD = 12.56

FILE_RE = re.compile(
    r"motor(?P<pair>14|25|36)_(?P<stamp>\d{8}_\d{6})_"
    r"kp(?P<kp>\d+(?:p\d+)?)_kd(?P<kd>\d+(?:p\d+)?)\.csv$"
)


@dataclass(frozen=True)
class SweepSpec:
    pair: str
    motor_ids: tuple[int, int]
    joint_ids: tuple[int, int]
    joint_group: str
    target_type: str
    center: float
    amplitude: float
    fixture_hold_kp: tuple[float, ...]
    fixture_hold_kd: tuple[float, ...]


SPECS = {
    "14": SweepSpec(
        "14", (1, 4), (0, 3), "thigh", "position", -0.5202, 0.20,
        (0.0, 20.0, 0.0, 0.0, 20.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
    ),
    "25": SweepSpec(
        "25", (2, 5), (1, 4), "calf", "position", 0.0, 0.15,
        (40.0, 0.0, 0.0, 40.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
    ),
    "36": SweepSpec(
        "36", (3, 6), (2, 5), "wheel", "velocity", 0.0, 5.0,
        (1.0, 1.0, 0.0, 1.0, 1.0, 0.0),
        (0.1, 0.1, 0.0, 0.1, 0.1, 0.0),
    ),
}

LEG_GROUPS = (
    (
        "thigh_kp2_kd0p1__calf_kp8_kd0p8",
        {"14": (2.0, 0.1), "25": (8.0, 0.8)},
    ),
    (
        "thigh_kp4_kd0p2__calf_kp4_kd0p2",
        {"14": (4.0, 0.2), "25": (4.0, 0.2)},
    ),
)
WHEEL_KD = (0.05, 0.1, 0.2)


def parse_filename(path: Path) -> tuple[SweepSpec, float, float]:
    match = FILE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unsupported ESD-Link sweep filename: {path.name}")
    kp = float(match.group("kp").replace("p", "."))
    kd = float(match.group("kd").replace("p", "."))
    return SPECS[match.group("pair")], kp, kd


def read_numeric_columns(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    values = {name: [] for name in names}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV header: {path}")
        missing = sorted(names.difference(reader.fieldnames))
        if missing:
            raise KeyError(f"{path.name}: missing columns {missing}")
        for row_number, row in enumerate(reader, start=2):
            for name in names:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path.name}:{row_number}: {name} is not numeric"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path.name}:{row_number}: {name} is not finite")
                values[name].append(value)
    if len(next(iter(values.values()), [])) < 100:
        raise ValueError(f"{path.name}: fewer than 100 numeric samples")
    return {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}


def required_columns(spec: SweepSpec) -> set[str]:
    names = {
        "host_monotonic_ns",
        "device_sample_time_us",
        "state_sample_seq",
        "source_state_sample_seq",
        "last_applied_command_seq",
        "command_status_flags",
    }
    for motor_id in range(1, 7):
        names.update(
            {
                f"p{motor_id}_cmd_q",
                f"p{motor_id}_cmd_dq",
                f"p{motor_id}_kp",
                f"p{motor_id}_kd",
                f"p{motor_id}_q",
                f"p{motor_id}_dq",
                f"p{motor_id}_tau",
            }
        )
    return names


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    return [(int(start), int(stop - 1)) for start, stop in edges]


def select_formal_segment(
    data: dict[str, np.ndarray], spec: SweepSpec
) -> tuple[np.ndarray, slice, dict[str, Any]]:
    clock = np.asarray(data["host_monotonic_ns"], dtype=np.float64) * 1.0e-9
    if not np.all(np.diff(clock) > 0.0):
        raise ValueError("host_monotonic_ns must be strictly increasing")
    command_suffix = "cmd_dq" if spec.target_type == "velocity" else "cmd_q"
    command = data[f"p{spec.motor_ids[0]}_{command_suffix}"]
    at_center = np.abs(command - spec.center) <= POSITION_CENTER_EPS_RAD
    stable_runs = [
        (start, stop)
        for start, stop in true_runs(at_center)
        if clock[stop] - clock[start] >= POST_HOLD_MIN_S
    ]
    if not stable_runs:
        raise ValueError("cannot find the stable post-sweep center hold")
    post_start, post_stop = stable_runs[-1]
    formal_end = post_start - 1
    if formal_end <= 0:
        raise ValueError("post-sweep hold leaves no formal chirp samples")
    formal_start = int(np.searchsorted(clock, clock[post_start] - CHIRP_DURATION_S))
    if formal_start >= formal_end:
        raise ValueError("formal chirp segment is empty")
    selected = slice(formal_start, formal_end + 1)
    time = clock[selected] - clock[formal_start]
    duration = float(time[-1])
    if not 39.8 <= duration <= 40.05:
        raise ValueError(f"formal chirp duration is {duration:.6f}s, expected about 40s")
    return time, selected, {
        "selection": "40_seconds_immediately_before_final_stable_command_center_hold",
        "selected_start_row_zero_based": formal_start,
        "selected_end_row_zero_based_inclusive": formal_end,
        "post_hold_start_row_zero_based": post_start,
        "post_hold_end_row_zero_based_inclusive": post_stop,
        "selected_raw_samples": int(formal_end - formal_start + 1),
        "selected_duration_s": duration,
        "selected_dt_median_s": float(np.median(np.diff(time))),
    }


def sequence_gap_count(values: np.ndarray) -> int:
    delta = np.diff(np.asarray(values, dtype=np.int64))
    return int(np.sum(np.clip(delta - 1, 0, None)))


def quality_report(
    path: Path,
    data: dict[str, np.ndarray],
    selected: slice,
    time: np.ndarray,
) -> dict[str, Any]:
    dt = np.diff(time)
    device_dt = np.diff(data["device_sample_time_us"][selected]) * 1.0e-6
    return {
        "source_csv": str(path.resolve()),
        "source_csv_sha256": sha256_file(path),
        "raw_samples": int(next(iter(data.values())).size),
        "formal_samples": int(time.size),
        "formal_duration_s": float(time[-1]),
        "effective_sample_rate_hz": float((time.size - 1) / time[-1]),
        "host_dt_median_s": float(np.median(dt)),
        "host_dt_p99_s": float(np.quantile(dt, 0.99)),
        "host_dt_max_s": float(np.max(dt)),
        "device_dt_median_s": float(np.median(device_dt)),
        "device_dt_max_s": float(np.max(device_dt)),
        "state_sample_sequence_missing_count": sequence_gap_count(
            data["state_sample_seq"][selected]
        ),
        "source_state_sequence_missing_count": sequence_gap_count(
            data["source_state_sample_seq"][selected]
        ),
        "command_status_flags": sorted(
            {int(value) for value in data["command_status_flags"][selected]}
        ),
        "delay_confidence": "reduced_due_to_irregular_observation_sampling",
    }


def unwrap_wheel(position: np.ndarray) -> np.ndarray:
    radians = np.asarray(position, dtype=np.float64) * (2.0 * np.pi / WHEEL_WRAP_PERIOD_RAD)
    unwrapped = np.unwrap(radians) * (WHEEL_WRAP_PERIOD_RAD / (2.0 * np.pi))
    return unwrapped - unwrapped[0]


def validate_linear_chirp(
    path: Path,
    time: np.ndarray,
    command: np.ndarray,
    spec: SweepSpec,
) -> dict[str, float]:
    phase = 2.0 * np.pi * (
        CHIRP_START_HZ * time
        + 0.5
        * (CHIRP_END_HZ - CHIRP_START_HZ)
        / CHIRP_DURATION_S
        * np.square(time)
    )
    expected = spec.center + spec.amplitude * np.sin(phase)
    correlation = float(np.corrcoef(command, expected)[0, 1])
    rms_error = float(np.sqrt(np.mean(np.square(command - expected))))
    normalized_rms_error = rms_error / spec.amplitude
    if not math.isfinite(correlation) or correlation < 0.99:
        raise ValueError(
            f"{path.name}: command does not match a 0.1->5 Hz linear chirp "
            f"(correlation={correlation:.6f})"
        )
    if normalized_rms_error > 0.10:
        raise ValueError(
            f"{path.name}: chirp normalized RMS error is {normalized_rms_error:.6f}"
        )
    return {
        "expected_start_hz": CHIRP_START_HZ,
        "expected_end_hz": CHIRP_END_HZ,
        "expected_duration_s": CHIRP_DURATION_S,
        "waveform_correlation": correlation,
        "waveform_rms_error": rms_error,
        "waveform_normalized_rms_error": normalized_rms_error,
    }


def write_trace(
    path: Path,
    time: np.ndarray,
    active_joint_ids: tuple[int, int],
    desired: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    torque: np.ndarray,
    target_type: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        target_label = "target_velocity" if target_type == "velocity" else "target_position"
        header = ["time_s"]
        for joint_id in active_joint_ids:
            joint = JOINT_NAMES[joint_id]
            header += [
                f"{target_label}_{joint}",
                f"position_{joint}",
                f"velocity_{joint}",
                f"torque_{joint}",
            ]
        writer.writerow(header)
        for row in range(time.size):
            values: list[float] = [float(time[row])]
            for column in range(len(active_joint_ids)):
                values += [
                    float(desired[row, column]),
                    float(position[row, column]),
                    float(velocity[row, column]),
                    float(torque[row, column]),
                ]
            writer.writerow(values)


def import_source(path: Path, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, kp, kd = parse_filename(path)
    data = read_numeric_columns(path, required_columns(spec))
    raw_time, selected, segment_meta = select_formal_segment(data, spec)
    command_suffix = "cmd_dq" if spec.target_type == "velocity" else "cmd_q"
    trace_names: set[str] = set()
    command_names: set[str] = set()
    for motor_id in range(1, 7):
        for suffix in ("cmd_q", "cmd_dq", "q", "dq", "tau"):
            name = f"p{motor_id}_{suffix}"
            trace_names.add(name)
            if suffix in {"cmd_q", "cmd_dq"}:
                command_names.add(name)
    raw_traces = {name: data[name][selected] for name in trace_names}
    time, sampled = resample_uniform(
        raw_time,
        raw_traces,
        RESAMPLE_DT_S,
        zero_order_hold_traces=command_names,
    )

    commands = np.column_stack(
        [sampled[f"p{motor_id}_{command_suffix}"] for motor_id in spec.motor_ids]
    )
    paired_error = float(np.max(np.abs(commands[:, 0] - commands[:, 1])))
    if paired_error > 1.0e-6:
        raise ValueError(f"{path.name}: paired commands differ by {paired_error:.9g}")
    command_min = float(np.min(commands))
    command_max = float(np.max(commands))
    measured_center = 0.5 * (command_min + command_max)
    measured_amplitude = 0.5 * (command_max - command_min)
    if abs(measured_center - spec.center) > 1.0e-3:
        raise ValueError(f"{path.name}: command center {measured_center:g} is invalid")
    if abs(measured_amplitude - spec.amplitude) > 1.0e-3:
        raise ValueError(f"{path.name}: command amplitude {measured_amplitude:g} is invalid")
    chirp_validation = validate_linear_chirp(path, time, commands[:, 0], spec)
    for motor_id in spec.motor_ids:
        if not np.allclose(data[f"p{motor_id}_kp"][selected], kp, atol=1.0e-6):
            raise ValueError(f"{path.name}: P{motor_id} Kp does not match filename")
        if not np.allclose(data[f"p{motor_id}_kd"][selected], kd, atol=1.0e-6):
            raise ValueError(f"{path.name}: P{motor_id} Kd does not match filename")

    dof_pos = np.tile(DEFAULT_ANGLES, (time.size, 1))
    des_dof_pos = np.tile(DEFAULT_ANGLES, (time.size, 1))
    dof_vel = np.zeros_like(dof_pos)
    des_dof_vel = np.zeros_like(dof_pos)
    active_position = np.empty((time.size, 2), dtype=np.float64)
    active_velocity = np.empty_like(active_position)
    active_torque = np.empty_like(active_position)
    for column, (motor_id, joint_id) in enumerate(
        zip(spec.motor_ids, spec.joint_ids, strict=True)
    ):
        position = sampled[f"p{motor_id}_q"]
        if spec.target_type == "velocity":
            position = unwrap_wheel(position)
            des_dof_vel[:, joint_id] = sampled[f"p{motor_id}_cmd_dq"]
        else:
            position = DEFAULT_ANGLES[joint_id] + position
            des_dof_pos[:, joint_id] = DEFAULT_ANGLES[joint_id] + sampled[f"p{motor_id}_cmd_q"]
        dof_pos[:, joint_id] = position
        dof_vel[:, joint_id] = sampled[f"p{motor_id}_dq"]
        active_position[:, column] = position
        active_velocity[:, column] = sampled[f"p{motor_id}_dq"]
        active_torque[:, column] = sampled[f"p{motor_id}_tau"]

    stem = f"esd_link_{spec.joint_group}_paired_chirp_data"
    pt_path = save_chirp_data(
        out_dir / f"{stem}.pt",
        time,
        dof_pos,
        des_dof_pos,
        dof_vel=dof_vel,
        des_dof_vel=des_dof_vel,
    )
    trace_path = out_dir / f"{stem}.csv"
    active_desired = (
        des_dof_vel[:, spec.joint_ids]
        if spec.target_type == "velocity"
        else des_dof_pos[:, spec.joint_ids]
    )
    write_trace(
        trace_path,
        time,
        spec.joint_ids,
        active_desired,
        active_position,
        active_velocity,
        active_torque,
        spec.target_type,
    )
    quality = quality_report(path, data, selected, raw_time)
    source = {
        "motor_ids": list(spec.motor_ids),
        "active_joint_ids": list(spec.joint_ids),
        "active_joints": [JOINT_NAMES[index] for index in spec.joint_ids],
        "joint_group": spec.joint_group,
        "joint_id": spec.joint_ids[0],
        "joint": JOINT_NAMES[spec.joint_ids[0]],
        "target_type": spec.target_type,
        "kp": kp,
        "kd": kd,
        "fixture_hold_kp": list(spec.fixture_hold_kp),
        "fixture_hold_kd": list(spec.fixture_hold_kd),
        "source_csv": str(path.resolve()),
        "source_csv_sha256": sha256_file(path),
        "pt": pt_path.name,
        "pt_sha256": sha256_file(pt_path),
        "trace_csv": trace_path.name,
        "trace_csv_sha256": sha256_file(trace_path),
        "num_samples": int(time.size),
        "duration_s": float(time[-1]),
        "sample_rate_hz": 200.0,
        "formal_segment": segment_meta,
        "amplitude_validation": {
            "expected_center": spec.center,
            "measured_center": measured_center,
            "expected_amplitude": spec.amplitude,
            "measured_amplitude": measured_amplitude,
            "paired_command_max_abs_difference": paired_error,
        },
        "chirp_validation": chirp_validation,
        "quality": quality,
    }
    return source, quality


def manifest_base(csv_dir: Path, control_mode: str) -> dict[str, Any]:
    return {
        "control_mode": control_mode,
        "source_dir": str(csv_dir.resolve()),
        "joint_names": list(JOINT_NAMES),
        "default_angles": DEFAULT_ANGLES.tolist(),
        "sample_rate_hz": 200.0,
        "resample_dt_s": RESAMPLE_DT_S,
        "resampling": resampling_metadata(RESAMPLE_DT_S),
        "chirp": {
            "type": "linear",
            "start_hz": CHIRP_START_HZ,
            "end_hz": CHIRP_END_HZ,
            "duration_s": CHIRP_DURATION_S,
        },
        "coordinate_contract": {
            "input": "ESD-Link bridge policy coordinates",
            "position": "q_mujoco = mujoco_default_angle + pN_q",
            "velocity": "qd_mujoco = pN_dq",
            "effort": "tau_mujoco = pN_tau",
            "wheel_position": "unwrap period 12.56 rad, then relative to first formal sample",
        },
    }


def discover(csv_dir: Path) -> dict[tuple[str, float, float], Path]:
    found: dict[tuple[str, float, float], Path] = {}
    for path in sorted(csv_dir.glob("motor*.csv")):
        if FILE_RE.fullmatch(path.name) is None:
            continue
        spec, kp, kd = parse_filename(path)
        key = spec.pair, kp, kd
        if key in found:
            raise ValueError(f"duplicate ESD-Link sweep {key}: {found[key]} and {path}")
        found[key] = path
    required = {
        ("14", 2.0, 0.1), ("14", 4.0, 0.2),
        ("25", 8.0, 0.8), ("25", 4.0, 0.2),
        ("36", 0.0, 0.05), ("36", 0.0, 0.1), ("36", 0.0, 0.2),
    }
    missing = sorted(required.difference(found))
    if missing:
        raise FileNotFoundError(f"missing ESD-Link sweeps: {missing}")
    return found


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--out-root", default="outputs/we11_esd_link_sweep_import")
    parser.add_argument("--run-name", default="we11_esd_link_paired_sweeps")
    args = parser.parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(csv_dir)
    selected = discover(csv_dir)
    out_dir = make_out_dir(resolve_repo_path(args.out_root), args.run_name)
    index_groups: list[dict[str, Any]] = []
    all_quality: list[dict[str, Any]] = []

    for group_name, selectors in LEG_GROUPS:
        group_dir = out_dir / group_name
        group_dir.mkdir()
        sources = []
        for pair in ("14", "25"):
            kp, kd = selectors[pair]
            source, quality = import_source(selected[(pair, kp, kd)], group_dir)
            sources.append(source)
            all_quality.append(quality)
        manifest = {
            "format": "dr002_real_sweep_import_v2_paired",
            "case": f"esd_link_{group_name}",
            "paired_group": group_name,
            **manifest_base(csv_dir, "position"),
            "sources": sources,
        }
        manifest_path = group_dir / "chirp_source_manifest.json"
        write_json(manifest_path, manifest)
        index_groups.append(
            {
                "name": group_name,
                "directory": group_dir.name,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )

    for kd in WHEEL_KD:
        group_name = f"wheel_kd{str(kd).replace('.', 'p')}"
        group_dir = out_dir / group_name
        group_dir.mkdir()
        source, quality = import_source(selected[("36", 0.0, kd)], group_dir)
        all_quality.append(quality)
        manifest = {
            "format": "we11_esd_link_wheel_velocity_sweep_v1",
            **manifest_base(csv_dir, "mixed"),
            "controller": "MIT_D_only_velocity",
            "sources": [source],
        }
        manifest_path = group_dir / "chirp_source_manifest.json"
        write_json(manifest_path, manifest)
        index_groups.append(
            {
                "name": group_name,
                "directory": group_dir.name,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )

    quality_path = out_dir / "data_quality_report.json"
    write_json(
        quality_path,
        {
            "format": "we11_esd_link_sweep_quality_v1",
            "sources": all_quality,
            "warning": "irregular observations were causally resampled to 200 Hz",
        },
    )
    index_path = out_dir / "sweep_import_index.json"
    write_json(
        index_path,
        {
            "format": "we11_esd_link_sweep_import_index_v1",
            "source_dir": str(csv_dir),
            "output_dir": str(out_dir),
            "groups": index_groups,
            "quality_report": quality_path.name,
            "quality_report_sha256": sha256_file(quality_path),
        },
    )
    print(f"[DONE] imported ESD-Link sweeps: {out_dir}", flush=True)
    print(f"[DONE] index: {index_path}", flush=True)


if __name__ == "__main__":
    main()
