#!/usr/bin/env python3
"""Build phase-aligned WE11 wrench replay assets from the 2026-07-28 sweep.

The raw logger records one approximately 200 Hz six-axis force snapshot per
row together with one motor's asynchronous feedback. The WE11 runtime expects a
canonical ``time,Fx,Fy,Fz,Mx,My,Mz`` file spanning a seamless 10-second period.
This converter phase-averages the same 17-cycle steady window as the previous
replay assets, then aligns the new motor feedback to the retained motor7 wing
observation so wrench and wing observation keep a common replay phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RAW_FILES = {
    1.0: "010_1hz_20cyc.csv",
    2.0: "020_2hz_20cyc.csv",
    3.0: "030_3hz_20cyc.csv",
}
RAW_WRENCH_COLUMNS = ("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm")
OUTPUT_COLUMNS = ("time", "Fx", "Fy", "Fz", "Mx", "My", "Mz")
STEADY_CYCLE_START = 2
STEADY_CYCLE_STOP = 19
OUTPUT_DURATION_S = 10.0
OUTPUT_SAMPLE_RATE_HZ = 1000


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_phase_reference_dir() -> Path:
    return (
        _repository_root() / "src/unilab/assets/robots/dr002/we11/training_data/wing_angle_20260713"
    )


def _default_output_dir() -> Path:
    return (
        _repository_root()
        / "src/unilab/assets/robots/dr002/we11/training_data/measured_wrench_20260728_skin"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frequency_filename(frequency_hz: float) -> str:
    return f"{frequency_hz:g}hz.csv"


def _fit_fundamental(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
) -> dict[str, float | int]:
    if time_s.ndim != 1 or values.shape != time_s.shape or time_s.size < 3:
        raise ValueError("fundamental fit requires matching one-dimensional samples")
    omega_t = 2.0 * np.pi * frequency_hz * time_s
    design = np.column_stack(
        (
            np.ones_like(time_s),
            np.cos(omega_t),
            np.sin(omega_t),
        )
    )
    dc, cosine, sine = np.linalg.lstsq(design, values, rcond=None)[0]
    amplitude = float(math.hypot(float(cosine), float(sine)))
    phase_rad = float(math.atan2(-float(sine), float(cosine)))
    return {
        "samples": int(time_s.size),
        "dc": float(dc),
        "amplitude": amplitude,
        "phase_deg": float(math.degrees(phase_rad)),
    }


def _load_raw_acquisition(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, dict[str, Any]]:
    required = {
        "force_frame_count",
        "force_t_abs_s",
        "force_age_s",
        "m1_rx_seq",
        "m1_rx_abs_s",
        "m1_pos_deg",
        *RAW_WRENCH_COLUMNS,
    }
    frame_counts: list[int] = []
    force_times_abs: list[float] = []
    force_ages: list[float] = []
    wrench_rows: list[list[float]] = []
    motor_sequences: list[int] = []
    motor_times_abs: list[float] = []
    motor_positions_deg: list[float] = []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV is missing required columns {missing}: {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_count = int(row["force_frame_count"])
                force_time_abs = float(row["force_t_abs_s"])
                force_age = float(row["force_age_s"])
                wrench = [float(row[column]) for column in RAW_WRENCH_COLUMNS]
                motor_sequence = int(row["m1_rx_seq"])
                motor_time_abs = float(row["m1_rx_abs_s"])
                motor_position_deg = float(row["m1_pos_deg"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value at {path}:{line_number}") from exc
            numeric_values = (
                force_time_abs,
                force_age,
                motor_time_abs,
                motor_position_deg,
                *wrench,
            )
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(f"Non-finite acquisition sample at {path}:{line_number}")
            frame_counts.append(frame_count)
            force_times_abs.append(force_time_abs)
            force_ages.append(force_age)
            wrench_rows.append(wrench)
            motor_sequences.append(motor_sequence)
            motor_times_abs.append(motor_time_abs)
            motor_positions_deg.append(motor_position_deg)

    if len(frame_counts) < 3:
        raise ValueError(f"CSV needs at least three acquisition rows: {path}")

    frame_array = np.asarray(frame_counts, dtype=np.int64)
    force_time_array = np.asarray(force_times_abs, dtype=np.float64)
    force_age_array = np.asarray(force_ages, dtype=np.float64)
    wrench_array = np.asarray(wrench_rows, dtype=np.float64)
    repeated_force = frame_array[1:] == frame_array[:-1]
    if np.any(force_time_array[1:][repeated_force] != force_time_array[:-1][repeated_force]):
        raise ValueError(f"Repeated force frame has inconsistent timestamp: {path}")
    if np.any(wrench_array[1:][repeated_force] != wrench_array[:-1][repeated_force]):
        raise ValueError(f"Repeated force frame has inconsistent wrench: {path}")
    keep_force = np.concatenate(([True], ~repeated_force))
    deduplicated_frames = frame_array[keep_force]
    if np.unique(deduplicated_frames).size != deduplicated_frames.size:
        raise ValueError(f"Force frame IDs repeat non-consecutively: {path}")
    if np.any(np.diff(deduplicated_frames) <= 0):
        raise ValueError(f"Force frame IDs are not strictly increasing: {path}")

    force_time_array = force_time_array[keep_force]
    force_age_array = force_age_array[keep_force]
    wrench_array = wrench_array[keep_force]
    if np.any(np.diff(force_time_array) <= 0.0):
        raise ValueError(f"Force timestamps are not strictly increasing: {path}")
    force_time_zero_abs = float(force_time_array[0])
    force_time_s = force_time_array - force_time_zero_abs

    motor_sequence_array = np.asarray(motor_sequences, dtype=np.int64)
    motor_time_array = np.asarray(motor_times_abs, dtype=np.float64)
    motor_position_array = np.asarray(motor_positions_deg, dtype=np.float64)
    repeated_motor = motor_sequence_array[1:] == motor_sequence_array[:-1]
    if np.any(motor_time_array[1:][repeated_motor] != motor_time_array[:-1][repeated_motor]):
        raise ValueError(f"Repeated motor feedback has inconsistent timestamp: {path}")
    if np.any(
        motor_position_array[1:][repeated_motor] != motor_position_array[:-1][repeated_motor]
    ):
        raise ValueError(f"Repeated motor feedback has inconsistent position: {path}")
    keep_motor = np.concatenate(([True], ~repeated_motor))
    motor_sequence_array = motor_sequence_array[keep_motor]
    motor_time_s = motor_time_array[keep_motor] - force_time_zero_abs
    motor_position_array = motor_position_array[keep_motor]
    if np.unique(motor_sequence_array).size != motor_sequence_array.size:
        raise ValueError(f"Motor feedback IDs repeat non-consecutively: {path}")
    if np.any(np.diff(motor_sequence_array) <= 0) or np.any(np.diff(motor_time_s) <= 0.0):
        raise ValueError(f"Motor feedback is not strictly increasing: {path}")

    metadata: dict[str, Any] = {
        "raw_rows": len(frame_counts),
        "retained_force_frames": int(force_time_s.size),
        "discarded_repeated_force_rows": int(len(frame_counts) - force_time_s.size),
        "retained_motor_feedback_rows": int(motor_time_s.size),
        "discarded_repeated_motor_rows": int(len(frame_counts) - motor_time_s.size),
        "drop_initial_stale_force_frame": False,
        "first_force_age_s": float(force_age_array[0]),
        "force_age_s": {
            "min": float(np.min(force_age_array)),
            "median": float(np.median(force_age_array)),
            "p99": float(np.quantile(force_age_array, 0.99)),
            "max": float(np.max(force_age_array)),
        },
        "first_force_t_abs_s": force_time_zero_abs,
        "retained_force_duration_s": float(force_time_s[-1]),
        "force_sample_rate_hz": float((force_time_s.size - 1) / force_time_s[-1]),
        "max_force_timestamp_gap_s": float(np.max(np.diff(force_time_s))),
    }
    return (
        force_time_s,
        wrench_array,
        force_time_zero_abs,
        motor_time_s,
        motor_position_array,
        metadata,
    )


def _load_old_motor7_phase(
    path: Path,
    frequency_hz: float,
) -> tuple[dict[str, float | int], dict[str, float]]:
    times: list[float] = []
    positions: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"time_s", "motor7_deg"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Phase reference must contain time_s,motor7_deg: {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                time_s = float(row["time_s"])
                position_deg = float(row["motor7_deg"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid phase reference at {path}:{line_number}") from exc
            if not math.isfinite(time_s) or not math.isfinite(position_deg):
                raise ValueError(f"Non-finite phase reference at {path}:{line_number}")
            times.append(time_s)
            positions.append(position_deg)
    time_array = np.asarray(times, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    if time_array.size < 3 or np.any(np.diff(time_array) <= 0.0):
        raise ValueError(f"Phase reference timestamps must be strictly increasing: {path}")
    one_cycle = (time_array >= 0.0) & (time_array < 1.0 / frequency_hz)
    fit = _fit_fundamental(time_array[one_cycle], position_array[one_cycle], frequency_hz)
    initial = {
        "position_deg": float(position_array[0]),
        "forward_slope_deg_s": float(
            (position_array[1] - position_array[0]) / (time_array[1] - time_array[0])
        ),
    }
    return fit, initial


def _phase_average(
    force_time_s: np.ndarray,
    wrench: np.ndarray,
    frequency_hz: float,
    source_time_advance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    frequency_integer = int(round(frequency_hz))
    if frequency_integer <= 0 or not math.isclose(
        float(frequency_integer), frequency_hz, abs_tol=1.0e-12
    ):
        raise ValueError("This U9 converter supports positive integer frequencies")

    sample_count = int(round(OUTPUT_DURATION_S * OUTPUT_SAMPLE_RATE_HZ))
    sample_indices = np.arange(sample_count + 1, dtype=np.int64)
    output_time = sample_indices.astype(np.float64) / OUTPUT_SAMPLE_RATE_HZ
    output_phase_cycles = ((sample_indices * frequency_integer) % OUTPUT_SAMPLE_RATE_HZ).astype(
        np.float64
    ) / OUTPUT_SAMPLE_RATE_HZ
    phase_advance_cycles = (source_time_advance_s * frequency_hz) % 1.0
    source_phase_cycles = (output_phase_cycles + phase_advance_cycles) % 1.0

    cycle_numbers = np.arange(STEADY_CYCLE_START, STEADY_CYCLE_STOP, dtype=np.float64)
    query_times = (cycle_numbers[:, None] + source_phase_cycles[None, :]) / frequency_hz
    query_min = float(np.min(query_times))
    query_max = float(np.max(query_times))
    if force_time_s[0] > query_min or force_time_s[-1] < query_max:
        raise ValueError(
            "Force recording does not cover the requested steady-state window: "
            f"available=[{force_time_s[0]:.6f}, {force_time_s[-1]:.6f}], "
            f"required=[{query_min:.6f}, {query_max:.6f}]"
        )

    output_wrench = np.empty((output_time.size, wrench.shape[1]), dtype=np.float64)
    flat_query = query_times.reshape(-1)
    for column in range(wrench.shape[1]):
        sampled_cycles = np.interp(
            flat_query,
            force_time_s,
            wrench[:, column],
        ).reshape(query_times.shape)
        output_wrench[:, column] = np.mean(sampled_cycles, axis=0)
    output_wrench[-1] = output_wrench[0]
    if not np.all(np.isfinite(output_wrench)):
        raise ValueError("Phase averaging produced a non-finite wrench")
    return output_time, output_wrench


def _write_wrench_csv(path: Path, time_s: np.ndarray, wrench: np.ndarray) -> None:
    output = np.column_stack((time_s, wrench))
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savetxt(
        temporary,
        output,
        delimiter=",",
        header=",".join(OUTPUT_COLUMNS),
        comments="",
        fmt=("%.6f", "%.9f", "%.9f", "%.9f", "%.9f", "%.9f", "%.9f"),
    )
    temporary.replace(path)


def _channel_stats(
    wrench: np.ndarray,
    frequency_hz: float,
    time_s: np.ndarray,
) -> dict[str, dict[str, float]]:
    values = wrench[:-1]
    time = time_s[:-1]
    stats: dict[str, dict[str, float]] = {}
    for index, column in enumerate(OUTPUT_COLUMNS[1:]):
        fit = _fit_fundamental(time, values[:, index], frequency_hz)
        stats[column] = {
            "min": float(np.min(values[:, index])),
            "max": float(np.max(values[:, index])),
            "mean": float(np.mean(values[:, index])),
            "rms": float(np.sqrt(np.mean(np.square(values[:, index])))),
            "peak_to_peak": float(np.ptp(values[:, index])),
            "fundamental_amplitude": float(fit["amplitude"]),
            "fundamental_phase_deg": float(fit["phase_deg"]),
        }
    return stats


def build_artifacts(
    source_dir: Path,
    phase_reference_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    phase_reference_dir = phase_reference_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "artifact_version": 1,
        "source_batch": source_dir.name,
        "schema": list(OUTPUT_COLUMNS),
        "units": {
            "time": "s",
            "Fx": "N",
            "Fy": "N",
            "Fz": "N",
            "Mx": "N*m",
            "My": "N*m",
            "Mz": "N*m",
        },
        "coordinate_frame": {
            "output": "force-sensor local channel convention consumed by WE11",
            "evidence": "logger channel names match the previous sensor-local acquisition",
            "caveat": "the latest raw CSV does not independently declare sensor mounting/frame",
        },
        "preprocessing": {
            "deduplicate_force_key": "force_frame_count",
            "drop_initial_stale_force_frame": False,
            "drop_reason": "latest first force frames are fresh; stale-frame drop was specific to the old batch",
            "time_source": "force_t_abs_s relative to first retained force frame",
            "steady_cycle_interval": f"[{STEADY_CYCLE_START}/f, {STEADY_CYCLE_STOP}/f)",
            "averaged_cycles": STEADY_CYCLE_STOP - STEADY_CYCLE_START,
            "remove_per_file_mean": False,
            "output_duration_s": OUTPUT_DURATION_S,
            "output_sample_rate_hz": OUTPUT_SAMPLE_RATE_HZ,
            "endpoint_policy": "10 s endpoint duplicates 0 s for an exact periodic seam",
        },
        "phase_alignment": {
            "target": "retained old motor7 wing-angle replay fundamental phase",
            "source": "latest m1_pos_deg timed by m1_rx_abs_s relative to force_t_abs_s",
            "convention": (
                "positive source_time_advance means aligned(t) = "
                "unaligned((t + advance) modulo one cycle)"
            ),
            "identity_caveat": (
                "latest m1 has no CAN ID; its positive envelope is a high-confidence, "
                "not independently proven, match to old motor7"
            ),
        },
        "files": {},
    }

    for frequency_hz, source_filename in RAW_FILES.items():
        source_path = source_dir / source_filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing raw sweep file: {source_path}")
        phase_reference_path = phase_reference_dir / _frequency_filename(frequency_hz)
        if not phase_reference_path.is_file():
            raise FileNotFoundError(f"Missing phase reference: {phase_reference_path}")

        (
            force_time_s,
            wrench,
            _force_time_zero_abs,
            motor_time_s,
            motor_position_deg,
            source_metadata,
        ) = _load_raw_acquisition(source_path)
        steady_motor = (motor_time_s >= STEADY_CYCLE_START / frequency_hz) & (
            motor_time_s < STEADY_CYCLE_STOP / frequency_hz
        )
        latest_motor_fit = _fit_fundamental(
            motor_time_s[steady_motor],
            motor_position_deg[steady_motor],
            frequency_hz,
        )
        old_motor_fit, old_motor_initial = _load_old_motor7_phase(
            phase_reference_path,
            frequency_hz,
        )
        phase_advance_deg = (
            float(old_motor_fit["phase_deg"]) - float(latest_motor_fit["phase_deg"])
        ) % 360.0
        source_time_advance_s = phase_advance_deg / (360.0 * frequency_hz)
        output_time, output_wrench = _phase_average(
            force_time_s,
            wrench,
            frequency_hz,
            source_time_advance_s,
        )
        output_path = output_dir / _frequency_filename(frequency_hz)
        _write_wrench_csv(output_path, output_time, output_wrench)
        seam_step = np.abs(output_wrench[0] - output_wrench[-2])
        max_internal_step = np.max(np.abs(np.diff(output_wrench[:-1], axis=0)), axis=0)
        manifest["files"][output_path.name] = {
            "frequency_hz": frequency_hz,
            "source": {
                "filename": source_filename,
                "sha256": _sha256(source_path),
                **source_metadata,
            },
            "phase_reference": {
                "filename": phase_reference_path.name,
                "sha256": _sha256(phase_reference_path),
                "old_motor7_fit": old_motor_fit,
                "old_motor7_initial": old_motor_initial,
                "latest_m1_fit": latest_motor_fit,
                "source_time_advance_s": source_time_advance_s,
                "phase_advance_deg": phase_advance_deg,
            },
            "output": {
                "filename": output_path.name,
                "sha256": _sha256(output_path),
                "rows": int(output_time.size),
                "max_seam_abs": float(np.max(np.abs(output_wrench[-1] - output_wrench[0]))),
                "seam_step_abs": dict(zip(OUTPUT_COLUMNS[1:], seam_step.tolist(), strict=True)),
                "max_internal_step_abs": dict(
                    zip(OUTPUT_COLUMNS[1:], max_internal_step.tolist(), strict=True)
                ),
                "channels": _channel_stats(output_wrench, frequency_hz, output_time),
            },
        }

    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Raw directory containing 010/020/030 *_20cyc.csv recordings",
    )
    parser.add_argument(
        "--phase-reference-dir",
        type=Path,
        default=_default_phase_reference_dir(),
        help="Existing 1/2/3 Hz wing-angle replay directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Versioned canonical wrench asset directory",
    )
    args = parser.parse_args()
    manifest = build_artifacts(
        args.source_dir,
        args.phase_reference_dir,
        args.output_dir,
    )
    print(
        f"Wrote {len(manifest['files'])} phase-aligned replay CSVs and manifest "
        f"to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
