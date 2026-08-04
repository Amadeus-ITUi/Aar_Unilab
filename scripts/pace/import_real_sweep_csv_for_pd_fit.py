#!/usr/bin/env python3
"""Import DR002 deploy sweep CSV logs into the six-joint PD-fit layout.

The single-motor mode keeps the retained 200 Hz sweep contract:
- position control targets around the physical sweep pose
  [0.8, -1.6, 0, 0.8, -1.6, 0]
- 200 Hz resampling (dt = 0.005 s), matching the deploy motor loop
- command targets resampled with causal zero-order hold; measured position/velocity linearly interpolated
- first 0.5 s used to remove measured-position DC offset

The July-30 paired mode is intentionally stricter:
- a source contains both physical sides of a thigh or calf sweep
- only the formal segment after the final timestamp rollback is imported
- 200 Hz resampling is mandatory
- the measured response is never DC-aligned
- policy-to-MuJoCo absolute-qpos calibration must be supplied explicitly
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from mujoco_dr002_common import DEFAULT_ANGLES, JOINT_NAMES, resolve_repo_path, save_chirp_data

MOTOR_TO_JOINT = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}
CSV_RE = re.compile(r"sweep_motor_(?P<motor>[1-6])_(?P<joint>.+?)_(?P<stamp>\d{8}_\d{6}_\d+)\.csv$")
PAIRED_CSV_RE = re.compile(
    r"motor(?P<pair>14|25)_(?P<stamp>\d{8}_\d{6})_"
    r"kp(?P<kp>\d+(?:p\d+)?)_kd(?P<kd>\d+(?:p\d+)?)\.csv$"
)
CONTROL_MODE = "position"
DEFAULT_ALIGNMENT_WINDOW_S = 0.5
DEFAULT_RESAMPLE_DT_S = 0.005
DEFAULT_SAMPLE_RATE_HZ = 1.0 / DEFAULT_RESAMPLE_DT_S
DEFAULT_SWEEP_ZERO_POSE = DEFAULT_ANGLES.astype(np.float64).copy()
NON_ACTIVE_COMMAND_EPS = 1e-4
RESAMPLE_METHODS = {
    "command_position": "zero_order_hold_previous_sample",
    "measured_position": "linear_interpolation",
    "measured_velocity": "linear_interpolation",
}
SERIALIZATION_DTYPES = {
    "time": "float64",
    "signals": "float32",
}
JULY30_RAW_ENCODER_CENTER = np.asarray(
    [0.52020, 0.41338, 0.0, -0.52020, -0.41338, 0.0],
    dtype=np.float64,
)
JULY30_POLICY_SWEEP_CENTER = np.asarray(
    [-0.52020, 0.41338, 0.0, -0.52020, 0.41338, 0.0],
    dtype=np.float64,
)
# Only leg entries are part of this import contract. Wheel signs are kept at
# +1 solely to make the six-vector serializable and are marked unverified in
# the manifest; paired mode never reads motor 3/6.
JULY30_RAW_TO_POLICY_SIGN = np.asarray([-1.0, 1.0, 1.0, 1.0, -1.0, 1.0])
JULY30_PAIRED_SPECS: dict[str, dict[str, Any]] = {
    "14": {
        "pair": "14",
        "group": "thigh",
        "motor_ids": [1, 4],
        "active_joint_ids": [0, 3],
        "expected_amplitude_rad": 0.20,
    },
    "25": {
        "pair": "25",
        "group": "calf",
        "motor_ids": [2, 5],
        "active_joint_ids": [1, 4],
        "expected_amplitude_rad": 0.25,
    },
}
JULY30_PD_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "thigh_kp2_kd0p1__calf_kp8_kd0p8",
        "selectors": {
            "14": {"kp": 2.0, "kd": 0.1},
            "25": {"kp": 8.0, "kd": 0.8},
        },
    },
    {
        "name": "thigh_kp4_kd0p2__calf_kp4_kd0p2",
        "selectors": {
            "14": {"kp": 4.0, "kd": 0.2},
            "25": {"kp": 4.0, "kd": 0.2},
        },
    },
)
PAIRED_AMPLITUDE_TOLERANCE_RAD = 5.0e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coerce_six_vector(values: list[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"{name} must contain 6 values, got {arr.size}")
    return arr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-format",
        choices=("legacy-single", "july30-paired"),
        default="legacy-single",
        help="CSV filename/schema contract to import.",
    )
    parser.add_argument(
        "--csv-dir",
        default="plots",
        help="Directory containing legacy sweep_motor_*.csv or July-30 motor14/motor25 CSVs.",
    )
    parser.add_argument("--out-root", default="logs/dr002_real_sweep_import", help="Output root.")
    parser.add_argument("--run-name", default="real_sweep_for_pd_fit", help="Output suffix.")
    parser.add_argument(
        "--case", default="real_sweep", help="Case name used by fit_isaaclab_pd_to_pseudo_real.py."
    )
    parser.add_argument(
        "--alignment-window-s",
        type=float,
        default=DEFAULT_ALIGNMENT_WINDOW_S,
        help="Initial window used to remove encoder/default DC offset from measured positions.",
    )
    parser.add_argument(
        "--no-align-initial",
        action="store_true",
        help="Do not remove the initial measured-position offset.",
    )
    parser.add_argument(
        "--resample-dt",
        type=float,
        default=DEFAULT_RESAMPLE_DT_S,
        help="Resample CSV traces to a common uniform dt. Use <=0 to keep original timestamps.",
    )
    parser.add_argument(
        "--default-angles",
        type=float,
        nargs=6,
        default=DEFAULT_SWEEP_ZERO_POSE.tolist(),
        help="Six-joint sweep zero pose used as the absolute MuJoCo pose. Default: policy default angles.",
    )
    parser.add_argument(
        "--strict-single-motor",
        action="store_true",
        help="Fail if a CSV named for one motor contains non-zero command columns for other motors.",
    )
    parser.add_argument(
        "--mujoco-qpos-at-policy-center",
        type=float,
        nargs=6,
        default=None,
        help=(
            "Required in july30-paired mode: absolute MuJoCo qpos corresponding exactly to "
            "the documented six-joint policy sweep center."
        ),
    )
    parser.add_argument(
        "--policy-delta-to-mujoco-sign",
        type=float,
        nargs=6,
        default=None,
        help=(
            "Required in july30-paired mode: six explicit +/-1 direction signs mapping "
            "policy-coordinate deltas to MuJoCo qpos deltas."
        ),
    )
    parser.add_argument(
        "--runtime-joint-default",
        type=float,
        nargs=6,
        default=None,
        help=(
            "Required in july30-paired mode: deployment motors.yaml joint_default "
            "vector d used by the runtime topic relation u = p - d."
        ),
    )
    parser.add_argument(
        "--amplitude-tolerance-rad",
        type=float,
        default=PAIRED_AMPLITUDE_TOLERANCE_RAD,
        help="Allowed absolute error when validating July-30 0.20/0.25 rad command amplitudes.",
    )
    return parser.parse_args(argv)


def make_out_dir(root: str | Path, run_name: str) -> Path:
    root_path = resolve_repo_path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_name).strip("._-")
    for index in range(1, 1000):
        out = root_path / f"{stamp}_{index:03d}_{safe}"
        try:
            out.mkdir()
            return out
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate output directory under {root_path}")


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    out: dict[str, list[float]] = {name: [] for name in rows[0].keys()}
    for row in rows:
        for name in out:
            out[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=np.float64) for name, values in out.items()}


def active_motor_from_name(path: Path) -> tuple[int, str]:
    match = CSV_RE.match(path.name)
    if not match:
        raise ValueError(f"CSV name must match sweep_motor_<id>_<joint>_<stamp>.csv: {path.name}")
    motor_id = int(match.group("motor"))
    joint_id = MOTOR_TO_JOINT[motor_id]
    expected = JOINT_NAMES[joint_id]
    joint_name = match.group("joint")
    if joint_name != expected:
        raise ValueError(f"{path.name}: motor {motor_id} maps to {expected}, got {joint_name}")
    return motor_id, expected


def parse_paired_csv_name(path: Path) -> dict[str, Any]:
    match = PAIRED_CSV_RE.fullmatch(path.name)
    if not match:
        raise ValueError(
            "paired CSV name must match "
            "motor14|motor25_<YYYYMMDD_HHMMSS>_kp<Kp>_kd<Kd>.csv: "
            f"{path.name}"
        )

    def parse_gain(token: str) -> float:
        return float(token.replace("p", "."))

    pair = match.group("pair")
    spec = JULY30_PAIRED_SPECS[pair]
    return {
        **spec,
        "stamp": match.group("stamp"),
        "kp": parse_gain(match.group("kp")),
        "kd": parse_gain(match.group("kd")),
    }


def build_coordinate_contract(
    mujoco_qpos_at_policy_center: list[float] | np.ndarray | None,
    policy_delta_to_mujoco_sign: list[float] | np.ndarray | None,
    runtime_joint_default: list[float] | np.ndarray | None,
) -> dict[str, Any]:
    if mujoco_qpos_at_policy_center is None:
        raise ValueError(
            "july30-paired import requires --mujoco-qpos-at-policy-center; "
            "policy sweep coordinates are not assumed to be absolute MuJoCo qpos"
        )
    if policy_delta_to_mujoco_sign is None:
        raise ValueError(
            "july30-paired import requires --policy-delta-to-mujoco-sign; "
            "MuJoCo joint directions are not inferred"
        )
    if runtime_joint_default is None:
        raise ValueError(
            "july30-paired import requires --runtime-joint-default; "
            "the sign-normalized absolute sweep coordinate p must not be confused "
            "with the runtime relative topic coordinate u = p - d"
        )
    mujoco_center = coerce_six_vector(
        list(mujoco_qpos_at_policy_center),
        "mujoco_qpos_at_policy_center",
    )
    mujoco_sign = coerce_six_vector(
        list(policy_delta_to_mujoco_sign),
        "policy_delta_to_mujoco_sign",
    )
    runtime_default = coerce_six_vector(
        list(runtime_joint_default),
        "runtime_joint_default",
    )
    if not np.all(np.isfinite(mujoco_center)):
        raise ValueError("mujoco_qpos_at_policy_center must contain only finite values")
    if not np.all(np.isfinite(runtime_default)):
        raise ValueError("runtime_joint_default must contain only finite values")
    if not np.all(np.isfinite(mujoco_sign)) or not np.all(
        np.isclose(np.abs(mujoco_sign), 1.0, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("policy_delta_to_mujoco_sign entries must each be exactly +1 or -1")

    joints: list[dict[str, Any]] = []
    active_leg_joint_ids = {0, 1, 3, 4}
    for joint_id, joint_name in enumerate(JOINT_NAMES):
        joints.append(
            {
                "joint_id": joint_id,
                "joint": joint_name,
                "motor_id": joint_id + 1,
                "used_by_paired_leg_import": joint_id in active_leg_joint_ids,
                "raw_encoder_center_rad": float(JULY30_RAW_ENCODER_CENTER[joint_id]),
                "raw_to_policy_sign": (
                    float(JULY30_RAW_TO_POLICY_SIGN[joint_id])
                    if joint_id in active_leg_joint_ids
                    else None
                ),
                "policy_sweep_center_rad": float(JULY30_POLICY_SWEEP_CENTER[joint_id]),
                "runtime_joint_default_rad": float(runtime_default[joint_id]),
                "runtime_topic_center_rad": float(
                    JULY30_POLICY_SWEEP_CENTER[joint_id] - runtime_default[joint_id]
                ),
                "policy_delta_to_mujoco_sign": float(mujoco_sign[joint_id]),
                "mujoco_qpos_at_policy_center_rad": float(mujoco_center[joint_id]),
            }
        )
    payload: dict[str, Any] = {
        "format": "dr002_july30_coordinate_contract_v1",
        "joint_order": list(JOINT_NAMES),
        "raw_encoder_center_rad": JULY30_RAW_ENCODER_CENTER.tolist(),
        "policy_sweep_center_rad": JULY30_POLICY_SWEEP_CENTER.tolist(),
        "policy_named_coordinate_semantics": (
            "sign_normalized_absolute_motor_coordinate_p_not_runtime_relative_u"
        ),
        "raw_to_policy_sign": [
            float(JULY30_RAW_TO_POLICY_SIGN[joint_id]) if joint_id in active_leg_joint_ids else None
            for joint_id in range(len(JOINT_NAMES))
        ],
        "runtime_joint_default_rad": runtime_default.tolist(),
        "runtime_topic_center_rad": (JULY30_POLICY_SWEEP_CENTER - runtime_default).tolist(),
        "mujoco_qpos_at_policy_center_rad": mujoco_center.tolist(),
        "policy_delta_to_mujoco_sign": mujoco_sign.tolist(),
        "position_formulas": {
            "raw_to_policy": (
                "p = policy_named_center + raw_to_policy_sign * (q_raw - raw_encoder_center)"
            ),
            "policy_named_to_runtime_topic": "u = p - runtime_joint_default",
            "runtime_topic_to_mujoco": (
                "q_mujoco = mujoco_qpos_at_policy_center + "
                "policy_delta_to_mujoco_sign * (u - runtime_topic_center)"
            ),
            "policy_named_to_mujoco_direct": (
                "q_mujoco = mujoco_qpos_at_policy_center + "
                "policy_delta_to_mujoco_sign * (p - policy_named_center)"
            ),
        },
        "velocity_formulas": {
            "raw_to_policy": "qd_policy = raw_to_policy_sign * qd_raw",
            "policy_to_mujoco": ("qd_mujoco = policy_delta_to_mujoco_sign * qd_policy"),
        },
        "coordinate_layers": {
            "raw_encoder": {
                "symbol": "q_raw",
                "center_rad": JULY30_RAW_ENCODER_CENTER.tolist(),
            },
            "policy_named": {
                "symbol": "p",
                "semantics": "sign_normalized_absolute_motor_coordinate",
                "center_rad": JULY30_POLICY_SWEEP_CENTER.tolist(),
            },
            "runtime_topic": {
                "symbol": "u",
                "semantics": "relative_position_published_to_policy_joint_states",
                "joint_default_rad": runtime_default.tolist(),
                "center_rad": (JULY30_POLICY_SWEEP_CENTER - runtime_default).tolist(),
            },
            "mujoco_absolute_qpos": {
                "symbol": "q_mujoco",
                "center_rad": mujoco_center.tolist(),
            },
        },
        "absolute_offset_provenance": {
            "source": "caller_supplied_mujoco_qpos_at_policy_center",
            "encoder_zero_was_not_modified_during_acquisition": True,
            "acquisition_evidence": "July-30 sweep README: no mechanical zero was written",
            "runtime_joint_default_source": (
                "Walking_Eagle-Deploy_Refactor deployment motors.yaml"
            ),
            "assumption": (
                "the caller-supplied runtime configuration is the same calibration "
                "version retained by the acquisition hardware"
            ),
            "identifiability_note": (
                "the chirp response cannot identify an absolute MuJoCo qpos offset"
            ),
        },
        "wheel_raw_to_policy_mapping": "not_used_not_verified",
        "joints": joints,
    }
    payload["sha256"] = sha256_json(payload)
    return payload


def slice_after_last_timestamp_reset(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if "timestamp" not in data:
        raise KeyError("paired CSV is missing timestamp")
    timestamps = np.asarray(data["timestamp"], dtype=np.float64)
    if timestamps.size < 2 or not np.all(np.isfinite(timestamps)):
        raise ValueError("paired CSV timestamps must contain at least two finite samples")
    reset_indices = np.flatnonzero(np.diff(timestamps) < 0.0) + 1
    if reset_indices.size == 0:
        raise ValueError(
            "paired CSV contains no timestamp rollback; cannot identify the formal sweep segment"
        )
    start_index = int(reset_indices[-1])
    segment = {
        name: np.asarray(values[start_index:], dtype=np.float64) for name, values in data.items()
    }
    segment_timestamps = segment["timestamp"]
    if segment_timestamps.size < 2:
        raise ValueError("formal sweep segment contains fewer than two samples")
    segment_diffs = np.diff(segment_timestamps)
    if not np.all(segment_diffs > 0.0):
        raise ValueError("timestamps after the final rollback must be strictly increasing")
    time = segment_timestamps - float(segment_timestamps[0])
    return (
        time,
        segment,
        {
            "selection": "rows_from_final_timestamp_rollback_inclusive",
            "reset_row_indices_zero_based": reset_indices.astype(int).tolist(),
            "selected_start_row_zero_based": start_index,
            "discarded_prefix_samples": start_index,
            "selected_raw_samples": int(time.size),
            "selected_duration_s": float(time[-1]),
            "selected_dt_median_s": float(np.median(np.diff(time))),
        },
    )


def raw_position_to_policy(
    values: np.ndarray,
    joint_id: int,
) -> np.ndarray:
    return JULY30_POLICY_SWEEP_CENTER[joint_id] + JULY30_RAW_TO_POLICY_SIGN[joint_id] * (
        values - JULY30_RAW_ENCODER_CENTER[joint_id]
    )


def policy_position_to_mujoco(
    values: np.ndarray,
    joint_id: int,
    coordinate_contract: dict[str, Any],
) -> np.ndarray:
    mujoco_center = float(coordinate_contract["mujoco_qpos_at_policy_center_rad"][joint_id])
    sign = float(coordinate_contract["policy_delta_to_mujoco_sign"][joint_id])
    return mujoco_center + sign * (values - JULY30_POLICY_SWEEP_CENTER[joint_id])


def raw_velocity_to_mujoco(
    values: np.ndarray,
    joint_id: int,
    coordinate_contract: dict[str, Any],
) -> np.ndarray:
    return (
        JULY30_RAW_TO_POLICY_SIGN[joint_id]
        * float(coordinate_contract["policy_delta_to_mujoco_sign"][joint_id])
        * values
    )


def initial_mean(time: np.ndarray, values: np.ndarray, window_s: float) -> float:
    if window_s <= 0:
        return float(values[0])
    mask = time <= time[0] + window_s
    if not np.any(mask):
        return float(values[0])
    return float(np.mean(values[mask]))


def resampling_metadata(dt: float) -> dict[str, Any]:
    enabled = dt > 0.0
    methods = (
        RESAMPLE_METHODS
        if enabled
        else {name: "none_original_samples" for name in RESAMPLE_METHODS}
    )
    return {
        "enabled": enabled,
        "target_dt_s": float(dt),
        "target_sample_rate_hz": float(1.0 / dt) if enabled else 0.0,
        "time_grid": "integer_tick_divided_by_sample_rate_hz_float64"
        if enabled
        else "original_timestamps",
        "methods": dict(methods),
    }


def resample_uniform(
    time: np.ndarray,
    traces: dict[str, np.ndarray],
    dt: float,
    zero_order_hold_traces: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dt <= 0:
        return time, traces
    duration = float(time[-1])
    if duration <= 0:
        return time, traces
    # Build the clock the same way the replay builds controller ticks
    # (integer tick / frequency).  ``np.arange(..., step=dt)`` is effectively
    # tick * dt and can differ by one float64 ulp from tick / control_hz; a
    # strict causal ZOH then incorrectly selects the preceding source frame.
    sample_rate_hz = 1.0 / dt
    final_tick = int(np.floor(duration * sample_rate_hz + 0.5))
    uniform_time = np.arange(final_tick + 1, dtype=np.float64) / sample_rate_hz
    zero_order_hold_traces = zero_order_hold_traces or set()
    hold_indices = np.searchsorted(time, uniform_time, side="right") - 1
    hold_indices = np.clip(hold_indices, 0, time.shape[0] - 1)
    resampled = {
        name: values[hold_indices]
        if name in zero_order_hold_traces
        else np.interp(uniform_time, time, values)
        for name, values in traces.items()
    }
    return uniform_time, resampled


def write_truth_csv(
    path: Path,
    time: np.ndarray,
    des_pos: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
) -> None:
    header = (
        ["time"]
        + [f"q_{name}" for name in JOINT_NAMES]
        + [f"des_{name}" for name in JOINT_NAMES]
        + [f"tau_{name}" for name in JOINT_NAMES]
        + [f"qd_{name}" for name in JOINT_NAMES]
        + [f"desvel_{name}" for name in JOINT_NAMES]
    )
    zeros = np.zeros_like(dof_pos)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t in enumerate(time):
            writer.writerow(
                [float(t)]
                + dof_pos[i].tolist()
                + des_pos[i].tolist()
                + zeros[i].tolist()
                + dof_vel[i].tolist()
                + zeros[i].tolist()
            )


def import_paired_source(
    path: Path,
    out_dir: Path,
    case: str,
    coordinate_contract: dict[str, Any],
    amplitude_tolerance_rad: float = PAIRED_AMPLITUDE_TOLERANCE_RAD,
    resample_dt: float = DEFAULT_RESAMPLE_DT_S,
) -> dict[str, Any]:
    if not np.isclose(
        resample_dt,
        DEFAULT_RESAMPLE_DT_S,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "july30-paired import requires exact 200 Hz resampling "
            f"(--resample-dt {DEFAULT_RESAMPLE_DT_S})"
        )
    if not np.isfinite(amplitude_tolerance_rad) or amplitude_tolerance_rad <= 0.0:
        raise ValueError("amplitude_tolerance_rad must be finite and positive")

    file_meta = parse_paired_csv_name(path)
    motor_ids = [int(value) for value in file_meta["motor_ids"]]
    active_joint_ids = [int(value) for value in file_meta["active_joint_ids"]]
    expected_amplitude = float(file_meta["expected_amplitude_rad"])
    raw_data = read_csv(path)
    raw_time, data, segment_meta = slice_after_last_timestamp_reset(raw_data)

    required_columns = [
        column
        for motor_id in motor_ids
        for column in (
            f"motor_{motor_id}_target_raw",
            f"motor_{motor_id}_pos_raw",
            f"motor_{motor_id}_vel_raw",
        )
    ]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise KeyError(f"{path.name}: missing paired columns: {missing_columns}")

    traces: dict[str, np.ndarray] = {}
    command_trace_names: set[str] = set()
    amplitude_checks: list[dict[str, Any]] = []
    for motor_id, joint_id in zip(motor_ids, active_joint_ids, strict=True):
        command_name = f"motor_{motor_id}_command_raw"
        position_name = f"motor_{motor_id}_position_raw"
        velocity_name = f"motor_{motor_id}_velocity_raw"
        command = np.asarray(data[f"motor_{motor_id}_target_raw"], dtype=np.float64)
        position = np.asarray(data[f"motor_{motor_id}_pos_raw"], dtype=np.float64)
        velocity = np.asarray(data[f"motor_{motor_id}_vel_raw"], dtype=np.float64)
        if not (
            np.all(np.isfinite(command))
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(velocity))
        ):
            raise ValueError(f"{path.name}: motor {motor_id} contains non-finite samples")
        command_min = float(np.min(command))
        command_max = float(np.max(command))
        measured_center = 0.5 * (command_min + command_max)
        measured_amplitude = 0.5 * (command_max - command_min)
        expected_center = float(JULY30_RAW_ENCODER_CENTER[joint_id])
        center_error = measured_center - expected_center
        amplitude_error = measured_amplitude - expected_amplitude
        if abs(center_error) > amplitude_tolerance_rad:
            raise ValueError(
                f"{path.name}: motor {motor_id} raw command center "
                f"{measured_center:+.9f} differs from expected {expected_center:+.9f} "
                f"by {center_error:+.9f} rad"
            )
        if abs(amplitude_error) > amplitude_tolerance_rad:
            raise ValueError(
                f"{path.name}: motor {motor_id} command amplitude "
                f"{measured_amplitude:.9f} differs from expected "
                f"{expected_amplitude:.9f} by {amplitude_error:+.9f} rad"
            )
        amplitude_checks.append(
            {
                "motor_id": motor_id,
                "joint_id": joint_id,
                "joint": JOINT_NAMES[joint_id],
                "raw_command_min_rad": command_min,
                "raw_command_max_rad": command_max,
                "measured_raw_center_rad": measured_center,
                "expected_raw_center_rad": expected_center,
                "center_error_rad": center_error,
                "measured_amplitude_rad": measured_amplitude,
                "expected_amplitude_rad": expected_amplitude,
                "amplitude_error_rad": amplitude_error,
                "passed": True,
            }
        )
        traces[command_name] = command
        traces[position_name] = position
        traces[velocity_name] = velocity
        command_trace_names.add(command_name)

    time, resampled = resample_uniform(
        raw_time,
        traces,
        resample_dt,
        zero_order_hold_traces=command_trace_names,
    )
    mujoco_center = np.asarray(
        coordinate_contract["mujoco_qpos_at_policy_center_rad"],
        dtype=np.float64,
    )
    des_pos = np.tile(mujoco_center.reshape(1, -1), (time.shape[0], 1))
    dof_pos = des_pos.copy()
    dof_vel = np.zeros_like(des_pos)
    policy_commands: list[np.ndarray] = []
    trace_contract: list[dict[str, Any]] = []
    for motor_id, joint_id in zip(motor_ids, active_joint_ids, strict=True):
        command_raw = resampled[f"motor_{motor_id}_command_raw"]
        position_raw = resampled[f"motor_{motor_id}_position_raw"]
        velocity_raw = resampled[f"motor_{motor_id}_velocity_raw"]
        command_policy = raw_position_to_policy(command_raw, joint_id)
        position_policy = raw_position_to_policy(position_raw, joint_id)
        command_mujoco = policy_position_to_mujoco(
            command_policy,
            joint_id,
            coordinate_contract,
        )
        position_mujoco = policy_position_to_mujoco(
            position_policy,
            joint_id,
            coordinate_contract,
        )
        velocity_mujoco = raw_velocity_to_mujoco(
            velocity_raw,
            joint_id,
            coordinate_contract,
        )
        des_pos[:, joint_id] = command_mujoco
        dof_pos[:, joint_id] = position_mujoco
        dof_vel[:, joint_id] = velocity_mujoco
        policy_commands.append(command_policy)
        trace_contract.append(
            {
                "motor_id": motor_id,
                "joint_id": joint_id,
                "joint": JOINT_NAMES[joint_id],
                "raw_columns": {
                    "command": f"motor_{motor_id}_target_raw",
                    "position": f"motor_{motor_id}_pos_raw",
                    "velocity": f"motor_{motor_id}_vel_raw",
                },
                "payload_columns": {
                    "command": f"des_dof_pos[:, {joint_id}]",
                    "position": f"dof_pos[:, {joint_id}]",
                    "velocity": f"dof_vel[:, {joint_id}]",
                },
            }
        )

    policy_command_difference = float(np.max(np.abs(policy_commands[0] - policy_commands[1])))
    if policy_command_difference > amplitude_tolerance_rad:
        raise ValueError(
            f"{path.name}: paired policy commands differ by up to "
            f"{policy_command_difference:.9f} rad after coordinate conversion"
        )

    joint_group = str(file_meta["group"])
    active_label = "_".join(f"j{joint_id}" for joint_id in active_joint_ids)
    stem = f"real_sweep_paired_{joint_group}_{active_label}_chirp_data"
    pt_path = save_chirp_data(
        out_dir / f"{stem}.pt",
        time,
        dof_pos,
        des_pos,
        dof_vel=dof_vel,
        des_dof_vel=np.zeros_like(des_pos),
    )
    truth_csv = out_dir / f"mujoco_six_joint_paired_{joint_group}_{case}.csv"
    write_truth_csv(truth_csv, time, des_pos, dof_pos, dof_vel)

    representative_joint_id = active_joint_ids[0]
    return {
        "motor_ids": motor_ids,
        "active_joint_ids": active_joint_ids,
        "active_joint_indices_1_based": [joint_id + 1 for joint_id in active_joint_ids],
        "active_joints": [JOINT_NAMES[joint_id] for joint_id in active_joint_ids],
        "joint_group": joint_group,
        "representative_joint_id": representative_joint_id,
        "joint_id": representative_joint_id,
        "joint_index": representative_joint_id + 1,
        "joint": JOINT_NAMES[representative_joint_id],
        "kp": float(file_meta["kp"]),
        "kd": float(file_meta["kd"]),
        "source_csv": str(path.resolve()),
        "source_csv_sha256": sha256_file(path),
        "pt": pt_path.name,
        "pt_sha256": sha256_file(pt_path),
        "truth_csv": truth_csv.name,
        "truth_csv_sha256": sha256_file(truth_csv),
        "serialization_dtypes": dict(SERIALIZATION_DTYPES),
        "num_samples": int(time.shape[0]),
        "duration_s": float(time[-1]) if time.shape[0] else 0.0,
        "dt_median_s": float(np.median(np.diff(time))) if time.shape[0] > 1 else 0.0,
        "raw_num_samples": int(raw_time.shape[0]),
        "raw_dt_median_s": (float(np.median(np.diff(raw_time))) if raw_time.shape[0] > 1 else 0.0),
        "resample_dt_s": float(resample_dt),
        "sample_rate_hz": float(1.0 / resample_dt),
        "resampling": resampling_metadata(resample_dt),
        "target_type": "position",
        "coordinate_contract_sha256": coordinate_contract["sha256"],
        "trace_coordinate_mapping": trace_contract,
        "formal_segment": segment_meta,
        "amplitude_validation": {
            "passed": True,
            "tolerance_rad": float(amplitude_tolerance_rad),
            "expected_amplitude_rad": expected_amplitude,
            "per_motor": amplitude_checks,
            "paired_policy_command_max_abs_difference_rad": policy_command_difference,
        },
        "initial_dc_alignment": {
            "enabled": False,
            "window_s": 0.0,
            "position_offset_removed_rad": [0.0, 0.0],
            "reason": "preserve_measured_response_under_explicit_coordinate_contract",
        },
        "aligned": False,
        "position_offset_removed": 0.0,
    }


def _select_paired_csvs(
    paths: list[Path],
) -> dict[tuple[str, float, float], Path]:
    selected: dict[tuple[str, float, float], Path] = {}
    for path in paths:
        if not (path.name.startswith("motor14_") or path.name.startswith("motor25_")):
            continue
        meta = parse_paired_csv_name(path)
        key = (str(meta["pair"]), float(meta["kp"]), float(meta["kd"]))
        if key in selected:
            raise ValueError(f"duplicate paired sweep selector {key}: {selected[key]} and {path}")
        selected[key] = path
    return selected


def import_july30_paired_directory(
    csv_dir: Path,
    out_dir: Path,
    case: str,
    mujoco_qpos_at_policy_center: list[float] | np.ndarray | None,
    policy_delta_to_mujoco_sign: list[float] | np.ndarray | None,
    runtime_joint_default: list[float] | np.ndarray | None,
    amplitude_tolerance_rad: float = PAIRED_AMPLITUDE_TOLERANCE_RAD,
    resample_dt: float = DEFAULT_RESAMPLE_DT_S,
) -> dict[str, Any]:
    if not np.isclose(
        resample_dt,
        DEFAULT_RESAMPLE_DT_S,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("july30-paired import requires resample_dt=0.005 for exact 200 Hz output")
    coordinate_contract = build_coordinate_contract(
        mujoco_qpos_at_policy_center,
        policy_delta_to_mujoco_sign,
        runtime_joint_default,
    )
    candidates = sorted(csv_dir.glob("motor*.csv"))
    selector_to_path = _select_paired_csvs(candidates)
    if not selector_to_path:
        raise FileNotFoundError(f"no motor14/motor25 paired CSV files found under {csv_dir}")
    missing_selectors: list[str] = []
    for group_spec in JULY30_PD_GROUPS:
        for pair in ("14", "25"):
            selector = group_spec["selectors"][pair]
            key = (pair, float(selector["kp"]), float(selector["kd"]))
            if key not in selector_to_path:
                missing_selectors.append(
                    f"{group_spec['name']}: pair={pair}, Kp={selector['kp']}, Kd={selector['kd']}"
                )
    if missing_selectors:
        raise FileNotFoundError(
            f"missing required July-30 paired sweeps under {csv_dir}: {missing_selectors}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    group_index_entries: list[dict[str, Any]] = []
    for group_spec in JULY30_PD_GROUPS:
        group_name = str(group_spec["name"])
        group_dir = out_dir / group_name
        group_dir.mkdir()
        sources: list[dict[str, Any]] = []
        selected_pd: dict[str, Any] = {}
        for pair in ("14", "25"):
            selector = group_spec["selectors"][pair]
            key = (pair, float(selector["kp"]), float(selector["kd"]))
            source = import_paired_source(
                selector_to_path[key],
                group_dir,
                f"{case}_{group_name}",
                coordinate_contract,
                amplitude_tolerance_rad=amplitude_tolerance_rad,
                resample_dt=resample_dt,
            )
            sources.append(source)
            selected_pd[str(source["joint_group"])] = {
                "kp": float(selector["kp"]),
                "kd": float(selector["kd"]),
                "source_csv": source["source_csv"],
                "source_csv_sha256": source["source_csv_sha256"],
            }

        manifest = {
            "format": "dr002_real_sweep_import_v2_paired",
            "case": f"{case}_{group_name}",
            "paired_group": group_name,
            "source_dir": str(csv_dir.resolve()),
            "control_mode": CONTROL_MODE,
            "joint_names": list(JOINT_NAMES),
            "default_angles": list(coordinate_contract["mujoco_qpos_at_policy_center_rad"]),
            "coordinate_contract": coordinate_contract,
            "pd_configuration": selected_pd,
            "alignment_window_s": 0.0,
            "align_initial": False,
            "initial_dc_alignment": "disabled",
            "resample_dt_s": float(resample_dt),
            "sample_rate_hz": float(1.0 / resample_dt),
            "resampling": resampling_metadata(resample_dt),
            "serialization_dtypes": dict(SERIALIZATION_DTYPES),
            "amplitude_contract_rad": {"thigh": 0.20, "calf": 0.25},
            "amplitude_tolerance_rad": float(amplitude_tolerance_rad),
            "sources": sources,
        }
        manifest_path = group_dir / "chirp_source_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        group_index_entries.append(
            {
                "name": group_name,
                "directory": group_dir.name,
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
                "pd_configuration": selected_pd,
            }
        )

    index_manifest = {
        "format": "dr002_real_sweep_paired_group_index_v1",
        "case": case,
        "source_dir": str(csv_dir.resolve()),
        "coordinate_contract": coordinate_contract,
        "resample_dt_s": float(resample_dt),
        "sample_rate_hz": float(1.0 / resample_dt),
        "align_initial": False,
        "groups": group_index_entries,
    }
    index_path = out_dir / "paired_group_index.json"
    index_path.write_text(
        json.dumps(index_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return index_manifest


def import_one(
    path: Path,
    out_dir: Path,
    case: str,
    default_angles: np.ndarray,
    align_initial: bool,
    alignment_window_s: float,
    resample_dt: float,
    strict_single_motor: bool,
) -> dict[str, Any]:
    motor_id, joint_name = active_motor_from_name(path)
    joint_id = MOTOR_TO_JOINT[motor_id]
    data = read_csv(path)
    raw_time = data["timestamp"] - float(data["timestamp"][0])
    cmd = data[f"motor_{motor_id}_cmd_pos"]
    act = data[f"motor_{motor_id}_act_pos"]
    vel = data.get(f"motor_{motor_id}_act_vel", np.zeros_like(act))
    raw_dt_median = float(np.median(np.diff(raw_time))) if raw_time.shape[0] > 1 else 0.0
    non_active_commands: list[dict[str, float | int | str]] = []
    for other_motor_id in range(1, 7):
        if other_motor_id == motor_id:
            continue
        key = f"motor_{other_motor_id}_cmd_pos"
        if key not in data:
            continue
        values = data[key]
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        if max_abs > NON_ACTIVE_COMMAND_EPS:
            non_active_commands.append(
                {
                    "motor_id": other_motor_id,
                    "joint": JOINT_NAMES[MOTOR_TO_JOINT[other_motor_id]],
                    "column": key,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "max_abs": max_abs,
                }
            )
    if strict_single_motor and non_active_commands:
        columns = ", ".join(str(item["column"]) for item in non_active_commands)
        raise ValueError(
            f"{path.name}: expected single-motor sweep, but non-active command columns are non-zero: {columns}"
        )

    time, resampled = resample_uniform(
        raw_time,
        {"cmd": cmd, "act": act, "vel": vel},
        resample_dt,
        zero_order_hold_traces={"cmd"},
    )
    cmd = resampled["cmd"]
    act = resampled["act"]
    vel = resampled["vel"]

    cmd0 = initial_mean(time, cmd, alignment_window_s)
    act0 = initial_mean(time, act, alignment_window_s)
    offset_removed = act0 - cmd0 if align_initial else 0.0
    act_aligned = act - offset_removed

    des_pos = np.tile(default_angles.reshape(1, -1), (time.shape[0], 1))
    dof_pos = des_pos.copy()
    dof_vel = np.zeros_like(des_pos)
    des_pos[:, joint_id] = default_angles[joint_id] + cmd
    dof_pos[:, joint_id] = default_angles[joint_id] + act_aligned
    dof_vel[:, joint_id] = vel

    stem = f"real_sweep_joint{joint_id + 1:02d}_{joint_name}_chirp_data"
    pt_path = save_chirp_data(
        out_dir / f"{stem}.pt",
        time,
        dof_pos,
        des_pos,
        dof_vel=dof_vel,
        des_dof_vel=np.zeros_like(des_pos),
    )
    truth_csv = out_dir / f"mujoco_six_joint_joint{joint_id + 1:02d}_{joint_name}_{case}.csv"
    write_truth_csv(truth_csv, time, des_pos, dof_pos, dof_vel)

    meta = {
        "motor_id": motor_id,
        "joint_index": joint_id + 1,
        "joint_id": joint_id,
        "joint": joint_name,
        "source_csv": str(path),
        "source_csv_sha256": sha256_file(path),
        "pt": pt_path.name,
        "pt_sha256": sha256_file(pt_path),
        "truth_csv": truth_csv.name,
        "truth_csv_sha256": sha256_file(truth_csv),
        "serialization_dtypes": dict(SERIALIZATION_DTYPES),
        "num_samples": int(time.shape[0]),
        "duration_s": float(time[-1]) if time.shape[0] else 0.0,
        "dt_median_s": float(np.median(np.diff(time))) if time.shape[0] > 1 else 0.0,
        "raw_num_samples": int(raw_time.shape[0]),
        "raw_dt_median_s": raw_dt_median,
        "resample_dt_s": float(resample_dt),
        "sample_rate_hz": float(1.0 / resample_dt) if resample_dt > 0.0 else 0.0,
        "resampling": resampling_metadata(resample_dt),
        "target_type": "position",
        "cmd_initial_mean": cmd0,
        "act_initial_mean": act0,
        "position_offset_removed": offset_removed,
        "aligned": bool(align_initial),
        "non_active_command_columns": non_active_commands,
    }
    return meta


def main() -> None:
    args = parse_args()
    csv_dir = resolve_repo_path(args.csv_dir)
    if args.input_format == "july30-paired":
        if not np.isclose(
            args.resample_dt,
            DEFAULT_RESAMPLE_DT_S,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "july30-paired import requires --resample-dt 0.005 for exact 200 Hz output"
            )
        build_coordinate_contract(
            args.mujoco_qpos_at_policy_center,
            args.policy_delta_to_mujoco_sign,
            args.runtime_joint_default,
        )
        out_dir = make_out_dir(args.out_root, args.run_name)
        index = import_july30_paired_directory(
            csv_dir=csv_dir,
            out_dir=out_dir,
            case=args.case,
            mujoco_qpos_at_policy_center=args.mujoco_qpos_at_policy_center,
            policy_delta_to_mujoco_sign=args.policy_delta_to_mujoco_sign,
            runtime_joint_default=args.runtime_joint_default,
            amplitude_tolerance_rad=args.amplitude_tolerance_rad,
            resample_dt=args.resample_dt,
        )
        print(f"[DONE] wrote paired real sweep directory: {out_dir}")
        print(
            "[CONFIG] "
            f"control_mode={CONTROL_MODE} "
            f"sample_rate={index['sample_rate_hz']:.1f}Hz "
            "align_initial=False "
            f"coordinate_contract_sha256={index['coordinate_contract']['sha256']}"
        )
        for group in index["groups"]:
            print(
                "[GROUP] "
                f"{group['name']} "
                f"manifest={out_dir / group['directory'] / group['manifest']} "
                f"sha256={group['manifest_sha256']}"
            )
        return

    paths = sorted(csv_dir.glob("sweep_motor_*_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no sweep_motor_*.csv files found under {csv_dir}")

    out_dir = make_out_dir(args.out_root, args.run_name)
    align_initial = not args.no_align_initial
    default_angles = coerce_six_vector(args.default_angles, "default_angles")
    sources = [
        import_one(
            path,
            out_dir,
            args.case,
            default_angles,
            align_initial,
            args.alignment_window_s,
            args.resample_dt,
            args.strict_single_motor,
        )
        for path in paths
    ]

    manifest = {
        "format": "dr002_real_sweep_import_v1",
        "case": args.case,
        "source_dir": str(csv_dir),
        "control_mode": CONTROL_MODE,
        "joint_names": JOINT_NAMES,
        "default_angles": default_angles.tolist(),
        "alignment_window_s": args.alignment_window_s,
        "align_initial": align_initial,
        "resample_dt_s": args.resample_dt,
        "sample_rate_hz": float(1.0 / args.resample_dt) if args.resample_dt > 0.0 else 0.0,
        "resampling": resampling_metadata(args.resample_dt),
        "serialization_dtypes": dict(SERIALIZATION_DTYPES),
        "strict_single_motor": bool(args.strict_single_motor),
        "sources": sources,
    }
    (out_dir / "chirp_source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[DONE] wrote imported real sweep directory: {out_dir}")
    print(
        "[CONFIG] "
        f"control_mode={CONTROL_MODE} "
        f"default_angles={default_angles.tolist()} "
        f"resample_dt={args.resample_dt:.6f}s "
        f"sample_rate={manifest['sample_rate_hz']:.1f}Hz "
        f"alignment_window={args.alignment_window_s:.3f}s "
        f"align_initial={align_initial}"
    )
    for source in sources:
        print(
            "[SOURCE] "
            f"joint{source['joint_index']:02d} {source['joint']} "
            f"N={source['num_samples']} duration={source['duration_s']:.3f}s "
            f"dt={source['dt_median_s']:.6f}s rate={source['sample_rate_hz']:.1f}Hz "
            f"offset_removed={source['position_offset_removed']:+.6f}"
        )
        for item in source.get("non_active_command_columns", []):
            print(
                "[WARN] non-active command column is not zero: "
                f"{item['column']} ({item['joint']}) "
                f"min={item['min']:+.6f} max={item['max']:+.6f}"
            )


if __name__ == "__main__":
    main()
