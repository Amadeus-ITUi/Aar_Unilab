#!/usr/bin/env python3
"""Fit DR002 PACE actuator parameters with time-domain MSE and CMA-ES.

It consumes imported sweep artifacts without importing Deploy or replay source
code. It keeps PD gains fixed and identifies per-joint armature, viscous damping,
Coulomb friction, encoder bias, plus one shared command or torque delay. CMA-ES
is the only optimizer.  Every candidate is replayed over the complete recorded
chirp, while the time-domain mean squared error is evaluated only over the
requested strict linear-chirp frequency window. Bode metrics are generated
after identification as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cmaes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chirp_frequency_response import (  # noqa: E402
    DEFAULT_COMMAND_PSD_THRESHOLD_DB,
    DEFAULT_WELCH_SEGMENT_DURATION_S,
    LinearChirpTimeWindow,
    SharedCommandDelayBuffer,
    bode_error_components,
    coerce_shared_delay_steps,
    linear_chirp_time_window,
    reference_excitation_mask,
    reference_excitation_summary,
    transfer_bode,
)
from mujoco_dr002_common import (  # noqa: E402
    DEFAULT_ANGLES,
    DEFAULT_MODEL,
    JOINT_NAMES,
    PARAM_VECTOR_ORDER,
    coerce_vector,
    ensure_parent,
    load_chirp_data,
    load_mujoco_model_with_mesh_fallback,
    load_params_file,
    make_fixed_base_mjcf,
    make_self_contained_mjcf,
    normalize_params,
    params_to_vector,
    resolve_repo_path,
    write_json,
)
from run_mujoco_dr002_sim import (  # noqa: E402
    DEFAULT_EFFORT_LIMIT,
    LEG_JOINT_IDS,
    WHEEL_JOINT_IDS,
    free_root_addresses,
    joint_addresses,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_lockable_joint_mjcf(model_path: str | Path, *, fixed_base: bool) -> Path:
    """Return an MJCF copy with inactive-by-default equality locks on all six joints."""

    source = resolve_repo_path(model_path)
    if fixed_base:
        source = make_fixed_base_mjcf(source)
    else:
        source = make_self_contained_mjcf(source)
    if source.suffix.lower() not in {".xml", ".mjcf"}:
        raise ValueError("locking non-source joints requires an MJCF/XML model")
    digest = hashlib.sha256(source.read_bytes() + b":pace_lockable_six_joint_v1").hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / "dr002_mujoco_lockable"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{source.stem}_{digest}_lockable.xml"

    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir", ""))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((source.parent / meshdir).resolve()))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing_names = {element.get("name") for element in equality}
    for joint_name in JOINT_NAMES:
        lock_name = f"pace_lock_{joint_name}"
        if lock_name in existing_names:
            continue
        ET.SubElement(
            equality,
            "joint",
            {
                "name": lock_name,
                "joint1": joint_name,
                "polycoef": "0 0 0 0 0",
                "active": "false",
                "solref": "0.002 1",
                "solimp": "0.9999 0.9999 0.0001 0.5 2",
            },
        )
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


IDENTIFIED_KP = np.asarray([3.5, 3.7, 0.0, 3.5, 3.7, 0.0], dtype=np.float64)
IDENTIFIED_KD = np.asarray([0.15, 0.25, 0.05, 0.15, 0.25, 0.05], dtype=np.float64)
DEFAULT_CMA_INITIAL_ARMATURE = np.asarray(
    [0.0076577, 0.01738699, 0.0008, 0.0076577, 0.01738699, 0.0008],
    dtype=np.float64,
)
FIXTURE_MODES = ("free", "equality", "high-impedance")
DEFAULT_FIXTURE_HOLD_KP = np.asarray([20.0, 20.0, 0.0, 20.0, 20.0, 0.0], dtype=np.float64)
# Wheel rotors have much lower inertia than the leg joints.  At a 200 Hz
# sampled controller, Kd=1 is numerically unstable for them, so retain the
# deploy-like stiff leg hold while using a stable wheel velocity damper.
DEFAULT_FIXTURE_HOLD_KD = np.asarray([1.0, 1.0, 0.1, 1.0, 1.0, 0.1], dtype=np.float64)
EPS = 1.0e-12
EQUALITY_FIXTURE_DRIFT_TOL_RAD = 1.0e-4
EQUALITY_FIXTURE_VELOCITY_TOL_RAD_S = 1.0e-3
WHEEL_LOCK_DRIFT_TOL_RAD = 1.0e-12
WHEEL_LOCK_VELOCITY_TOL_RAD_S = 1.0e-12
SIDE_PAIRS = ((0, 3), (1, 4), (2, 5))
DYNAMIC_PARAM_KEYS = ("armature", "viscous_friction", "coulomb_friction")
CANDIDATE_CSV_FIELDS = (
    "eval",
    "seed",
    "generation",
    "member",
    "time_mse",
    "mean_rmse",
    "source_mse",
    "delay_steps",
    "normalized_x",
    "x",
    "params",
)
OPTIMIZER_CSV_FIELDS = (
    "seed",
    "generation",
    "best_time_mse",
    "mean_time_mse",
    "worst_time_mse",
    "relative_score_spread",
    "sigma",
    "cma_should_stop",
    "epsilon_converged",
)


def delay_artifact_fields(delay_steps: int, semantics: str) -> dict[str, int]:
    """Return unambiguous top-level FIFO delay fields for a fit artifact."""

    delay_steps = coerce_shared_delay_steps(delay_steps)
    if semantics == "command":
        return {"command_delay_steps": delay_steps, "torque_delay_steps": 0}
    if semantics == "torque":
        return {"command_delay_steps": 0, "torque_delay_steps": delay_steps}
    raise ValueError(f"unsupported delay semantics: {semantics}")


@dataclass
class SourceData:
    source: dict[str, Any]
    payload: dict[str, np.ndarray]
    target_kind: str
    score_time_window: LinearChirpTimeWindow | None = None


def coerce_active_joint_ids(
    value: Any,
    *,
    name: str = "active_joint_ids",
) -> tuple[int, ...]:
    """Validate one or more unique joint ids while preserving their order."""

    raw_values = np.asarray(value, dtype=object).reshape(-1)
    if raw_values.size == 0:
        raise ValueError(f"{name} must contain at least one joint id")
    joint_ids: list[int] = []
    for raw in raw_values:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integer joint ids, got {raw!r}")
        try:
            scalar = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integer joint ids, got {raw!r}") from exc
        rounded = round(scalar)
        if not np.isfinite(scalar) or not np.isclose(scalar, rounded, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{name} must contain integer joint ids, got {raw!r}")
        joint_id = int(rounded)
        if joint_id < 0 or joint_id >= len(JOINT_NAMES):
            raise ValueError(
                f"{name} contains joint id {joint_id} outside [0,{len(JOINT_NAMES) - 1}]"
            )
        if joint_id in joint_ids:
            raise ValueError(f"{name} contains duplicate joint id {joint_id}")
        joint_ids.append(joint_id)
    return tuple(joint_ids)


def resolve_active_joint_arguments(
    *,
    active_joint_id: int | None,
    active_joint_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve the legacy scalar and paired active-joint call forms."""

    if active_joint_ids is None:
        if active_joint_id is None:
            raise ValueError("active_joint_id or active_joint_ids is required")
        return coerce_active_joint_ids([active_joint_id])
    resolved = coerce_active_joint_ids(active_joint_ids)
    if active_joint_id is not None and resolved != coerce_active_joint_ids(
        [active_joint_id],
        name="active_joint_id",
    ):
        raise ValueError("active_joint_id conflicts with active_joint_ids")
    return resolved


def source_active_joint_ids(source: dict[str, Any]) -> tuple[int, ...]:
    """Return a source's active joints, preferring the paired-source field."""

    if "active_joint_ids" in source:
        active_joint_ids = coerce_active_joint_ids(
            source["active_joint_ids"],
            name="source.active_joint_ids",
        )
        if "joint_id" in source:
            representative = coerce_active_joint_ids(
                [source["joint_id"]],
                name="source.joint_id",
            )[0]
            if representative not in active_joint_ids:
                raise ValueError("source.joint_id must name one of source.active_joint_ids")
        return active_joint_ids
    if "joint_id" not in source:
        raise ValueError("source must contain active_joint_ids or legacy joint_id")
    return coerce_active_joint_ids([source["joint_id"]], name="source.joint_id")


def source_active_joint_names(source: dict[str, Any]) -> tuple[str, ...]:
    return tuple(JOINT_NAMES[joint_id] for joint_id in source_active_joint_ids(source))


def source_label(source: dict[str, Any]) -> str:
    return "+".join(source_active_joint_names(source))


def source_scalar_metadata(source: dict[str, Any], key: str) -> float | None:
    """Return scalar source metadata without misrepresenting vector-valued fields."""

    value = source.get(key)
    if value is None:
        return None
    raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if raw.size != 1 or not np.isfinite(raw[0]):
        return None
    return float(raw[0])


def source_artifact_token(source_index: int, source: dict[str, Any]) -> str:
    """Build a stable token so repeated sources for one joint cannot overwrite files."""

    token = f"source{source_index:02d}"
    kd = source_scalar_metadata(source, "kd")
    if kd is not None:
        kd_token = format(kd, ".9g").replace("-", "m").replace(".", "p")
        token += f"_kd{kd_token}"
    return token


def source_controller_gain(
    source: dict[str, Any],
    key: str,
    base: np.ndarray,
) -> np.ndarray:
    """Apply an optional scalar/source-specific controller gain to active joints."""

    values = np.asarray(base, dtype=np.float64).copy()
    override = source.get(key)
    if override is None:
        return values
    raw = np.asarray(override, dtype=np.float64).reshape(-1)
    if raw.size == 1:
        values[np.asarray(source_active_joint_ids(source), dtype=np.int32)] = float(raw[0])
    elif raw.size == len(JOINT_NAMES):
        values[:] = raw
    else:
        raise ValueError(f"source.{key} must be a scalar or a {len(JOINT_NAMES)}-joint vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"source.{key} must contain finite non-negative values")
    return values


def source_fixture_gain(
    source: dict[str, Any],
    key: str,
    fallback: np.ndarray,
) -> np.ndarray:
    """Resolve a per-acquisition non-source fixture gain vector.

    ESD-Link acquisitions use different holding gains depending on which joint
    pair is swept.  Keeping this separate from ``source_controller_gain`` is
    intentional: fixture gains always describe all six joints, including the
    inactive joints that are not represented by the scalar source Kp/Kd.
    """

    values = np.asarray(source.get(key, fallback), dtype=np.float64).reshape(-1)
    if values.shape != (len(JOINT_NAMES),):
        raise ValueError(f"source.{key} must contain {len(JOINT_NAMES)} values")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"source.{key} must contain finite non-negative values")
    return values.copy()


def canonicalize_mirrored_fit_joints(
    fit_joints: Sequence[int],
    *,
    mirror_side_dynamics: bool,
) -> list[int]:
    """Fit one side of a requested symmetric pair when dynamics are mirrored."""

    requested = set(coerce_active_joint_ids(fit_joints, name="fit_joints"))
    if not mirror_side_dynamics:
        return sorted(requested)
    effective = set(requested)
    for left_id, right_id in SIDE_PAIRS:
        if left_id in effective and right_id in effective:
            effective.remove(right_id)
    return sorted(effective)


def source_result_trace(
    item: SourceData,
    result: dict[str, np.ndarray],
    key: str,
    joint_id: int,
) -> np.ndarray:
    """Select one active-joint trace from a replay result."""

    active_joint_ids = source_active_joint_ids(item.source)
    if joint_id not in active_joint_ids:
        raise ValueError(f"joint {joint_id} is not active for source {source_label(item.source)}")
    values = np.asarray(result[key], dtype=np.float64)
    if len(active_joint_ids) == 1:
        if values.ndim != 1:
            raise ValueError(f"single-joint replay {key} must be one-dimensional")
        return values
    if values.ndim != 2 or values.shape[1] != len(active_joint_ids):
        raise ValueError(
            f"paired replay {key} must have shape (samples,{len(active_joint_ids)}), "
            f"got {values.shape}"
        )
    return values[:, active_joint_ids.index(joint_id)]


@dataclass(frozen=True)
class CmaSearchSpace:
    """Map normalized CMA coordinates to bounded physical parameters."""

    continuous_bounds: np.ndarray
    delay_values: tuple[int, ...]

    def __post_init__(self) -> None:
        bounds = np.asarray(self.continuous_bounds, dtype=np.float64)
        delays = tuple(sorted(set(int(value) for value in self.delay_values)))
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("continuous_bounds must have shape (N, 2)")
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] < bounds[:, 0]):
            raise ValueError("continuous_bounds must contain finite increasing pairs")
        if not delays or any(value < 0 for value in delays):
            raise ValueError("delay_values must contain non-negative integers")
        object.__setattr__(self, "continuous_bounds", bounds)
        object.__setattr__(self, "delay_values", delays)

    @property
    def physical_bounds(self) -> np.ndarray:
        delay_bounds = np.asarray([[self.delay_values[0], self.delay_values[-1]]], dtype=np.float64)
        return np.concatenate((self.continuous_bounds, delay_bounds), axis=0)

    @property
    def active_mask(self) -> np.ndarray:
        bounds = self.physical_bounds
        return bounds[:, 1] > bounds[:, 0]

    @property
    def dimension(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def normalized_bounds(self) -> np.ndarray:
        return np.tile(np.asarray([[-1.0, 1.0]], dtype=np.float64), (self.dimension, 1))

    def encode(self, continuous: np.ndarray, delay_steps: int) -> np.ndarray:
        values = np.asarray(continuous, dtype=np.float64).reshape(-1)
        if values.size != self.continuous_bounds.shape[0]:
            raise ValueError(
                f"continuous vector must contain {self.continuous_bounds.shape[0]} values, got {values.size}"
            )
        nearest_delay = min(
            self.delay_values, key=lambda value: (abs(value - int(delay_steps)), value)
        )
        physical = np.concatenate((values, np.asarray([nearest_delay], dtype=np.float64)))
        bounds = self.physical_bounds
        physical = np.clip(physical, bounds[:, 0], bounds[:, 1])
        active = self.active_mask
        return (
            2.0 * (physical[active] - bounds[active, 0]) / (bounds[active, 1] - bounds[active, 0])
            - 1.0
        )

    def decode(self, normalized: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(normalized, dtype=np.float64).reshape(-1)
        if values.size != self.dimension:
            raise ValueError(
                f"normalized vector must contain {self.dimension} values, got {values.size}"
            )
        bounds = self.physical_bounds
        active = self.active_mask
        physical = np.mean(bounds, axis=1)
        clipped = np.clip(values, -1.0, 1.0)
        physical[active] = bounds[active, 0] + 0.5 * (clipped + 1.0) * (
            bounds[active, 1] - bounds[active, 0]
        )
        delay_coordinate = float(physical[-1])
        delay_steps = min(
            self.delay_values, key=lambda value: (abs(value - delay_coordinate), value)
        )
        return physical[:-1], int(delay_steps)


def default_cma_mean(
    search_space: CmaSearchSpace,
    fit_joints: list[int],
    initial_armature: np.ndarray,
) -> np.ndarray:
    """Build the default CMA mean with explicit armature and midpoint remaining parameters."""

    armature = coerce_vector(initial_armature, len(JOINT_NAMES), "initial_armature")
    expected_continuous_size = 4 * len(fit_joints)
    if search_space.continuous_bounds.shape[0] != expected_continuous_size:
        raise ValueError(
            f"search space must contain {expected_continuous_size} continuous values for {len(fit_joints)} joints"
        )
    continuous = np.mean(search_space.continuous_bounds, axis=1)
    continuous[: len(fit_joints)] = armature[fit_joints]
    delay_midpoint = 0.5 * (search_space.delay_values[0] + search_space.delay_values[-1])
    initial_delay = min(
        search_space.delay_values,
        key=lambda value: (abs(value - delay_midpoint), value),
    )
    return search_space.encode(continuous, initial_delay)


def source_bode_time_window(
    item: SourceData,
    time: np.ndarray,
    *,
    chirp_source_frequency_range_hz: tuple[float, float],
    fmin: float,
    fmax: float,
    fit_start: float,
    fit_end_margin: float,
) -> LinearChirpTimeWindow:
    duration_s = float(item.source.get("duration_s", float(time[-1] - time[0])))
    chirp_start_hz, chirp_end_hz = map(float, chirp_source_frequency_range_hz)
    return linear_chirp_time_window(
        time,
        chirp_start_hz=chirp_start_hz,
        chirp_end_hz=chirp_end_hz,
        chirp_duration_s=duration_s,
        fmin=fmin,
        fmax=fmax,
        fit_start=fit_start,
        fit_end_margin=fit_end_margin,
    )


def resolve_fixture_mode(
    fixture_mode: str | None,
    lock_non_source_joints: bool | None,
    *,
    default: str = "free",
) -> str:
    """Resolve the explicit fixture mode while preserving the legacy lock flag."""

    if default not in FIXTURE_MODES:
        raise ValueError(f"unsupported default fixture mode: {default!r}")
    if fixture_mode is not None and fixture_mode not in FIXTURE_MODES:
        raise ValueError(f"unsupported fixture mode: {fixture_mode!r}")
    if lock_non_source_joints is None:
        return default if fixture_mode is None else fixture_mode
    legacy_mode = "equality" if lock_non_source_joints else "free"
    if fixture_mode is not None and fixture_mode != legacy_mode:
        raise ValueError(
            f"fixture_mode={fixture_mode!r} conflicts with legacy "
            f"lock_non_source_joints={lock_non_source_joints}"
        )
    return legacy_mode


def simulation_schedule(
    time: np.ndarray,
    sim_hz: float,
    control_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-step physics times and the controller update schedule.

    Controller updates are placed on the first physics boundary at or after
    each nominal control tick.  This supports non-integer rate ratios such as
    1000/400: update intervals alternate between three and two physics steps.
    The returned control ticks are logical controller indices.  Replay uses the
    corresponding physics-boundary times for both command and state sampling,
    then holds the resulting torque until the next update.
    """
    samples = np.asarray(time, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("time must be a non-empty one-dimensional array")
    if samples.size > 1 and np.any(np.diff(samples) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if not np.isfinite(sim_hz) or not np.isfinite(control_hz) or sim_hz <= 0.0 or control_hz <= 0.0:
        raise ValueError("sim_hz and control_hz must be finite and positive")
    if control_hz > sim_hz + EPS:
        raise ValueError("control_hz cannot exceed sim_hz")

    duration = max(0.0, float(samples[-1] - samples[0]))
    num_sim_steps = int(math.ceil(duration * sim_hz - 1.0e-10))
    sim_times = np.arange(num_sim_steps + 1, dtype=np.float64) / float(sim_hz)
    sim_indices = np.arange(num_sim_steps + 1, dtype=np.float64)
    control_tick_at_step = np.floor(
        sim_indices * float(control_hz) / float(sim_hz) + 1.0e-12
    ).astype(np.int64)
    update_indices = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(control_tick_at_step)) + 1)
    )
    control_ticks = control_tick_at_step[update_indices]
    return sim_times, update_indices, control_ticks


def causal_previous_sample_zoh(
    sample_time: np.ndarray,
    values: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    """Sample a command causally using the latest source value at each query time."""

    source_time = np.asarray(sample_time, dtype=np.float64)
    source_values = np.asarray(values)
    queries = np.asarray(query_time, dtype=np.float64)
    if source_time.ndim != 1 or source_time.size == 0:
        raise ValueError("sample_time must be a non-empty one-dimensional array")
    if source_values.ndim == 0 or source_values.shape[0] != source_time.size:
        raise ValueError("values must have sample_time as its first dimension")
    if queries.ndim != 1:
        raise ValueError("query_time must be one-dimensional")
    if np.any(np.diff(source_time) <= 0.0):
        raise ValueError("sample_time must be strictly increasing")
    if not np.all(np.isfinite(source_time)) or not np.all(np.isfinite(queries)):
        raise ValueError("sample_time and query_time must contain only finite values")
    indices = np.searchsorted(source_time, queries, side="right") - 1
    indices = np.clip(indices, 0, source_time.size - 1)
    return source_values[indices]


def parse_float_pair(text: str) -> tuple[float, float]:
    parts = [float(item) for item in text.replace(" ", "").split(",") if item]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected LOW,HIGH, got {text!r}")
    if parts[1] < parts[0]:
        raise argparse.ArgumentTypeError(f"upper bound must be >= lower bound, got {text!r}")
    return float(parts[0]), float(parts[1])


def parse_int_list(text: str) -> list[int]:
    values = [int(item) for item in text.replace(" ", "").split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return sorted(set(values))


def make_out_dir(root: str | Path, run_name: str) -> Path:
    root_path = resolve_repo_path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = (
        "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_name).strip("._-")
        or "mujoco_pace_fit"
    )
    for index in range(1, 1000):
        out = root_path / f"{stamp}_{index:03d}_{safe}"
        try:
            out.mkdir()
            return out
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate output directory under {root_path}")


def atomic_write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    destination = ensure_parent(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = ensure_parent(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def candidate_progress_summary(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "seed": int(candidate["seed"]),
        "generation": int(candidate["generation"]),
        "member": int(candidate["member"]),
        "time_mse": float(candidate["score"]),
        "mean_rmse": float(candidate["mean_rmse"]),
        "source_mse": [float(value) for value in candidate["source_mse"]],
        "delay_steps": int(candidate["delay_steps"]),
        "normalized_x": np.asarray(candidate["normalized_x"], dtype=np.float64).tolist(),
        "x": np.asarray(candidate["x"], dtype=np.float64).tolist(),
        "params": candidate["params"],
    }


def write_generation_progress(
    *,
    candidate_csv: Path,
    optimizer_csv: Path,
    progress_json: Path,
    eval_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    best_candidate: dict[str, Any] | None,
    status: str,
    seed: int,
    generation: int,
) -> None:
    atomic_write_csv(candidate_csv, CANDIDATE_CSV_FIELDS, eval_rows)
    atomic_write_csv(optimizer_csv, OPTIMIZER_CSV_FIELDS, optimizer_rows)
    atomic_write_json(
        progress_json,
        {
            "format": "dr002_mujoco_pace_cma_progress_v1",
            "status": status,
            "seed": int(seed),
            "generation": int(generation),
            "completed_objective_evals": len(eval_rows),
            "best_candidate": candidate_progress_summary(best_candidate),
        },
    )


def load_manifest(truth_run_dir: Path) -> dict[str, Any]:
    path = truth_run_dir / "chirp_source_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_default_angles(manifest: dict[str, Any]) -> np.ndarray:
    return coerce_vector(
        manifest.get("default_angles", DEFAULT_ANGLES.tolist()), len(JOINT_NAMES), "default_angles"
    )


def truncate_payload(payload: dict[str, np.ndarray], limit_steps: int) -> dict[str, np.ndarray]:
    if limit_steps <= 0:
        return payload
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] >= limit_steps:
            out[key] = value[:limit_steps]
        else:
            out[key] = value
    return out


def source_target_kind(source: dict[str, Any], payload: dict[str, np.ndarray]) -> str:
    value = source.get("target_type")
    if value in {"position", "velocity"}:
        return str(value)
    active_joint_ids = source_active_joint_ids(source)
    if (
        any(joint_id in set(WHEEL_JOINT_IDS.tolist()) for joint_id in active_joint_ids)
        and payload.get("dof_vel") is not None
    ):
        return "velocity"
    return "position"


def parse_fit_joints(text: str, sources: list[SourceData]) -> list[int]:
    text = text.strip().lower()
    if text in {"active", "sources"}:
        return sorted(
            {joint_id for item in sources for joint_id in source_active_joint_ids(item.source)}
        )
    if text in {"all", "*"}:
        return list(range(len(JOINT_NAMES)))
    ids: list[int] = []
    name_to_id = {name: index for index, name in enumerate(JOINT_NAMES)}
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        if item.isdigit():
            ids.append(int(item))
        elif item in name_to_id:
            ids.append(name_to_id[item])
        else:
            raise ValueError(f"unknown fit joint {item!r}; use active/all/0..5/{JOINT_NAMES}")
    bad = [idx for idx in ids if idx < 0 or idx >= len(JOINT_NAMES)]
    if bad:
        raise ValueError(f"fit joint ids out of range [0,{len(JOINT_NAMES) - 1}]: {bad}")
    return sorted(set(ids))


def model_native_params(model_path: str | Path, *, fixed_base: bool) -> dict[str, Any]:
    model, _ = load_mujoco_model_with_mesh_fallback(model_path, fixed_base=fixed_base)
    _, qvel_ids = joint_addresses(model, JOINT_NAMES)
    return normalize_params(
        {
            "armature": np.asarray(model.dof_armature[qvel_ids], dtype=np.float64),
            "viscous_friction": np.asarray(model.dof_damping[qvel_ids], dtype=np.float64),
            "coulomb_friction": np.asarray(model.dof_frictionloss[qvel_ids], dtype=np.float64),
            "encoder_bias": np.zeros(len(JOINT_NAMES), dtype=np.float64),
            "motor_strength": np.ones(len(JOINT_NAMES), dtype=np.float64),
            "command_delay_steps": 0,
        },
        len(JOINT_NAMES),
    )


def with_candidate_params(
    base: dict[str, Any],
    fit_joints: list[int],
    x: np.ndarray,
    *,
    delay_steps: int | None = None,
    mirror_side_dynamics: bool = False,
    mirror_encoder_bias: bool = False,
) -> dict[str, Any]:
    n = len(JOINT_NAMES)
    params = normalize_params(base, n)
    m = len(fit_joints)
    cursor = 0
    for key in ("armature", "viscous_friction", "coulomb_friction", "encoder_bias"):
        arr = np.asarray(params[key], dtype=np.float64)
        arr[fit_joints] = x[cursor : cursor + m]
        params[key] = arr.tolist()
        cursor += m
    if delay_steps is None:
        if x.size != cursor + 1:
            raise ValueError(
                f"candidate must contain {cursor + 1} values when delay is embedded, got {x.size}"
            )
        params["command_delay_steps"] = int(round(float(x[cursor])))
    else:
        if x.size != cursor:
            raise ValueError(f"continuous candidate must contain {cursor} values, got {x.size}")
        params["command_delay_steps"] = int(delay_steps)

    if mirror_side_dynamics or mirror_encoder_bias:
        fit_set = set(fit_joints)
        keys = list(DYNAMIC_PARAM_KEYS) if mirror_side_dynamics else []
        if mirror_encoder_bias:
            keys.append("encoder_bias")
        for left_id, right_id in SIDE_PAIRS:
            if right_id in fit_set and left_id not in fit_set:
                source_id, target_id = right_id, left_id
            elif left_id in fit_set and right_id not in fit_set:
                source_id, target_id = left_id, right_id
            else:
                continue
            for key in keys:
                values = np.asarray(params[key], dtype=np.float64)
                values[target_id] = values[source_id]
                params[key] = values.tolist()
    return normalize_params(params, n)


def params_to_named_dict(params: dict[str, Any], key: str) -> dict[str, float]:
    values = np.asarray(params[key], dtype=np.float64)
    return {name: float(values[i]) for i, name in enumerate(JOINT_NAMES)}


class MujocoPaceReplay:
    def __init__(
        self,
        model_path: str | Path,
        *,
        sim_hz: float,
        control_hz: float,
        delay_semantics: str,
        control_mode: str,
        kp: np.ndarray,
        kd: np.ndarray,
        effort_limit: np.ndarray,
        torque_clip: bool,
        free_base: bool,
        root_z: float,
        enable_gravity: bool,
        fixture_mode: str | None = None,
        fixture_hold_kp: np.ndarray | None = None,
        fixture_hold_kd: np.ndarray | None = None,
        lock_wheel_positions: bool = True,
        lock_non_source_joints: bool | None = None,
        initial_joint_qpos: np.ndarray | None = None,
        initial_joint_qvel: np.ndarray | None = None,
        model_prepared: bool = False,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.fixture_mode = resolve_fixture_mode(fixture_mode, lock_non_source_joints)
        self.lock_non_source_joints = self.fixture_mode == "equality"
        self.lock_wheel_positions = bool(lock_wheel_positions)
        self.needs_joint_equalities = self.lock_non_source_joints or self.lock_wheel_positions
        if model_prepared:
            self.model, self.loaded_path = load_mujoco_model_with_mesh_fallback(
                model_path, fixed_base=False
            )
        elif self.needs_joint_equalities:
            lockable_path = make_lockable_joint_mjcf(model_path, fixed_base=not free_base)
            self.model, self.loaded_path = load_mujoco_model_with_mesh_fallback(
                lockable_path, fixed_base=False
            )
        else:
            self.model, self.loaded_path = load_mujoco_model_with_mesh_fallback(
                model_path, fixed_base=not free_base
            )
        self.sim_hz = float(sim_hz)
        self.control_hz = float(control_hz)
        if delay_semantics not in {"command", "torque"}:
            raise ValueError(f"unsupported delay semantics: {delay_semantics}")
        self.delay_semantics = delay_semantics
        self.model.opt.timestep = 1.0 / self.sim_hz
        if not enable_gravity:
            self.model.opt.gravity[:] = 0.0
        self.data = mujoco.MjData(self.model)
        self.qpos_ids, self.qvel_ids = joint_addresses(self.model, JOINT_NAMES)
        if self.needs_joint_equalities:
            self.lock_equality_ids = np.asarray(
                [
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_EQUALITY,
                        f"pace_lock_{joint_name}",
                    )
                    for joint_name in JOINT_NAMES
                ],
                dtype=np.int32,
            )
            if np.any(self.lock_equality_ids < 0):
                raise ValueError("lockable MJCF is missing one or more joint equality constraints")
        else:
            self.lock_equality_ids = np.zeros(0, dtype=np.int32)
        self.free_root = free_root_addresses(self.model)
        self.control_mode = control_mode
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.effort_limit = np.asarray(effort_limit, dtype=np.float64)
        self.torque_clip = bool(torque_clip)
        self.free_base = bool(free_base)
        self.root_z = float(root_z)
        self.initial_joint_qpos = self._optional_joint_state(
            initial_joint_qpos, "initial_joint_qpos"
        )
        self.initial_joint_qvel = self._optional_joint_state(
            initial_joint_qvel, "initial_joint_qvel"
        )
        self.fixture_hold_kp = self._fixture_gain_vector(
            fixture_hold_kp,
            DEFAULT_FIXTURE_HOLD_KP,
            "fixture_hold_kp",
        )
        self.fixture_hold_kd = self._fixture_gain_vector(
            fixture_hold_kd,
            DEFAULT_FIXTURE_HOLD_KD,
            "fixture_hold_kd",
        )
        self.fixture_target_qpos: np.ndarray | None = None
        self.fixture_active_joint_id: int | None = None
        self.fixture_active_joint_ids: tuple[int, ...] = ()
        self.locked_wheel_joint_ids = WHEEL_JOINT_IDS.copy()
        self._schedule_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _optional_joint_state(values: np.ndarray | None, name: str) -> np.ndarray | None:
        if values is None:
            return None
        state = np.asarray(values, dtype=np.float64).reshape(-1)
        if state.shape != (len(JOINT_NAMES),):
            raise ValueError(f"{name} must contain {len(JOINT_NAMES)} values")
        if not np.all(np.isfinite(state)):
            raise ValueError(f"{name} must contain only finite values")
        return state.copy()

    @staticmethod
    def _fixture_gain_vector(
        values: np.ndarray | None,
        default: np.ndarray,
        name: str,
    ) -> np.ndarray:
        gains = np.asarray(default if values is None else values, dtype=np.float64).reshape(-1)
        if gains.shape != (len(JOINT_NAMES),):
            raise ValueError(f"{name} must contain {len(JOINT_NAMES)} values")
        if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
            raise ValueError(f"{name} must contain finite non-negative values")
        return gains.copy()

    def pin_root(self) -> None:
        if self.free_base or self.free_root is None:
            return
        qadr, vadr = self.free_root
        self.data.qpos[qadr : qadr + 7] = np.array([0.0, 0.0, self.root_z, 1.0, 0.0, 0.0, 0.0])
        self.data.qvel[vadr : vadr + 6] = 0.0

    def project_locked_joint_states(self) -> None:
        """Project every strictly locked joint exactly onto its reset state."""

        if self.fixture_target_qpos is None:
            raise RuntimeError("reset must be called before projecting locked joint states")
        locked_ids = set(
            int(joint_id)
            for joint_id in (self.locked_wheel_joint_ids if self.lock_wheel_positions else ())
        )
        if self.lock_non_source_joints:
            locked_ids.update(
                joint_id
                for joint_id in range(len(JOINT_NAMES))
                if joint_id not in set(self.fixture_active_joint_ids)
            )
        if not locked_ids:
            return
        indices = np.asarray(sorted(locked_ids), dtype=np.int32)
        self.data.qpos[self.qpos_ids[indices]] = self.fixture_target_qpos[indices]
        self.data.qvel[self.qvel_ids[indices]] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        initial_measured: np.ndarray,
        params: dict[str, Any],
        *,
        active_joint_id: int | None = None,
        active_joint_ids: Sequence[int] | None = None,
        initial_measured_velocity: np.ndarray | None = None,
    ) -> None:
        active_ids = resolve_active_joint_arguments(
            active_joint_id=active_joint_id,
            active_joint_ids=active_joint_ids,
        )
        active_index = np.asarray(active_ids, dtype=np.int32)
        self.locked_wheel_joint_ids = np.asarray(
            [joint_id for joint_id in WHEEL_JOINT_IDS if joint_id not in set(active_ids)],
            dtype=np.int32,
        )
        self.mujoco.mj_resetData(self.model, self.data)
        arm = np.asarray(params["armature"], dtype=np.float64)
        visc = np.asarray(params["viscous_friction"], dtype=np.float64)
        fric = np.asarray(params["coulomb_friction"], dtype=np.float64)
        bias = np.asarray(params["encoder_bias"], dtype=np.float64)
        self.model.dof_armature[self.qvel_ids] = arm
        self.model.dof_damping[self.qvel_ids] = visc
        self.model.dof_frictionloss[self.qvel_ids] = fric
        self.pin_root()
        measured_qpos = self._optional_joint_state(initial_measured, "initial_measured")
        assert measured_qpos is not None
        if self.initial_joint_qpos is None:
            reset_qpos = measured_qpos + bias
        else:
            reset_qpos = self.initial_joint_qpos.copy()
            reset_qpos[active_index] = measured_qpos[active_index] + bias[active_index]
        self.data.qpos[self.qpos_ids] = reset_qpos
        reset_qvel = (
            np.zeros(len(JOINT_NAMES), dtype=np.float64)
            if self.initial_joint_qvel is None
            else self.initial_joint_qvel.copy()
        )
        if initial_measured_velocity is not None:
            measured_qvel = self._optional_joint_state(
                initial_measured_velocity,
                "initial_measured_velocity",
            )
            assert measured_qvel is not None
            reset_qvel[active_index] = measured_qvel[active_index]
        if self.lock_wheel_positions and self.locked_wheel_joint_ids.size:
            reset_qvel[self.locked_wheel_joint_ids] = 0.0
        self.data.qvel[self.qvel_ids] = reset_qvel
        self.fixture_active_joint_ids = active_ids
        self.fixture_active_joint_id = active_ids[0] if len(active_ids) == 1 else None
        self.fixture_target_qpos = self.data.qpos[self.qpos_ids].copy()
        if self.needs_joint_equalities:
            self.data.eq_active[self.lock_equality_ids] = False
            self.model.eq_data[self.lock_equality_ids, 0] = self.data.qpos[self.qpos_ids]
            if self.lock_non_source_joints:
                self.data.eq_active[self.lock_equality_ids] = True
                self.data.eq_active[self.lock_equality_ids[active_index]] = False
            elif self.lock_wheel_positions and self.locked_wheel_joint_ids.size:
                self.data.eq_active[self.lock_equality_ids[self.locked_wheel_joint_ids]] = True
        self.mujoco.mj_forward(self.model, self.data)
        self.project_locked_joint_states()

    def torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        des_pos: np.ndarray,
        des_vel: np.ndarray,
        bias: np.ndarray,
        *,
        active_joint_id: int | None = None,
        active_joint_ids: Sequence[int] | None = None,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        fixture_hold_kp: np.ndarray | None = None,
        fixture_hold_kd: np.ndarray | None = None,
    ) -> np.ndarray:
        kp_values = self.kp if kp is None else np.asarray(kp, dtype=np.float64)
        kd_values = self.kd if kd is None else np.asarray(kd, dtype=np.float64)
        if self.control_mode == "position":
            raw = kp_values * (des_pos + bias - q) - kd_values * qd
        elif self.control_mode == "mixed":
            raw = np.zeros_like(q)
            raw[LEG_JOINT_IDS] = (
                kp_values[LEG_JOINT_IDS]
                * (des_pos[LEG_JOINT_IDS] + bias[LEG_JOINT_IDS] - q[LEG_JOINT_IDS])
                - kd_values[LEG_JOINT_IDS] * qd[LEG_JOINT_IDS]
            )
            raw[WHEEL_JOINT_IDS] = kd_values[WHEEL_JOINT_IDS] * (
                des_vel[WHEEL_JOINT_IDS] - qd[WHEEL_JOINT_IDS]
            )
        else:
            raise ValueError(f"unsupported control mode: {self.control_mode}")
        if self.fixture_mode == "equality":
            active_ids = resolve_active_joint_arguments(
                active_joint_id=active_joint_id,
                active_joint_ids=active_joint_ids,
            )
            non_source = np.ones(len(JOINT_NAMES), dtype=bool)
            non_source[np.asarray(active_ids, dtype=np.int32)] = False
            raw[non_source] = 0.0
        elif self.fixture_mode == "high-impedance":
            active_ids = resolve_active_joint_arguments(
                active_joint_id=active_joint_id,
                active_joint_ids=active_joint_ids,
            )
            if self.fixture_target_qpos is None:
                raise RuntimeError(
                    "reset must be called before computing high-impedance fixture torque"
                )
            non_source = np.ones(len(JOINT_NAMES), dtype=bool)
            non_source[np.asarray(active_ids, dtype=np.int32)] = False
            hold_kp = self.fixture_hold_kp if fixture_hold_kp is None else fixture_hold_kp
            hold_kd = self.fixture_hold_kd if fixture_hold_kd is None else fixture_hold_kd
            hold_tau = hold_kp * (self.fixture_target_qpos - q) - hold_kd * qd
            raw[non_source] = hold_tau[non_source]
        if self.lock_wheel_positions and self.locked_wheel_joint_ids.size:
            raw[self.locked_wheel_joint_ids] = 0.0
        return np.clip(raw, -self.effort_limit, self.effort_limit) if self.torque_clip else raw

    def replay(
        self, item: SourceData, params: dict[str, Any], *, log_full: bool = False
    ) -> dict[str, np.ndarray]:
        payload = item.payload
        time = np.asarray(payload["time"], dtype=np.float64)
        des_pos = np.asarray(payload["des_dof_pos"], dtype=np.float64)
        des_vel = (
            np.zeros_like(des_pos)
            if payload.get("des_dof_vel") is None
            else np.asarray(payload["des_dof_vel"], dtype=np.float64)
        )
        real_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
        real_vel = (
            np.zeros_like(real_pos)
            if payload.get("dof_vel") is None
            else np.asarray(payload["dof_vel"], dtype=np.float64)
        )
        bias = np.asarray(params["encoder_bias"], dtype=np.float64)
        delay = coerce_shared_delay_steps(params["command_delay_steps"], "command_delay_steps")
        active_joint_ids = source_active_joint_ids(item.source)
        active_index = np.asarray(active_joint_ids, dtype=np.int32)
        source_kp = source_controller_gain(item.source, "kp", self.kp)
        source_kd = source_controller_gain(item.source, "kd", self.kd)
        source_fixture_kp = source_fixture_gain(
            item.source, "fixture_hold_kp", self.fixture_hold_kp
        )
        source_fixture_kd = source_fixture_gain(
            item.source, "fixture_hold_kd", self.fixture_hold_kd
        )
        self.reset(
            real_pos[0],
            params,
            active_joint_ids=active_joint_ids,
            initial_measured_velocity=real_vel[0],
        )
        relative_time = time - time[0]
        cache_key = id(item)
        cached = self._schedule_cache.get(cache_key)
        if cached is None:
            sim_time, update_indices, _ = simulation_schedule(
                relative_time, self.sim_hz, self.control_hz
            )
            control_time = sim_time[update_indices]
            control_des_pos = causal_previous_sample_zoh(relative_time, des_pos, control_time)
            control_des_vel = causal_previous_sample_zoh(relative_time, des_vel, control_time)
            cached = (sim_time, update_indices, control_des_pos, control_des_vel)
            self._schedule_cache[cache_key] = cached
        sim_time, update_indices, control_des_pos, control_des_vel = cached

        sim_response = np.zeros(
            (sim_time.size, len(active_joint_ids)),
            dtype=np.float64,
        )
        sim_q = np.zeros((sim_time.size, len(JOINT_NAMES)), dtype=np.float64) if log_full else None
        sim_qd = np.zeros_like(sim_q) if log_full else None
        sim_tau = np.zeros_like(sim_q) if log_full else None
        sim_applied_des = np.zeros_like(sim_q) if log_full else None
        sim_applied_desvel = np.zeros_like(sim_q) if log_full else None
        truth_matrix = (
            real_vel[:, active_index]
            if item.target_kind == "velocity"
            else real_pos[:, active_index]
        )
        torque_delay = deque(np.zeros(len(JOINT_NAMES), dtype=np.float64) for _ in range(delay))
        command_delay = SharedCommandDelayBuffer(delay if self.delay_semantics == "command" else 0)
        applied_tau = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        applied_des = control_des_pos[0].copy()
        applied_desvel = control_des_vel[0].copy()
        update_cursor = 0

        for step in range(sim_time.size):
            q = self.data.qpos[self.qpos_ids].copy()
            qd = self.data.qvel[self.qvel_ids].copy()
            measured_q = q - bias
            sim_response[step] = (
                qd[active_index] if item.target_kind == "velocity" else measured_q[active_index]
            )
            if log_full:
                assert sim_q is not None and sim_qd is not None
                sim_q[step] = measured_q
                sim_qd[step] = qd

            if update_cursor < update_indices.size and step == int(update_indices[update_cursor]):
                command_frame = np.column_stack(
                    (control_des_pos[update_cursor], control_des_vel[update_cursor])
                )
                delayed_frame = command_delay.push(command_frame)
                applied_des = delayed_frame[:, 0]
                applied_desvel = delayed_frame[:, 1]
                motor_tau = self.torque(
                    q,
                    qd,
                    applied_des,
                    applied_desvel,
                    bias,
                    active_joint_ids=active_joint_ids,
                    kp=source_kp,
                    kd=source_kd,
                    fixture_hold_kp=source_fixture_kp,
                    fixture_hold_kd=source_fixture_kd,
                )
                if self.delay_semantics == "torque" and delay > 0:
                    torque_delay.append(motor_tau)
                    applied_tau = torque_delay.popleft()
                else:
                    applied_tau = motor_tau
                update_cursor += 1
            if log_full:
                assert (
                    sim_tau is not None
                    and sim_applied_des is not None
                    and sim_applied_desvel is not None
                )
                sim_tau[step] = applied_tau
                sim_applied_des[step] = applied_des
                sim_applied_desvel[step] = applied_desvel

            if step + 1 == sim_time.size:
                break
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.qvel_ids] = applied_tau
            self.mujoco.mj_step(self.model, self.data)
            self.pin_root()
            self.project_locked_joint_states()

        response_matrix = np.column_stack(
            [
                np.interp(relative_time, sim_time, sim_response[:, column])
                for column in range(len(active_joint_ids))
            ]
        )
        response: np.ndarray
        truth: np.ndarray
        if len(active_joint_ids) == 1:
            response = response_matrix[:, 0]
            truth = truth_matrix[:, 0]
        else:
            response = response_matrix
            truth = truth_matrix
        out = {
            "time": time,
            "response": response,
            "truth": truth,
            "active_joint_ids": active_index.copy(),
        }
        if log_full:
            assert (
                sim_q is not None
                and sim_qd is not None
                and sim_tau is not None
                and sim_applied_des is not None
                and sim_applied_desvel is not None
            )
            q_log = np.column_stack(
                [
                    np.interp(relative_time, sim_time, sim_q[:, idx])
                    for idx in range(len(JOINT_NAMES))
                ]
            )
            qd_log = np.column_stack(
                [
                    np.interp(relative_time, sim_time, sim_qd[:, idx])
                    for idx in range(len(JOINT_NAMES))
                ]
            )
            tau_indices = np.searchsorted(sim_time, relative_time, side="right") - 1
            tau_indices = np.clip(tau_indices, 0, sim_time.size - 1)
            tau_log = sim_tau[tau_indices]
            out.update(
                {
                    "q": q_log,
                    "qd": qd_log,
                    "tau": tau_log,
                    "des": des_pos,
                    "desvel": des_vel,
                    "applied_des": sim_applied_des[tau_indices],
                    "applied_desvel": sim_applied_desvel[tau_indices],
                }
            )
        return out


def _masked_trajectory_pair(
    response: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(response.shape[0], truth.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    response_values = np.asarray(response[:n], dtype=np.float64)
    y = np.asarray(truth[:n], dtype=np.float64)
    if mask is None:
        return response_values, y
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if selected.shape[0] != n:
        raise ValueError(f"trajectory score mask must contain {n} values, got {selected.shape[0]}")
    return response_values[selected], y[selected]


def normalized_mse(
    response: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    response_values, y = _masked_trajectory_pair(response, truth, mask)
    if y.size == 0:
        return math.inf
    err = response_values - y
    scale = max(float(np.std(y)), 1.0e-3)
    return float(np.mean(np.square(err / scale)))


def raw_mse(
    response: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Return the PACE trajectory loss for one source and score window."""

    response_values, truth_values = _masked_trajectory_pair(response, truth, mask)
    if truth_values.size == 0:
        return math.inf
    err = response_values - truth_values
    return float(np.mean(np.square(err)))


def raw_rmse(
    response: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    return float(np.sqrt(raw_mse(response, truth, mask)))


@dataclass
class CandidateWorkerContext:
    replay: MujocoPaceReplay
    search_space: CmaSearchSpace
    base_params: dict[str, Any]
    fit_joints: tuple[int, ...]
    sources: tuple[SourceData, ...]
    mirror_side_dynamics: bool
    mirror_encoder_bias: bool


_CANDIDATE_WORKER_CONTEXT: CandidateWorkerContext | None = None


def evaluate_candidate_with_context(
    normalized_vector: np.ndarray,
    context: CandidateWorkerContext,
) -> dict[str, Any]:
    vector, delay_steps = context.search_space.decode(normalized_vector)
    params = with_candidate_params(
        context.base_params,
        list(context.fit_joints),
        vector,
        delay_steps=delay_steps,
        mirror_side_dynamics=context.mirror_side_dynamics,
        mirror_encoder_bias=context.mirror_encoder_bias,
    )
    mse_scores = []
    rmses = []
    for item in context.sources:
        result = context.replay.replay(item, params, log_full=False)
        if item.score_time_window is None:
            raise RuntimeError(
                f"source {source_label(item.source)} is missing its time-domain score window"
            )
        score_mask = item.score_time_window.mask
        for joint_id in source_active_joint_ids(item.source):
            response = source_result_trace(item, result, "response", joint_id)
            truth = source_result_trace(item, result, "truth", joint_id)
            mse_scores.append(raw_mse(response, truth, score_mask))
            rmses.append(raw_rmse(response, truth, score_mask))
    score = float(np.mean(mse_scores))
    if not math.isfinite(score):
        score = math.inf
    return {
        "score": score,
        "mean_rmse": float(np.mean(rmses)),
        "source_mse": mse_scores,
        "delay_steps": int(delay_steps),
        "normalized_x": np.asarray(normalized_vector, dtype=np.float64).copy(),
        "x": vector,
        "params": params,
    }


def initialize_candidate_worker(
    replay_kwargs: dict[str, Any],
    search_space: CmaSearchSpace,
    base_params: dict[str, Any],
    fit_joints: tuple[int, ...],
    sources: tuple[SourceData, ...],
    mirror_side_dynamics: bool,
    mirror_encoder_bias: bool,
    inherit_parent_context: bool,
) -> None:
    """Build one MuJoCo replay instance and retain it for the worker lifetime."""

    global _CANDIDATE_WORKER_CONTEXT
    if inherit_parent_context:
        if _CANDIDATE_WORKER_CONTEXT is None:
            raise RuntimeError("forked candidate worker did not inherit the parent replay context")
        return
    _CANDIDATE_WORKER_CONTEXT = CandidateWorkerContext(
        replay=MujocoPaceReplay(**replay_kwargs),
        search_space=search_space,
        base_params=base_params,
        fit_joints=fit_joints,
        sources=sources,
        mirror_side_dynamics=mirror_side_dynamics,
        mirror_encoder_bias=mirror_encoder_bias,
    )


def evaluate_candidate_in_worker(task: tuple[int, np.ndarray]) -> tuple[int, dict[str, Any]]:
    """Evaluate one population member in a persistent process worker."""

    if _CANDIDATE_WORKER_CONTEXT is None:
        raise RuntimeError("candidate worker was not initialized")
    member, normalized_vector = task
    return int(member), evaluate_candidate_with_context(
        normalized_vector,
        _CANDIDATE_WORKER_CONTEXT,
    )


def write_replay_csv(path: Path, replay: dict[str, np.ndarray]) -> None:
    header = (
        ["time"]
        + [f"q_{name}" for name in JOINT_NAMES]
        + [f"qd_{name}" for name in JOINT_NAMES]
        + [f"des_{name}" for name in JOINT_NAMES]
        + [f"desvel_{name}" for name in JOINT_NAMES]
        + [f"applied_des_{name}" for name in JOINT_NAMES]
        + [f"applied_desvel_{name}" for name in JOINT_NAMES]
        + [f"tau_{name}" for name in JOINT_NAMES]
    )
    time = np.asarray(replay["time"])
    q = np.asarray(replay["q"])
    qd = np.asarray(replay["qd"])
    des = np.asarray(replay["des"])
    desvel = np.asarray(replay["desvel"])
    applied_des = np.asarray(replay["applied_des"])
    applied_desvel = np.asarray(replay["applied_desvel"])
    tau = np.asarray(replay["tau"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t in enumerate(time):
            writer.writerow(
                [float(t)]
                + q[i].tolist()
                + qd[i].tolist()
                + des[i].tolist()
                + desvel[i].tolist()
                + applied_des[i].tolist()
                + applied_desvel[i].tolist()
                + tau[i].tolist()
            )


def write_bode_report(
    path: Path,
    comparisons: list[dict[str, Any]],
    fmin: float,
    fmax: float,
    *,
    title: str,
) -> None:
    """Write time, magnitude, and phase comparisons for fitted sources."""
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-we11-pace")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(comparisons), 3, figsize=(20.0, 4.8 * len(comparisons)), squeeze=False
    )
    for row, comparison in enumerate(comparisons):
        time = np.asarray(comparison["time"], dtype=np.float64)
        command = np.asarray(comparison["command"], dtype=np.float64)
        applied_command = np.asarray(comparison["applied_command"], dtype=np.float64)
        truth = np.asarray(comparison["truth"], dtype=np.float64)
        response = np.asarray(comparison["response"], dtype=np.float64)
        fit_mask = np.asarray(comparison["fit_mask"], dtype=bool)
        joint = str(comparison["joint"])
        target_kind = str(comparison["target_kind"])

        axes[row, 0].plot(time, command, color="0.65", linewidth=0.7, label="Raw command")
        axes[row, 0].plot(
            time,
            applied_command,
            color="#CC79A7",
            linewidth=0.8,
            linestyle="--",
            label="Applied command (after FIFO)",
        )
        axes[row, 0].plot(time, truth, color="#0072B2", linewidth=0.9, label="Real")
        axes[row, 0].plot(time, response, color="#D55E00", linewidth=0.9, label="MuJoCo")
        axes[row, 0].set_title(f"{joint}: {target_kind} response")
        axes[row, 0].set_xlabel("Time [s]")
        axes[row, 0].set_ylabel(
            "Velocity [rad/s]" if target_kind == "velocity" else "Position [rad]"
        )
        axes[row, 0].legend(loc="best")

        real_frequency, real_magnitude, real_phase = transfer_bode(
            command[fit_mask], truth[fit_mask], time[fit_mask]
        )
        sim_frequency, sim_magnitude, sim_phase = transfer_bode(
            command[fit_mask], response[fit_mask], time[fit_mask]
        )
        real_mask = (real_frequency >= fmin) & (real_frequency <= fmax)
        sim_mask = (sim_frequency >= fmin) & (sim_frequency <= fmax)
        axes[row, 1].semilogx(
            real_frequency[real_mask], real_magnitude[real_mask], color="#0072B2", label="Real"
        )
        axes[row, 1].semilogx(
            sim_frequency[sim_mask], sim_magnitude[sim_mask], color="#D55E00", label="MuJoCo"
        )
        axes[row, 1].set_title(f"{joint}: Bode magnitude")
        axes[row, 1].set_xlabel("Frequency [Hz]")
        axes[row, 1].set_ylabel("Magnitude [dB]")
        axes[row, 1].legend(loc="best")

        axes[row, 2].semilogx(
            real_frequency[real_mask], real_phase[real_mask], color="#0072B2", label="Real"
        )
        axes[row, 2].semilogx(
            sim_frequency[sim_mask], sim_phase[sim_mask], color="#D55E00", label="MuJoCo"
        )
        axes[row, 2].set_title(f"{joint}: Bode phase")
        axes[row, 2].set_xlabel("Frequency [Hz]")
        axes[row, 2].set_ylabel("Phase [deg]")
        axes[row, 2].legend(loc="best")
        for axis in axes[row]:
            axis.grid(True, which="both", alpha=0.25)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth-run-dir",
        required=True,
        help="Imported real/pseudo-real sweep directory with chirp_source_manifest.json.",
    )
    parser.add_argument("--out-root", default="logs/dr002_mujoco_pace_fit")
    parser.add_argument("--run-name", default="mujoco_pace_fit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MJCF path.")
    parser.add_argument("--control-mode", choices=("auto", "position", "mixed"), default="auto")
    parser.add_argument(
        "--source-joints",
        default="active",
        help="Manifest sources to score: active/all or comma-separated 0-based joint ids/names.",
    )
    parser.add_argument(
        "--fit-joints", default="active", help="active/all or comma-separated joint ids/names."
    )
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Legacy physics timestep override. Prefer --sim-hz; cannot be combined with it.",
    )
    parser.add_argument(
        "--sim-hz",
        type=float,
        default=None,
        help="MuJoCo physics rate. Defaults to the imported truth sample rate for legacy compatibility.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="PD update rate and delay-tick rate. Defaults to the imported truth sample rate.",
    )
    parser.add_argument(
        "--delay-semantics",
        choices=("command", "torque"),
        default="command",
        help=(
            "Delay the complete six-joint command frame before PD (the WE default), "
            "or explicitly select the legacy clipped-motor-torque comparison."
        ),
    )
    parser.add_argument(
        "--bode-freq-range",
        type=float,
        nargs=2,
        default=[0.5, 4.5],
        metavar=("FMIN", "FMAX"),
    )
    parser.add_argument(
        "--time-score-freq-range",
        "--score-freq-range",
        dest="time_score_freq_range",
        type=float,
        nargs=2,
        default=[0.5, 4.5],
        metavar=("FMIN", "FMAX"),
        help=(
            "Strict linear-chirp frequency window used by the time-domain CMA objective. "
            "The full trajectory is still replayed before applying this score mask."
        ),
    )
    parser.add_argument(
        "--chirp-source-freq-range",
        type=float,
        nargs=2,
        default=[0.1, 5.0],
        metavar=("START_HZ", "END_HZ"),
        help=(
            "Linear chirp start/end frequencies used to map both the time-domain "
            "score window and the post-fit Bode diagnostic window."
        ),
    )
    parser.add_argument(
        "--bode-command-psd-threshold-db",
        type=float,
        default=DEFAULT_COMMAND_PSD_THRESHOLD_DB,
        help="Keep only reference-command Welch bins at least this many dB below its PSD peak.",
    )
    parser.add_argument("--bode-fit-start", type=float, default=0.5)
    parser.add_argument("--bode-fit-end-margin", type=float, default=0.25)
    parser.add_argument("--root-z", type=float, default=0.35)
    parser.add_argument("--free-base", action="store_true")
    parser.add_argument(
        "--initial-joint-qpos",
        type=float,
        nargs=6,
        default=None,
        metavar=("L_THIGH", "L_CALF", "L_WHEEL", "R_THIGH", "R_CALF", "R_WHEEL"),
        help="Physical MuJoCo joint qpos used before every replay; no encoder bias is added.",
    )
    parser.add_argument(
        "--initial-joint-qvel",
        type=float,
        nargs=6,
        default=None,
        metavar=("L_THIGH", "L_CALF", "L_WHEEL", "R_THIGH", "R_CALF", "R_WHEEL"),
        help="Physical MuJoCo joint qvel used before every replay.",
    )
    parser.add_argument(
        "--fixture-mode",
        choices=FIXTURE_MODES,
        default=None,
        help=(
            "Non-source joint fixture: free, equality constraints, or independent high-impedance "
            "PD about each replay's reset pose. Single-joint fits can use equality locks."
        ),
    )
    parser.add_argument(
        "--fixture-hold-kp",
        type=float,
        nargs=6,
        default=DEFAULT_FIXTURE_HOLD_KP.tolist(),
        metavar=("L_THIGH", "L_CALF", "L_WHEEL", "R_THIGH", "R_CALF", "R_WHEEL"),
        help="Independent non-source hold Kp; defaults to 20 for legs and 0 for wheels.",
    )
    parser.add_argument(
        "--fixture-hold-kd",
        type=float,
        nargs=6,
        default=DEFAULT_FIXTURE_HOLD_KD.tolist(),
        metavar=("L_THIGH", "L_CALF", "L_WHEEL", "R_THIGH", "R_CALF", "R_WHEEL"),
        help="Independent non-source hold Kd; defaults to 1 for legs and 0.1 for wheels.",
    )
    parser.add_argument(
        "--lock-wheel-positions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Lock every non-source wheel position using MuJoCo joint equalities plus "
            "exact state projection after every physics step "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--require-rk4",
        action="store_true",
        help="Fail unless the loaded MJCF explicitly resolves to MuJoCo's RK4 integrator.",
    )
    parser.add_argument(
        "--lock-non-source-joints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Legacy compatibility flag: true selects --fixture-mode equality and false selects free. "
            "Do not combine it with a conflicting explicit fixture mode."
        ),
    )
    parser.add_argument("--enable-gravity", action="store_true")
    parser.add_argument("--no-torque-clip", action="store_true")
    parser.add_argument("--kp", type=float, nargs=6, default=IDENTIFIED_KP.tolist())
    parser.add_argument("--kd", type=float, nargs=6, default=IDENTIFIED_KD.tolist())
    parser.add_argument(
        "--effort-limit", type=float, nargs=6, default=DEFAULT_EFFORT_LIMIT.tolist()
    )
    parser.add_argument(
        "--initial-armature",
        type=float,
        nargs=6,
        default=DEFAULT_CMA_INITIAL_ARMATURE.tolist(),
        metavar=("L_THIGH", "L_CALF", "L_WHEEL", "R_THIGH", "R_CALF", "R_WHEEL"),
        help="Default CMA mean for all six armatures; ignored when --initial-params is supplied.",
    )
    parser.add_argument("--armature-bounds", type=parse_float_pair, default=(1.0e-5, 0.08))
    parser.add_argument("--viscous-bounds", type=parse_float_pair, default=(0.0, 1.5))
    parser.add_argument("--friction-bounds", type=parse_float_pair, default=(0.0, 0.35))
    parser.add_argument("--bias-bounds", type=parse_float_pair, default=(-0.12, 0.12))
    parser.add_argument("--delay-bounds", type=parse_float_pair, default=(0.0, 12.0))
    parser.add_argument(
        "--delay-values",
        type=parse_int_list,
        default=None,
        help=(
            "Explicit delay candidates in control ticks, e.g. 8,9,10. Their location is selected "
            "by --delay-semantics; defaults to every integer in --delay-bounds."
        ),
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=64,
        help="CMA-ES candidates per generation.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=40,
        help="Maximum CMA-ES generations per seed.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.35,
        help="Initial CMA-ES sigma in normalized [-1,1] coordinates.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Stop when one generation's relative score spread falls below this value; set 0 to disable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Persistent MuJoCo worker processes. Each process owns an independent model/data instance.",
    )
    available_start_methods = multiprocessing.get_all_start_methods()
    parser.add_argument(
        "--mp-start-method",
        choices=available_start_methods,
        default="fork" if "fork" in available_start_methods else "spawn",
        help="Multiprocessing start method. Linux defaults to fork for lower startup time and shared read-only pages.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=None,
        help="CMA-ES seeds. Defaults to the single --seed value.",
    )
    parser.add_argument(
        "--initial-params",
        nargs="*",
        default=None,
        help="Optional prior PACE JSON/PT files used as CMA means, cycled across seeds.",
    )
    parser.add_argument(
        "--mirror-side-dynamics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror fitted armature/damping/friction to the unfitted symmetric side (default: enabled).",
    )
    parser.add_argument(
        "--mirror-encoder-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also mirror fitted encoder bias; disabled by default because sensor zero offsets are joint-specific.",
    )
    return parser.parse_args()


def main() -> None:
    global _CANDIDATE_WORKER_CONTEXT

    args = parse_args()
    fixture_mode = resolve_fixture_mode(
        args.fixture_mode,
        args.lock_non_source_joints,
        default="equality",
    )
    truth_run_dir = resolve_repo_path(args.truth_run_dir)
    manifest = load_manifest(truth_run_dir)
    control_mode = (
        manifest.get("control_mode", "position")
        if args.control_mode == "auto"
        else args.control_mode
    )
    default_angles = manifest_default_angles(manifest)
    initial_joint_qpos = (
        None
        if args.initial_joint_qpos is None
        else coerce_vector(args.initial_joint_qpos, len(JOINT_NAMES), "initial_joint_qpos")
    )
    initial_joint_qvel = (
        None
        if args.initial_joint_qvel is None
        else coerce_vector(args.initial_joint_qvel, len(JOINT_NAMES), "initial_joint_qvel")
    )
    if initial_joint_qpos is not None and not np.allclose(
        initial_joint_qpos, default_angles, atol=1.0e-9
    ):
        raise ValueError(
            "--initial-joint-qpos must match the imported manifest default_angles; "
            "re-import the raw sweep at the requested physical fixture pose"
        )
    kp = coerce_vector(args.kp, len(JOINT_NAMES), "kp")
    kd = coerce_vector(args.kd, len(JOINT_NAMES), "kd")
    effort_limit = coerce_vector(args.effort_limit, len(JOINT_NAMES), "effort_limit")

    sources: list[SourceData] = []
    dts: list[float] = []
    for source in manifest.get("sources", []):
        payload = truncate_payload(load_chirp_data(truth_run_dir / source["pt"]), args.limit_steps)
        kind = source_target_kind(source, payload)
        sources.append(SourceData(source=source, payload=payload, target_kind=kind))
        time = np.asarray(payload["time"], dtype=np.float64)
        if time.shape[0] > 1:
            dts.append(float(np.median(np.diff(time))))
    if not sources:
        raise ValueError(f"no sources found in {truth_run_dir / 'chirp_source_manifest.json'}")
    source_joint_ids = set(parse_fit_joints(args.source_joints, sources))
    sources = [
        item
        for item in sources
        if source_joint_ids.intersection(source_active_joint_ids(item.source))
    ]
    if not sources:
        raise ValueError(f"--source-joints={args.source_joints!r} selected no manifest sources")
    dts = []
    for item in sources:
        time = np.asarray(item.payload["time"], dtype=np.float64)
        if time.shape[0] > 1:
            dts.append(float(np.median(np.diff(time))))
    if not dts:
        raise ValueError(
            "cannot infer truth sample rate from fewer than two samples; pass data with at least two timestamps"
        )
    truth_dt = float(np.median(dts))
    truth_sample_hz = 1.0 / truth_dt
    if args.dt is not None and args.sim_hz is not None:
        raise ValueError("--dt and --sim-hz are mutually exclusive")
    if args.dt is not None and args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    sim_hz = (
        float(args.sim_hz)
        if args.sim_hz is not None
        else (1.0 / float(args.dt) if args.dt is not None else truth_sample_hz)
    )
    control_hz = float(args.control_hz) if args.control_hz is not None else truth_sample_hz
    if not np.isfinite(sim_hz) or not np.isfinite(control_hz) or sim_hz <= 0.0 or control_hz <= 0.0:
        raise ValueError("--sim-hz and --control-hz must be finite and positive")
    if control_hz > sim_hz + EPS:
        raise ValueError("--control-hz cannot exceed --sim-hz")
    bode_fmin, bode_fmax = map(float, args.bode_freq_range)
    if bode_fmin <= 0.0 or bode_fmax <= bode_fmin:
        raise ValueError("--bode-freq-range must be positive and increasing")
    score_fmin, score_fmax = map(float, args.time_score_freq_range)
    if score_fmin <= 0.0 or score_fmax <= score_fmin:
        raise ValueError("--time-score-freq-range must be positive and increasing")
    chirp_start_hz, chirp_end_hz = map(float, args.chirp_source_freq_range)
    if (
        not all(math.isfinite(value) and value > 0.0 for value in (chirp_start_hz, chirp_end_hz))
        or chirp_start_hz == chirp_end_hz
    ):
        raise ValueError("--chirp-source-freq-range values must be positive, finite, and different")
    if bode_fmin < min(chirp_start_hz, chirp_end_hz) or bode_fmax > max(
        chirp_start_hz, chirp_end_hz
    ):
        raise ValueError("--bode-freq-range must lie inside --chirp-source-freq-range")
    if score_fmin < min(chirp_start_hz, chirp_end_hz) or score_fmax > max(
        chirp_start_hz, chirp_end_hz
    ):
        raise ValueError("--time-score-freq-range must lie inside --chirp-source-freq-range")
    if (
        not math.isfinite(args.bode_command_psd_threshold_db)
        or args.bode_command_psd_threshold_db > 0.0
    ):
        raise ValueError("--bode-command-psd-threshold-db must be finite and no greater than 0")
    if args.bode_fit_start < 0.0 or args.bode_fit_end_margin < 0.0:
        raise ValueError("Bode fit-window margins must be non-negative")
    requested_fit_joints = parse_fit_joints(args.fit_joints, sources)
    fit_joints = canonicalize_mirrored_fit_joints(
        requested_fit_joints,
        mirror_side_dynamics=bool(args.mirror_side_dynamics),
    )
    wheel_source_ids = sorted(
        {
            joint_id
            for item in sources
            for joint_id in source_active_joint_ids(item.source)
            if joint_id in set(WHEEL_JOINT_IDS.tolist())
        }
    )
    if wheel_source_ids and control_mode != "mixed":
        raise ValueError("wheel source fitting requires --control-mode mixed")
    for item in sources:
        active_ids = source_active_joint_ids(item.source)
        if any(joint_id in set(WHEEL_JOINT_IDS.tolist()) for joint_id in active_ids):
            if item.target_kind != "velocity":
                raise ValueError(
                    f"wheel source {source_label(item.source)} must declare target_type=velocity"
                )
    chirp_source_frequency_range = tuple(map(float, args.chirp_source_freq_range))
    time_score_reference_windows: list[dict[str, Any]] = []
    for item in sources:
        active_joint_ids = source_active_joint_ids(item.source)
        time = np.asarray(item.payload["time"], dtype=np.float64)
        score_time_window = source_bode_time_window(
            item,
            time,
            chirp_source_frequency_range_hz=chirp_source_frequency_range,
            fmin=score_fmin,
            fmax=score_fmax,
            fit_start=0.0,
            fit_end_margin=0.0,
        )
        item.score_time_window = score_time_window
        for joint_id in active_joint_ids:
            time_score_reference_windows.append(
                {
                    "joint": JOINT_NAMES[joint_id],
                    "joint_id": joint_id,
                    "active_joint_ids": list(active_joint_ids),
                    "active_joints": [JOINT_NAMES[idx] for idx in active_joint_ids],
                    "target_type": item.target_kind,
                    **score_time_window.summary(),
                }
            )

    out_dir = make_out_dir(args.out_root, args.run_name)
    model_path = resolve_repo_path(args.model)
    model_sha256 = sha256_file(model_path)
    model_snapshot_path = out_dir / f"input_model_snapshot{model_path.suffix or '.xml'}"
    shutil.copy2(model_path, model_snapshot_path)
    base_params = model_native_params(model_path, fixed_base=not args.free_base)

    continuous_bounds: list[tuple[float, float]] = []
    for pair in (args.armature_bounds, args.viscous_bounds, args.friction_bounds, args.bias_bounds):
        continuous_bounds.extend([pair] * len(fit_joints))
    if args.delay_values is None:
        delay_low = int(math.ceil(float(args.delay_bounds[0])))
        delay_high = int(math.floor(float(args.delay_bounds[1])))
        delay_values = list(range(delay_low, delay_high + 1))
    else:
        delay_values = list(args.delay_values)
    if not delay_values or any(value < 0 for value in delay_values):
        raise ValueError(f"delay candidates must contain non-negative integers, got {delay_values}")
    seeds = list(args.seeds) if args.seeds is not None else [int(args.seed)]
    if args.population_size < 4 or args.max_generations <= 0 or args.workers <= 0:
        raise ValueError(
            "--population-size must be at least 4 and --max-generations/--workers must be positive"
        )
    if not math.isfinite(args.sigma) or args.sigma <= 0.0:
        raise ValueError("--sigma must be finite and positive")
    if not math.isfinite(args.epsilon) or args.epsilon < 0.0:
        raise ValueError("--epsilon must be finite and non-negative")

    def params_to_continuous(params: dict[str, Any]) -> np.ndarray:
        normalized = normalize_params(params, len(JOINT_NAMES))
        return np.concatenate(
            [
                np.asarray(normalized[key], dtype=np.float64)[fit_joints]
                for key in DYNAMIC_PARAM_KEYS + ("encoder_bias",)
            ]
        )

    search_space = CmaSearchSpace(
        np.asarray(continuous_bounds, dtype=np.float64),
        tuple(delay_values),
    )
    if search_space.dimension == 0:
        raise ValueError(
            "all physical bounds and delay values are fixed; CMA-ES has no active parameter"
        )
    initial_armature = coerce_vector(args.initial_armature, len(JOINT_NAMES), "initial_armature")
    default_initial_mean = default_cma_mean(search_space, fit_joints, initial_armature)
    initial_param_files = [resolve_repo_path(path) for path in (args.initial_params or [])]
    initial_means: list[np.ndarray] = []
    for path in initial_param_files:
        prior = load_params_file(path, len(JOINT_NAMES))
        initial_means.append(
            search_space.encode(
                params_to_continuous(prior),
                int(prior[f"{args.delay_semantics}_delay_steps"]),
            )
        )

    replay_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "sim_hz": sim_hz,
        "control_hz": control_hz,
        "delay_semantics": args.delay_semantics,
        "control_mode": control_mode,
        "kp": kp,
        "kd": kd,
        "effort_limit": effort_limit,
        "torque_clip": not args.no_torque_clip,
        "free_base": args.free_base,
        "root_z": args.root_z,
        "enable_gravity": args.enable_gravity,
        "fixture_mode": fixture_mode,
        "fixture_hold_kp": np.asarray(args.fixture_hold_kp, dtype=np.float64),
        "fixture_hold_kd": np.asarray(args.fixture_hold_kd, dtype=np.float64),
        "lock_wheel_positions": bool(args.lock_wheel_positions),
        "initial_joint_qpos": initial_joint_qpos,
        "initial_joint_qvel": initial_joint_qvel,
    }
    replay = MujocoPaceReplay(**replay_kwargs)
    if args.require_rk4 and replay.model.opt.integrator != replay.mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("--require-rk4 was set, but the loaded MJCF does not use RK4")
    identification_model_snapshot_path = out_dir / "identification_model_snapshot.xml"
    shutil.copy2(Path(replay.loaded_path), identification_model_snapshot_path)
    identification_model_sha256 = sha256_file(identification_model_snapshot_path)
    worker_replay_kwargs = {
        **replay_kwargs,
        "model_path": identification_model_snapshot_path,
        "model_prepared": True,
    }

    bode_reference_validity: list[dict[str, Any]] = []
    for item in sources:
        time = np.asarray(item.payload["time"], dtype=np.float64)
        time_window = source_bode_time_window(
            item,
            time,
            chirp_source_frequency_range_hz=chirp_source_frequency_range,
            fmin=bode_fmin,
            fmax=bode_fmax,
            fit_start=args.bode_fit_start,
            fit_end_margin=args.bode_fit_end_margin,
        )
        fit_mask = time_window.mask
        active_joint_ids = source_active_joint_ids(item.source)
        for joint_id in active_joint_ids:
            if item.target_kind == "velocity":
                command = np.asarray(item.payload["des_dof_vel"], dtype=np.float64)[:, joint_id]
                real_velocity = item.payload.get("dof_vel")
                truth = (
                    np.zeros_like(command)
                    if real_velocity is None
                    else np.asarray(real_velocity, dtype=np.float64)[:, joint_id]
                )
            else:
                command = np.asarray(item.payload["des_dof_pos"], dtype=np.float64)[:, joint_id]
                truth = np.asarray(item.payload["dof_pos"], dtype=np.float64)[:, joint_id]
            reference_frequency, _, _ = transfer_bode(
                command[fit_mask], truth[fit_mask], time[fit_mask]
            )
            frequency_mask, command_psd_relative_db = reference_excitation_mask(
                command[fit_mask],
                time[fit_mask],
                reference_frequency,
                bode_fmin,
                bode_fmax,
                command_psd_threshold_db=args.bode_command_psd_threshold_db,
            )
            validity = reference_excitation_summary(
                reference_frequency,
                frequency_mask,
                command_psd_relative_db,
                bode_fmin,
                bode_fmax,
                args.bode_command_psd_threshold_db,
            )
            bode_reference_validity.append(
                {
                    "joint": JOINT_NAMES[joint_id],
                    "joint_id": joint_id,
                    "active_joint_ids": list(active_joint_ids),
                    "active_joints": [JOINT_NAMES[idx] for idx in active_joint_ids],
                    "target_type": item.target_kind,
                    "time_window": time_window.summary(),
                    **validity,
                }
            )

    eval_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    eval_counter = 0
    best_score = math.inf
    best_candidate: dict[str, Any] | None = None
    candidate_csv = out_dir / "pace_candidates_summary.csv"
    optimizer_csv = out_dir / "pace_optimizer_runs.csv"
    progress_json = out_dir / "pace_progress.json"
    local_candidate_context = CandidateWorkerContext(
        replay=replay,
        search_space=search_space,
        base_params=base_params,
        fit_joints=tuple(fit_joints),
        sources=tuple(sources),
        mirror_side_dynamics=bool(args.mirror_side_dynamics),
        mirror_encoder_bias=bool(args.mirror_encoder_bias),
    )
    effective_workers = min(int(args.workers), int(args.population_size))
    inherit_parent_context = effective_workers > 1 and args.mp_start_method == "fork"

    def source_bode_components(
        item: SourceData,
        result: dict[str, np.ndarray],
        joint_id: int,
    ) -> tuple[float, float]:
        time = np.asarray(result["time"], dtype=np.float64)
        if item.target_kind == "velocity":
            command = np.asarray(item.payload["des_dof_vel"], dtype=np.float64)[:, joint_id]
        else:
            command = np.asarray(item.payload["des_dof_pos"], dtype=np.float64)[:, joint_id]
        time_window = source_bode_time_window(
            item,
            time,
            chirp_source_frequency_range_hz=chirp_source_frequency_range,
            fmin=bode_fmin,
            fmax=bode_fmax,
            fit_start=args.bode_fit_start,
            fit_end_margin=args.bode_fit_end_margin,
        )
        mask = time_window.mask
        if np.count_nonzero(mask) < 16:
            return float("nan"), float("nan")
        return bode_error_components(
            command[mask],
            source_result_trace(item, result, "response", joint_id)[mask],
            source_result_trace(item, result, "truth", joint_id)[mask],
            time[mask],
            bode_fmin,
            bode_fmax,
            command_psd_threshold_db=args.bode_command_psd_threshold_db,
        )

    def record_candidate(
        result: dict[str, Any],
        *,
        seed: int,
        generation: int,
        member: int,
    ) -> None:
        nonlocal eval_counter, best_score, best_candidate
        eval_counter += 1
        score = float(result["score"])
        if score < best_score:
            best_score = score
            best_candidate = {
                **result,
                "seed": int(seed),
                "generation": int(generation),
                "member": int(member),
            }
            print(
                f"[BEST] eval={eval_counter} seed={seed} generation={generation} "
                f"time_mse={score:.8g} rmse={float(result['mean_rmse']):.8g} "
                f"delay={int(result['delay_steps'])}",
                flush=True,
            )
        eval_rows.append(
            {
                "eval": eval_counter,
                "seed": int(seed),
                "generation": int(generation),
                "member": int(member),
                "time_mse": score,
                "mean_rmse": float(result["mean_rmse"]),
                "source_mse": json.dumps([float(value) for value in result["source_mse"]]),
                "delay_steps": int(result["delay_steps"]),
                "normalized_x": json.dumps(result["normalized_x"].astype(float).tolist()),
                "x": json.dumps(result["x"].astype(float).tolist()),
                "params": json.dumps(result["params"]),
            }
        )

    print(f"[INFO] truth_run_dir={truth_run_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] model={model_path}, sha256={model_sha256}")
    print(f"[INFO] loaded_model={replay.loaded_path}")
    print(f"[INFO] integrator={replay.mujoco.mjtIntegrator(replay.model.opt.integrator).name}")
    print(
        f"[INFO] control_mode={control_mode}, truth_hz={truth_sample_hz:.9g}, "
        f"sim_hz={sim_hz:g}, control_hz={control_hz:g}, limit_steps={args.limit_steps}"
    )
    print(f"[INFO] default_angles={default_angles.tolist()}")
    print(
        "[INFO] initial_joint_qpos="
        f"{None if initial_joint_qpos is None else initial_joint_qpos.tolist()}, "
        "initial_joint_qvel="
        f"{None if initial_joint_qvel is None else initial_joint_qvel.tolist()}"
    )
    print(
        "[INFO] source_joints="
        f"{[(list(source_active_joint_ids(item.source)), list(source_active_joint_names(item.source)), item.source.get('kp', 'cli'), item.source.get('kd', 'cli')) for item in sources]}"
    )
    print(
        f"[INFO] requested_fit_joints={[(idx, JOINT_NAMES[idx]) for idx in requested_fit_joints]}"
    )
    print(f"[INFO] fit_joints={[(idx, JOINT_NAMES[idx]) for idx in fit_joints]}")
    delay_label = (
        "pre_controller_command_fifo"
        if args.delay_semantics == "command"
        else "post_controller_motor_torque_fifo"
    )
    print(f"[INFO] delay_semantics={delay_label}")
    print(
        "[INFO] optimizer=cma_es, score_metric=band_limited_time_mse, "
        f"time_score_range=[{score_fmin:g}, {score_fmax:g}] Hz, "
        f"bode_diagnostic_range=[{bode_fmin:g}, {bode_fmax:g}] Hz, "
        f"chirp_source_freq_range={list(chirp_source_frequency_range)} Hz, "
        f"bode_command_psd_threshold_db={args.bode_command_psd_threshold_db:g}"
    )
    print(f"[INFO] time_score_reference_windows={time_score_reference_windows}")
    print(f"[INFO] bode_reference_validity={bode_reference_validity}")
    print(f"[INFO] kp={kp.tolist()}, kd={kd.tolist()}, torque_clip={not args.no_torque_clip}")
    print(f"[INFO] continuous_bounds={continuous_bounds}")
    print(f"[INFO] delay_values={delay_values}, seeds={seeds}")
    print(f"[INFO] initial_params={[str(path) for path in initial_param_files]}")
    print(
        f"[INFO] default_initial_armature={initial_armature.tolist()}, "
        f"default_initial_normalized_mean={default_initial_mean.tolist()}"
    )
    print(
        f"[INFO] cma_es population_size={args.population_size}, "
        f"max_generations={args.max_generations}, sigma={args.sigma:g}, "
        f"epsilon={args.epsilon:g}, requested_workers={args.workers}, "
        f"effective_workers={effective_workers}, parallel_backend="
        f"{'process' if effective_workers > 1 else 'sequential'}, "
        f"mp_start_method={args.mp_start_method}, "
        f"worker_model_init={'fork_cow' if inherit_parent_context else 'independent'}, "
        f"active_dimension={search_space.dimension}"
    )
    print(
        f"[INFO] mirror_side_dynamics={args.mirror_side_dynamics}, "
        f"mirror_encoder_bias={args.mirror_encoder_bias}"
    )
    print(
        f"[INFO] fixture_mode={fixture_mode}, "
        f"lock_wheel_positions={replay.lock_wheel_positions}, "
        f"fixture_hold_kp={replay.fixture_hold_kp.tolist()}, "
        f"fixture_hold_kd={replay.fixture_hold_kd.tolist()}"
    )

    optimizer_runs: list[dict[str, Any]] = []
    executor = None
    if effective_workers > 1:
        if inherit_parent_context:
            _CANDIDATE_WORKER_CONTEXT = local_candidate_context
        multiprocessing_context = multiprocessing.get_context(args.mp_start_method)
        executor = ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=multiprocessing_context,
            initializer=initialize_candidate_worker,
            initargs=(
                worker_replay_kwargs,
                search_space,
                base_params,
                tuple(fit_joints),
                tuple(sources),
                bool(args.mirror_side_dynamics),
                bool(args.mirror_encoder_bias),
                inherit_parent_context,
            ),
        )
    try:
        for seed_index, seed in enumerate(seeds):
            mean = (
                initial_means[seed_index % len(initial_means)].copy()
                if initial_means
                else default_initial_mean.copy()
            )
            optimizer = cmaes.CMA(
                mean=mean,
                sigma=float(args.sigma),
                bounds=search_space.normalized_bounds,
                seed=int(seed),
                population_size=int(args.population_size),
            )
            run_best = math.inf
            run_best_normalized: np.ndarray | None = None
            termination_reason = "max_generations"
            converged = False
            completed_generations = 0
            for generation in range(1, int(args.max_generations) + 1):
                population = [optimizer.ask() for _ in range(int(args.population_size))]
                generation_results: list[dict[str, Any] | None] = [None] * len(population)
                if executor is None:
                    for completed, (member, normalized_vector) in enumerate(
                        enumerate(population),
                        start=1,
                    ):
                        result = evaluate_candidate_with_context(
                            normalized_vector,
                            local_candidate_context,
                        )
                        generation_results[member] = result
                        print(
                            f"[EVAL] seed={seed} generation={generation}/{args.max_generations} "
                            f"completed={completed}/{len(population)} member={member} "
                            f"time_mse={float(result['score']):.8g} "
                            f"rmse={float(result['mean_rmse']):.8g}",
                            flush=True,
                        )
                else:
                    future_to_member = {
                        executor.submit(
                            evaluate_candidate_in_worker,
                            (member, normalized_vector),
                        ): member
                        for member, normalized_vector in enumerate(population)
                    }
                    for completed, future in enumerate(as_completed(future_to_member), start=1):
                        expected_member = future_to_member[future]
                        member, result = future.result()
                        if member != expected_member:
                            raise RuntimeError(
                                f"worker returned member {member}, expected {expected_member}"
                            )
                        generation_results[member] = result
                        print(
                            f"[EVAL] seed={seed} generation={generation}/{args.max_generations} "
                            f"completed={completed}/{len(population)} member={member} "
                            f"time_mse={float(result['score']):.8g} "
                            f"rmse={float(result['mean_rmse']):.8g}",
                            flush=True,
                        )
                if any(result is None for result in generation_results):
                    raise RuntimeError(
                        f"one or more CMA-ES candidates produced no result for seed {seed}, generation {generation}"
                    )
                results = [result for result in generation_results if result is not None]
                solutions: list[tuple[np.ndarray, float]] = []
                generation_scores = np.asarray(
                    [result["score"] for result in results], dtype=np.float64
                )
                for member, (normalized_vector, result) in enumerate(
                    zip(population, results, strict=True)
                ):
                    score = float(result["score"])
                    solutions.append((normalized_vector, score))
                    record_candidate(
                        result,
                        seed=int(seed),
                        generation=generation,
                        member=member,
                    )
                    if score < run_best:
                        run_best = score
                        run_best_normalized = normalized_vector.copy()
                if not np.any(np.isfinite(generation_scores)):
                    raise RuntimeError(
                        f"all CMA-ES candidates were non-finite for seed {seed}, generation {generation}"
                    )
                optimizer.tell(solutions)
                completed_generations = generation
                generation_min = float(np.min(generation_scores))
                generation_max = float(np.max(generation_scores))
                relative_spread = (generation_max - generation_min) / max(abs(generation_min), EPS)
                library_stop = bool(optimizer.should_stop())
                converged = bool(args.epsilon > 0.0 and relative_spread < args.epsilon)
                optimizer_rows.append(
                    {
                        "seed": int(seed),
                        "generation": generation,
                        "best_time_mse": generation_min,
                        "mean_time_mse": float(np.mean(generation_scores)),
                        "worst_time_mse": generation_max,
                        "relative_score_spread": relative_spread,
                        "sigma": float(getattr(optimizer, "_sigma", math.nan)),
                        "cma_should_stop": library_stop,
                        "epsilon_converged": converged,
                    }
                )
                print(
                    f"[GEN] seed={seed} generation={generation}/{args.max_generations} "
                    f"best_time_mse={generation_min:.8g} mean_time_mse={float(np.mean(generation_scores)):.8g} "
                    f"spread={relative_spread:.8g}",
                    flush=True,
                )
                write_generation_progress(
                    candidate_csv=candidate_csv,
                    optimizer_csv=optimizer_csv,
                    progress_json=progress_json,
                    eval_rows=eval_rows,
                    optimizer_rows=optimizer_rows,
                    best_candidate=best_candidate,
                    status="running",
                    seed=int(seed),
                    generation=generation,
                )
                if converged:
                    termination_reason = "epsilon"
                    break
                if library_stop:
                    termination_reason = "cma_should_stop"
                    break
            if run_best_normalized is None:
                raise RuntimeError(f"CMA-ES seed {seed} did not produce a finite candidate")
            run_record = {
                "run_id": f"seed{seed}",
                "seed": int(seed),
                "termination_reason": termination_reason,
                "converged": bool(converged or termination_reason == "cma_should_stop"),
                "best_time_mse": float(run_best),
                "nfev": int(completed_generations * args.population_size),
                "generations": int(completed_generations),
                "best_normalized_x": run_best_normalized.tolist(),
            }
            optimizer_runs.append(run_record)
            print(
                f"[RUN] seed={seed} best_time_mse={run_best:.8g} "
                f"generations={completed_generations} termination={termination_reason}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _CANDIDATE_WORKER_CONTEXT = None

    if best_candidate is None or not math.isfinite(best_score):
        raise RuntimeError("CMA-ES did not produce a finite candidate")
    best_x = np.asarray(best_candidate["x"], dtype=np.float64)
    best_delay = int(best_candidate["delay_steps"])
    final_score = float(best_candidate["score"])
    selected_run = next(
        row for row in optimizer_runs if int(row["seed"]) == int(best_candidate["seed"])
    )
    calibration_converged = bool(selected_run["converged"])
    best_params = with_candidate_params(
        base_params,
        fit_joints,
        best_x,
        delay_steps=best_delay,
        mirror_side_dynamics=args.mirror_side_dynamics,
        mirror_encoder_bias=args.mirror_encoder_bias,
    )

    atomic_write_csv(candidate_csv, CANDIDATE_CSV_FIELDS, eval_rows)
    atomic_write_csv(optimizer_csv, OPTIMIZER_CSV_FIELDS, optimizer_rows)

    best_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for source_index, item in enumerate(sources):
        active_joint_ids = source_active_joint_ids(item.source)
        active_joint_names = source_active_joint_names(item.source)
        source_kp = source_scalar_metadata(item.source, "kp")
        source_kd = source_scalar_metadata(item.source, "kd")
        artifact_token = source_artifact_token(source_index, item.source)
        replay_result = replay.replay(item, best_params, log_full=True)
        if item.score_time_window is None:
            raise RuntimeError(
                f"source {source_label(item.source)} is missing its time-domain score window"
            )
        score_mask = item.score_time_window.mask
        non_source_mask = np.ones(len(JOINT_NAMES), dtype=bool)
        non_source_mask[np.asarray(active_joint_ids, dtype=np.int32)] = False
        non_source_position_drift = float(
            np.max(
                np.abs(
                    np.asarray(replay_result["q"], dtype=np.float64)[:, non_source_mask]
                    - np.asarray(replay_result["q"], dtype=np.float64)[0, non_source_mask]
                )
            )
        )
        non_source_velocity_max = float(
            np.max(np.abs(np.asarray(replay_result["qd"], dtype=np.float64)[:, non_source_mask]))
        )
        if not np.isfinite(non_source_position_drift) or not np.isfinite(non_source_velocity_max):
            raise RuntimeError(
                f"non-source fixture state became non-finite for {source_label(item.source)}"
            )
        if fixture_mode == "equality" and (
            non_source_position_drift > EQUALITY_FIXTURE_DRIFT_TOL_RAD
            or non_source_velocity_max > EQUALITY_FIXTURE_VELOCITY_TOL_RAD_S
        ):
            raise RuntimeError(
                "non-source joint equality lock tolerance exceeded: "
                f"source={source_label(item.source)} drift={non_source_position_drift:.9g} rad "
                f"velocity={non_source_velocity_max:.9g} rad/s"
            )
        locked_wheel_ids = np.asarray(
            [joint_id for joint_id in WHEEL_JOINT_IDS if joint_id not in set(active_joint_ids)],
            dtype=np.int32,
        )
        if locked_wheel_ids.size:
            wheel_q = np.asarray(replay_result["q"], dtype=np.float64)[:, locked_wheel_ids]
            wheel_qd = np.asarray(replay_result["qd"], dtype=np.float64)[:, locked_wheel_ids]
            wheel_position_drift = float(np.max(np.abs(wheel_q - wheel_q[0])))
            wheel_velocity_max = float(np.max(np.abs(wheel_qd)))
        else:
            wheel_position_drift = 0.0
            wheel_velocity_max = 0.0
        if (
            args.lock_wheel_positions
            and locked_wheel_ids.size
            and (
                wheel_position_drift > WHEEL_LOCK_DRIFT_TOL_RAD
                or wheel_velocity_max > WHEEL_LOCK_VELOCITY_TOL_RAD_S
            )
        ):
            raise RuntimeError(
                "non-source wheel position lock drift exceeded its exact projection tolerance: "
                f"source={source_label(item.source)} position_drift={wheel_position_drift:.9g} "
                f"velocity={wheel_velocity_max:.9g}"
            )
        command_key = "des_dof_vel" if item.target_kind == "velocity" else "des_dof_pos"
        applied_command_key = "applied_desvel" if item.target_kind == "velocity" else "applied_des"
        comparison_time = np.asarray(replay_result["time"], dtype=np.float64)
        comparison_time_window = source_bode_time_window(
            item,
            comparison_time,
            chirp_source_frequency_range_hz=chirp_source_frequency_range,
            fmin=bode_fmin,
            fmax=bode_fmax,
            fit_start=args.bode_fit_start,
            fit_end_margin=args.bode_fit_end_margin,
        )
        for joint_id in active_joint_ids:
            response = source_result_trace(
                item,
                replay_result,
                "response",
                joint_id,
            )
            truth = source_result_trace(item, replay_result, "truth", joint_id)
            time_mse = raw_mse(response, truth, score_mask)
            rmse = raw_rmse(response, truth, score_mask)
            nmse = normalized_mse(response, truth, score_mask)
            bode_magnitude_mse, bode_phase_mse = source_bode_components(
                item,
                replay_result,
                joint_id,
            )
            best_rows.append(
                {
                    "source_index": source_index,
                    "source_kp": source_kp,
                    "source_kd": source_kd,
                    "source_csv": item.source.get("source_csv", item.source.get("csv")),
                    "source_pt": item.source.get("pt"),
                    "joint": JOINT_NAMES[joint_id],
                    "joint_id": joint_id,
                    "active_joint_ids": list(active_joint_ids),
                    "active_joints": list(active_joint_names),
                    "target_type": item.target_kind,
                    "time_mse": time_mse,
                    "rmse": rmse,
                    "normalized_mse": nmse,
                    "bode_mag_mse_db2": bode_magnitude_mse,
                    "bode_mag_rmse_db": math.sqrt(bode_magnitude_mse),
                    "bode_phase_mse_deg2": bode_phase_mse,
                    "bode_phase_rmse_deg": math.sqrt(bode_phase_mse),
                    "selection_score": time_mse,
                    "non_source_joint_max_abs_position_drift_rad": (non_source_position_drift),
                    "non_source_joint_max_abs_velocity_rad_s": (non_source_velocity_max),
                    "locked_joint_max_abs_position_drift_rad": (
                        non_source_position_drift if fixture_mode == "equality" else None
                    ),
                    "locked_joint_max_abs_velocity_rad_s": (
                        non_source_velocity_max if fixture_mode == "equality" else None
                    ),
                    "wheel_max_abs_position_drift_rad": wheel_position_drift,
                    "wheel_max_abs_velocity_rad_s": wheel_velocity_max,
                }
            )
            comparisons.append(
                {
                    "joint": (
                        f"{JOINT_NAMES[joint_id]} | Kd={source_kd:g}"
                        if source_kd is not None
                        else f"{JOINT_NAMES[joint_id]} | source={source_index}"
                    ),
                    "target_kind": item.target_kind,
                    "time": replay_result["time"],
                    "command": np.asarray(
                        item.payload[command_key],
                        dtype=np.float64,
                    )[:, joint_id],
                    "applied_command": np.asarray(
                        replay_result[applied_command_key],
                        dtype=np.float64,
                    )[:, joint_id],
                    "truth": truth,
                    "response": response,
                    "fit_mask": comparison_time_window.mask,
                }
            )
        if len(active_joint_ids) == 1:
            replay_name = (
                f"best_replay_{artifact_token}_joint{active_joint_ids[0] + 1:02d}_"
                f"{active_joint_names[0]}.csv"
            )
        else:
            joint_token = "-".join(f"{joint_id + 1:02d}" for joint_id in active_joint_ids)
            name_token = "_".join(active_joint_names)
            replay_name = f"best_replay_{artifact_token}_joints{joint_token}_{name_token}.csv"
        write_replay_csv(
            out_dir / replay_name,
            replay_result,
        )

    best_summary_csv = out_dir / "best_fit_summary.csv"
    with best_summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_index",
            "source_kp",
            "source_kd",
            "source_csv",
            "source_pt",
            "joint",
            "joint_id",
            "active_joint_ids",
            "active_joints",
            "target_type",
            "time_mse",
            "rmse",
            "normalized_mse",
            "bode_mag_mse_db2",
            "bode_mag_rmse_db",
            "bode_phase_mse_deg2",
            "bode_phase_rmse_deg",
            "selection_score",
            "non_source_joint_max_abs_position_drift_rad",
            "non_source_joint_max_abs_velocity_rad_s",
            "locked_joint_max_abs_position_drift_rad",
            "locked_joint_max_abs_velocity_rad_s",
            "wheel_max_abs_position_drift_rad",
            "wheel_max_abs_velocity_rad_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    bode_report_path = out_dir / "pace_bode_report.png"
    write_bode_report(
        bode_report_path,
        comparisons,
        bode_fmin,
        bode_fmax,
        title=f"PACE Bode fit ({delay_label}, {sim_hz:g} Hz sim / {control_hz:g} Hz control)",
    )

    delay_steps = int(best_params["command_delay_steps"])
    serialized_delay = delay_artifact_fields(delay_steps, args.delay_semantics)
    parameter_vector_order = list(PARAM_VECTOR_ORDER)
    parameter_vector_order[-1] = (
        "command_delay_steps[1]" if args.delay_semantics == "command" else "torque_delay_steps[1]"
    )
    final_score = float(np.mean([row["selection_score"] for row in best_rows]))
    calibration_status = "converged" if calibration_converged else "completed_max_generations"
    best_json = {
        "format": "dr002_mujoco_pace_fit_v4",
        "calibration_status": calibration_status,
        "joint_order": JOINT_NAMES,
        "active_source_joint_ids": [list(source_active_joint_ids(item.source)) for item in sources],
        "active_source_joints": [list(source_active_joint_names(item.source)) for item in sources],
        "parameter_vector_order": parameter_vector_order,
        "parameter_vector": params_to_vector(best_params, len(JOINT_NAMES)).tolist(),
        "armature": params_to_named_dict(best_params, "armature"),
        "viscous_friction": params_to_named_dict(best_params, "viscous_friction"),
        "viscous_damping": params_to_named_dict(best_params, "viscous_friction"),
        "coulomb_friction": params_to_named_dict(best_params, "coulomb_friction"),
        "friction": params_to_named_dict(best_params, "coulomb_friction"),
        "encoder_bias": params_to_named_dict(best_params, "encoder_bias"),
        "bias": params_to_named_dict(best_params, "encoder_bias"),
        "delay_steps": delay_steps,
        **serialized_delay,
        "delay_semantics": delay_label,
        "delay_scope": "all_six_joints",
        "shared_command_delay_steps": (delay_steps if args.delay_semantics == "command" else 0),
        "delay_s": delay_steps / float(control_hz),
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "meta": {
            "truth_run_dir": str(truth_run_dir),
            "model": str(model_path),
            "model_sha256_at_fit": model_sha256,
            "model_snapshot": str(model_snapshot_path),
            "loaded_model": str(replay.loaded_path),
            "identification_model_snapshot": str(identification_model_snapshot_path),
            "identification_model_sha256": identification_model_sha256,
            "control_mode": control_mode,
            "integrator": replay.mujoco.mjtIntegrator(replay.model.opt.integrator).name,
            "require_rk4": bool(args.require_rk4),
            "dt": truth_dt,
            "truth_dt": truth_dt,
            "truth_sample_hz": truth_sample_hz,
            "sim_dt": 1.0 / sim_hz,
            "sim_hz": sim_hz,
            "control_dt": 1.0 / control_hz,
            "control_hz": control_hz,
            "control_schedule": "first_physics_boundary_at_or_after_control_tick_zoh",
            "command_resampling": "causal_previous_source_sample_zoh",
            "score_metric": "band_limited_time_mse",
            "score_reduction": "equal_mean_of_per_source_active_joint_mean_squared_error",
            "score_simulation_window": "full_replay_trajectory",
            "score_time_window": "strict_linear_chirp_time_window",
            "score_frequency_range_hz": [score_fmin, score_fmax],
            "score_chirp_source_frequency_range_hz": list(chirp_source_frequency_range),
            "score_trace_order": [
                {
                    "source_index": source_index,
                    "joint_id": joint_id,
                    "joint": JOINT_NAMES[joint_id],
                }
                for source_index, item in enumerate(sources)
                for joint_id in source_active_joint_ids(item.source)
            ],
            "source_controller_gains": [
                {
                    "source_index": source_index,
                    "active_joint_ids": list(source_active_joint_ids(item.source)),
                    "active_joints": list(source_active_joint_names(item.source)),
                    "kp": source_controller_gain(item.source, "kp", kp).tolist(),
                    "kd": source_controller_gain(item.source, "kd", kd).tolist(),
                }
                for source_index, item in enumerate(sources)
            ],
            "source_fixture_gains": [
                {
                    "source_index": source_index,
                    "active_joint_ids": list(source_active_joint_ids(item.source)),
                    "active_joints": list(source_active_joint_names(item.source)),
                    "fixture_hold_kp": source_fixture_gain(
                        item.source, "fixture_hold_kp", replay.fixture_hold_kp
                    ).tolist(),
                    "fixture_hold_kd": source_fixture_gain(
                        item.source, "fixture_hold_kd", replay.fixture_hold_kd
                    ).tolist(),
                }
                for source_index, item in enumerate(sources)
            ],
            "score_reference_windows": time_score_reference_windows,
            "bode_role": "post_fit_diagnostic_only",
            "bode_estimator": "welch_csd_fixed_physical_window",
            "bode_welch_segment_duration_s": DEFAULT_WELCH_SEGMENT_DURATION_S,
            "bode_frequency_range_hz": [bode_fmin, bode_fmax],
            "bode_chirp_source_frequency_range_hz": list(chirp_source_frequency_range),
            "bode_fit_time_window_mode": "strict_linear_chirp_time_window",
            "bode_command_psd_threshold_db_relative_to_peak": float(
                args.bode_command_psd_threshold_db
            ),
            "bode_reference_validity": bode_reference_validity,
            "bode_fit_window_margins_s": [
                float(args.bode_fit_start),
                float(args.bode_fit_end_margin),
            ],
            "limit_steps": args.limit_steps,
            "requested_fit_joint_ids": requested_fit_joints,
            "requested_fit_joints": [JOINT_NAMES[idx] for idx in requested_fit_joints],
            "fit_joint_ids": fit_joints,
            "fit_joints": [JOINT_NAMES[idx] for idx in fit_joints],
            "mirrored_fit_joint_canonicalization": (
                "left_side_when_both_sides_requested" if args.mirror_side_dynamics else "disabled"
            ),
            "enable_gravity": bool(args.enable_gravity),
            "free_base": bool(args.free_base),
            "fixture_mode": fixture_mode,
            "fixture_target": (
                "explicit_initial_joint_qpos_for_non_source_and_per_source_initial_measured_plus_encoder_bias_for_active"
                if initial_joint_qpos is not None
                else "per_source_initial_measured_plus_encoder_bias"
            ),
            "active_joint_initial_qpos_source": "per_source_initial_measured_plus_encoder_bias",
            "active_joint_initial_qvel_source": "per_source_initial_measured_velocity",
            "non_source_initial_qpos_source": (
                "explicit_initial_joint_qpos"
                if initial_joint_qpos is not None
                else "per_source_initial_measured_plus_encoder_bias"
            ),
            "non_source_initial_qvel_source": (
                "explicit_initial_joint_qvel" if initial_joint_qvel is not None else "zero"
            ),
            "fixture_hold_kp": replay.fixture_hold_kp.tolist(),
            "fixture_hold_kd": replay.fixture_hold_kd.tolist(),
            "fixture_active_joint_controller": "identified_or_user_kp_kd",
            "fixture_non_source_controller": (
                "reset_pose_pd" if fixture_mode == "high-impedance" else "not_applied"
            ),
            "fixture_wheel_behavior": (
                "non_source_wheels_use_mujoco_joint_equality_plus_exact_state_projection"
                if replay.lock_wheel_positions
                else (
                    "zero_stiffness_velocity_damping"
                    if fixture_mode == "high-impedance"
                    else "not_applied"
                )
            ),
            "lock_wheel_positions": replay.lock_wheel_positions,
            "wheel_lock_method": (
                "non_source_wheels_use_mujoco_joint_equality_plus_exact_state_projection"
                if replay.lock_wheel_positions
                else "none"
            ),
            "wheel_lock_scope": "non_source_wheels_only",
            "lock_non_source_joints": fixture_mode == "equality",
            "joint_lock_method": (
                "inactive_by_default_mujoco_joint_equality_constraints"
                if fixture_mode == "equality"
                else ("independent_reset_pose_pd" if fixture_mode == "high-impedance" else "none")
            ),
            "root_z": float(args.root_z),
            "initial_joint_qpos": None
            if initial_joint_qpos is None
            else initial_joint_qpos.tolist(),
            "initial_joint_qvel": None
            if initial_joint_qvel is None
            else initial_joint_qvel.tolist(),
            "torque_clip": not args.no_torque_clip,
            "effort_limit": effort_limit.tolist(),
            "continuous_bounds": {
                "armature": list(args.armature_bounds),
                "viscous_friction": list(args.viscous_bounds),
                "coulomb_friction": list(args.friction_bounds),
                "encoder_bias": list(args.bias_bounds),
            },
            "delay_values": delay_values,
            "initial_params": [str(path) for path in initial_param_files],
            "default_initial_armature": initial_armature.tolist(),
            "default_initial_mean_used": not bool(initial_param_files),
            "mirror_side_dynamics": bool(args.mirror_side_dynamics),
            "mirror_encoder_bias": bool(args.mirror_encoder_bias),
            "optimizer": {
                "name": "pace_cma_es_time_domain_mse",
                "library": "cmaes",
                "library_version": getattr(cmaes, "__version__", "unknown"),
                "coordinate_system": "normalized_minus_one_to_one",
                "active_dimension": int(search_space.dimension),
                "population_size": int(args.population_size),
                "max_generations": int(args.max_generations),
                "sigma": float(args.sigma),
                "epsilon": float(args.epsilon),
                "workers": int(effective_workers),
                "requested_workers": int(args.workers),
                "parallel_backend": "process" if effective_workers > 1 else "sequential",
                "multiprocessing_start_method": args.mp_start_method,
                "worker_model_initialization": (
                    "fork_copy_on_write_parent_context"
                    if inherit_parent_context
                    else "independent_worker_initialization"
                ),
                "seeds": seeds,
                "calibration_converged": bool(calibration_converged),
                "selected_seed": int(best_candidate["seed"]),
                "selected_generation": int(best_candidate["generation"]),
                "selected_member": int(best_candidate["member"]),
                "best_time_mse": final_score,
                "total_objective_evals": int(eval_counter),
                "runs": optimizer_runs,
            },
            "sources": [item.source for item in sources],
            "best_fit_summary": best_rows,
            "artifacts": {
                "bode_report_png": str(bode_report_path),
                "bode_report_pdf": str(bode_report_path.with_suffix(".pdf")),
                "input_model_snapshot": str(model_snapshot_path),
                "identification_model_snapshot": str(identification_model_snapshot_path),
                "progress_json": str(progress_json),
            },
        },
    }
    best_json_path = out_dir / "pace_best_params.json"
    write_json(best_json_path, best_json)
    final_optimizer_row = optimizer_rows[-1]
    write_generation_progress(
        candidate_csv=candidate_csv,
        optimizer_csv=optimizer_csv,
        progress_json=progress_json,
        eval_rows=eval_rows,
        optimizer_rows=optimizer_rows,
        best_candidate=best_candidate,
        status="complete",
        seed=int(final_optimizer_row["seed"]),
        generation=int(final_optimizer_row["generation"]),
    )

    print(f"[DONE] wrote candidates: {candidate_csv}")
    print(f"[DONE] wrote optimizer runs: {optimizer_csv}")
    print(f"[DONE] wrote progress checkpoint: {progress_json}")
    print(f"[DONE] wrote best summary: {best_summary_csv}")
    print(f"[DONE] wrote Bode report: {bode_report_path}")
    print(f"[DONE] wrote PACE params: {best_json_path}")
    print(
        f"[BEST] status={calibration_status} score={final_score:.8g}, "
        f"delay_steps={delay_steps}, delay_s={delay_steps / float(control_hz):.6g}"
    )
    for idx in fit_joints:
        print(
            f"[BEST] {JOINT_NAMES[idx]} arm={best_params['armature'][idx]:.8g} "
            f"visc={best_params['viscous_friction'][idx]:.8g} "
            f"fric={best_params['coulomb_friction'][idx]:.8g} "
            f"bias={best_params['encoder_bias'][idx]:.8g}"
        )


if __name__ == "__main__":
    main()
