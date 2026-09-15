#!/usr/bin/env python3
"""Fit symmetric WE11 leg Kp/Kd against paired real chirp recordings.

This scanner consumes one completed ``dr002_mujoco_pace_fit_v4`` artifact
directly.  It freezes the identified PACE dynamics and shared command FIFO,
replays each bilateral thigh/calf source once, and scores the left/right
responses with equal weight. The PACE artifact freezes the physical parameters,
model hash and shared command-delay contract before the gain search begins.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import signal
import shlex
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACE_DIR.parents[1]
sys.path.insert(0, str(PACE_DIR))

from bode_reference import BodeReference  # noqa: E402
from chirp_frequency_response import transfer_bode  # noqa: E402
from fit_mujoco_pace_params import (  # noqa: E402
    MujocoPaceReplay,
    SourceData,
    atomic_write_csv,
    atomic_write_json,
    make_out_dir,
    sha256_file,
)
from mujoco_dr002_common import (  # noqa: E402
    JOINT_NAMES,
    load_chirp_data,
    normalize_params,
)

GROUPS = ("thigh", "calf")
GROUP_ACTIVE_IDS = {"thigh": (0, 3), "calf": (1, 4)}
GROUP_BASELINE = {"thigh": (2.0, 0.1), "calf": (8.0, 0.8)}
SIDE_LABELS = ("left", "right")
METRIC_KEYS = (
    "bode_magnitude_mse_db2",
    "bode_magnitude_rmse_db",
    "bode_phase_mse_deg2",
    "bode_phase_rmse_deg",
    "bode_complex_score",
    "time_rmse",
    "time_normalized_mse",
)
TIMEOUT_PENALTY_SCORE = 1.0e12
PRIMARY_METRICS = {
    "bode-mag": "bode_magnitude_mse_db2",
    "bode-complex": "bode_complex_score",
}
CANDIDATE_FIELDS = (
    "group",
    "stage",
    "kp",
    "kd",
    *METRIC_KEYS,
    *(f"{side}_{key}" for side in SIDE_LABELS for key in METRIC_KEYS),
)


@dataclass(frozen=True)
class SearchRange:
    kp_min: float
    kp_max: float
    kp_step: float
    kd_min: float
    kd_max: float
    kd_step: float


@dataclass
class PairedContext:
    group: str
    active_joint_ids: tuple[int, int]
    source_metadata: dict[str, Any]
    source: SourceData
    references: tuple[BodeReference, BodeReference]
    params: dict[str, Any]
    replay: MujocoPaceReplay
    candidate_timeout_s: float = 0.0


@dataclass
class FinalTrace:
    group: str
    kp: float
    kd: float
    metrics: dict[str, float]
    result: dict[str, np.ndarray]
    wheel_position_drift_rad: float
    wheel_max_abs_velocity_rad_s: float
    non_source_position_drift_rad: float
    max_abs_torque_nm: float
    saturation_count: int


class CandidateStore:
    def __init__(self, path: Path, group: str) -> None:
        self.path = path
        self.group = group
        self.rows: list[dict[str, Any]] = []
        self.by_gain: dict[tuple[float, float], dict[str, Any]] = {}
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=CANDIDATE_FIELDS)
        self._writer.writeheader()
        self._handle.flush()

    @staticmethod
    def key(kp: float, kd: float) -> tuple[float, float]:
        return round(float(kp), 10), round(float(kd), 10)

    def get(self, kp: float, kd: float) -> dict[str, Any] | None:
        return self.by_gain.get(self.key(kp, kd))

    def add(self, row: dict[str, Any]) -> None:
        key = self.key(float(row["kp"]), float(row["kd"]))
        if key in self.by_gain:
            return
        self.by_gain[key] = row
        self.rows.append(row)
        self._writer.writerow({field: row.get(field, "") for field in CANDIDATE_FIELDS})
        self._handle.flush()

    def best(self, score_metric: str) -> dict[str, Any]:
        metric = PRIMARY_METRICS[score_metric]
        finite = [row for row in self.rows if math.isfinite(float(row[metric]))]
        if not finite:
            raise RuntimeError(f"no finite candidates for {self.group}/{score_metric}")
        return min(
            finite,
            key=lambda row: (float(row[metric]), float(row["kp"]), float(row["kd"])),
        )

    def close(self) -> None:
        self._handle.close()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def resolve_path(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def inclusive_grid(low: float, high: float, step: float) -> np.ndarray:
    if not all(math.isfinite(value) for value in (low, high, step)):
        raise ValueError("grid bounds and step must be finite")
    if step <= 0.0 or high < low:
        raise ValueError(f"invalid grid [{low}, {high}] step {step}")
    count = int(math.floor((high - low) / step + 1.0e-9))
    values = low + step * np.arange(count + 1, dtype=np.float64)
    if values.size == 0 or values[-1] < high - step * 1.0e-8:
        values = np.append(values, high)
    else:
        values[-1] = high
    return np.round(values, 10)


def centered_grid(center: float, radius: float, step: float, low: float, high: float) -> np.ndarray:
    return inclusive_grid(max(low, center - radius), min(high, center + radius), step)


def cartesian(kp_values: np.ndarray, kd_values: np.ndarray) -> list[tuple[float, float]]:
    return [(float(kp), float(kd)) for kp in kp_values for kd in kd_values]


def selected_groups(values: Iterable[str]) -> tuple[str, ...]:
    groups = tuple(dict.fromkeys(values))
    if not groups or any(group not in GROUPS for group in groups):
        raise ValueError(f"groups must be a non-empty subset of {GROUPS}")
    return groups


def group_search_range(args: argparse.Namespace, group: str) -> SearchRange:
    return SearchRange(
        kp_min=float(getattr(args, f"{group}_kp_min")),
        kp_max=float(getattr(args, f"{group}_kp_max")),
        kp_step=float(getattr(args, f"{group}_kp_step")),
        kd_min=float(getattr(args, f"{group}_kd_min")),
        kd_max=float(getattr(args, f"{group}_kd_max")),
        kd_step=float(getattr(args, f"{group}_kd_step")),
    )


def validate_search_range(group: str, search: SearchRange, baseline: tuple[float, float]) -> None:
    for name, value in search.__dict__.items():
        if not math.isfinite(value):
            raise ValueError(f"{group} {name} must be finite")
    if search.kp_step <= 0.0 or search.kd_step <= 0.0:
        raise ValueError(f"{group} search steps must be positive")
    if search.kp_max < search.kp_min or search.kd_max < search.kd_min:
        raise ValueError(f"{group} search bounds must be increasing")
    kp0, kd0 = baseline
    if not (search.kp_min <= kp0 <= search.kp_max):
        raise ValueError(f"{group} Kp range does not contain acquisition baseline {kp0:g}")
    if not (search.kd_min <= kd0 <= search.kd_max):
        raise ValueError(f"{group} Kd range does not contain acquisition baseline {kd0:g}")


def validate_rk4_model(model_path: Path, expected_sha256: str) -> None:
    if sha256_file(model_path) != expected_sha256:
        raise ValueError("WE11 model SHA-256 does not match the completed PACE fit")
    option = ET.parse(model_path).getroot().find("option")
    if option is None or option.get("integrator", "").strip().lower() != "rk4":
        raise ValueError("WE11 paired Kp/Kd replay requires an explicit RK4 MJCF")


def validate_pace_contract(pace: dict[str, Any], model_path: Path) -> dict[str, Any]:
    if pace.get("format") != "dr002_mujoco_pace_fit_v4":
        raise ValueError(f"unsupported PACE format: {pace.get('format')!r}")
    if pace.get("calibration_status") not in {"converged", "completed_max_generations"}:
        raise ValueError("PACE artifact is not a completed usable calibration")
    if pace.get("joint_order") != JOINT_NAMES:
        raise ValueError("PACE joint order does not match WE policy order")
    meta = pace.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("PACE artifact is missing meta")
    for key, expected in (("sim_hz", 400.0), ("control_hz", 200.0)):
        if not math.isclose(float(meta.get(key, float("nan"))), expected, abs_tol=1.0e-12):
            raise ValueError(f"PACE {key} must be {expected:g}")
    if meta.get("control_mode") != "position":
        raise ValueError("PACE control mode must be position")
    if meta.get("enable_gravity") is not True or meta.get("free_base") is not False:
        raise ValueError("PACE replay must use gravity with a fixed base")
    if meta.get("fixture_mode") != "high-impedance":
        raise ValueError("PACE fixture mode must be high-impedance")
    pace_sources = meta.get("sources")
    if not isinstance(pace_sources, list) or not pace_sources:
        raise ValueError("PACE artifact is missing per-source acquisition metadata")
    for source in pace_sources:
        if not isinstance(source, dict):
            raise ValueError("PACE source metadata must be an object")
        for key in ("fixture_hold_kp", "fixture_hold_kd"):
            gain = np.asarray(source.get(key, []), dtype=np.float64).reshape(-1)
            if gain.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(gain)):
                raise ValueError(f"PACE source {key} must contain six finite values")
            if np.any(gain < 0.0):
                raise ValueError(f"PACE source {key} must be non-negative")
    if meta.get("lock_wheel_positions") is not True:
        raise ValueError("PACE artifact must declare exact wheel position locking")
    if pace.get("delay_semantics") != "pre_controller_command_fifo":
        raise ValueError("PACE artifact must use a pre-controller command FIFO")
    delay = pace.get("shared_command_delay_steps", pace.get("command_delay_steps"))
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError("PACE shared command delay must be a non-negative integer")
    model_sha = meta.get("model_sha256_at_fit")
    if not isinstance(model_sha, str) or len(model_sha) != 64:
        raise ValueError("PACE artifact is missing model_sha256_at_fit")
    validate_rk4_model(model_path, model_sha)
    return meta


def resolve_paired_sources(
    truth_run_dir: Path,
    pace_meta: dict[str, Any],
    groups: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, Any]]:
    manifest_path = truth_run_dir / "chirp_source_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("format") != "dr002_real_sweep_import_v2_paired":
        raise ValueError("truth manifest must use dr002_real_sweep_import_v2_paired")
    if manifest.get("control_mode") != "position":
        raise ValueError("paired PD fitting requires position-control truth")
    if manifest.get("joint_names") != JOINT_NAMES:
        raise ValueError("truth joint order does not match WE policy order")
    if not math.isclose(float(manifest.get("sample_rate_hz", 0.0)), 200.0, abs_tol=1.0e-12):
        raise ValueError("truth data must be exactly 200 Hz")
    sources = manifest.get("sources")
    pace_sources = pace_meta.get("sources")
    if not isinstance(sources, list) or not isinstance(pace_sources, list):
        raise ValueError("truth/PACE source metadata is missing")
    by_group = {
        str(source.get("joint_group")): source for source in sources if isinstance(source, dict)
    }
    pace_by_group = {
        str(source.get("joint_group")): source
        for source in pace_sources
        if isinstance(source, dict)
    }
    resolved: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for group in groups:
        if group not in by_group or group not in pace_by_group:
            raise ValueError(f"missing paired {group} source")
        source = by_group[group]
        frozen = pace_by_group[group]
        active = tuple(int(value) for value in source.get("active_joint_ids", []))
        if active != GROUP_ACTIVE_IDS[group]:
            raise ValueError(
                f"{group} active_joint_ids={active}, expected {GROUP_ACTIVE_IDS[group]}"
            )
        if source.get("target_type") != "position":
            raise ValueError(f"{group} target_type must be position")
        baseline = (float(source.get("kp", math.nan)), float(source.get("kd", math.nan)))
        if not np.allclose(baseline, GROUP_BASELINE[group], atol=1.0e-12, rtol=0.0):
            raise ValueError(
                f"{group} acquisition gains={baseline}, expected {GROUP_BASELINE[group]}"
            )
        data_path = (truth_run_dir / str(source.get("pt", ""))).resolve()
        data_sha = source.get("pt_sha256")
        if not isinstance(data_sha, str) or sha256_file(data_path) != data_sha:
            raise ValueError(f"{group} truth PT SHA-256 mismatch")
        if frozen.get("pt_sha256") != data_sha:
            raise ValueError(f"{group} truth is not the frozen Group A PACE source")
        for key in ("fixture_hold_kp", "fixture_hold_kd"):
            actual = np.asarray(source.get(key, []), dtype=np.float64)
            expected = np.asarray(frozen.get(key, []), dtype=np.float64)
            if actual.shape != (len(JOINT_NAMES),) or not np.allclose(
                actual, expected, atol=1.0e-12, rtol=0.0
            ):
                raise ValueError(f"{group} {key} does not match the frozen PACE source")
        resolved[group] = source
        paths[group] = data_path
    return resolved, paths, manifest


def reduce_paired_metrics(
    references: tuple[BodeReference, BodeReference],
    response: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"paired response must have shape (N,2), got {values.shape}")
    side_metrics = [
        reference.metrics(values[:, index]) for index, reference in enumerate(references)
    ]
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        out[key] = float(np.mean([metrics[key] for metrics in side_metrics]))
    for side, metrics in zip(SIDE_LABELS, side_metrics, strict=True):
        for key in METRIC_KEYS:
            out[f"{side}_{key}"] = float(metrics[key])
    return out


def set_group_gains(replay: MujocoPaceReplay, group: str, kp: float, kd: float) -> None:
    ids = np.asarray(GROUP_ACTIVE_IDS[group], dtype=np.int64)
    kp_vector = replay.kp.copy()
    kd_vector = replay.kd.copy()
    kp_vector[ids] = float(kp)
    kd_vector[ids] = float(kd)
    replay.kp = kp_vector
    replay.kd = kd_vector


class CandidateTimeoutError(TimeoutError):
    pass


def candidate_timeout_handler(_signum: int, _frame: Any) -> None:
    raise CandidateTimeoutError("candidate evaluation exceeded timeout")


def timeout_penalty_row(context: PairedContext, kp: float, kd: float, stage: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": context.group,
        "stage": f"{stage}_timeout",
        "kp": float(kp),
        "kd": float(kd),
    }
    for key in METRIC_KEYS:
        row[key] = TIMEOUT_PENALTY_SCORE
    for side in SIDE_LABELS:
        for key in METRIC_KEYS:
            row[f"{side}_{key}"] = TIMEOUT_PENALTY_SCORE
    return row


def evaluate_candidate(
    context: PairedContext,
    kp: float,
    kd: float,
    stage: str,
) -> dict[str, Any]:
    timeout = float(getattr(context, "candidate_timeout_s", 0.0))
    previous_handler = None
    if timeout > 0.0:
        previous_handler = signal.signal(signal.SIGALRM, candidate_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    set_group_gains(context.replay, context.group, kp, kd)
    try:
        candidate_source_metadata = dict(context.source.source)
        candidate_source_metadata["kp"] = np.asarray(context.replay.kp, dtype=np.float64).tolist()
        candidate_source_metadata["kd"] = np.asarray(context.replay.kd, dtype=np.float64).tolist()
        candidate_source = SourceData(
            source=candidate_source_metadata,
            payload=context.source.payload,
            target_kind=context.source.target_kind,
            score_time_window=context.source.score_time_window,
        )
        result = context.replay.replay(candidate_source, context.params, log_full=False)
        metrics = reduce_paired_metrics(
            context.references,
            np.asarray(result["response"], dtype=np.float64),
        )
        return {
            "group": context.group,
            "stage": stage,
            "kp": float(kp),
            "kd": float(kd),
            **metrics,
        }
    except CandidateTimeoutError:
        return timeout_penalty_row(context, kp, kd, stage)
    finally:
        if timeout > 0.0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


_WORKER_CONTEXT: PairedContext | None = None


def evaluate_worker(task: tuple[float, float, str]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("paired PD worker context was not initialized")
    kp, kd, stage = task
    return evaluate_candidate(_WORKER_CONTEXT, kp, kd, stage)


def evaluate_grid(
    context: PairedContext,
    store: CandidateStore,
    candidates: Iterable[tuple[float, float]],
    *,
    stage: str,
    score_metric: str,
    workers: int,
    progress_every: int,
) -> None:
    pending = [
        (float(kp), float(kd), stage)
        for kp, kd in candidates
        if store.get(float(kp), float(kd)) is None
    ]
    print(
        f"[SCAN] WE11/{context.group} stage={stage} new_candidates={len(pending)} "
        f"total_before={len(store.rows)} workers={workers}",
        flush=True,
    )
    if not pending:
        return
    if workers == 1:
        rows = (evaluate_candidate(context, kp, kd, stage) for kp, kd, _ in pending)
        executor = None
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel paired PD scan requires multiprocessing start method fork")
        global _WORKER_CONTEXT
        _WORKER_CONTEXT = context
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        )
        futures = [executor.submit(evaluate_worker, task) for task in pending]
        rows = (future.result() for future in as_completed(futures))
    try:
        for index, row in enumerate(rows, start=1):
            store.add(row)
            if index == 1 or index % progress_every == 0 or index == len(pending):
                best = store.best(score_metric)
                metric = PRIMARY_METRICS[score_metric]
                print(
                    f"[PROGRESS] WE11/{context.group} {stage} {index}/{len(pending)} "
                    f"best kp={best['kp']:.6g} kd={best['kd']:.6g} "
                    f"{score_metric}={best[metric]:.9g}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)


def final_trace(
    context: PairedContext,
    best: dict[str, Any],
    effort_limit: np.ndarray,
) -> FinalTrace:
    kp = float(best["kp"])
    kd = float(best["kd"])
    set_group_gains(context.replay, context.group, kp, kd)
    result = context.replay.replay(context.source, context.params, log_full=True)
    response = np.asarray(result["response"], dtype=np.float64)
    metrics = reduce_paired_metrics(context.references, response)
    q = np.asarray(result["q"], dtype=np.float64)
    qd = np.asarray(result["qd"], dtype=np.float64)
    tau = np.asarray(result["tau"], dtype=np.float64)
    active = np.asarray(context.active_joint_ids, dtype=np.int64)
    wheels = np.asarray([2, 5], dtype=np.int64)
    non_source = np.asarray(
        [index for index in range(len(JOINT_NAMES)) if index not in context.active_joint_ids],
        dtype=np.int64,
    )
    active_tau = tau[:, active]
    active_limits = effort_limit[active]
    saturation_count = int(
        np.count_nonzero(np.abs(active_tau) >= active_limits[np.newaxis, :] - 1.0e-9)
    )
    return FinalTrace(
        group=context.group,
        kp=kp,
        kd=kd,
        metrics=metrics,
        result=result,
        wheel_position_drift_rad=float(np.max(np.abs(q[:, wheels] - q[0, wheels]))),
        wheel_max_abs_velocity_rad_s=float(np.max(np.abs(qd[:, wheels]))),
        non_source_position_drift_rad=float(np.max(np.abs(q[:, non_source] - q[0, non_source]))),
        max_abs_torque_nm=float(np.max(np.abs(active_tau))),
        saturation_count=saturation_count,
    )


def write_replay_csv(path: Path, trace: FinalTrace) -> None:
    active = GROUP_ACTIVE_IDS[trace.group]
    result = trace.result
    time = np.asarray(result["time"], dtype=np.float64)
    command = np.asarray(result["des"], dtype=np.float64)
    applied = np.asarray(result["applied_des"], dtype=np.float64)
    truth = np.asarray(result["truth"], dtype=np.float64)
    response = np.asarray(result["response"], dtype=np.float64)
    tau = np.asarray(result["tau"], dtype=np.float64)
    fieldnames = (
        "time_s",
        "left_command_rad",
        "left_applied_command_rad",
        "left_truth_rad",
        "left_mujoco_rad",
        "left_torque_nm",
        "right_command_rad",
        "right_applied_command_rad",
        "right_truth_rad",
        "right_mujoco_rad",
        "right_torque_nm",
    )
    rows = []
    for index, timestamp in enumerate(time):
        rows.append(
            {
                "time_s": float(timestamp),
                "left_command_rad": float(command[index, active[0]]),
                "left_applied_command_rad": float(applied[index, active[0]]),
                "left_truth_rad": float(truth[index, 0]),
                "left_mujoco_rad": float(response[index, 0]),
                "left_torque_nm": float(tau[index, active[0]]),
                "right_command_rad": float(command[index, active[1]]),
                "right_applied_command_rad": float(applied[index, active[1]]),
                "right_truth_rad": float(truth[index, 1]),
                "right_mujoco_rad": float(response[index, 1]),
                "right_torque_nm": float(tau[index, active[1]]),
            }
        )
    atomic_write_csv(path, fieldnames, rows)


def write_report(
    path: Path, contexts: dict[str, PairedContext], traces: dict[str, FinalTrace]
) -> None:
    import matplotlib.pyplot as plt

    rows = sum(len(contexts[group].active_joint_ids) for group in contexts)
    figure, axes = plt.subplots(rows, 3, figsize=(19, 4.2 * rows), squeeze=False)
    row = 0
    for group, context in contexts.items():
        trace = traces[group]
        response = np.asarray(trace.result["response"], dtype=np.float64)
        for side_index, joint_id in enumerate(context.active_joint_ids):
            reference = context.references[side_index]
            axis = axes[row]
            axis[0].plot(
                reference.time, reference.command, color="0.75", linewidth=0.7, label="Command"
            )
            axis[0].plot(reference.time, reference.truth, linewidth=1.0, label="Real")
            axis[0].plot(reference.time, response[:, side_index], linewidth=1.0, label="MuJoCo")
            axis[0].set_title(f"{JOINT_NAMES[joint_id]} | Kp={trace.kp:g}, Kd={trace.kd:g}")
            axis[0].set_xlabel("Time [s]")
            axis[0].set_ylabel("Position [rad]")
            axis[0].grid(True, alpha=0.25)
            axis[0].legend()
            sim_frequency, sim_magnitude, sim_phase = transfer_bode(
                reference.command[reference.fit_mask],
                response[reference.fit_mask, side_index],
                reference.time[reference.fit_mask],
            )
            axis[1].plot(reference.frequency, reference.magnitude_db, label="Real")
            axis[1].plot(sim_frequency, sim_magnitude, label="MuJoCo")
            axis[1].set_xlim(reference.fmin, reference.fmax)
            axis[1].set_title("Bode magnitude")
            axis[1].set_xlabel("Frequency [Hz]")
            axis[1].set_ylabel("Magnitude [dB]")
            axis[1].grid(True, alpha=0.25)
            axis[1].legend()
            axis[2].plot(reference.frequency, reference.phase_deg, label="Real")
            axis[2].plot(sim_frequency, sim_phase, label="MuJoCo")
            axis[2].set_xlim(reference.fmin, reference.fmax)
            axis[2].set_title("Bode phase")
            axis[2].set_xlabel("Frequency [Hz]")
            axis[2].set_ylabel("Phase [deg]")
            axis[2].grid(True, alpha=0.25)
            axis[2].legend()
            row += 1
    delay = int(next(iter(contexts.values())).params["command_delay_steps"])
    figure.suptitle(
        f"WE11 paired leg Kp/Kd fit | RK4 400 Hz / PD 200 Hz / delay {delay}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def build_contexts(
    args: argparse.Namespace,
    pace: dict[str, Any],
    pace_meta: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    data_paths: dict[str, Path],
    model_path: Path,
    groups: tuple[str, ...],
) -> tuple[dict[str, PairedContext], dict[str, Any], np.ndarray]:
    params = normalize_params(pace, len(JOINT_NAMES))
    delay = int(pace["shared_command_delay_steps"])
    params["command_delay_steps"] = delay
    params["torque_delay_steps"] = 0
    initial_qpos = np.asarray(pace_meta["initial_joint_qpos"], dtype=np.float64)
    initial_qvel = np.asarray(pace_meta["initial_joint_qvel"], dtype=np.float64)
    effort_limit = np.asarray(pace_meta["effort_limit"], dtype=np.float64)
    baseline_kp = np.asarray(pace["kp"], dtype=np.float64)
    baseline_kd = np.asarray(pace["kd"], dtype=np.float64)
    contexts: dict[str, PairedContext] = {}
    for group in groups:
        source_metadata = sources[group]
        hold_kp = np.asarray(source_metadata["fixture_hold_kp"], dtype=np.float64)
        hold_kd = np.asarray(source_metadata["fixture_hold_kd"], dtype=np.float64)
        active = GROUP_ACTIVE_IDS[group]
        payload = load_chirp_data(data_paths[group])
        source = SourceData(
            source=source_metadata,
            payload=payload,
            target_kind="position",
        )
        time = np.asarray(payload["time"], dtype=np.float64)
        command = np.asarray(payload["des_dof_pos"], dtype=np.float64)
        truth = np.asarray(payload["dof_pos"], dtype=np.float64)
        references = tuple(
            BodeReference.create(
                time,
                command[:, joint_id],
                truth[:, joint_id],
                fit_start=args.fit_start,
                fit_end_margin=args.fit_end_margin,
                fmin=args.freq_min,
                fmax=args.freq_max,
                phase_weight=args.phase_weight,
                command_psd_threshold_db=args.command_psd_threshold_db,
                chirp_source_frequency_range_hz=(0.1, 5.0),
                chirp_duration_s=float(source_metadata["duration_s"]),
            )
            for joint_id in active
        )
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=400.0,
            control_hz=200.0,
            delay_semantics="command",
            control_mode="position",
            kp=baseline_kp.copy(),
            kd=baseline_kd.copy(),
            effort_limit=effort_limit,
            torque_clip=True,
            free_base=False,
            root_z=float(pace_meta.get("root_z", 0.35)),
            enable_gravity=True,
            fixture_mode="high-impedance",
            fixture_hold_kp=hold_kp,
            fixture_hold_kd=hold_kd,
            lock_wheel_positions=True,
            initial_joint_qpos=initial_qpos,
            initial_joint_qvel=initial_qvel,
        )
        contexts[group] = PairedContext(
            group=group,
            active_joint_ids=active,
            source_metadata=source_metadata,
            source=source,
            references=references,  # type: ignore[arg-type]
            params=params,
            replay=replay,
            candidate_timeout_s=float(getattr(args, "candidate_timeout_s", 0.0)),
        )
    return contexts, params, effort_limit


def scan_group(
    args: argparse.Namespace,
    context: PairedContext,
    store: CandidateStore,
) -> dict[str, Any]:
    search = group_search_range(args, context.group)
    baseline = GROUP_BASELINE[context.group]
    validate_search_range(context.group, search, baseline)
    evaluate_grid(
        context,
        store,
        [baseline],
        stage="acquisition_baseline",
        score_metric=args.score_metric,
        workers=1,
        progress_every=args.progress_every,
    )
    evaluate_grid(
        context,
        store,
        cartesian(
            inclusive_grid(search.kp_min, search.kp_max, search.kp_step),
            inclusive_grid(search.kd_min, search.kd_max, search.kd_step),
        ),
        stage="coarse",
        score_metric=args.score_metric,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    if not args.no_refine:
        coarse_best = store.best(args.score_metric)
        evaluate_grid(
            context,
            store,
            cartesian(
                centered_grid(
                    float(coarse_best["kp"]),
                    args.fine_kp_radius,
                    args.fine_kp_step,
                    search.kp_min,
                    search.kp_max,
                ),
                centered_grid(
                    float(coarse_best["kd"]),
                    args.fine_kd_radius,
                    args.fine_kd_step,
                    search.kd_min,
                    search.kd_max,
                ),
            ),
            stage="fine",
            score_metric=args.score_metric,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        fine_best = store.best(args.score_metric)
        evaluate_grid(
            context,
            store,
            cartesian(
                centered_grid(
                    float(fine_best["kp"]),
                    args.ultra_kp_radius,
                    args.ultra_kp_step,
                    search.kp_min,
                    search.kp_max,
                ),
                centered_grid(
                    float(fine_best["kd"]),
                    args.ultra_kd_radius,
                    args.ultra_kd_step,
                    search.kd_min,
                    search.kd_max,
                ),
            ),
            stage="ultra",
            score_metric=args.score_metric,
            workers=args.workers,
            progress_every=args.progress_every,
        )
    best = store.best(args.score_metric)
    tolerance_kp = 0.5 * args.ultra_kp_step
    tolerance_kd = 0.5 * args.ultra_kd_step
    best["at_kp_boundary"] = bool(
        abs(float(best["kp"]) - search.kp_min) <= tolerance_kp
        or abs(float(best["kp"]) - search.kp_max) <= tolerance_kp
    )
    best["at_kd_boundary"] = bool(
        abs(float(best["kd"]) - search.kd_min) <= tolerance_kd
        or abs(float(best["kd"]) - search.kd_max) <= tolerance_kd
    )
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-params", type=Path, required=True)
    parser.add_argument(
        "--truth-run-dir",
        type=Path,
        default=None,
        help="Defaults to the frozen Group A truth directory stored in the PACE artifact.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Defaults to the exact WE11 model stored in the PACE artifact.",
    )
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument(
        "--out-root", type=Path, default=REPO_ROOT / "outputs/we11_paired_kp_kd_fit"
    )
    parser.add_argument("--run-name", default="we11_paired_kp_kd_rk4")
    parser.add_argument("--score-metric", choices=tuple(PRIMARY_METRICS), default="bode-complex")
    parser.add_argument("--freq-min", type=float, default=0.5)
    parser.add_argument("--freq-max", type=float, default=4.5)
    parser.add_argument("--phase-weight", type=float, default=0.01)
    parser.add_argument("--command-psd-threshold-db", type=float, default=-20.0)
    parser.add_argument("--fit-start", type=float, default=0.5)
    parser.add_argument("--fit-end-margin", type=float, default=0.25)
    parser.add_argument("--thigh-kp-min", type=float, default=1.0)
    parser.add_argument("--thigh-kp-max", type=float, default=3.0)
    parser.add_argument("--thigh-kp-step", type=float, default=0.1)
    parser.add_argument("--thigh-kd-min", type=float, default=0.03)
    parser.add_argument("--thigh-kd-max", type=float, default=0.25)
    parser.add_argument("--thigh-kd-step", type=float, default=0.01)
    parser.add_argument("--calf-kp-min", type=float, default=6.0)
    parser.add_argument("--calf-kp-max", type=float, default=10.0)
    parser.add_argument("--calf-kp-step", type=float, default=0.2)
    parser.add_argument("--calf-kd-min", type=float, default=0.4)
    parser.add_argument("--calf-kd-max", type=float, default=1.2)
    parser.add_argument("--calf-kd-step", type=float, default=0.04)
    parser.add_argument("--fine-kp-radius", type=float, default=0.2)
    parser.add_argument("--fine-kp-step", type=float, default=0.05)
    parser.add_argument("--fine-kd-radius", type=float, default=0.04)
    parser.add_argument("--fine-kd-step", type=float, default=0.01)
    parser.add_argument("--ultra-kp-radius", type=float, default=0.05)
    parser.add_argument("--ultra-kp-step", type=float, default=0.01)
    parser.add_argument("--ultra-kd-radius", type=float, default=0.01)
    parser.add_argument("--ultra-kd-step", type=float, default=0.002)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--candidate-timeout-s",
        type=float,
        default=0.0,
        help="Per-candidate wall-clock timeout inside each worker; 0 disables timeout.",
    )
    parser.add_argument("--no-refine", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, groups: tuple[str, ...]) -> None:
    if args.workers <= 0 or args.progress_every <= 0:
        raise ValueError("workers and progress-every must be positive")
    if not math.isfinite(args.candidate_timeout_s) or args.candidate_timeout_s < 0.0:
        raise ValueError("--candidate-timeout-s must be finite and non-negative")
    if args.freq_min <= 0.0 or args.freq_max <= args.freq_min:
        raise ValueError("frequency range must be positive and increasing")
    if args.freq_min < 0.1 or args.freq_max > 5.0:
        raise ValueError("score range must stay inside the 0.1–5.0 Hz source chirp")
    if args.phase_weight < 0.0 or args.command_psd_threshold_db > 0.0:
        raise ValueError("phase weight/command PSD threshold is invalid")
    for name in (
        "fine_kp_radius",
        "fine_kp_step",
        "fine_kd_radius",
        "fine_kd_step",
        "ultra_kp_radius",
        "ultra_kp_step",
        "ultra_kd_radius",
        "ultra_kd_step",
    ):
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for group in groups:
        validate_search_range(group, group_search_range(args, group), GROUP_BASELINE[group])


def main() -> None:
    args = parse_args()
    groups = selected_groups(args.groups)
    validate_args(args, groups)
    pace_path = resolve_path(args.pace_params)
    pace = load_json(pace_path)
    preliminary_meta = pace.get("meta")
    if not isinstance(preliminary_meta, dict):
        raise ValueError("PACE artifact is missing meta")
    model_path = resolve_path(args.model or preliminary_meta.get("model"))
    pace_meta = validate_pace_contract(pace, model_path)
    truth_run_dir = resolve_path(args.truth_run_dir or pace_meta.get("truth_run_dir"))
    sources, data_paths, manifest = resolve_paired_sources(truth_run_dir, pace_meta, groups)
    contexts, params, effort_limit = build_contexts(
        args,
        pace,
        pace_meta,
        sources,
        data_paths,
        model_path,
        groups,
    )
    out_dir = make_out_dir(args.out_root, args.run_name)
    stores = {
        group: CandidateStore(out_dir / f"{group}_kp_kd_candidates.csv", group) for group in groups
    }
    print(f"[INFO] out_dir={out_dir}", flush=True)
    print(f"[INFO] model={model_path}, integrator=RK4, sim/control=400/200 Hz", flush=True)
    delay_steps = int(pace["shared_command_delay_steps"])
    print(
        f"[INFO] pace_params={pace_path}, shared_command_delay_steps={delay_steps}",
        flush=True,
    )
    print(f"[INFO] truth_run_dir={truth_run_dir}, groups={list(groups)}", flush=True)
    print(
        "[INFO] gravity=true fixed_base=true fixture=high-impedance "
        "wheel_lock=exact score_range=[0.5,4.5]Hz paired_reduction=equal_mean",
        flush=True,
    )
    best_rows: dict[str, dict[str, Any]] = {}
    traces: dict[str, FinalTrace] = {}
    try:
        for group in groups:
            best_rows[group] = scan_group(args, contexts[group], stores[group])
            traces[group] = final_trace(contexts[group], best_rows[group], effort_limit)
            write_replay_csv(out_dir / f"{group}_best_paired_replay.csv", traces[group])
    finally:
        for store in stores.values():
            store.close()
    write_report(out_dir / "paired_pd_report.png", contexts, traces)
    summary_rows = []
    for group in groups:
        best = best_rows[group]
        trace = traces[group]
        summary_rows.append(
            {
                "group": group,
                "kp": float(best["kp"]),
                "kd": float(best["kd"]),
                "score_metric": args.score_metric,
                "score": float(best[PRIMARY_METRICS[args.score_metric]]),
                "bode_magnitude_mse_db2": float(best["bode_magnitude_mse_db2"]),
                "bode_phase_mse_deg2": float(best["bode_phase_mse_deg2"]),
                "time_rmse": float(best["time_rmse"]),
                "at_kp_boundary": bool(best["at_kp_boundary"]),
                "at_kd_boundary": bool(best["at_kd_boundary"]),
                "wheel_position_drift_rad": trace.wheel_position_drift_rad,
                "wheel_max_abs_velocity_rad_s": trace.wheel_max_abs_velocity_rad_s,
                "non_source_position_drift_rad": trace.non_source_position_drift_rad,
                "max_abs_torque_nm": trace.max_abs_torque_nm,
                "saturation_count": trace.saturation_count,
            }
        )
    atomic_write_csv(
        out_dir / "best_kp_kd_summary.csv",
        tuple(summary_rows[0]),
        summary_rows,
    )
    payload = {
        "format": "we11_paired_leg_pd_fit_v1",
        "status": "completed",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "integrator": "RK4",
        "pace_params": str(pace_path),
        "pace_params_sha256": sha256_file(pace_path),
        "truth_run_dir": str(truth_run_dir),
        "truth_manifest_sha256": sha256_file(truth_run_dir / "chirp_source_manifest.json"),
        "groups": {
            row["group"]: {
                **row,
                "baseline_kp": GROUP_BASELINE[row["group"]][0],
                "baseline_kd": GROUP_BASELINE[row["group"]][1],
                "candidate_count": len(stores[row["group"]].rows),
                "active_joint_ids": list(GROUP_ACTIVE_IDS[row["group"]]),
                "source_pt": str(data_paths[row["group"]]),
                "source_pt_sha256": sha256_file(data_paths[row["group"]]),
            }
            for row in summary_rows
        },
        "runtime_contract": {
            "sim_hz": 400.0,
            "control_hz": 200.0,
            "gravity_enabled": True,
            "fixed_base": True,
            "fixture_mode": "high-impedance",
            "fixture_gain_source": "truth_manifest_per_source",
            "source_fixture_gains": {
                group: {
                    "fixture_hold_kp": sources[group]["fixture_hold_kp"],
                    "fixture_hold_kd": sources[group]["fixture_hold_kd"],
                }
                for group in groups
            },
            "lock_wheel_positions": True,
            "delay_semantics": "pre_controller_command_fifo",
            "shared_command_delay_steps": delay_steps,
            "delay_s": delay_steps / 200.0,
            "initial_joint_qpos": pace_meta["initial_joint_qpos"],
            "initial_joint_qvel": pace_meta["initial_joint_qvel"],
            "effort_limit": pace_meta["effort_limit"],
            "frozen_physical_parameters": params,
        },
        "score_contract": {
            "primary_metric": args.score_metric,
            "phase_weight": args.phase_weight,
            "frequency_range_hz": [args.freq_min, args.freq_max],
            "chirp_source_frequency_range_hz": [0.1, 5.0],
            "paired_reduction": "equal_mean_of_left_and_right_trace_metrics",
        },
        "command": shlex.join(sys.argv),
        "artifacts": {
            "summary_csv": str(out_dir / "best_kp_kd_summary.csv"),
            "report_png": str(out_dir / "paired_pd_report.png"),
            "report_pdf": str(out_dir / "paired_pd_report.pdf"),
            "candidate_csvs": {
                group: str(out_dir / f"{group}_kp_kd_candidates.csv") for group in groups
            },
            "replay_csvs": {
                group: str(out_dir / f"{group}_best_paired_replay.csv") for group in groups
            },
        },
    }
    atomic_write_json(out_dir / "best_kp_kd.json", payload)
    print(f"[DONE] wrote result: {out_dir / 'best_kp_kd.json'}", flush=True)
    for row in summary_rows:
        print(
            f"[BEST] {row['group']} kp={row['kp']:.6g} kd={row['kd']:.6g} "
            f"{row['score_metric']}={row['score']:.9g} "
            f"boundary=({row['at_kp_boundary']},{row['at_kd_boundary']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
