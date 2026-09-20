#!/usr/bin/env python3
"""Replay frozen WE11 PACE parameters against an independent sweep group."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bode_reference import BodeReference
from fit_mujoco_pace_params import (
    MujocoPaceReplay,
    SourceData,
    make_out_dir,
    raw_mse,
    raw_rmse,
    sha256_file,
    source_active_joint_ids,
    source_bode_time_window,
    source_result_trace,
    write_bode_report,
    write_replay_csv,
)
from mujoco_dr002_common import JOINT_NAMES, load_chirp_data, normalize_params, resolve_repo_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-params", required=True)
    parser.add_argument("--truth-run-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-root", default="outputs/we11_pace_validation")
    parser.add_argument("--run-name", default="we11_pace_validation")
    parser.add_argument("--freq-range", nargs=2, type=float, default=(0.5, 4.5))
    parser.add_argument(
        "--failure-ratio",
        type=float,
        default=2.0,
        help="Mark generalization failed when validation MSE/train MSE exceeds this ratio.",
    )
    args = parser.parse_args()

    pace_path = resolve_repo_path(args.pace_params)
    truth_dir = resolve_repo_path(args.truth_run_dir)
    pace = load_json(pace_path)
    meta = pace.get("meta")
    if pace.get("format") != "dr002_mujoco_pace_fit_v4" or not isinstance(meta, dict):
        raise ValueError("--pace-params must be a completed dr002_mujoco_pace_fit_v4 artifact")
    model_path = resolve_repo_path(args.model or meta["model"])
    expected_model_sha = str(meta.get("model_sha256_at_fit", ""))
    if sha256_file(model_path) != expected_model_sha:
        raise ValueError("validation model SHA-256 differs from the frozen PACE model")
    manifest_path = truth_dir / "chirp_source_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("joint_names") != JOINT_NAMES:
        raise ValueError("truth joint order does not match WE11 policy order")
    if not math.isclose(float(manifest.get("sample_rate_hz", 0.0)), 200.0, abs_tol=1e-12):
        raise ValueError("validation truth must be resampled to exactly 200 Hz")

    params = normalize_params(pace, len(JOINT_NAMES))
    delay = int(pace["shared_command_delay_steps"])
    params["command_delay_steps"] = delay
    params["torque_delay_steps"] = 0
    initial_qpos_value = meta.get("initial_joint_qpos")
    initial_qvel_value = meta.get("initial_joint_qvel")
    replay = MujocoPaceReplay(
        model_path,
        sim_hz=400.0,
        control_hz=200.0,
        delay_semantics="command",
        control_mode=str(manifest.get("control_mode", meta["control_mode"])),
        kp=finite_vector(pace["kp"], "pace.kp"),
        kd=finite_vector(pace["kd"], "pace.kd"),
        effort_limit=finite_vector(meta["effort_limit"], "meta.effort_limit"),
        torque_clip=bool(meta.get("torque_clip", True)),
        free_base=False,
        root_z=float(meta.get("root_z", 0.35)),
        enable_gravity=True,
        fixture_mode="high-impedance",
        fixture_hold_kp=finite_vector(meta["fixture_hold_kp"], "meta.fixture_hold_kp"),
        fixture_hold_kd=finite_vector(meta["fixture_hold_kd"], "meta.fixture_hold_kd"),
        lock_wheel_positions=bool(meta.get("lock_wheel_positions", True)),
        initial_joint_qpos=None
        if initial_qpos_value is None
        else finite_vector(initial_qpos_value, "meta.initial_joint_qpos"),
        initial_joint_qvel=None
        if initial_qvel_value is None
        else finite_vector(initial_qvel_value, "meta.initial_joint_qvel"),
    )

    out_dir = make_out_dir(resolve_repo_path(args.out_root), args.run_name)
    comparisons: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    fmin, fmax = map(float, args.freq_range)
    effort_limit = finite_vector(meta["effort_limit"], "meta.effort_limit")
    for source_index, source_meta in enumerate(manifest.get("sources", [])):
        pt_path = truth_dir / str(source_meta["pt"])
        if sha256_file(pt_path) != source_meta.get("pt_sha256"):
            raise ValueError(f"source {source_index} PT SHA-256 mismatch")
        payload = load_chirp_data(pt_path)
        item = SourceData(
            source=source_meta,
            payload=payload,
            target_kind=str(source_meta.get("target_type", "position")),
        )
        time = np.asarray(payload["time"], dtype=np.float64)
        window = source_bode_time_window(
            item,
            time,
            chirp_source_frequency_range_hz=(0.1, 5.0),
            fmin=fmin,
            fmax=fmax,
            fit_start=0.0,
            fit_end_margin=0.0,
        )
        result = replay.replay(item, params, log_full=True)
        write_replay_csv(out_dir / f"source{source_index:02d}_replay.csv", result)
        active_ids = source_active_joint_ids(source_meta)
        command_key = "des_dof_vel" if item.target_kind == "velocity" else "des_dof_pos"
        applied_key = "applied_desvel" if item.target_kind == "velocity" else "applied_des"
        source_mses: list[float] = []
        for joint_id in active_ids:
            response = source_result_trace(item, result, "response", joint_id)
            truth = source_result_trace(item, result, "truth", joint_id)
            reference = BodeReference.create(
                time,
                np.asarray(payload[command_key])[:, joint_id],
                truth,
                fit_start=0.0,
                fit_end_margin=0.0,
                fmin=fmin,
                fmax=fmax,
                phase_weight=0.0,
                command_psd_threshold_db=-20.0,
                chirp_source_frequency_range_hz=(0.1, 5.0),
                chirp_duration_s=float(source_meta["duration_s"]),
            )
            bode = reference.metrics(response)
            mse = raw_mse(response, truth, window.mask)
            source_mses.append(mse)
            tau = np.asarray(result["tau"], dtype=np.float64)[:, joint_id]
            saturation_count = int(
                np.count_nonzero(np.isclose(np.abs(tau), effort_limit[joint_id], atol=1e-8))
            )
            rows.append(
                {
                    "source_index": source_index,
                    "joint_id": joint_id,
                    "joint": JOINT_NAMES[joint_id],
                    "kp": source_meta.get("kp"),
                    "kd": source_meta.get("kd"),
                    "time_mse": mse,
                    "time_rmse": raw_rmse(response, truth, window.mask),
                    "bode_magnitude_mse_db2": bode["bode_magnitude_mse_db2"],
                    "bode_phase_mse_deg2": bode["bode_phase_mse_deg2"],
                    "saturation_count": saturation_count,
                    "max_abs_torque_nm": float(np.max(np.abs(tau))),
                }
            )
            comparisons.append(
                {
                    "joint": JOINT_NAMES[joint_id],
                    "target_kind": item.target_kind,
                    "time": time,
                    "command": np.asarray(payload[command_key])[:, joint_id],
                    "applied_command": np.asarray(result[applied_key])[:, joint_id],
                    "truth": truth,
                    "response": response,
                    "fit_mask": window.mask,
                }
            )
        if len(source_mses) == 2:
            difference = abs(source_mses[0] - source_mses[1])
            for row in rows[-2:]:
                row["paired_left_right_mse_abs_difference"] = difference

    fieldnames = tuple(rows[0])
    with (out_dir / "validation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_bode_report(
        out_dir / "pace_validation_report.png",
        comparisons,
        fmin,
        fmax,
        title=f"Frozen PACE validation | delay={delay} ticks | {fmin:g}-{fmax:g} Hz",
    )
    validation_mse = float(np.mean([row["time_mse"] for row in rows]))
    optimizer_meta = meta.get("optimizer", {})
    training_mse = float(
        optimizer_meta.get("best_time_mse", math.nan)
        if isinstance(optimizer_meta, dict)
        else math.nan
    )
    ratio = validation_mse / training_mse if training_mse > 0.0 else math.inf
    failed = not math.isfinite(ratio) or ratio > float(args.failure_ratio)
    report = {
        "format": "we11_mujoco_pace_validation_v1",
        "status": "completed",
        "conclusion": "model_generalization_failed" if failed else "model_generalization_passed",
        "criterion": f"validation_mse / training_mse <= {float(args.failure_ratio):g}",
        "training_mse": training_mse,
        "validation_mse": validation_mse,
        "validation_to_training_mse_ratio": ratio,
        "pace_params": str(pace_path),
        "pace_params_sha256": sha256_file(pace_path),
        "truth_manifest": str(manifest_path),
        "truth_manifest_sha256": sha256_file(manifest_path),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "shared_command_delay_steps": delay,
        "rows": rows,
    }
    (out_dir / "pace_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[DONE] validation: {out_dir / 'pace_validation.json'}", flush=True)
    print(
        f"[RESULT] {report['conclusion']} validation/train MSE ratio={ratio:.6g}", flush=True
    )


if __name__ == "__main__":
    main()
