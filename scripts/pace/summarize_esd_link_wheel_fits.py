#!/usr/bin/env python3
"""Summarize independent WE11 ESD-Link wheel PACE fits.

The three wheel Kd sweeps are intentionally fitted independently.  This helper
does not select or merge parameters; it records whether the three independent
fits tell a consistent story.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


WHEEL_JOINTS = ("left_foot_joint", "right_foot_joint")
PARAM_KEYS = ("armature", "viscous_friction", "coulomb_friction", "encoder_bias")
BOUND_TOL = 1.0e-6


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def resolve_fit_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "pace_best_params.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def infer_kd_label(path: Path, payload: dict[str, Any]) -> str:
    text = " ".join(
        str(value)
        for value in (
            path,
            payload.get("meta", {}).get("truth_run_dir", ""),
            payload.get("meta", {}).get("run_name", ""),
        )
    )
    match = re.search(r"kd0p(\d+)", text)
    if match:
        return "0." + match.group(1)
    kd = payload.get("kd")
    if isinstance(kd, list) and len(kd) >= 6:
        values = {float(kd[index]) for index in (2, 5)}
        if len(values) == 1:
            return f"{values.pop():g}"
    return path.parent.name


def metric_value(payload: dict[str, Any]) -> float | None:
    meta = payload.get("meta", {})
    optimizer = meta.get("optimizer", {}) if isinstance(meta, dict) else {}
    for key in ("best_time_mse", "best_score", "time_mse"):
        value = optimizer.get(key) if isinstance(optimizer, dict) else None
        if value is None:
            value = payload.get(key)
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            return scalar
    return None


def optimizer_bounds(payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    optimizer = payload.get("meta", {}).get("optimizer", {})
    if not isinstance(optimizer, dict):
        return {}
    raw = optimizer.get("bounds", {})
    if not isinstance(raw, dict):
        return {}
    bounds: dict[str, tuple[float, float]] = {}
    key_map = {
        "armature": "armature",
        "viscous_friction": "viscous_friction",
        "viscous": "viscous_friction",
        "friction": "coulomb_friction",
        "coulomb_friction": "coulomb_friction",
        "encoder_bias": "encoder_bias",
        "bias": "encoder_bias",
    }
    for raw_key, mapped in key_map.items():
        value = raw.get(raw_key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                bounds[mapped] = (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                pass
    return bounds


def extract_row(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    row: dict[str, Any] = {
        "kd_label": infer_kd_label(path, payload),
        "fit_path": str(path),
        "delay_steps": payload.get("delay_steps", payload.get("command_delay_steps")),
        "delay_s": payload.get("delay_s"),
        "time_mse": metric_value(payload),
    }
    bounds = optimizer_bounds(payload)
    boundary_hits: list[str] = []
    for key in PARAM_KEYS:
        values = payload.get(key, {})
        if not isinstance(values, dict):
            values = {}
        wheel_values = []
        for joint in WHEEL_JOINTS:
            value = values.get(joint)
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                scalar = math.nan
            row[f"{key}_{joint}"] = scalar
            wheel_values.append(scalar)
            if key in bounds and math.isfinite(scalar):
                low, high = bounds[key]
                if abs(scalar - low) <= BOUND_TOL or abs(scalar - high) <= BOUND_TOL:
                    boundary_hits.append(f"{key}:{joint}")
        finite_values = [value for value in wheel_values if math.isfinite(value)]
        if finite_values:
            row[f"{key}_wheel_mean"] = sum(finite_values) / len(finite_values)
            row[f"{key}_left_right_abs_diff"] = (
                abs(finite_values[0] - finite_values[1]) if len(finite_values) == 2 else math.nan
            )
        else:
            row[f"{key}_wheel_mean"] = math.nan
            row[f"{key}_left_right_abs_diff"] = math.nan
    row["boundary_hits"] = ";".join(boundary_hits)
    return row


def consistency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "fit_count": len(rows),
        "kd_labels": [row["kd_label"] for row in rows],
        "delay_steps": [row["delay_steps"] for row in rows],
    }
    for key in PARAM_KEYS:
        means = [float(row[f"{key}_wheel_mean"]) for row in rows]
        means = [value for value in means if math.isfinite(value)]
        if not means:
            continue
        mean = sum(means) / len(means)
        span = max(means) - min(means)
        cv = (math.sqrt(sum((value - mean) ** 2 for value in means) / len(means)) / abs(mean)) if mean else math.inf
        summary[key] = {
            "mean_across_kd": mean,
            "min": min(means),
            "max": max(means),
            "span": span,
            "coefficient_of_variation": cv,
        }
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits", nargs="+", type=Path, help="Fit dirs or pace_best_params.json files.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.fits) != 3:
        raise ValueError("expected exactly three independent wheel fits")

    paths = [resolve_fit_path(path) for path in args.fits]
    rows = sorted((extract_row(path) for path in paths), key=lambda row: row["kd_label"])
    payload = {
        "format": "we11_esd_link_wheel_consistency_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "rows": rows,
        "summary": consistency_summary(rows),
    }
    write_json(args.out_dir / "wheel_fit_consistency.json", payload)
    write_csv(args.out_dir / "wheel_fit_consistency.csv", rows)
    print(f"[OK] wrote {args.out_dir / 'wheel_fit_consistency.json'}")
    print(f"[OK] wrote {args.out_dir / 'wheel_fit_consistency.csv'}")


if __name__ == "__main__":
    main()
