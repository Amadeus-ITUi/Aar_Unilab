"""Generate collision-checked WE11 forward/backward get-up reset poses offline.

The generated bank is deliberately a cold-path artifact. Training reset only
needs to sample and copy a saved qpos; no MuJoCo kinematics or contact solve is
required in the reset hot loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = (
    ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "dr002" / "we11" / "scene_flat_we11.xml"
)
DEFAULT_OUTPUT = DEFAULT_SCENE.with_name("getup_pose_bank_v2.npz")

FAMILY_FRONT = 0
FAMILY_BACK = 1
FAMILY_NAMES = {FAMILY_FRONT: "front", FAMILY_BACK: "back"}


@dataclass(frozen=True)
class FamilySpec:
    family_id: int
    target_geom: str
    thigh_range: tuple[float, float]
    pitch_range: tuple[float, float]


FAMILY_SPECS = {
    "front": FamilySpec(
        family_id=FAMILY_FRONT,
        target_geom="base_shoulder_collision",
        thigh_range=(1.57, 1.57),
        pitch_range=(0.02, 1.56),
    ),
    "back": FamilySpec(
        family_id=FAMILY_BACK,
        target_geom="U2_collision",
        thigh_range=(-0.13, 0.60),
        pitch_range=(-1.56, -0.02),
    ),
}


def _inclusive_grid(low: float, high: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("angle grid step must be positive")
    if high < low:
        raise ValueError("angle grid upper bound must be >= lower bound")
    if np.isclose(low, high):
        return np.asarray([low], dtype=np.float64)
    count = int(np.floor((high - low) / step))
    values = low + np.arange(count + 1, dtype=np.float64) * step
    if values[-1] < high - 1.0e-12:
        values = np.append(values, high)
    else:
        values[-1] = high
    return values


def _object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object {name!r} ({object_type.name}) was not found")
    return int(object_id)


def _geom_vertical_extent(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    geom_type = int(model.geom_type[geom_id])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(np.dot(np.abs(rotation[2]), size[:3]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        axis_z = float(rotation[2, 2])
        radial_z = float(size[0]) * float(np.sqrt(max(0.0, 1.0 - axis_z * axis_z)))
        return radial_z + float(size[1]) * abs(axis_z)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(size[0]) + float(size[1]) * abs(float(rotation[2, 2]))
    raise ValueError(
        "pose-bank validation only supports box/cylinder/sphere/capsule collision geoms; "
        f"geom {geom_id} has type {geom_type}"
    )


def _geom_bottom(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    return float(data.geom_xpos[geom_id, 2]) - _geom_vertical_extent(model, data, geom_id)


def _set_base_pitch(qpos: np.ndarray, pitch: float) -> None:
    half = 0.5 * pitch
    qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]


def _find_pitch_roots(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    *,
    wheel_geom_ids: tuple[int, int],
    target_geom_id: int,
    pitch_range: tuple[float, float],
    pitch_samples: int,
) -> list[float]:
    if pitch_samples < 3:
        raise ValueError("pitch_samples must be at least 3")

    def gap(pitch: float) -> float:
        data.qpos[:] = qpos
        data.qpos[2] = 0.0
        _set_base_pitch(data.qpos, pitch)
        mujoco.mj_kinematics(model, data)
        wheel_bottom = 0.5 * sum(_geom_bottom(model, data, gid) for gid in wheel_geom_ids)
        return _geom_bottom(model, data, target_geom_id) - wheel_bottom

    pitches = np.linspace(pitch_range[0], pitch_range[1], pitch_samples)
    gaps = np.asarray([gap(float(pitch)) for pitch in pitches])
    roots: list[float] = []
    for index in range(pitches.size - 1):
        left_pitch = float(pitches[index])
        right_pitch = float(pitches[index + 1])
        left_gap = float(gaps[index])
        right_gap = float(gaps[index + 1])
        if abs(left_gap) <= 1.0e-10:
            roots.append(left_pitch)
            continue
        if left_gap * right_gap > 0.0:
            continue
        for _ in range(48):
            middle = 0.5 * (left_pitch + right_pitch)
            middle_gap = gap(middle)
            if left_gap * middle_gap <= 0.0:
                right_pitch = middle
                right_gap = middle_gap
            else:
                left_pitch = middle
                left_gap = middle_gap
        roots.append(0.5 * (left_pitch + right_pitch))
    if abs(float(gaps[-1])) <= 1.0e-10:
        roots.append(float(pitches[-1]))
    return sorted(set(round(root, 12) for root in roots), key=abs)


def _collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    result: list[int] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if int(model.geom_contype[geom_id]) == 0:
            continue
        result.append(geom_id)
    return result


def _validate_ground_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    *,
    collision_geom_ids: list[int],
    wheel_geom_ids: tuple[int, int],
    target_geom_id: int,
    contact_depth: float,
    contact_tolerance: float,
    max_penetration: float,
) -> tuple[bool, str, float]:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    bottoms = {geom_id: _geom_bottom(model, data, geom_id) for geom_id in collision_geom_ids}
    target_bottom = bottoms[target_geom_id]
    wheel_bottoms = [bottoms[geom_id] for geom_id in wheel_geom_ids]
    expected = -contact_depth
    if abs(target_bottom - expected) > contact_tolerance:
        return False, "target_not_grounded", min(bottoms.values())
    if max(abs(bottom - expected) for bottom in wheel_bottoms) > contact_tolerance:
        return False, "wheel_not_grounded", min(bottoms.values())
    minimum = min(bottoms.values())
    if minimum < -max_penetration:
        return False, "other_geom_penetration", minimum
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qacc)):
        return False, "nonfinite_state", minimum
    return True, "valid", minimum


def generate_pose_bank(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    scene_path = Path(args.scene).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    home_key = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home_qpos = np.asarray(model.key_qpos[home_key], dtype=np.float64).copy()

    joint_names = (
        "left_thigh_joint",
        "left_calf_joint",
        "right_thigh_joint",
        "right_calf_joint",
        "left_wing_joint",
        "right_wing_joint",
    )
    joint_qpos = {
        name: int(model.jnt_qposadr[_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in joint_names
    }
    wing_joint_ids = np.asarray(
        [
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wing_joint"),
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wing_joint"),
        ],
        dtype=np.int32,
    )
    wing_limits = np.asarray(model.jnt_range[wing_joint_ids], dtype=np.float64)
    wheel_geom_ids = (
        _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
        _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"),
    )
    collision_geom_ids = _collision_geom_ids(model)
    calf_joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_calf_joint")
    calf_low, calf_high = (float(value) for value in model.jnt_range[calf_joint_id])
    calf_values = _inclusive_grid(calf_low, calf_high, args.calf_step)
    if args.wing_validation_samples < 2:
        raise ValueError("wing_validation_samples must be at least 2")
    wing_validation_values = [
        np.linspace(limits[0], limits[1], args.wing_validation_samples) for limits in wing_limits
    ]
    canonical_wing_qpos = np.mean(wing_limits, axis=1)

    rows: list[np.ndarray] = []
    family_ids: list[int] = []
    thighs: list[float] = []
    calves: list[float] = []
    pitches: list[float] = []
    minimum_clearances: list[float] = []
    rejected: Counter[str] = Counter()
    attempted_leg_configs = 0
    attempted_wing_checks = 0

    family_names = tuple(FAMILY_SPECS) if args.family == "both" else (args.family,)
    for family_name in family_names:
        spec = FAMILY_SPECS[family_name]
        target_geom_id = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.target_geom)
        thigh_values = _inclusive_grid(*spec.thigh_range, args.thigh_step)
        for thigh in thigh_values:
            for calf in calf_values:
                attempted_leg_configs += 1
                candidate = home_qpos.copy()
                candidate[0:2] = 0.0
                candidate[joint_qpos["left_thigh_joint"]] = thigh
                candidate[joint_qpos["right_thigh_joint"]] = thigh
                candidate[joint_qpos["left_calf_joint"]] = calf
                candidate[joint_qpos["right_calf_joint"]] = calf
                roots = _find_pitch_roots(
                    model,
                    data,
                    candidate,
                    wheel_geom_ids=wheel_geom_ids,
                    target_geom_id=target_geom_id,
                    pitch_range=spec.pitch_range,
                    pitch_samples=args.pitch_samples,
                )
                if not roots:
                    rejected["no_support_pitch"] += 1
                    continue
                pitch = roots[0]
                _set_base_pitch(candidate, pitch)
                candidate[2] = 0.0
                data.qpos[:] = candidate
                mujoco.mj_kinematics(model, data)
                support_bottom = 0.5 * sum(
                    _geom_bottom(model, data, geom_id) for geom_id in wheel_geom_ids
                )
                candidate[2] = -support_bottom - args.contact_depth

                minimum = np.inf
                invalid_reason: str | None = None
                for left_wing in wing_validation_values[0]:
                    for right_wing in wing_validation_values[1]:
                        attempted_wing_checks += 1
                        pose = candidate.copy()
                        pose[joint_qpos["left_wing_joint"]] = left_wing
                        pose[joint_qpos["right_wing_joint"]] = right_wing
                        valid, reason, pose_minimum = _validate_ground_pose(
                            model,
                            data,
                            pose,
                            collision_geom_ids=collision_geom_ids,
                            wheel_geom_ids=wheel_geom_ids,
                            target_geom_id=target_geom_id,
                            contact_depth=args.contact_depth,
                            contact_tolerance=args.contact_tolerance,
                            max_penetration=args.max_penetration,
                        )
                        minimum = min(minimum, pose_minimum)
                        if not valid:
                            invalid_reason = reason
                            break
                    if invalid_reason is not None:
                        break
                if invalid_reason is not None:
                    rejected[f"wing_sweep_{invalid_reason}"] += 1
                    continue

                pose = candidate.copy()
                pose[joint_qpos["left_wing_joint"]] = canonical_wing_qpos[0]
                pose[joint_qpos["right_wing_joint"]] = canonical_wing_qpos[1]
                rows.append(pose)
                family_ids.append(spec.family_id)
                thighs.append(float(thigh))
                calves.append(float(calf))
                pitches.append(float(pitch))
                minimum_clearances.append(float(minimum))

    qpos = np.asarray(rows, dtype=np.float64).reshape(-1, model.nq)
    arrays = {
        "qpos": qpos,
        "family": np.asarray(family_ids, dtype=np.int8),
        "thigh": np.asarray(thighs, dtype=np.float64),
        "calf": np.asarray(calves, dtype=np.float64),
        "base_pitch": np.asarray(pitches, dtype=np.float64),
        "minimum_clearance": np.asarray(minimum_clearances, dtype=np.float64),
        "wing_joint_lower": np.asarray(wing_limits[:, 0], dtype=np.float64),
        "wing_joint_upper": np.asarray(wing_limits[:, 1], dtype=np.float64),
    }
    family_counts = {
        name: int(np.count_nonzero(arrays["family"] == family_id))
        for family_id, name in FAMILY_NAMES.items()
    }
    summary = {
        "schema_version": 2,
        "scene": str(scene_path),
        "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "nq": int(model.nq),
        "angle_grid": {
            "thigh_step": float(args.thigh_step),
            "calf_step": float(args.calf_step),
        },
        "contact": {
            "depth": float(args.contact_depth),
            "tolerance": float(args.contact_tolerance),
            "max_penetration": float(args.max_penetration),
        },
        "attempted_leg_configs": attempted_leg_configs,
        "attempted_wing_checks": attempted_wing_checks,
        "valid_poses": int(qpos.shape[0]),
        "family_counts": family_counts,
        "wing_reset": {
            "mode": "independent_uniform",
            "joint_lower": arrays["wing_joint_lower"].tolist(),
            "joint_upper": arrays["wing_joint_upper"].tolist(),
            "offline_grid_samples_per_joint": int(args.wing_validation_samples),
        },
        "rejected": dict(sorted(rejected.items())),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(summary, sort_keys=True))
    return arrays, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--family", choices=("front", "back", "both"), default="both")
    parser.add_argument("--thigh-step", type=float, default=0.02)
    parser.add_argument("--calf-step", type=float, default=0.02)
    parser.add_argument("--pitch-samples", type=int, default=321)
    parser.add_argument("--wing-validation-samples", type=int, default=9)
    parser.add_argument("--contact-depth", type=float, default=0.0005)
    parser.add_argument("--contact-tolerance", type=float, default=0.001)
    parser.add_argument("--max-penetration", type=float, default=0.002)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Run collection and print diagnostics without writing the pose bank.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arrays, summary = generate_pose_bank(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_only:
        print("[generate_we11_getup_pose_bank] summary-only: no files written")
        return
    if arrays["qpos"].shape[0] == 0:
        raise RuntimeError("no valid poses were generated; refusing to write an empty bank")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[generate_we11_getup_pose_bank] wrote {output}")
    print(f"[generate_we11_getup_pose_bank] wrote {summary_path}")


if __name__ == "__main__":
    main()
