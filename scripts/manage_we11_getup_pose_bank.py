"""Inspect, review, and export WE11 get-up pose banks with MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = (
    ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "dr002" / "we11" / "scene_flat_we11.xml"
)
DEFAULT_OUTPUT = DEFAULT_SCENE.with_name("getup_pose_bank_v3.npz")

FAMILY_NAMES = {
    0: "front",
    1: "back",
    2: "home",
    3: "getup",
    4: "getup_to_front",
    5: "home_to_getup",
}
FAMILY_IDS = {name: family_id for family_id, name in FAMILY_NAMES.items()}
REVIEW_STATUSES = ("unreviewed", "approved", "rejected")
ROW_FIELDS = {
    "qpos",
    "family",
    "thigh",
    "calf",
    "base_pitch",
    "minimum_clearance",
    "progress",
    # Kept for non-destructive compatibility when reviewing a v1 bank.
    "wing_mode",
    "sampling_weight",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_id(qpos: np.ndarray) -> str:
    # Wings are randomized independently at reset and are not part of the
    # review identity. WE11's two wing qpos entries are the final two entries.
    canonical = np.ascontiguousarray(qpos[:-2], dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def load_bank(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"pose bank does not exist: {path}\n"
            "Generate it first with: python scripts/generate_we11_getup_pose_bank.py"
        )
    with np.load(path, allow_pickle=False) as archive:
        bank = {name: archive[name].copy() for name in archive.files}
    qpos = bank.get("qpos")
    if qpos is None or qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError("pose bank qpos must be a non-empty rank-2 array")
    for required in ("family", "thigh", "calf", "base_pitch"):
        values = bank.get(required)
        if values is None or values.shape != (qpos.shape[0],):
            raise ValueError(f"pose bank field {required!r} must have shape ({qpos.shape[0]},)")
    return bank


def _empty_review(bank_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bank": str(bank_path.resolve()),
        "bank_sha256": _sha256_file(bank_path),
        "poses": {},
    }


def load_review(review_path: Path, bank_path: Path) -> dict[str, Any]:
    if not review_path.exists():
        return _empty_review(bank_path)
    payload_obj = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"unsupported or malformed review file: {review_path}")
    payload: dict[str, Any] = payload_obj
    if payload.get("schema_version") != 1 or not isinstance(payload.get("poses"), dict):
        raise ValueError(f"unsupported or malformed review file: {review_path}")
    current_hash = _sha256_file(bank_path)
    if payload.get("bank_sha256") != current_hash:
        print(
            "[pose-bank] warning: bank SHA256 changed; preserving reviews only for matching "
            "qpos pose IDs"
        )
        payload["bank"] = str(bank_path.resolve())
        payload["bank_sha256"] = current_hash
    return payload


def save_review(review_path: Path, review: dict[str, Any]) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = review_path.with_suffix(review_path.suffix + ".tmp")
    temporary.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(review_path)


def review_status(review: dict[str, Any], qpos: np.ndarray) -> str:
    entry = review["poses"].get(pose_id(qpos))
    if not isinstance(entry, dict):
        return "unreviewed"
    status = str(entry.get("status", "unreviewed"))
    return status if status in REVIEW_STATUSES else "unreviewed"


def set_review_status(review: dict[str, Any], qpos: np.ndarray, status: str, *, index: int) -> None:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"invalid review status: {status}")
    key = pose_id(qpos)
    if status == "unreviewed":
        review["poses"].pop(key, None)
        return
    review["poses"][key] = {
        "status": status,
        "source_index": int(index),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def filtered_indices(
    bank: dict[str, np.ndarray],
    review: dict[str, Any],
    *,
    family: str,
    status: str,
) -> np.ndarray:
    count = bank["qpos"].shape[0]
    mask = np.ones((count,), dtype=np.bool_)
    if family != "all":
        mask &= bank["family"] == FAMILY_IDS[family]
    if status != "all":
        statuses = np.asarray([review_status(review, qpos) for qpos in bank["qpos"]], dtype="U10")
        mask &= statuses == status
    return np.flatnonzero(mask).astype(np.int32)


def review_stats(bank: dict[str, np.ndarray], review: dict[str, Any]) -> dict[str, Any]:
    statuses = [review_status(review, qpos) for qpos in bank["qpos"]]
    result: dict[str, Any] = {
        "total": len(statuses),
        "statuses": {name: statuses.count(name) for name in REVIEW_STATUSES},
        "families": {},
    }
    for family_id, family_name in FAMILY_NAMES.items():
        rows = np.flatnonzero(bank["family"] == family_id)
        family_statuses = [statuses[int(row)] for row in rows]
        result["families"][family_name] = {
            "total": int(rows.size),
            **{name: family_statuses.count(name) for name in REVIEW_STATUSES},
        }
    return result


def export_approved_bank(
    bank: dict[str, np.ndarray],
    review: dict[str, Any],
    source_path: Path,
    output_path: Path,
) -> int:
    approved = np.asarray(
        [review_status(review, qpos) == "approved" for qpos in bank["qpos"]], dtype=np.bool_
    )
    count = int(np.count_nonzero(approved))
    if count == 0:
        raise ValueError("no approved poses are available to export")
    source_count = int(bank["qpos"].shape[0])
    output: dict[str, np.ndarray] = {}
    for name, values in bank.items():
        if name in ROW_FIELDS and values.ndim >= 1 and values.shape[0] == source_count:
            output[name] = values[approved]
        elif name != "metadata_json":
            output[name] = values
    metadata: dict[str, Any] = {}
    if "metadata_json" in bank:
        metadata = json.loads(str(bank["metadata_json"].item()))
    metadata.update(
        {
            "review_export": True,
            "source_bank": str(source_path.resolve()),
            "source_bank_sha256": _sha256_file(source_path),
            "source_pose_count": source_count,
            "approved_pose_count": count,
        }
    )
    output["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(output_path)
    return count


def _describe_pose(bank: dict[str, np.ndarray], review: dict[str, Any], index: int) -> str:
    family = FAMILY_NAMES[int(bank["family"][index])]
    status = review_status(review, bank["qpos"][index])
    progress = ""
    if "progress" in bank and np.isfinite(bank["progress"][index]):
        progress = f" progress={bank['progress'][index]:.3f}"
    return (
        f"index={index} family={family} status={status} "
        f"thigh={bank['thigh'][index]:.4f} calf={bank['calf'][index]:.4f} "
        f"pitch={bank['base_pitch'][index]:.4f}{progress}"
    )


def run_viewer(
    model: mujoco.MjModel,
    bank: dict[str, np.ndarray],
    review: dict[str, Any],
    review_path: Path,
    indices: np.ndarray,
    *,
    start_index: int | None,
    seed: int,
    simulate_passive: bool,
) -> None:
    if indices.size == 0:
        raise ValueError("the selected family/status filters contain no poses")
    if start_index is None:
        cursor = 0
    else:
        matches = np.flatnonzero(indices == start_index)
        if matches.size == 0:
            raise ValueError(f"start index {start_index} is excluded by the active filters")
        cursor = int(matches[0])

    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    pending: list[str] = []
    wing_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("left_wing_joint", "right_wing_joint")
    ]
    if any(joint_id < 0 for joint_id in wing_joint_ids):
        raise ValueError("viewer scene is missing WE11 wing joints")
    wing_qpos_indices = np.asarray([model.jnt_qposadr[joint_id] for joint_id in wing_joint_ids])
    wing_limits = np.asarray(model.jnt_range[wing_joint_ids], dtype=np.float64)
    displayed_wings = rng.uniform(wing_limits[:, 0], wing_limits[:, 1])
    passive_running = simulate_passive
    last_step_time = time.monotonic()

    def on_key(keycode: int) -> None:
        mapping = {
            ord("N"): "next",
            ord("P"): "previous",
            ord("A"): "approved",
            ord("X"): "rejected",
            ord("U"): "unreviewed",
            ord("R"): "random",
            ord("V"): "rerandomize_wings",
            ord("S"): "toggle_passive",
            262: "next",  # GLFW_KEY_RIGHT
            263: "previous",  # GLFW_KEY_LEFT
        }
        action = mapping.get(keycode)
        if action is not None:
            pending.append(action)

    def load_current(*, rerandomize_wings: bool) -> None:
        nonlocal displayed_wings, last_step_time
        index = int(indices[cursor])
        data.qpos[:] = bank["qpos"][index]
        if rerandomize_wings:
            displayed_wings = rng.uniform(wing_limits[:, 0], wing_limits[:, 1])
        data.qpos[wing_qpos_indices] = displayed_wings
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        last_step_time = time.monotonic()
        print(
            f"[pose-bank] {_describe_pose(bank, review, index)} "
            f"wings=[{displayed_wings[0]:.4f}, {displayed_wings[1]:.4f}]"
        )

    print(
        "[pose-bank] controls: N/Right next, P/Left previous, R random, "
        "V new wing angles, S toggle passive physics"
    )
    print("[pose-bank] review: A approve, X reject, U clear review; close viewer to exit")
    print(
        "[pose-bank] passive physics "
        f"{'ON' if passive_running else 'OFF'} (zero actuator command; damping/friction only)"
    )
    load_current(rerandomize_wings=True)
    with mujoco.viewer.launch_passive(model=model, data=data, key_callback=on_key) as viewer:
        viewer.sync()
        while viewer.is_running():
            changed = False
            rerandomize_wings = False
            while pending:
                action = pending.pop(0)
                if action == "next":
                    cursor = (cursor + 1) % int(indices.size)
                    changed = True
                    rerandomize_wings = True
                elif action == "previous":
                    cursor = (cursor - 1) % int(indices.size)
                    changed = True
                    rerandomize_wings = True
                elif action == "random":
                    cursor = int(rng.integers(0, indices.size))
                    changed = True
                    rerandomize_wings = True
                elif action == "rerandomize_wings":
                    changed = True
                    rerandomize_wings = True
                elif action == "toggle_passive":
                    passive_running = not passive_running
                    last_step_time = time.monotonic()
                    print(f"[pose-bank] passive physics {'ON' if passive_running else 'OFF'}")
                else:
                    index = int(indices[cursor])
                    set_review_status(review, bank["qpos"][index], action, index=index)
                    save_review(review_path, review)
                    print(f"[pose-bank] saved {action}: index={index} -> {review_path}")
                    changed = True
            if changed:
                load_current(rerandomize_wings=rerandomize_wings)
                viewer.sync()
            if passive_running:
                now = time.monotonic()
                num_steps = min(int((now - last_step_time) / model.opt.timestep), 100)
                for _ in range(num_steps):
                    mujoco.mj_step(model, data)
                if num_steps:
                    last_step_time += num_steps * model.opt.timestep
                    viewer.sync()
            time.sleep(0.02)


def _parse_indices(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not result or any(index < 0 for index in result):
        raise argparse.ArgumentTypeError("indices must contain nonnegative integers")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument(
        "--pose-type",
        "--family",
        dest="family",
        choices=(
            "all",
            "home",
            "home_to_getup",
            "getup",
            "getup_to_front",
            "front",
            "back",
        ),
        default="all",
        help="Filter by reset-pose type; --family is retained as a compatibility alias.",
    )
    parser.add_argument("--status", choices=("all", *REVIEW_STATUSES), default="all")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--simulate-passive",
        action="store_true",
        help="Advance zero-control MuJoCo dynamics to inspect passive damping/friction stability.",
    )
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--set-status", choices=REVIEW_STATUSES)
    parser.add_argument("--indices", type=_parse_indices)
    parser.add_argument("--export-approved", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bank_path = Path(args.bank).resolve()
    review_path = (
        Path(args.review_file).resolve()
        if args.review_file is not None
        else bank_path.with_suffix(".review.json")
    )
    bank = load_bank(bank_path)
    review = load_review(review_path, bank_path)

    if args.set_status is not None:
        if args.indices is None:
            raise ValueError("--set-status requires --indices")
        for index in args.indices:
            if index >= bank["qpos"].shape[0]:
                raise IndexError(f"pose index {index} is outside the bank")
            set_review_status(review, bank["qpos"][index], args.set_status, index=index)
        save_review(review_path, review)
        print(f"[pose-bank] updated {len(args.indices)} poses in {review_path}")

    if args.export_approved is not None:
        count = export_approved_bank(bank, review, bank_path, args.export_approved.resolve())
        print(f"[pose-bank] exported {count} approved poses to {args.export_approved.resolve()}")

    stats = review_stats(bank, review)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.stats_only or args.set_status is not None or args.export_approved is not None:
        return

    indices = filtered_indices(
        bank,
        review,
        family=args.family,
        status=args.status,
    )
    model = mujoco.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    if model.nq != bank["qpos"].shape[1]:
        raise ValueError(
            f"scene nq={model.nq} does not match bank qpos width={bank['qpos'].shape[1]}"
        )
    run_viewer(
        model,
        bank,
        review,
        review_path,
        indices,
        start_index=args.start_index,
        seed=args.seed,
        simulate_passive=args.simulate_passive,
    )


if __name__ == "__main__":
    main()
