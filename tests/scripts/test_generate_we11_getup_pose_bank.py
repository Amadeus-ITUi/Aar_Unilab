from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
from scripts.generate_we11_getup_pose_bank import (
    DEFAULT_SCENE,
    FAMILY_BACK,
    FAMILY_FRONT,
    generate_pose_bank,
)
from scripts.manage_we11_getup_pose_bank import (
    export_approved_bank,
    filtered_indices,
    load_bank,
    load_review,
    review_stats,
    save_review,
    set_review_status,
)


def test_coarse_we11_getup_pose_bank_contains_valid_front_and_back_families(
    tmp_path: Path,
) -> None:
    args = Namespace(
        scene=DEFAULT_SCENE,
        output=tmp_path / "unused.npz",
        family="both",
        thigh_step=0.25,
        calf_step=0.25,
        pitch_samples=121,
        wing_validation_samples=3,
        contact_depth=0.0005,
        contact_tolerance=0.001,
        max_penetration=0.002,
        summary_only=True,
    )

    arrays, summary = generate_pose_bank(args)

    assert arrays["qpos"].shape[1] == 15
    assert np.all(np.isfinite(arrays["qpos"]))
    assert np.any(arrays["family"] == FAMILY_FRONT)
    assert np.any(arrays["family"] == FAMILY_BACK)
    np.testing.assert_allclose(arrays["thigh"][arrays["family"] == FAMILY_FRONT], 1.57)
    back_thigh = arrays["thigh"][arrays["family"] == FAMILY_BACK]
    assert np.all((back_thigh >= -0.13) & (back_thigh <= 0.60))
    assert np.all((arrays["calf"] >= -2.60) & (arrays["calf"] <= -0.17))
    assert np.all(arrays["minimum_clearance"] >= -args.max_penetration)
    np.testing.assert_allclose(arrays["wing_joint_lower"], [-1.5708, -1.5708])
    np.testing.assert_allclose(arrays["wing_joint_upper"], [0.0, 0.0])
    # One row represents one lower-body pose; stored wings are only a canonical
    # midpoint and are randomized independently by reset/viewer consumers.
    assert np.unique(arrays["qpos"][:, :13], axis=0).shape[0] == arrays["qpos"].shape[0]
    assert summary["valid_poses"] == arrays["qpos"].shape[0]
    assert summary["rejected"].get("wing_sweep_other_geom_penetration", 0) > 0


def test_pose_bank_review_is_non_destructive_and_exports_only_approved(tmp_path: Path) -> None:
    source = tmp_path / "poses.npz"
    qpos = np.asarray(
        [
            [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.57, -2.6, 0.0, 1.57, -2.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.1, -1.0, 0.0, 0.1, -1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        source,
        qpos=qpos,
        family=np.asarray([FAMILY_FRONT, FAMILY_BACK], dtype=np.int8),
        thigh=np.asarray([1.57, 0.1]),
        calf=np.asarray([-2.6, -1.0]),
        base_pitch=np.asarray([1.0, -1.0]),
        wing_joint_lower=np.asarray([-1.5708, -1.5708]),
        wing_joint_upper=np.asarray([0.0, 0.0]),
        metadata_json=np.asarray('{"schema_version": 2}'),
    )
    bank = load_bank(source)
    review_path = tmp_path / "poses.review.json"
    review = load_review(review_path, source)

    set_review_status(review, bank["qpos"][0], "approved", index=0)
    set_review_status(review, bank["qpos"][1], "rejected", index=1)
    save_review(review_path, review)
    restored = load_review(review_path, source)

    assert review_stats(bank, restored)["statuses"] == {
        "unreviewed": 0,
        "approved": 1,
        "rejected": 1,
    }
    np.testing.assert_array_equal(
        filtered_indices(
            bank,
            restored,
            family="front",
            status="approved",
        ),
        [0],
    )

    approved_path = tmp_path / "approved.npz"
    assert export_approved_bank(bank, restored, source, approved_path) == 1
    approved = load_bank(approved_path)
    np.testing.assert_array_equal(approved["qpos"], qpos[:1])
    np.testing.assert_array_equal(approved["wing_joint_lower"], [-1.5708, -1.5708])
    np.testing.assert_array_equal(approved["wing_joint_upper"], [0.0, 0.0])
    assert source.is_file()
