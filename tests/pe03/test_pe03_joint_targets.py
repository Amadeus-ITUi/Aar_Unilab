"""Measured mechanical ranges, final PD targets and action history contracts."""

import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from unilab.base.backend.mujoco.batched_robot import compile_robot_scene
from unilab.base.backend.mujoco.foot_workspace import FootWorkspace, workspace_fingerprint
from unilab.envs.locomotion.pe03.config import ROOT, load_config
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.vector_env import PE03VectorEnv

MEASURED = {
    "L_hip_": [-0.20, 1.50],
    "L_thigh_": [-1.44, 0.00],
    "L_calf_": [-2.02, 0.00],
    "R_hip_": [-1.50, 0.20],
    "R_thigh_": [0.00, 1.44],
    "R_calf_": [0.00, 2.02],
}


def config(*overrides):
    return load_config(
        [
            "+experiment=gait_fixed",
            "algo.num_envs=2",
            "training.device=cpu",
            "training.mujoco_threads=1",
            *overrides,
        ]
    )


@pytest.mark.parametrize(
    "robot,stem,scene",
    [
        ("pe03", "pe03", "scene.xml"),
        ("pe03", "pe03_joint_limits", "scene_joint_limits.xml"),
        ("pe04", "pe04", "scene.xml"),
        ("pe05", "pe05", "scene.xml"),
    ],
)
def test_all_current_models_and_urdfs_use_measured_limits(robot, stem, scene):
    cfg = config()
    assets = ROOT / "src/unilab/assets/robots" / robot
    new = compile_robot_scene(assets / scene, tuple(cfg.env.body_names), visual=False)
    old = compile_robot_scene(
        ROOT / "src/unilab/assets/robots/pe03/scene.xml",
        tuple(cfg.env.body_names),
        visual=False,
    )
    for field in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "jnt_axis",
        "dof_armature",
        "actuator_gear",
        "key_qpos",
    ):
        np.testing.assert_array_equal(getattr(new, field), getattr(old, field))
    urdf = ET.parse(assets / "urdf" / f"{stem}.urdf")
    for name, expected in MEASURED.items():
        joint = new.joint(name)
        np.testing.assert_array_equal(joint.range, expected)
        assert expected[0] <= new.key("home").qpos[joint.qposadr[0]] <= expected[1]
        limit = urdf.find(f"joint[@name='{name}']/limit")
        np.testing.assert_allclose(expected, [float(limit.get("lower")), float(limit.get("upper"))])


def test_final_pd_targets_and_history_use_clamped_actions(monkeypatch):
    env = PE03GaitEnv(config(), evaluation=True)
    received = []
    try:
        monkeypatch.setattr(
            env.backend, "step", lambda offset, *_, **kw: received.append(offset.copy())
        )
        limits = env.backend.joint_range
        # Both signs on every joint catch asymmetric left/right and offset mistakes.
        first = env.step(np.array([[100.0] * 6, [-100.0] * 6]))
        targets = env.backend.default_position + received[-1]
        np.testing.assert_allclose(targets, np.stack((limits[:, 1], limits[:, 0])), atol=1e-15)
        applied = received[-1] / env.config.control.action_scale
        np.testing.assert_allclose(first.obs["frame"][:, 18:24], applied, atol=1e-6)
        assert first.info["diagnostics"]["control/target_clipped_fraction"] == 1
        second = env.step(np.zeros((2, 6)))
        np.testing.assert_allclose(received[-1], 0, atol=1e-15)
        np.testing.assert_allclose(second.obs["frame"][:, 24:30], applied, atol=1e-6)
        np.testing.assert_allclose(second.obs["frame"][:, 18:24], 0, atol=1e-15)
        np.testing.assert_allclose(
            second.obs["actor"].reshape(2, 30, 38)[:, -2, 18:24], applied, atol=1e-6
        )
        assert (
            (env.home[7:] >= env.soft_limits[:, 0]) & (env.home[7:] <= env.soft_limits[:, 1])
        ).all()
    finally:
        env.close()


def test_explicit_unclipped_contract_remains_available_for_old_release_playback(monkeypatch):
    cfg = config(
        "~control.clip_joint_targets",
        "env.model_path=src/unilab/assets/robots/pe03/scene.xml",
        "env.workspace_path=src/unilab/assets/robots/pe03/analysis/gait_workspace.npz",
    )
    env = PE03GaitEnv(cfg, evaluation=True)
    received = []
    try:
        monkeypatch.setattr(
            env.backend, "step", lambda offset, *_, **kw: received.append(offset.copy())
        )
        env.step(np.full((2, 6), 100.0))
        assert not env.clip_joint_targets
        assert env.backend.joint_range[0, 0] == -0.20
        target = env.backend.default_position + received[-1]
        assert (target > env.backend.joint_range[:, 1]).any()
    finally:
        env.close()


def test_clip_targets_requires_boolean():
    with pytest.raises(ValueError, match="clip_joint_targets"):
        config("control.clip_joint_targets=3")


@pytest.mark.parametrize("experiment", ["standing", "walking"])
def test_flat_targets_include_motor_offsets_and_record_applied_actions(monkeypatch, experiment):
    cfg = load_config([f"+experiment={experiment}", "algo.num_envs=2", "training.mujoco_threads=1"])
    env = PE03VectorEnv(cfg, evaluation=True)
    received = []
    try:
        assert env.clip_joint_targets
        env.backend.default_position += np.array([[0.02] * 6, [-0.02] * 6])
        monkeypatch.setattr(
            env.backend, "step", lambda offset, *_, **kw: received.append(offset.copy())
        )
        result = env.step(np.array([[100.0] * 6, [-100.0] * 6]))
        limits = env.backend.joint_range
        target = env.backend.default_position + received[-1]
        np.testing.assert_allclose(target, np.stack((limits[:, 1], limits[:, 0])), atol=1e-15)
        applied = received[-1] / cfg.control.action_scale
        np.testing.assert_allclose(result.obs["frame"][:, 18:24], applied, atol=1e-6)
        np.testing.assert_allclose(env.actions, applied)
        width = np.diff(limits, axis=1)[:, 0]
        margin = width * (1 - cfg.reward.soft_joint_limit) / 2
        np.testing.assert_allclose(env.soft_limits, limits + np.column_stack((margin, -margin)))
        env.cfg["env"]["joint_reset_range"] = [-10, 10]
        env.evaluation = False
        for _ in range(8):
            env._reset_ids(np.arange(2))
            assert (env.backend.qpos[:, 7:] >= limits[:, 0]).all()
            assert (env.backend.qpos[:, 7:] <= limits[:, 1]).all()
    finally:
        env.close()


@pytest.mark.parametrize("experiment", ["standing", "walking"])
def test_flat_clip_targets_requires_boolean(experiment):
    with pytest.raises(ValueError, match="clip_joint_targets"):
        load_config([f"+experiment={experiment}", "control.clip_joint_targets=3"])


@pytest.mark.parametrize("suffix", ["", "_joint_limits"])
def test_rebuilt_workspace_matches_scene_and_measured_bounds(suffix, tmp_path):
    assets = ROOT / "src/unilab/assets/robots/pe03"
    scene = assets / f"scene{suffix}.xml"
    with np.load(assets / f"analysis/gait_workspace{suffix}.npz", allow_pickle=False) as data:
        assert str(data["fingerprint"]) == workspace_fingerprint(scene)
        assert int(data["resolution"]) == 41
        bounds = np.array(list(MEASURED.values())).reshape(2, 3, 2)
        for side in range(2):
            joints = data[f"joints_{side}"]
            assert len(joints) == len(data[f"points_{side}"]) > 0
            assert (joints >= bounds[side, :, 0]).all()
            assert (joints <= bounds[side, :, 1]).all()
        stale = {name: data[name] for name in data.files}
    stale["fingerprint"] = np.array("old-asset-fingerprint")
    np.savez(tmp_path / "stale.npz", **stale)
    with pytest.raises(ValueError, match="rebuild"):
        FootWorkspace(tmp_path / "stale.npz", scene, stale["reference_points"])


@pytest.mark.parametrize("robot", ["pe04", "pe05"])
def test_measured_asset_provenance_matches_independent_snapshot(robot):
    assets = ROOT / "src/unilab/assets/robots"
    manifest = json.loads((assets / robot / "provenance.json").read_text())
    assert manifest["source"] == "PE03 pe03-cnc-joint-limits-v4"
    assert manifest["original_source"] == "PE03 pe03-cnc-joint-limits-v3"
    assert manifest["joint_limit_revision"]["limits"] == MEASURED
    renamed = {
        "pe03_joint_limits.xml": f"{robot}.xml",
        "scene_joint_limits.xml": "scene.xml",
        "urdf/pe03_joint_limits.urdf": f"urdf/{robot}.urdf",
    }
    for name, digest in manifest["sha256"].items():
        source = assets / "pe03" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        target = assets / robot / renamed.get(name, name)
        if source.suffix in (".xml", ".urdf"):
            expected = source.read_text().replace("pe03_joint_limits", robot).replace("pe03", robot)
            assert target.read_text() == expected
        else:
            assert target.read_bytes() == source.read_bytes()
