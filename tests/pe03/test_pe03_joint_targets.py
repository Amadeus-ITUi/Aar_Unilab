"""Per-joint PD limits, actual action histories and legacy model compatibility."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from unilab.base.backend.mujoco.batched_robot import compile_robot_scene
from unilab.envs.locomotion.pe03.config import ROOT, load_config
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv


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


def test_new_model_changes_only_confirmed_hip_limits():
    cfg = config()
    new = compile_robot_scene(ROOT / cfg.env.model_path, tuple(cfg.env.body_names), visual=False)
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
    expected = old.jnt_range.copy()
    inward_limit = np.deg2rad(11.1)
    expected[1], expected[4] = [-inward_limit, 1.57], [-1.57, inward_limit]
    np.testing.assert_array_equal(new.jnt_range, expected)
    urdf = ET.parse(ROOT / "src/unilab/assets/robots/pe03/urdf/pe03_joint_limits.urdf")
    for index, name in enumerate(cfg.env.joint_order, start=1):
        limit = urdf.find(f"joint[@name='{name}']/limit")
        np.testing.assert_allclose(
            new.jnt_range[index], [float(limit.get("lower")), float(limit.get("upper"))]
        )


def test_final_pd_targets_and_history_use_clamped_actions(monkeypatch):
    env = PE03GaitEnv(config(), evaluation=True)
    received = []
    try:
        monkeypatch.setattr(env.backend, "step", lambda offset, *_: received.append(offset.copy()))
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


def test_old_checkpoint_contract_keeps_original_assets_and_unclipped_targets(monkeypatch):
    cfg = config(
        "~control.clip_joint_targets",
        "env.model_path=src/unilab/assets/robots/pe03/scene.xml",
        "env.workspace_path=src/unilab/assets/robots/pe03/analysis/gait_workspace.npz",
    )
    env = PE03GaitEnv(cfg, evaluation=True)
    received = []
    try:
        monkeypatch.setattr(env.backend, "step", lambda offset, *_: received.append(offset.copy()))
        env.step(np.full((2, 6), 100.0))
        assert not env.clip_joint_targets
        assert env.backend.joint_range[0, 0] == 0
        target = env.backend.default_position + received[-1]
        assert (target > env.backend.joint_range[:, 1]).any()
    finally:
        env.close()


def test_clip_targets_requires_boolean():
    with pytest.raises(ValueError, match="clip_joint_targets"):
        config("control.clip_joint_targets=3")
