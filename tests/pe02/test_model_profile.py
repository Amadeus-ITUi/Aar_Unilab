import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.catalog import catalog
from unilab.catalog.cli import build_legacy_command
from unilab.envs.locomotion.pe02 import PE02Env
from unilab.envs.locomotion.pe02.config import load_legacy_config as load_config

ROOT = Path(__file__).resolve().parents[2]
ROBOT = catalog.robots["pe02"]
SELECTORS = [
    "robot=pe02",
    "task=pe02_flat",
    "observation=pe02_v1",
    "policy=pe02_encoder_mlp",
    "algorithm=pe02_custom_ppo",
    "simulator=mujoco",
]


@pytest.mark.parametrize("mode", ["train", "play"])
def test_pe02_routes_to_independent_adapter(mode):
    command = build_legacy_command(mode, SELECTORS, ROOT)
    assert command[1] == str(ROOT / "scripts/train_pe02.py")
    assert "robot=pe02" in command
    assert "task=pe02_flat" in command
    assert ("mode=play" in command) == (mode == "play")
    selected = catalog.resolve(dict(item.split("=", 1) for item in SELECTORS))
    assert selected.observation is catalog.observations["pe02_v1"]
    assert selected.policy is catalog.policies["pe02_encoder_mlp"]
    assert selected.algorithm is catalog.algorithms["pe02_custom_ppo"]
    assert selected.task.owner_config == "pe02/task/pe02_flat"
    assert (ROOT / "conf" / f"{selected.task.owner_config}.yaml").is_file()


def test_pe02_urdf_and_runtime_model_preserve_action_contract():
    urdf = ET.parse(ROOT / ROBOT.asset).getroot()
    assert urdf.attrib["name"] == "pe02"
    active_joints = [joint for joint in urdf.findall("joint") if joint.attrib["type"] != "fixed"]
    assert tuple(joint.attrib["name"] for joint in active_joints) == ROBOT.joints
    env = PE02Env()
    assert (env.model.nq, env.model.nv, env.model.nu) == (13, 12, 6)
    for index, joint in enumerate(active_joints):
        joint_id = int(env.model.actuator_trnid[index, 0])
        assert (
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            == joint.attrib["name"]
        )
        assert (
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            == ROBOT.actuators[index]
        )
        np.testing.assert_allclose(
            env.model.jnt_axis[joint_id], np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        )
        limit = joint.find("limit").attrib
        np.testing.assert_allclose(
            env.model.jnt_range[joint_id], [float(limit["lower"]), float(limit["upper"])]
        )
    assert np.isclose(
        env.model.body_mass.sum(),
        sum(float(mass.attrib["value"]) for mass in urdf.findall(".//mass")),
    )
    np.testing.assert_allclose(env.model.actuator_gear[:, 0], [5.5, 5.5, 14, 5.5, 5.5, 14])
    np.testing.assert_allclose(env.model.actuator_ctrlrange, [[-1, 1]] * 6)
    np.testing.assert_allclose(env.model.dof_damping[6:], 0.2)
    np.testing.assert_allclose(env.model.dof_armature[6:], [0.01, 0.01, 0.0193715] * 2)
    assert env.model.opt.timestep == 1.0 / 400
    observation = env.reset()
    assert observation.actor.shape == (300,)
    assert observation.critic.shape == (33,)
    for _ in range(5):
        observation, reward, done, _ = env.step(np.zeros(6, dtype=np.float32))
        assert np.isfinite(observation.actor).all()
        assert np.isfinite(observation.critic).all()
        assert np.isfinite(reward)
        assert isinstance(done, bool)
    assert np.isclose(env.data.time, 5 / env.policy_hz)


def test_pe02_rejects_pe01_selectors():
    values = dict(item.split("=", 1) for item in SELECTORS)
    with pytest.raises(ValueError, match="observation=.*does not support"):
        catalog.resolve({**values, "observation": "pe01_legacy"})
    with pytest.raises(ValueError, match="policy=.*does not support"):
        catalog.resolve({**values, "policy": "pe01_encoder_mlp", "algorithm": "pe01_custom_ppo"})


def test_pe02_hydra_config_controls_environment_and_is_not_shared():
    config = load_config(
        [
            "env.history_length=4",
            "env.frame_size=32",
            "control.physics_hz=200",
            "control.policy_hz=25",
            "env.initial_height=0.43",
            "control.action_scale=0.5",
            "reward.base_height=2.0",
            "reward.action_l2=-0.25",
        ]
    )
    env = PE02Env(config=config)
    observation = env.reset()
    assert observation.actor.shape == (128,)
    assert observation.critic.shape == (35,)
    assert env.data.qpos[2] == 0.43
    action = np.full(6, 2.0, dtype=np.float32)
    _, reward, _, info = env.step(action)
    np.testing.assert_allclose(env.data.ctrl, 0.5)
    assert np.isclose(reward, 2.0 * info["base_height"] - 0.25 * 24)
    assert np.isclose(env.data.time, 1.0 / 25)
    assert env.model.opt.timestep == 1.0 / 200
    assert load_config().env.history_length == 10


def test_pe02_assets_are_self_contained_and_original_meshes_are_preserved():
    root = (ROOT / ROBOT.scene).parent
    manifest = json.loads((root / "asset_manifest.json").read_text())
    assert manifest["source_asset_name"] == "dianzu23"
    assert manifest["visual_mode"] == "original_obj"
    assert len(manifest["meshes"]) == 9
    for mesh in manifest["meshes"]:
        for key in ("source", "runtime"):
            path = root / mesh[key]
            assert not path.is_symlink()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == mesh[f"{key}_sha256"]
        assert Path(mesh["runtime"]).suffix == ".obj"
        assert mesh["runtime_faces"] == mesh["source_faces"] > 0
    for mesh in ET.parse(ROOT / ROBOT.asset).findall(".//mesh"):
        uri = mesh.attrib["filename"]
        assert uri.startswith("package://pe02/")
        assert (root / uri.removeprefix("package://pe02/")).is_file()
