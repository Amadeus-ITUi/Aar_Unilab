import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.adapters.pe02_ppo import load_policy, train_minimal
from unilab.algos.torch.pe02 import PE02EncoderPolicy
from unilab.catalog import catalog
from unilab.envs.locomotion.pe02 import PE02Env
from unilab.envs.locomotion.pe02.config import load_legacy_config as load_config
from unilab.release_contract import load_manifest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("reset_mode", ["home", "height_override", "legacy"])
def test_pe02_training_and_relocated_release_use_pe02_assets(tmp_path, reset_mode):
    robot = catalog.robots["pe02"]
    config = load_config(
        [
            "training.steps=2",
            "env.history_length=4",
            "network.encoder_hidden_dims=[64]",
            "network.actor_hidden_dims=[32,32]",
            "network.critic_hidden_dims=[32]",
            "network.latent_dim=8",
            "control.physics_hz=200",
        ]
    )
    if reset_mode != "home":
        config.env.initial_height = 0.41
    if reset_mode == "legacy":
        del config.env.reset_keyframe
    result = train_minimal(tmp_path / "run", config=config)
    policy = load_policy(result.checkpoint)
    assert policy.history_dim == 120
    assert policy.encoder[0].out_features == 64
    assert policy.actor[0].out_features == 32
    assert torch.isfinite(
        policy.action_mean(torch.zeros(1, 120), torch.zeros(1, 30), torch.zeros(1, 3))
    ).all()
    # Match CLI isolation: the Torch 2.7 legacy tracer can crash inside pytest on Python 3.13.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, importlib.abc\n"
            "from pathlib import Path\n"
            "class RejectPE01(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if 'pe01' in fullname.lower(): raise AssertionError(fullname)\n"
            "sys.meta_path.insert(0, RejectPE01())\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from scripts.train_pe02 import export_release\n"
            "export_release(Path(sys.argv[2]), 'smoke', 'cpu')\n",
            str(ROOT),
            str(result.checkpoint),
        ],
        cwd=tmp_path,
        check=True,
        timeout=60,
    )
    release = tmp_path / "releases/pe02/pe02_flat/smoke"
    relocated = tmp_path / "relocated"
    shutil.move(release, relocated)
    manifest = load_manifest(relocated / "deployment_manifest.json")
    assert manifest["robot"]["id"] == "pe02"
    assert manifest["robot"]["joint_order"] == list(robot.joints)
    assert manifest["task"]["id"] == "pe02_flat"
    assert manifest["policy"]["observation_builder"] == "pe02_v1"
    assert manifest["policy"]["history"]["length"] == 4
    assert manifest["policy"]["inputs"][0]["shape"] == [1, 120]
    assert manifest["control"]["physics_hz"] == 200
    assert "robot: pe02" in (relocated / "runtime_config.yaml").read_text()
    assert len(list((relocated / "robot/runtime_meshes").glob("*.obj"))) == 9
    collision_meshes = list((relocated / "robot/collision_meshes").glob("*.STL"))
    assert (
        len(collision_meshes)
        == len(list((ROOT / robot.scene).parent.glob("collision_meshes/*.STL")))
        > 0
    )
    assert manifest["robot"]["asset_version"] == robot.asset_version
    assert not list((relocated / "robot").glob("pe01*"))
    restored_config = OmegaConf.load(relocated / "runtime_config.yaml")
    env = PE02Env(relocated / manifest["artifacts"]["scene_path"], config=restored_config)
    env.reset()
    assert env.model.opt.timestep == 1.0 / 200
    if reset_mode != "home":
        assert env.model.qpos0[2] == 0.41
        assert env.data.qpos[2] == 0.41
    # Reproduce native reset using only release metadata and the copied model.
    native_data = mujoco.MjData(env.model)
    if reset_mode != "legacy":
        assert manifest["task"]["reset_keyframe"] == "home"
        key_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(env.model, native_data, key_id)
        np.testing.assert_allclose(
            native_data.qpos[7:], [-native_data.qpos[i] for i in (10, 11, 12, 7, 8, 9)]
        )
    else:
        assert "reset_keyframe" not in manifest["task"]
        np.testing.assert_array_equal(native_data.qpos[7:], 0)
    np.testing.assert_array_equal(native_data.qpos, env.data.qpos)
    observation, reward, _, _ = env.step(np.zeros(6, dtype=np.float32))
    assert np.isfinite(observation.actor).all() and np.isfinite(reward)


@pytest.mark.parametrize("robot_id", [None, "pe01"])
def test_pe02_rejects_other_robot_checkpoints(tmp_path, robot_id):
    checkpoint = tmp_path / "other.pt"
    torch.save(
        {"actor_state_dict": PE02EncoderPolicy(load_config()).state_dict(), "robot_id": robot_id},
        checkpoint,
    )
    with pytest.raises(ValueError, match="PE02 requires"):
        load_policy(checkpoint)


def test_pe02_default_network_contract():
    policy = PE02EncoderPolicy(load_config())
    history, frame, commands = torch.zeros(2, 300), torch.zeros(2, 30), torch.zeros(2, 3)
    assert policy.action_mean(history, frame, commands).shape == (2, 6)
    assert policy.value(history, torch.zeros(2, 33), commands).shape == (2,)


def test_early_checkpoint_without_config_keeps_original_zero_reset(tmp_path):
    checkpoint = tmp_path / "early.pt"
    torch.save(
        {"actor_state_dict": PE02EncoderPolicy(load_config()).state_dict(), "robot_id": "pe02"},
        checkpoint,
    )
    policy = load_policy(checkpoint)
    env = PE02Env(config=policy.config)
    env.reset()
    assert env.data.qpos[2] == 0.38
    np.testing.assert_array_equal(env.data.qpos[7:], 0)
