from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from unilab.base import registry
from unilab.envs.locomotion.dr002.joystick import (
    DR002JoystickCfg,
    DR002JoystickFlatWE11Cfg,
    WE11ControlConfig,
    WE11JoystickSensor,
)
from unilab.training import BackendAdapter, create_env, ensure_registries

REPO_ROOT = Path(__file__).parents[4]
CONFIG_DIR = REPO_ROOT / "conf/ppo"
WE11_SELECTOR = "dr002_joystick_flat_we11/mujoco"
WE11_ROUGH_SELECTOR = "dr002_joystick_rough_we11/mujoco"
WE11_TASK_NAME = "DR002JoystickFlatWE11"
WE11_ROOT = REPO_ROOT / "src/unilab/assets/robots/dr002/we11"
WE11_SCENE = WE11_ROOT / "scene_flat_we11.xml"
WE11_GETUP_SCENE = WE11_ROOT / "scene_getup_alignment_we11.xml"
WE11_PACE = WE11_ROOT / "we11_pace_params.json"
WE11_TASK_ROOT = REPO_ROOT / "conf/ppo/task/dr002_joystick_flat_we11"
KP = [2.0, 7.59, 0.0, 2.0, 7.59, 0.0]
KD = [0.080, 0.682, 0.05, 0.080, 0.682, 0.05]
ARMATURE = [0.0045092746608505355, 0.0056654868268008396, 0.0008] * 2
DAMPING = [2.3658001235049574e-06, 1.7025403002922656e-05, 0.0] * 2
FRICTIONLOSS = [9.003633786813792e-06, 0.20614840564125578, 0.0] * 2
IDENTIFIED_KP = [1.86, 9.6, 0.0, 1.86, 9.6, 0.0]
IDENTIFIED_KD = [0.058, 0.11, 0.05, 0.058, 0.11, 0.05]
IDENTIFIED_ARMATURE = [0.008677222203964018, 0.030286671302744577, 0.004881044618994038] * 2
IDENTIFIED_DAMPING = [0.003964611000548929, 0.9333306854823012, 0.01824383576670055] * 2
IDENTIFIED_FRICTIONLOSS = [0.01165612207571664, 0.07282249089084407, 0.08020703598415967] * 2
CALF_RANGE = [-2.522524368, -1.285784060]


def _compose_task(selector: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"task={selector}"])


def test_we11_hydra_is_self_contained_with_network_obs_force_and_control() -> None:
    we11 = _compose_task(WE11_SELECTOR)

    assert we11.training.task_name == WE11_TASK_NAME
    assert we11.algo.actor.class_name == "unilab.algos.torch.dr002_mlp_adapt:MlpAdaptModel"
    assert list(we11.algo.actor.history_term_dims) == [3, 3, 4, 6, 6, 2, 2, 3]
    assert list(we11.algo.actor.actor_hidden_dims) == [128, 64, 32]
    assert list(we11.algo.critic.hidden_dims) == [256, 128, 64]
    assert we11.algo.algorithm.entropy_coef == 0.005
    assert list(we11.env.control_config.Kp) == KP
    assert list(we11.env.control_config.Kd) == KD
    assert we11.env.control_config.wheel_action_scale == 10.0
    assert we11.env.control_config.wheel_clip_actions == 3.5
    assert we11.env.control_config.use_native_command_delay_pd is True
    assert we11.env.control_config.action_delay_min_steps == 2
    assert we11.env.control_config.action_delay_max_steps == 8
    assert we11.env.control_config.resample_action_delay is True
    assert we11.env.wing_angle_obs.enabled is True
    assert we11.env.wing_angle_obs.gaussian_noise_relative_std == 0.05
    assert we11.env.wing_velocity_obs.enabled is True
    assert we11.env.wing_velocity_cmd.enabled is True
    assert we11.env.domain_rand.csv_force_enabled is False
    assert we11.env.domain_rand.csv_force_observation_include_measured_moment is False
    assert we11.env.reset_pose.mode == "mixed"
    assert we11.env.reset_pose.getup_probability == 0.30
    assert we11.env.reset_pose.getup_termination_grace_seconds == 3.0
    assert we11.training.play_start_pose == "getup"

    task_text = (WE11_TASK_ROOT / "mujoco.yaml").read_text(encoding="utf-8").lower()
    assert "dr002_joystick_flat_we9" not in task_text
    assert "dr002_joystick_flat_we10" not in task_text
    assert (WE11_TASK_ROOT / "base.yaml").is_file()
    assert not (REPO_ROOT / "conf/ppo/task/dr002_joystick_flat_we9").exists()
    assert not (REPO_ROOT / "conf/ppo/task/dr002_joystick_flat_we10").exists()


def test_we11_registry_scene_and_compiled_pace_are_exact() -> None:
    ensure_registries()
    cfg = DR002JoystickFlatWE11Cfg()
    we11 = mujoco.MjModel.from_xml_path(str(WE11_SCENE))

    assert DR002JoystickFlatWE11Cfg.__bases__ == (DR002JoystickCfg,)
    assert Path(cfg.scene.model_file) == WE11_SCENE
    assert isinstance(cfg.sensor, WE11JoystickSensor)
    assert isinstance(cfg.control_config, WE11ControlConfig)
    assert cfg.control_config.Kp == KP
    assert cfg.control_config.Kd == KD
    registered = registry.list_registered_envs()
    assert registered[WE11_TASK_NAME] == {
        "config_class": "DR002JoystickFlatWE11Cfg",
        "available_backends": ["mujoco"],
    }
    assert "DR002JoystickFlatWE9" not in registered
    assert "DR002JoystickFlatWE10" not in registered
    assert we11.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert we11.nbody > 0
    assert we11.nu == 8
    np.testing.assert_allclose(we11.dof_armature[6:12], ARMATURE, rtol=0, atol=1e-15)
    np.testing.assert_allclose(we11.dof_damping[6:12], DAMPING, rtol=0, atol=1e-15)
    np.testing.assert_allclose(we11.dof_frictionloss[6:12], FRICTIONLOSS, rtol=0, atol=1e-15)
    for joint_name in ("left_calf_joint", "right_calf_joint"):
        joint_id = mujoco.mj_name2id(we11, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        np.testing.assert_allclose(we11.jnt_range[joint_id], CALF_RANGE, rtol=0, atol=1e-12)

    urdf = ET.parse(WE11_ROOT / "urdf/we11_reviewed.urdf")
    for joint_name in ("left_calf_joint", "right_calf_joint"):
        limit = urdf.find(f".//joint[@name='{joint_name}']/limit")
        assert limit is not None
        np.testing.assert_allclose(
            [float(limit.attrib["lower"]), float(limit.attrib["upper"])],
            CALF_RANGE,
            rtol=0,
            atol=1e-12,
        )

    pace = json.loads(WE11_PACE.read_text())
    # Keep the latest identification artifact available without silently
    # injecting it into the Sep-09 training/play runtime contract.
    assert pace["armature"] == IDENTIFIED_ARMATURE
    assert pace["damping"] == IDENTIFIED_DAMPING
    assert pace["frictionloss"] == IDENTIFIED_FRICTIONLOSS
    assert pace["kp"] == IDENTIFIED_KP
    assert pace["kd"] == IDENTIFIED_KD
    assert (WE11_ROOT / "meshes_lod/base_link_part_00.STL").is_file()
    assert (WE11_ROOT / "urdf/we11_reviewed.urdf").is_file()
    assert not (REPO_ROOT / "src/unilab/assets/robots/dr002/u9").exists()
    assert not (REPO_ROOT / "src/unilab/assets/robots/dr002/we10").exists()


def test_flat_and_alignment_scenes_share_exact_getup_keyframe() -> None:
    flat = mujoco.MjModel.from_xml_path(str(WE11_SCENE))
    alignment = mujoco.MjModel.from_xml_path(str(WE11_GETUP_SCENE))
    flat_key = mujoco.mj_name2id(flat, mujoco.mjtObj.mjOBJ_KEY, "getup_start_v2")
    alignment_key = mujoco.mj_name2id(alignment, mujoco.mjtObj.mjOBJ_KEY, "getup_start_v2")

    assert flat_key >= 0
    assert alignment_key >= 0
    np.testing.assert_allclose(
        flat.key_qpos[flat_key], alignment.key_qpos[alignment_key], rtol=0, atol=0
    )
    for joint_name in ("left_calf_joint", "right_calf_joint"):
        joint_id = mujoco.mj_name2id(flat, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_index = int(flat.jnt_qposadr[joint_id])
        assert flat.key_qpos[flat_key, qpos_index] == CALF_RANGE[0]


def test_forced_getup_reset_uses_keyframe() -> None:
    cfg = _compose_task(WE11_SELECTOR)
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    override["reset_pose"] = {
        **override["reset_pose"],
        "mode": "getup",
    }
    override["domain_rand"] = {
        **override["domain_rand"],
        "init_xy_range": [0.0, 0.0],
        "randomize_init_yaw": False,
    }
    env = create_env(cfg, num_envs=1, env_cfg_override=override)

    try:
        env.init_state()
        key_qpos = env._backend.get_keyframe_qpos("getup_start_v2")
        assert env.episode_getup_mask().tolist() == [True]
        np.testing.assert_allclose(env._backend.get_base_pos()[0], key_qpos[:3], atol=1e-7)
        np.testing.assert_allclose(env._backend.get_base_quat()[0], key_qpos[3:7], atol=1e-7)
    finally:
        env.close()


def test_forced_upright_reset_preserves_complete_home_keyframe() -> None:
    cfg = _compose_task(WE11_SELECTOR)
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    override["reset_pose"] = {
        **override["reset_pose"],
        "mode": "upright",
    }
    override["domain_rand"] = {
        **override["domain_rand"],
        "init_xy_range": [0.0, 0.0],
        "randomize_init_yaw": False,
    }
    env = create_env(cfg, num_envs=1, env_cfg_override=override)

    try:
        home_qpos = env._backend.get_keyframe_qpos("home")
        np.testing.assert_allclose(env._init_qpos, home_qpos, rtol=0, atol=0)

        env.init_state()
        assert env.episode_getup_mask().tolist() == [False]
        np.testing.assert_allclose(env._backend.get_base_pos()[0], home_qpos[:3], atol=1e-7)
        np.testing.assert_allclose(env._backend.get_base_quat()[0], home_qpos[3:7], atol=1e-7)
        np.testing.assert_allclose(env.get_dof_pos()[0], [0.8, -1.6, 0.0] * 2, atol=1e-7)
    finally:
        env.close()


def test_mixed_getup_sampling_tracks_configured_probability() -> None:
    cfg = _compose_task(WE11_SELECTOR)
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=override)

    try:
        np.random.seed(1234)
        sampled = env.sample_reset_getup_mask(20_000)
        assert 0.29 < float(np.mean(sampled)) < 0.31
    finally:
        env.close()


def test_play_pose_override_defaults_getup_and_keeps_rough_upright() -> None:
    flat_cfg = _compose_task(WE11_SELECTOR)
    flat_adapter = BackendAdapter(flat_cfg, root_dir=REPO_ROOT, algo_name="ppo")
    assert flat_adapter.build_play_env_cfg_override()["reset_pose"]["mode"] == "getup"

    flat_cfg.training.play_start_pose = "upright"
    assert flat_adapter.build_play_env_cfg_override()["reset_pose"]["mode"] == "upright"

    rough_cfg = _compose_task(WE11_ROUGH_SELECTOR)
    assert rough_cfg.env.reset_pose.mode == "upright"
    rough_adapter = BackendAdapter(rough_cfg, root_dir=REPO_ROOT, algo_name="ppo")
    assert rough_adapter.build_play_env_cfg_override()["reset_pose"]["mode"] == "upright"


def test_we11_training_backend_keeps_rk4_through_first_step() -> None:
    cfg = _compose_task(WE11_SELECTOR)
    override = BackendAdapter(
        cfg,
        root_dir=REPO_ROOT,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=override)

    try:
        state = env.init_state()
        assert state.obs["obs"].shape[0] == 1
        assert env._backend._model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        np.testing.assert_allclose(env._base_motor_kp[:6], KP, rtol=0, atol=0)
        np.testing.assert_allclose(env._base_motor_kd[:6], KD, rtol=0, atol=0)
        np.testing.assert_array_equal(
            env._clip_policy_actions(np.full((1, 6), 10.0, dtype=np.float32)),
            [[10.0, 10.0, 3.5, 10.0, 10.0, 3.5]],
        )
        state = env.step(np.zeros((1, 6), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert env._backend._model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    finally:
        env.close()
