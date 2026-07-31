from __future__ import annotations

import json
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
WE11_TASK_NAME = "DR002JoystickFlatWE11"
WE11_ROOT = REPO_ROOT / "src/unilab/assets/robots/dr002/we11"
WE11_SCENE = WE11_ROOT / "scene_flat_we11.xml"
WE11_PACE = WE11_ROOT / "we11_pace_params.json"
WE11_TASK_ROOT = REPO_ROOT / "conf/ppo/task/dr002_joystick_flat_we11"
KP = [2.0, 7.59, 0.0, 2.0, 7.59, 0.0]
KD = [0.080, 0.682, 0.05, 0.080, 0.682, 0.05]
ARMATURE = [0.0045092746608505355, 0.0056654868268008396, 0.0008] * 2
DAMPING = [2.3658001235049574e-06, 1.7025403002922656e-05, 0.0] * 2
FRICTIONLOSS = [9.003633786813792e-06, 0.20614840564125578, 0.0] * 2


def _compose_task(selector: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"task={selector}"])


def test_we11_hydra_is_self_contained_with_network_obs_force_and_control() -> None:
    we11 = _compose_task(WE11_SELECTOR)

    assert we11.training.task_name == WE11_TASK_NAME
    assert we11.algo.actor.class_name == "unilab.algos.torch.dr002_mlp_adapt:MlpAdaptModel"
    assert list(we11.algo.actor.history_term_dims) == [3, 3, 4, 6, 6, 2, 3]
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
    assert list(we11.env.wing_angle_obs.curriculum_paths) == [
        "robots/dr002/we11/training_data/wing_angle_20260713/1hz.csv",
        "robots/dr002/we11/training_data/wing_angle_20260713/2hz.csv",
        "robots/dr002/we11/training_data/wing_angle_20260713/3hz.csv",
    ]
    assert list(we11.env.wing_angle_obs.csv_amplitude_scale_range) == [0.8, 1.2]
    assert we11.env.wing_angle_obs.gaussian_noise_relative_std == 0.05
    assert list(we11.env.domain_rand.csv_force_curriculum_hz) == [0, 1, 2, 3]
    assert list(we11.env.domain_rand.csv_force_amplitude_scale_range) == [0.5, 1.5]
    assert we11.env.domain_rand.csv_force_apply_measured_moment is True
    assert we11.env.domain_rand.csv_force_observation_include_measured_moment is True

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
    assert we11.nu == 6
    np.testing.assert_allclose(we11.dof_armature[6:], ARMATURE, rtol=0, atol=1e-15)
    np.testing.assert_allclose(we11.dof_damping[6:], DAMPING, rtol=0, atol=1e-15)
    np.testing.assert_allclose(we11.dof_frictionloss[6:], FRICTIONLOSS, rtol=0, atol=1e-15)

    pace = json.loads(WE11_PACE.read_text())
    assert pace["armature"] == ARMATURE
    assert pace["damping"] == DAMPING
    assert pace["frictionloss"] == FRICTIONLOSS
    assert pace["kp"] == KP
    assert pace["kd"] == KD
    assert (WE11_ROOT / "meshes_lod/base_link_part_00.STL").is_file()
    assert (WE11_ROOT / "urdf/we11_reviewed.urdf").is_file()
    assert not (REPO_ROOT / "src/unilab/assets/robots/dr002/u9").exists()
    assert not (REPO_ROOT / "src/unilab/assets/robots/dr002/we10").exists()


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
        np.testing.assert_allclose(env._base_motor_kp, KP, rtol=0, atol=0)
        np.testing.assert_allclose(env._base_motor_kd, KD, rtol=0, atol=0)
        np.testing.assert_array_equal(
            env._clip_policy_actions(np.full((1, 6), 10.0, dtype=np.float32)),
            [[10.0, 10.0, 3.5, 10.0, 10.0, 3.5]],
        )
        state = env.step(np.zeros((1, 6), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert env._backend._model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    finally:
        env.close()
