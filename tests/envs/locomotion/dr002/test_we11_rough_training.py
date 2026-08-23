from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from unilab.base import registry
from unilab.envs.locomotion.dr002.joystick import DR002JoystickFlatWE11Cfg
from unilab.envs.locomotion.dr002.rough import (
    DR002JoystickRoughEnv,
    DR002JoystickRoughWE11Cfg,
    WE11RoughJoystickSensor,
)
from unilab.training import BackendAdapter, create_env, ensure_registries

REPO_ROOT = Path(__file__).parents[4]
CONFIG_DIR = REPO_ROOT / "conf/ppo"
SELECTOR = "dr002_joystick_rough_we11/mujoco"
TASK_NAME = "DR002JoystickRoughWE11"
WE11_ROOT = REPO_ROOT / "src/unilab/assets/robots/dr002/we11"
WE11_ROBOT = WE11_ROOT / "we11.xml"
ROUGH_FRAGMENT = WE11_ROOT / "rough_locomotion_task.xml"
KP = [2.0, 7.59, 0.0, 2.0, 7.59, 0.0]
KD = [0.080, 0.682, 0.05, 0.080, 0.682, 0.05]


def _compose_task() -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"task={SELECTOR}"])


def _create_env(num_envs: int) -> DR002JoystickRoughEnv:
    cfg = _compose_task()
    override = BackendAdapter(
        cfg,
        root_dir=REPO_ROOT,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    return create_env(cfg, num_envs=num_envs, env_cfg_override=override)


def test_we11_rough_hydra_and_registry_preserve_flat_policy_contract() -> None:
    ensure_registries()
    hydra_cfg = _compose_task()
    cfg = DR002JoystickRoughWE11Cfg()

    assert hydra_cfg.training.task_name == TASK_NAME
    assert issubclass(DR002JoystickRoughWE11Cfg, DR002JoystickFlatWE11Cfg)
    assert Path(cfg.scene.model_file) == WE11_ROBOT
    assert [Path(path) for path in cfg.scene.fragment_files] == [ROUGH_FRAGMENT]
    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.generator is not None
    assert cfg.scene.terrain.generator.seed == 42
    assert cfg.terrain_curriculum.enabled is True
    assert cfg.terrain_curriculum.use_path_length is True
    assert cfg.terrain_curriculum.demote_requires_commanded_distance is True
    assert cfg.terrain_curriculum.unlock_after_force_curriculum is False
    terrain_proportions = {
        name: terrain.proportion
        for name, terrain in cfg.scene.terrain.generator.sub_terrains.items()
    }
    assert terrain_proportions == {
        "flat": 0.3,
        "hf_pyramid_slope": 0.2,
        "hf_pyramid_slope_inv": 0.2,
        "random_rough": 0.3,
    }
    random_rough = cfg.scene.terrain.generator.sub_terrains["random_rough"]
    assert random_rough.noise_range == (0.0, 0.006)
    assert random_rough.noise_step == 0.001
    assert cfg.scene.terrain.generator.vertical_scale == 0.001
    assert cfg.scene.terrain.generator.sub_terrains["hf_pyramid_slope"].slope_range == (
        0.0,
        0.4,
    )
    assert cfg.scene.terrain.generator.sub_terrains["hf_pyramid_slope_inv"].slope_range == (
        0.0,
        0.4,
    )
    assert cfg.control_config.Kp == KP
    assert cfg.control_config.Kd == KD
    assert isinstance(cfg.sensor, WE11RoughJoystickSensor)
    assert "left_calf_shaft_touch" in cfg.sensor.undesired_contacts
    assert "right_calf_shaft_touch" in cfg.sensor.undesired_contacts
    assert cfg.sensor.termination_contacts is not None
    assert "left_calf_shaft_touch" not in cfg.sensor.termination_contacts
    assert "right_calf_shaft_touch" not in cfg.sensor.termination_contacts
    assert "base_link_touch" in cfg.sensor.termination_contacts
    assert hydra_cfg.reward.scales.undesired_contacts == -3.0
    assert not cfg.control_config.use_native_batched_pd
    assert cfg.control_config.use_native_command_delay_pd
    assert list(hydra_cfg.algo.actor.history_term_dims) == [3, 3, 4, 6, 6, 2, 2, 3]
    assert hydra_cfg.env.commands.rel_standing_envs == 0.2
    assert hydra_cfg.env.control_config.wheel_clip_actions == 3.5
    assert registry.list_registered_envs()[TASK_NAME] == {
        "config_class": "DR002JoystickRoughWE11Cfg",
        "available_backends": ["mujoco"],
    }


def test_we11_rough_constructs_steps_and_restores_terrain_cells() -> None:
    env = _create_env(num_envs=2)
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 145)
        assert state.obs["critic"].shape == (2, 328)
        assert state.obs["privileged_target"].shape == (2, 3)
        assert env._height_scan_dim == 187
        assert env._backend._model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        floor_id = env._backend.get_geom_id("floor")
        assert env._backend._model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_HFIELD
        hfield_path = Path(env._backend.scene_artifacts_dir) / "hfields" / "hfield.hfield"
        assert hfield_path.is_file()
        assert np.unique(env._backend._model.hfield_data).size > 1000
        np.testing.assert_allclose(env._base_motor_kp[:6], KP, rtol=0, atol=0)
        np.testing.assert_allclose(env._base_motor_kd[:6], KD, rtol=0, atol=0)
        np.testing.assert_array_equal(env._spawn.levels, [0, 0])

        state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert all(np.isfinite(value).all() for value in state.obs.values())

        env._spawn.levels[:] = [2, 3]
        env._spawn.type_cols[:] = [0, 1]
        saved = env.training_state_dict()
        env._spawn.levels[:] = 0
        env._spawn.type_cols[:] = 0
        env.load_training_state_dict(saved)

        np.testing.assert_array_equal(env._spawn.levels, [2, 3])
        np.testing.assert_array_equal(env._spawn.type_cols, [0, 1])
        np.testing.assert_array_equal(env.state.info["steps"], [0, 0])
        base_xy = env._backend.get_base_pos()[:, :2]
        origins_xy = env._backend.terrain_origins[[2, 3], [0, 1], :2]
        assert np.all(np.abs(base_xy - origins_xy) <= 0.6)
        assert all(np.isfinite(value).all() for value in env.state.obs.values())
    finally:
        env.close()


def test_we11_level_nine_play_override_selects_last_terrain_row() -> None:
    cfg = _compose_task()
    cfg.env.terrain_curriculum.initial_level = 9
    override = BackendAdapter(
        cfg,
        root_dir=REPO_ROOT,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=4, env_cfg_override=override)
    try:
        env.init_state()
        np.testing.assert_array_equal(env._spawn.levels, [9, 9, 9, 9])
    finally:
        env.close()
