from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager, ResetPlan
from unilab.dr.dr_utils import zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    base_height_from_scan,
    height_scan_obs,
    init_height_scan_sensor,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.terrain_spawn import TerrainCurriculumCfg, TerrainSpawnManager
from unilab.envs.locomotion.dr002.joystick import (
    DR002Commands,
    DR002JoystickCfg,
    DR002JoystickDomainRandomizationProvider,
    DR002JoystickEnv,
    DR002JoystickFlatWE11Cfg,
    WE11JoystickSensor,
    _csv_force_zero_hz_mask,
    build_dr002_backend_reset_randomization,
)
from unilab.terrains import (
    HfRandomUniformTerrainCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
)


@dataclass
class DR002RoughCommands(DR002Commands):
    lin_vel_x: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    ang_vel_z: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    height: list[float] = field(default_factory=lambda: [0.28, 0.28])
    resampling_time: float = 5.0
    startup_stand_seconds: float = 3.0
    range_multiplier: list[float] = field(default_factory=lambda: [1.0, 2.0])
    ang_vel_z_range_multiplier: list[float] = field(default_factory=lambda: [0.3, 1.0])


@dataclass
class RoughTerminationConfig:
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0


@dataclass(kw_only=True)
class DR002CurriculumRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
    """Random rough terrain whose max height increases with curriculum difficulty."""

    def function(self, difficulty: float, rng: np.random.Generator):
        low, high = self.noise_range
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        scaled_high = low + difficulty * (high - low)
        return HfRandomUniformTerrainCfg.function(
            replace(self, noise_range=(low, scaled_high)),
            difficulty,
            rng,
        )


@dataclass(kw_only=True)
class DR002RoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 10
    num_cols: int = 20
    border_width: float = 20.0
    add_lights: bool = True
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    curriculum: bool = True

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.3),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.2,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.2,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.2,
            ),
            "random_rough": DR002CurriculumRandomUniformTerrainCfg(
                proportion=0.3,
                noise_range=(0.01, 0.05),
                noise_step=0.02,
                border_width=0.2,
            ),
        }
    )


def _we11_rough_terrain_generator() -> DR002RoughTerrainCfg:
    """Build WE11's millimeter-resolution runway roughness curriculum."""
    cfg = DR002RoughTerrainCfg(seed=42, vertical_scale=0.001)
    random_rough = cfg.sub_terrains["random_rough"]
    if not isinstance(random_rough, DR002CurriculumRandomUniformTerrainCfg):
        raise TypeError("WE11 random_rough must use the curriculum uniform terrain config")
    random_rough.noise_range = (0.0, 0.006)
    random_rough.noise_step = 0.001
    return cfg


@dataclass
class WE11RoughJoystickSensor(WE11JoystickSensor):
    """Penalize calf-shaft contact without using it as a terminal condition."""

    termination_contacts: tuple[str, ...] | None = (
        "base_link_touch",
        "left_thigh_touch",
        "right_thigh_touch",
        "ancient_upper_left_touch",
        "ancient_upper_right_touch",
        "ancient_front_center_touch",
    )


@registry.envcfg("DR002JoystickRough")
@dataclass
class DR002JoystickRoughCfg(DR002JoystickCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "dr002" / "dr002_latest.xml"),
            fragment_files=[],
            terrain=TerrainSceneCfg(
                generator=DR002RoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    commands: DR002RoughCommands = field(default_factory=DR002RoughCommands)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)


@registry.envcfg("DR002JoystickRoughWE11")
@dataclass
class DR002JoystickRoughWE11Cfg(DR002JoystickFlatWE11Cfg):
    """WE11's current policy contract on procedural curriculum terrain."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "dr002" / "we11" / "we11.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "dr002" / "we11" / "rough_locomotion_task.xml")
            ],
            terrain=TerrainSceneCfg(
                generator=_we11_rough_terrain_generator(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    sensor: WE11JoystickSensor = field(default_factory=WE11RoughJoystickSensor)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(
        default_factory=lambda: TerrainCurriculumCfg(
            enabled=True,
            use_path_length=True,
            demote_requires_commanded_distance=True,
            unlock_after_force_curriculum=True,
            seed=42,
        )
    )


class DR002JoystickRoughDomainRandomizationProvider(DR002JoystickDomainRandomizationProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        low_xy, high_xy = env.cfg.domain_rand.init_xy_range
        qpos[:, 0:2] += np.random.uniform(low_xy, high_xy, (num_reset, 2))
        yaw = np.zeros((num_reset,), dtype=get_global_dtype())
        if env.cfg.domain_rand.randomize_init_yaw:
            yaw_low, yaw_high = env.cfg.domain_rand.init_yaw_range
            yaw = np.random.uniform(yaw_low, yaw_high, (num_reset,))
            qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        if env._startup_stand_steps > 0:
            qvel[:, 0:6] = 0.0
        else:
            qvel_low, qvel_high = env.cfg.domain_rand.init_qvel_range
            qvel[:, 0:6] = np.asarray(
                np.random.uniform(qvel_low, qvel_high, size=(num_reset, 6)),
                dtype=get_global_dtype(),
            )

        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)
        torque_scale, default_joint_pos_offset = env.sample_reset_motor_runtime_randomization(
            num_reset
        )
        env.set_motor_runtime_randomization(env_ids, torque_scale, default_joint_pos_offset)
        standing_mask = env.sample_reset_standing_mask(env_ids)
        env.set_episode_standing_mask(env_ids, standing_mask)
        csv_force_levels = env.sample_reset_csv_force_levels(num_reset)
        csv_force_start_delay_steps = env.sample_reset_csv_force_start_delay_steps(num_reset)
        csv_force_amplitude_scales = env.sample_reset_csv_force_amplitude_scales(num_reset)
        wing_angle_obs_amplitude_scales = env.sample_reset_wing_angle_obs_amplitude_scales(
            num_reset
        )
        if (
            env._standing_envs_episode_persistent
            and not env.cfg.domain_rand.csv_force_apply_to_standing
        ):
            csv_force_levels[standing_mask] = 0
            csv_force_start_delay_steps[standing_mask] = 0
        zero_hz_wing_angle_obs = env.sample_reset_zero_hz_wing_angle_obs(
            _csv_force_zero_hz_mask(env.cfg.domain_rand, csv_force_levels)
        )
        env.set_zero_hz_wing_angle_obs(env_ids, zero_hz_wing_angle_obs)
        env.set_csv_force_active_levels(env_ids, csv_force_levels)
        env.set_csv_force_start_delay_steps(env_ids, csv_force_start_delay_steps)
        env.set_csv_force_amplitude_scales(env_ids, csv_force_amplitude_scales)
        env.set_wing_angle_obs_amplitude_scales(env_ids, wing_angle_obs_amplitude_scales)
        body_mass_multipliers = self._body_mass_multipliers_for_reset(env, env_ids)
        reset_randomization = build_dr002_backend_reset_randomization(
            env,
            num_reset,
            base_body_mass=self._base_body_mass,
            base_body_inertia=self._base_body_inertia,
            base_geom_friction=self._base_geom_friction,
            ground_geom_id=self._ground_geom_id,
            robot_geom_ids=self._robot_geom_ids,
            base_dof_armature=self._base_dof_armature,
            body_mass_multipliers=body_mass_multipliers,
        )
        env.set_privileged_reset_randomization(env_ids, reset_randomization)
        info_updates: dict[str, Any] = {
            "commands": (
                env.startup_commands(num_reset)
                if env._startup_stand_steps > 0
                else env.sample_commands(num_reset, env_ids=env_ids)
            ),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=reset_randomization,
        )


@registry.env("DR002JoystickRough", sim_backend="mujoco")
class DR002JoystickRoughEnv(DR002JoystickEnv):
    _cfg: DR002JoystickRoughCfg | DR002JoystickRoughWE11Cfg
    _height_scan_dim: int = 0

    def __init__(
        self,
        cfg: DR002JoystickRoughCfg | DR002JoystickRoughWE11Cfg,
        num_envs=1,
        backend_type="mujoco",
    ):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        terrain_origins = getattr(self._backend, "terrain_origins", None)
        terrain_generator = cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        if terrain_origins is not None and terrain_generator is not None:
            type_col_weights = None
            if getattr(terrain_generator, "curriculum", False):
                type_col_weights = np.asarray(
                    [sub_cfg.proportion for sub_cfg in terrain_generator.sub_terrains.values()],
                    dtype=np.float64,
                )
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=getattr(self._backend, "terrain_surface_sampler", None),
                type_col_weights=type_col_weights,
            )
        self._terrain_commanded_distance = np.zeros(num_envs, dtype=np.float64)
        self._terrain_all_env_ids = np.arange(num_envs, dtype=np.int32)
        self._dr_manager = DomainRandomizationManager(
            self,
            DR002JoystickRoughDomainRandomizationProvider(
                base_body_mass=self._dr_base_body_mass,
                base_body_inertia=self._dr_base_body_inertia,
                base_geom_friction=self._dr_base_geom_friction,
                ground_geom_id=self._dr_ground_geom_id,
                robot_geom_ids=self._dr_robot_geom_ids,
                base_dof_armature=self._dr_base_dof_armature,
            ),
        )
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        spec = super().obs_groups_spec
        return {**spec, "critic": spec["critic"] + self._height_scan_dim}

    def _compute_obs(self, *args, **kwargs) -> dict[str, np.ndarray]:
        obs = super()._compute_obs(*args, **kwargs)
        num_obs = obs["critic"].shape[0]
        if self._height_scan_dim > 0:
            scan = height_scan_obs(self, self._cfg.terrain_scan, num_obs)
            obs["critic"] = np.concatenate([obs["critic"], scan], axis=1, dtype=get_global_dtype())
        return obs

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        return base_height_from_scan(self, num_obs)

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        self._terrain_commanded_distance[env_ids] = 0.0
        return obs, info

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=np.float64)
        if isinstance(self._spawn, TerrainSpawnManager):
            self._spawn.record_step(self._terrain_all_env_ids, base_pos)
        commands = np.asarray(state.info.get("commands"), dtype=np.float64)
        if commands.shape == (self._num_envs, 3):
            self._terrain_commanded_distance += np.abs(commands[:, 0]) * float(self._cfg.ctrl_dt)

        truncated = super()._compute_truncated(state)
        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_cfg = self._cfg.scene.terrain.generator if self._cfg.scene.terrain else None
            if terrain_cfg is not None:
                truncated |= terrain_out_of_bounds(
                    self, terrain_cfg, self._cfg.termination_config.terrain_distance_buffer
                )
        done = state.terminated | truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            demote_eligible = None
            terrain_curriculum = self._cfg.terrain_curriculum
            if terrain_curriculum.demote_requires_commanded_distance:
                threshold = terrain_curriculum.demote_commanded_distance_frac * float(
                    self._cfg.scene.terrain.generator.size[0]
                )
                demote_eligible = self._terrain_commanded_distance[done_indices] >= threshold
            stats = self._spawn.update_on_done(
                done_indices,
                base_pos[done_indices],
                demote_eligible=demote_eligible,
                allow_progression=self._terrain_curriculum_is_unlocked(),
            )
            if stats:
                log = state.info.setdefault("log", {})
                for key, value in stats.items():
                    log[f"terrain_curriculum/{key}"] = float(value)
                unlocked = self._terrain_curriculum_is_unlocked()
                log["terrain_curriculum/unlocked"] = float(unlocked)
                log["terrain_curriculum/locked_by_force"] = float(not unlocked)
        return truncated

    def _terrain_curriculum_is_unlocked(self) -> bool:
        cfg = self._cfg.terrain_curriculum
        if not cfg.unlock_after_force_curriculum:
            return True
        return self._command_curriculum_is_full() and self._csv_force_curriculum_is_full()

    def training_state_dict(self) -> dict[str, Any]:
        state = super().training_state_dict()
        if isinstance(self._spawn, TerrainSpawnManager):
            state["terrain_curriculum"] = self._spawn.state_dict()
        return state

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        super().load_training_state_dict(state)
        terrain_state = state.get("terrain_curriculum")
        if terrain_state is None or not isinstance(self._spawn, TerrainSpawnManager):
            return
        self._spawn.load_state_dict(terrain_state)
        self._reset_all_after_terrain_restore()

    def _reset_all_after_terrain_restore(self) -> None:
        """Move live robots to restored cells before resumed rollout begins."""
        if self._state is None:
            return
        env_ids = self._terrain_all_env_ids
        self._state.terminated.fill(False)
        self._state.truncated.fill(False)
        self._state.info["steps"][env_ids] = 0
        obs, info_updates = self.reset(env_ids)
        for key, value in obs.items():
            self._state.obs[key][env_ids] = value
        for key, value in info_updates.items():
            if isinstance(value, np.ndarray):
                if key not in self._state.info:
                    full_shape = (self._num_envs,) + value.shape[1:]
                    self._state.info[key] = np.zeros(full_shape, dtype=value.dtype)
                self._state.info[key][env_ids] = value
            else:
                self._state.info[key] = value
        self._state.reward.fill(0.0)
        self._state.final_observation = None
        self._clear_step_final_observation()


registry.register_env("DR002JoystickRoughWE11", DR002JoystickRoughEnv, sim_backend="mujoco")
