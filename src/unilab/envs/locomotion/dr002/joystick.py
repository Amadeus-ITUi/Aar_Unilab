from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.dr002.base import (
    DEFAULT_DR002_ANGLES,
    JOINT_VEL_OBSERVATION_INDICES,
    LEG_ACTION_INDICES,
    NUM_DR002_ACTIONS,
    WHEEL_ACTION_INDICES,
    DR002BaseCfg,
    DR002BaseEnv,
    compute_dr002_motor_ctrl,
    stack_joint_sensors,
)

_HISTORY_LENGTH = 5
_TERM_DIMS = (3, 3, 4, 6, 6, 3)
_ACTOR_DIM = _HISTORY_LENGTH * sum(_TERM_DIMS)


@dataclass
class DR002Commands:
    lin_vel_x: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    ang_vel_z: list[float] = field(default_factory=lambda: [-3.14, 3.14])
    height: list[float] = field(default_factory=lambda: [0.12, 0.30])
    resampling_time: float = 5.0
    rel_standing_envs: float = 0.0
    curriculum: bool = True
    range_multiplier: list[float] = field(default_factory=lambda: [1.0, 2.5])
    ang_vel_z_range_multiplier: list[float] = field(default_factory=lambda: [0.3, 1.0])
    curriculum_threshold: float = 0.7
    curriculum_step: float = 0.1


@dataclass
class DR002DomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    init_xy_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    init_qvel_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])

    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    com_offset_y: list[float] = field(default_factory=lambda: [-0.01, 0.01])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.01, 0.01])

    randomize_torque_scale: bool = False
    torque_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    randomize_default_joint_pos: bool = False
    default_joint_pos_offset_range: list[float] = field(default_factory=lambda: [-0.02, 0.02])

    push_velocity_x_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    push_velocity_y_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_std: float = 0.05
    base_height_clip: float = 4.0
    only_positive_rewards: bool = False
    undesired_contact_threshold: float = 0.1
    termination_contact_threshold: float = 5.0
    termination_gravity_z_threshold: float = 0.7
    termination_fail_time_s: float = 0.5


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"
    undesired_contacts: tuple[str, ...] = (
        "base_link_touch",
        "left_thigh_touch",
        "left_calf_touch",
        "right_thigh_touch",
        "right_calf_touch",
    )


@registry.envcfg("DR002JoystickFlat")
@dataclass
class DR002JoystickCfg(DR002BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "dr002" / "scene_flat_latest.xml")
        )
    )
    max_episode_seconds: float = 20.0
    commands: DR002Commands = field(default_factory=DR002Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: DR002DomainRandConfig = field(default_factory=DR002DomainRandConfig)


def _sample_dr002_commands(
    cfg: DR002Commands,
    num_samples: int,
    lin_vel_x_range: tuple[float, float] | None = None,
    ang_vel_z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    lin_vel_x = tuple(cfg.lin_vel_x) if lin_vel_x_range is None else lin_vel_x_range
    ang_vel_z = tuple(cfg.ang_vel_z) if ang_vel_z_range is None else ang_vel_z_range
    low = np.asarray([lin_vel_x[0], ang_vel_z[0], cfg.height[0]], dtype=get_global_dtype())
    high = np.asarray([lin_vel_x[1], ang_vel_z[1], cfg.height[1]], dtype=get_global_dtype())
    commands = np.random.uniform(low=low, high=high, size=(num_samples, 3)).astype(get_global_dtype())
    standing_prob = float(getattr(cfg, "rel_standing_envs", 0.0))
    if standing_prob > 0.0:
        standing = np.random.uniform(size=(num_samples,)) < min(standing_prob, 1.0)
        commands[standing, 0:2] = 0.0
    return commands


def build_dr002_backend_reset_randomization(
    env: Any,
    num_reset: int,
    *,
    base_body_mass: np.ndarray | None = None,
    base_geom_friction: np.ndarray | None = None,
    ground_geom_id: int | None = None,
    base_dof_armature: np.ndarray | None = None,
) -> ResetRandomizationPayload | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()
    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))
    if getattr(domain_rand, "randomize_body_mass", False):
        if base_body_mass is None:
            raise ValueError("body mass randomization requires cached body mass")
        body_mass_template = np.asarray(base_body_mass, dtype=np.float64)
        low, high = domain_rand.body_mass_multiplier_range
        multipliers = np.random.uniform(low, high, size=(num_reset, body_mass_template.size))
        body_names = ("thigh_joint", "calf_joint", "foot_joint")
        try:
            base_body_id = env._backend.get_body_id(env.cfg.asset.base_name)
            multipliers[:, int(base_body_id)] = 1.0
            for suffix in body_names:
                left_id = env._backend.get_body_id(f"left_{suffix}")
                right_id = env._backend.get_body_id(f"right_{suffix}")
                pair_scale = np.random.uniform(low, high, size=(num_reset,))
                multipliers[:, int(left_id)] = pair_scale
                multipliers[:, int(right_id)] = pair_scale
        except Exception:
            pass
        body_mass = np.broadcast_to(body_mass_template, multipliers.shape).copy()
        randomized = body_mass_template > 0.0
        body_mass[:, randomized] *= multipliers[:, randomized]
        payload.body_mass = body_mass
    if getattr(domain_rand, "random_com", False):
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        low, high = domain_rand.com_offset_x
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        low, high = domain_rand.com_offset_y
        base_com_offset[:, 1] = np.random.uniform(low, high, size=(num_reset,))
        low, high = domain_rand.com_offset_z
        base_com_offset[:, 2] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset
    if getattr(domain_rand, "randomize_gravity", False):
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        low = np.minimum(gravity_range[0], gravity_range[1])
        high = np.maximum(gravity_range[0], gravity_range[1])
        payload.gravity = np.random.uniform(low=low, high=high, size=(num_reset, 3))
    if getattr(domain_rand, "randomize_ground_friction", False):
        if base_geom_friction is None or ground_geom_id is None:
            raise ValueError("ground friction randomization requires cached geom friction and ground geom id")
        geom_friction_template = np.asarray(base_geom_friction, dtype=np.float64)
        low, high = domain_rand.ground_friction_multiplier_range
        geom_friction = np.broadcast_to(
            geom_friction_template, (num_reset, *geom_friction_template.shape)
        ).copy()
        geom_friction[:, int(ground_geom_id), 0] *= np.random.uniform(low, high, size=(num_reset,))
        payload.geom_friction = geom_friction
    if getattr(domain_rand, "randomize_dof_armature", False):
        if base_dof_armature is None:
            raise ValueError("dof armature randomization requires cached dof armature")
        dof_armature_template = np.asarray(base_dof_armature, dtype=np.float64)
        low, high = domain_rand.dof_armature_multiplier_range
        dof_armature = np.broadcast_to(
            dof_armature_template, (num_reset, dof_armature_template.size)
        ).copy()
        randomized = dof_armature_template > 0.0
        dof_armature[:, randomized] *= np.random.uniform(
            low, high, size=(num_reset, int(np.count_nonzero(randomized)))
        )
        payload.dof_armature = dof_armature
    return None if payload.is_empty() else payload


class DR002JoystickDomainRandomizationProvider(LocomotionDRProvider):
    def __init__(
        self,
        *,
        base_body_mass: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
        base_dof_armature: np.ndarray | None = None,
    ) -> None:
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id
        self._base_dof_armature = base_dof_armature

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        payload = build_dr002_backend_reset_randomization(
            env,
            num_reset=1,
            base_body_mass=self._base_body_mass,
            base_geom_friction=self._base_geom_friction,
            ground_geom_id=self._ground_geom_id,
            base_dof_armature=self._base_dof_armature,
        )
        if payload is not None:
            unsupported = capabilities.get_unsupported_reset_terms(payload.requested_terms())
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support DR002 reset randomization terms: {names}"
                )
        if env.cfg.domain_rand.push_robots and not capabilities.supports_interval_body_velocity_delta:
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support interval body velocity perturbation"
            )

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        domain_rand = env.cfg.domain_rand
        if (
            not domain_rand.push_robots
            or step_counter <= 0
            or step_counter % domain_rand.push_interval != 0
        ):
            return None
        body_id = env._backend.get_body_id(env.cfg.asset.base_name)
        velocity_delta = np.zeros((env._num_envs, 1, 3), dtype=np.float64)
        low, high = domain_rand.push_velocity_x_range
        velocity_delta[:, 0, 0] = np.random.uniform(low, high, size=(env._num_envs,))
        low, high = domain_rand.push_velocity_y_range
        velocity_delta[:, 0, 1] = np.random.uniform(low, high, size=(env._num_envs,))
        return IntervalRandomizationPlan(
            body_ids=np.asarray([body_id], dtype=np.int32),
            body_linear_velocity_delta=velocity_delta,
        )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        low_xy, high_xy = env.cfg.domain_rand.init_xy_range
        qpos[:, 0:2] += np.random.uniform(low_xy, high_xy, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        if env.cfg.domain_rand.randomize_init_yaw:
            yaw_low, yaw_high = env.cfg.domain_rand.init_yaw_range
            yaw = np.random.uniform(yaw_low, yaw_high, (num_reset,))
            qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qvel_low, qvel_high = env.cfg.domain_rand.init_qvel_range
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(qvel_low, qvel_high, size=(num_reset, 6)),
            dtype=get_global_dtype(),
        )

        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)
        torque_scale, default_joint_pos_offset = env.sample_reset_motor_runtime_randomization(num_reset)
        env.set_motor_runtime_randomization(env_ids, torque_scale, default_joint_pos_offset)
        info_updates: dict[str, Any] = {
            "commands": env.sample_commands(num_reset),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_dr002_backend_reset_randomization(
                env,
                num_reset,
                base_body_mass=self._base_body_mass,
                base_geom_friction=self._base_geom_friction,
                ground_geom_id=self._ground_geom_id,
                base_dof_armature=self._base_dof_armature,
            ),
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
                linvel,
                gyro,
                gravity,
                dof_pos,
                dof_vel,
                env_ids=env_ids,
                reset_history=True,
            ),
        )


@registry.env("DR002JoystickFlat", sim_backend="mujoco")
class DR002JoystickEnv(DR002BaseEnv):
    _cfg: DR002JoystickCfg

    def __init__(self, cfg: DR002JoystickCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._validate_motor_control_contract(ctrl_range, num_envs)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        self._base_motor_kp = np.asarray(cfg.control_config.Kp, dtype=np.float64)
        self._base_motor_kd = np.asarray(cfg.control_config.Kd, dtype=np.float64)
        self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, NUM_DR002_ACTIONS)).copy()
        self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, NUM_DR002_ACTIONS)).copy()
        self._motor_torque_scale = np.ones((num_envs, NUM_DR002_ACTIONS), dtype=np.float64)
        self._default_joint_pos_offset = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=self._np_dtype)
        self._last_motor_ctrl = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=self._np_dtype)
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=get_global_dtype())
        self._action_delay_enabled = bool(cfg.control_config.simulate_action_latency)
        self._action_delay_min_steps = int(cfg.control_config.action_delay_min_steps)
        self._action_delay_max_steps = int(cfg.control_config.action_delay_max_steps)
        if self._action_delay_min_steps < 0 or self._action_delay_max_steps < self._action_delay_min_steps:
            raise ValueError(
                "DR002 action delay requires 0 <= action_delay_min_steps <= action_delay_max_steps, "
                f"got [{self._action_delay_min_steps}, {self._action_delay_max_steps}]"
            )
        self._action_delay_buffer = np.zeros(
            (num_envs, self._action_delay_max_steps + 1, NUM_DR002_ACTIONS), dtype=self._np_dtype
        )
        self._action_delay_indices = np.zeros((num_envs,), dtype=np.int32)
        self._reset_action_delay(np.arange(num_envs, dtype=np.int32), resample=True)
        self._joint_range = self._backend.get_joint_range()
        self._lingzu_prev_action = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=get_global_dtype())
        self._lingzu_prev_prev_action = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=get_global_dtype())
        self._lingzu_action_history_count = np.zeros((num_envs,), dtype=np.int32)
        self._lingzu_fail_steps = np.zeros((num_envs,), dtype=np.int32)
        self._base_command_lin_vel_x = np.asarray(cfg.commands.lin_vel_x, dtype=np.float64)
        self._base_command_ang_vel_z = np.asarray(cfg.commands.ang_vel_z, dtype=np.float64)
        multiplier = np.asarray(cfg.commands.range_multiplier, dtype=np.float64)
        if multiplier.shape != (2,):
            raise ValueError("commands.range_multiplier must contain [initial, final]")
        yaw_multiplier = np.asarray(cfg.commands.ang_vel_z_range_multiplier, dtype=np.float64)
        if yaw_multiplier.shape != (2,):
            raise ValueError("commands.ang_vel_z_range_multiplier must contain [initial, final]")
        self._command_curriculum_scale = float(multiplier[0])
        self._command_curriculum_final_scale = float(multiplier[1])
        self._command_curriculum_yaw_scale = float(yaw_multiplier[0])
        self._command_curriculum_yaw_final_scale = float(yaw_multiplier[1])
        self._last_command_curriculum_mean_tracking = np.nan
        self._episode_track_lin_vel_x_sum = np.zeros((num_envs,), dtype=np.float64)
        self._episode_track_ang_vel_z_sum = np.zeros((num_envs,), dtype=np.float64)
        self._episode_track_lin_vel_x_steps = np.zeros((num_envs,), dtype=np.int32)
        self._history_terms = [
            np.zeros((num_envs, _HISTORY_LENGTH, dim), dtype=get_global_dtype())
            for dim in _TERM_DIMS
        ]
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_reward_functions()
        self._dr_base_body_mass = (
            self._backend.get_body_mass() if cfg.domain_rand.randomize_body_mass else None
        )
        self._dr_base_geom_friction = None
        self._dr_ground_geom_id = None
        if cfg.domain_rand.randomize_ground_friction:
            self._dr_base_geom_friction = self._backend.get_geom_friction()
            self._dr_ground_geom_id = self._backend.get_geom_id(cfg.asset.ground)
        self._dr_base_dof_armature = (
            self._backend.get_dof_armature() if cfg.domain_rand.randomize_dof_armature else None
        )
        self._init_domain_randomization(
            DR002JoystickDomainRandomizationProvider(
                base_body_mass=self._dr_base_body_mass,
                base_geom_friction=self._dr_base_geom_friction,
                ground_geom_id=self._dr_ground_geom_id,
                base_dof_armature=self._dr_base_dof_armature,
            )
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": _ACTOR_DIM, "critic": 46, "privileged_target": 3}

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        if hasattr(self, "_episode_track_lin_vel_x_sum"):
            self._update_command_curriculum(env_ids)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        self._lingzu_prev_action[env_ids] = 0.0
        self._lingzu_prev_prev_action[env_ids] = 0.0
        self._lingzu_action_history_count[env_ids] = 0
        self._lingzu_fail_steps[env_ids] = 0
        self._reset_action_delay(env_ids, resample=self._cfg.control_config.resample_action_delay)
        self._episode_track_lin_vel_x_sum[env_ids] = 0.0
        self._episode_track_ang_vel_z_sum[env_ids] = 0.0
        self._episode_track_lin_vel_x_steps[env_ids] = 0
        return obs, info

    def _neutral_policy_ctrl(self, env_ids: np.ndarray) -> np.ndarray:
        env_ids = np.asarray(env_ids, dtype=np.int32)
        neutral = np.zeros((len(env_ids), NUM_DR002_ACTIONS), dtype=self._np_dtype)
        neutral[:, LEG_ACTION_INDICES] = (
            self.default_angles[LEG_ACTION_INDICES]
            + self._default_joint_pos_offset[env_ids][:, LEG_ACTION_INDICES]
        )
        return neutral

    def _reset_action_delay(self, env_ids: np.ndarray, *, resample: bool) -> None:
        if env_ids.size == 0:
            return
        neutral = self._neutral_policy_ctrl(env_ids)
        self._action_delay_buffer[env_ids] = neutral[:, None, :]
        if self._action_delay_enabled and resample:
            self._action_delay_indices[env_ids] = np.random.randint(
                self._action_delay_min_steps,
                self._action_delay_max_steps + 1,
                size=(len(env_ids),),
                dtype=np.int32,
            )
        elif not self._action_delay_enabled:
            self._action_delay_indices[env_ids] = 0

    def _delayed_policy_ctrl(self, policy_ctrl: np.ndarray) -> np.ndarray:
        if not self._action_delay_enabled:
            return policy_ctrl
        self._action_delay_buffer[:, 1:] = self._action_delay_buffer[:, :-1].copy()
        self._action_delay_buffer[:, 0] = policy_ctrl
        env_ids = np.arange(policy_ctrl.shape[0], dtype=np.int32)
        return self._action_delay_buffer[env_ids, self._action_delay_indices[env_ids]]

    def _validate_motor_control_contract(self, ctrl_range: np.ndarray, num_envs: int) -> None:
        if self._backend.num_actuators != NUM_DR002_ACTIONS:
            raise ValueError(
                f"DR002 requires {NUM_DR002_ACTIONS} motor actuators, got {self._backend.num_actuators}"
            )
        if ctrl_range.shape != (NUM_DR002_ACTIONS, 2):
            raise ValueError(
                f"DR002 actuator ctrl_range must have shape ({NUM_DR002_ACTIONS}, 2), got {ctrl_range.shape}"
            )
        expected_shape = (num_envs, NUM_DR002_ACTIONS)
        pos = stack_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)
        vel = stack_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)
        if pos.shape != expected_shape:
            raise ValueError(f"DR002 joint position sensor stack must have shape {expected_shape}")
        if vel.shape != expected_shape:
            raise ValueError(f"DR002 joint velocity sensor stack must have shape {expected_shape}")

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "track_lin_vel_x": self._reward_track_lin_vel_x,
            "track_ang_vel_z": self._reward_track_ang_vel_z,
            "track_lin_vel_x_enhance": self._reward_track_lin_vel_x_enhance,
            "lin_vel_z": self._reward_lin_vel_z_lingzu,
            "ang_vel_xy": self._reward_ang_vel_xy_lingzu,
            "orientation": self._reward_orientation_lingzu,
            "base_height": self._reward_base_height_cmd,
            "joint_torques_l2": self._reward_joint_torques_l2,
            "joint_torques_wheel_l2": self._reward_joint_torques_wheel_l2,
            "joint_vel_l2": self._reward_joint_vel_l2,
            "joint_acc_l2": self._reward_joint_acc_l2,
            "joint_acc_wheel_l2": self._reward_joint_acc_wheel_l2,
            "joint_pos_limits": self._reward_joint_pos_limits,
            "nominal_state_lingzu": self._reward_nominal_state_lingzu,
            "action_rate_l2": self._reward_action_rate_lingzu,
            "action_smooth_lingzu": self._reward_action_smooth_lingzu,
            "undesired_contacts": self._reward_undesired_contacts,
        }

    def sample_reset_motor_gains(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        kp = np.broadcast_to(self._base_motor_kp, (num_reset, NUM_DR002_ACTIONS)).copy()
        kd = np.broadcast_to(self._base_motor_kd, (num_reset, NUM_DR002_ACTIONS)).copy()
        domain_rand = self._cfg.domain_rand
        if domain_rand.randomize_kp:
            low, high = domain_rand.kp_multiplier_range
            kp *= np.random.uniform(low, high, size=(num_reset, 1))
        if domain_rand.randomize_kd:
            low, high = domain_rand.kd_multiplier_range
            kd *= np.random.uniform(low, high, size=(num_reset, 1))
        kp[:, WHEEL_ACTION_INDICES] = 0.0
        return kp, kd

    def sample_reset_motor_runtime_randomization(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        torque_scale = np.ones((num_reset, NUM_DR002_ACTIONS), dtype=np.float64)
        default_joint_pos_offset = np.zeros((num_reset, NUM_DR002_ACTIONS), dtype=self._np_dtype)
        domain_rand = self._cfg.domain_rand
        if domain_rand.randomize_torque_scale:
            low, high = domain_rand.torque_scale_range
            torque_scale *= np.random.uniform(low, high, size=(num_reset, NUM_DR002_ACTIONS))
        if domain_rand.randomize_default_joint_pos:
            low, high = domain_rand.default_joint_pos_offset_range
            default_joint_pos_offset[:, LEG_ACTION_INDICES] = np.random.uniform(
                low, high, size=(num_reset, len(LEG_ACTION_INDICES))
            )
        return torque_scale, default_joint_pos_offset

    def set_motor_gains(self, env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        self._motor_kp[env_ids] = np.asarray(kp, dtype=np.float64)
        self._motor_kd[env_ids] = np.asarray(kd, dtype=np.float64)

    def set_motor_runtime_randomization(
        self,
        env_ids: np.ndarray,
        torque_scale: np.ndarray,
        default_joint_pos_offset: np.ndarray,
    ) -> None:
        self._motor_torque_scale[env_ids] = np.asarray(torque_scale, dtype=np.float64)
        self._default_joint_pos_offset[env_ids] = np.asarray(
            default_joint_pos_offset, dtype=self._np_dtype
        )

    def _current_lin_vel_x_range(self) -> tuple[float, float]:
        scale = self._command_curriculum_scale if self._cfg.commands.curriculum else 1.0
        low, high = self._base_command_lin_vel_x * scale
        return float(low), float(high)

    def _current_ang_vel_z_range(self) -> tuple[float, float]:
        scale = self._command_curriculum_yaw_scale if self._cfg.commands.curriculum else 1.0
        low, high = self._base_command_ang_vel_z * scale
        return float(low), float(high)

    def sample_commands(self, num_samples: int) -> np.ndarray:
        return _sample_dr002_commands(
            self._cfg.commands,
            num_samples,
            lin_vel_x_range=self._current_lin_vel_x_range(),
            ang_vel_z_range=self._current_ang_vel_z_range(),
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(actions, -self._cfg.control_config.clip_actions, self._cfg.control_config.clip_actions),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped_actions))
        state.info["current_actions"] = clipped_actions

        ctrl = np.zeros((clipped_actions.shape[0], NUM_DR002_ACTIONS), dtype=self._np_dtype)
        num_actions = clipped_actions.shape[0]
        ctrl[:, LEG_ACTION_INDICES] = (
            clipped_actions[:, LEG_ACTION_INDICES] * self._cfg.control_config.action_scale
            + self.default_angles[LEG_ACTION_INDICES]
            + self._default_joint_pos_offset[:num_actions, LEG_ACTION_INDICES]
        )
        ctrl[:, WHEEL_ACTION_INDICES] = (
            clipped_actions[:, WHEEL_ACTION_INDICES] * self._cfg.control_config.wheel_action_scale
        )
        return ctrl

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        policy_ctrl = self._delayed_policy_ctrl(policy_ctrl)
        joint_pos = stack_joint_sensors(backend, "pos", dtype=self.default_angles.dtype)
        joint_vel = stack_joint_sensors(backend, "vel", dtype=self.default_angles.dtype)
        return compute_dr002_motor_ctrl(
            policy_ctrl,
            joint_pos,
            joint_vel,
            self._motor_kp,
            self._motor_kd,
            self._ctrl_lower,
            self._ctrl_upper,
            self._last_motor_ctrl,
            self._motor_torque_scale,
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._last_motor_ctrl.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._accumulate_command_curriculum(state.info, linvel)
        self._log_command_curriculum(state.info)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        failed_now = (
            gravity[:, 2] <= self._reward_cfg.termination_gravity_z_threshold
        ) | self._has_undesired_contact(
            gravity.shape[0], threshold=self._reward_cfg.termination_contact_threshold
        )
        self._lingzu_fail_steps = np.where(failed_now, self._lingzu_fail_steps + 1, 0)
        fail_steps = max(int(round(self._reward_cfg.termination_fail_time_s / self._cfg.ctrl_dt)), 1)
        return self._lingzu_fail_steps >= fail_steps

    def _undesired_contact_values(self, num_envs: int) -> np.ndarray:
        contacts = []
        for name in self._cfg.sensor.undesired_contacts:
            sensor = np.asarray(self._backend.get_sensor_data(name)).reshape(num_envs, -1)[:, 0]
            contacts.append(sensor)
        if not contacts:
            return np.zeros((num_envs, 0), dtype=get_global_dtype())
        return np.asarray(np.stack(contacts, axis=1), dtype=get_global_dtype())

    def _has_undesired_contact(self, num_envs: int, *, threshold: float) -> np.ndarray:
        contacts = self._undesired_contact_values(num_envs)
        if contacts.shape[1] == 0:
            return np.zeros((num_envs,), dtype=np.bool_)
        return np.any(contacts > float(threshold), axis=1)

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        interval = max(int(round(float(self._cfg.commands.resampling_time) / self._cfg.ctrl_dt)), 1)
        steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
        resample_mask = (steps > 0) & ((steps % interval) == 0)
        if np.any(resample_mask):
            commands_arr[resample_mask] = self.sample_commands(int(np.count_nonzero(resample_mask)))
        info["commands"] = commands_arr

    def _accumulate_command_curriculum(self, info: dict, linvel: np.ndarray) -> None:
        if not self._cfg.commands.curriculum:
            return
        commands = np.asarray(info.get("commands"), dtype=get_global_dtype())
        if commands.shape[0] != linvel.shape[0]:
            return
        lin_error = np.square(commands[:, 0] - linvel[:, 0])
        lin_tracking = np.exp(-lin_error / (self._reward_cfg.tracking_sigma**2))
        try:
            gyro = self.get_gyro()
        except Exception:
            gyro = None
        if isinstance(gyro, np.ndarray) and gyro.shape[0] == commands.shape[0]:
            yaw_error = np.square(commands[:, 1] - gyro[:, 2])
            yaw_tracking = np.exp(-yaw_error / (self._reward_cfg.tracking_sigma**2))
        else:
            yaw_tracking = lin_tracking
        self._episode_track_lin_vel_x_sum[: lin_tracking.shape[0]] += lin_tracking
        self._episode_track_ang_vel_z_sum[: yaw_tracking.shape[0]] += yaw_tracking
        self._episode_track_lin_vel_x_steps[: lin_tracking.shape[0]] += 1

    def _update_command_curriculum(self, env_ids: np.ndarray) -> None:
        if not self._cfg.commands.curriculum or env_ids.size == 0:
            return
        steps = self._episode_track_lin_vel_x_steps[env_ids]
        valid = steps > 0
        if not np.any(valid):
            return
        lin_tracking = self._episode_track_lin_vel_x_sum[env_ids][valid] / np.maximum(steps[valid], 1)
        yaw_tracking = self._episode_track_ang_vel_z_sum[env_ids][valid] / np.maximum(steps[valid], 1)
        mean_tracking = float(np.mean(0.5 * (lin_tracking + yaw_tracking)))
        self._last_command_curriculum_mean_tracking = float(mean_tracking)
        if mean_tracking > float(self._cfg.commands.curriculum_threshold):
            self._command_curriculum_scale = min(
                self._command_curriculum_final_scale,
                self._command_curriculum_scale + float(self._cfg.commands.curriculum_step),
            )
            self._command_curriculum_yaw_scale = min(
                self._command_curriculum_yaw_final_scale,
                self._command_curriculum_yaw_scale + float(self._cfg.commands.curriculum_step),
            )

    def _log_command_curriculum(self, info: dict) -> None:
        if not self._cfg.commands.curriculum:
            return
        low, high = self._current_lin_vel_x_range()
        yaw_low, yaw_high = self._current_ang_vel_z_range()
        log = info.setdefault("log", {})
        log["command_curriculum/scale"] = float(self._command_curriculum_scale)
        log["command_curriculum/lin_vel_x_min"] = low
        log["command_curriculum/lin_vel_x_max"] = high
        log["command_curriculum/yaw_scale"] = float(self._command_curriculum_yaw_scale)
        log["command_curriculum/ang_vel_z_min"] = yaw_low
        log["command_curriculum/ang_vel_z_max"] = yaw_high
        if np.isfinite(self._last_command_curriculum_mean_tracking):
            log["command_curriculum/last_mean_tracking"] = float(
                self._last_command_curriculum_mean_tracking
            )

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        *,
        env_ids: np.ndarray | None = None,
        reset_history: bool = False,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        num_obs = gyro.shape[0]
        current_actions = np.asarray(info.get("current_actions", np.zeros((num_obs, self._num_action))), dtype=get_global_dtype())
        dof_vel_obs = dof_vel[:, JOINT_VEL_OBSERVATION_INDICES]
        frame_terms = [
            self._obs_noise(gyro, noise_cfg.scale_gyro),
            self._obs_noise(-gravity, noise_cfg.scale_gravity),
            self._obs_noise(dof_pos[:, LEG_ACTION_INDICES] - self.default_angles[LEG_ACTION_INDICES], noise_cfg.scale_joint_angle),
            self._obs_noise(dof_vel_obs, noise_cfg.scale_joint_vel) * 0.1,
            current_actions,
            np.asarray(info["commands"], dtype=get_global_dtype()),
        ]
        actor = self._update_history(frame_terms, env_ids=env_ids, reset_history=reset_history)
        motor_ctrl = info.get("torques", np.zeros((num_obs, self._num_action), dtype=dof_pos.dtype))
        critic = np.concatenate(
            [
                linvel,
                gyro,
                -gravity,
                dof_pos[:, LEG_ACTION_INDICES] - self.default_angles[LEG_ACTION_INDICES],
                dof_vel_obs * 0.1,
                current_actions,
                np.asarray(info["commands"], dtype=get_global_dtype()),
                motor_ctrl,
                self._motor_kp[:num_obs],
                self._motor_kd[:num_obs],
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": actor, "critic": critic, "privileged_target": linvel.astype(get_global_dtype())}

    def _update_history(
        self,
        frame_terms: list[np.ndarray],
        *,
        env_ids: np.ndarray | None,
        reset_history: bool,
    ) -> np.ndarray:
        if env_ids is None:
            if reset_history:
                for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                    hist[:] = frame[:, None, :]
            else:
                for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                    hist[:, :-1] = hist[:, 1:]
                    hist[:, -1] = frame
            return np.concatenate([hist.reshape(self._num_envs, -1) for hist in self._history_terms], axis=1)

        env_ids = np.asarray(env_ids, dtype=np.intp)
        if reset_history:
            for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                hist[env_ids] = frame[:, None, :]
        else:
            for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                hist[env_ids, :-1] = hist[env_ids, 1:]
                hist[env_ids, -1] = frame
        return np.concatenate([hist[env_ids].reshape(len(env_ids), -1) for hist in self._history_terms], axis=1)

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        base_height = self._reward_base_height_values(linvel.shape[0])
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=linvel.shape[0],
            default_angles=DEFAULT_DR002_ANGLES.astype(get_global_dtype()),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height=base_height,
            gravity=gravity,
        )
        reward = rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )
        step_count = info.get("steps", np.zeros((ctx.num_envs,), dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0 and base_height.shape[0] == ctx.num_envs:
            target = np.asarray(info["commands"], dtype=get_global_dtype())[:, 2]
            height_error = base_height - target
            log = info.setdefault("log", {})
            log["base_height/mean"] = float(np.mean(base_height))
            log["base_height/target_mean"] = float(np.mean(target))
            log["base_height/error_mean"] = float(np.mean(height_error))
            log["base_height/abs_error_mean"] = float(np.mean(np.abs(height_error)))
        return reward

    def _clip_lingzu_reward(self, name: str, reward: np.ndarray, clip_single_reward: float = 1.0) -> np.ndarray:
        scale = float(self._reward_cfg.scales.get(name, 0.0))
        if scale == 0.0:
            return np.asarray(reward, dtype=get_global_dtype())
        weighted_dt = np.asarray(reward, dtype=get_global_dtype()) * scale * self._cfg.ctrl_dt
        clipped = np.clip(
            weighted_dt,
            -float(clip_single_reward) * self._cfg.ctrl_dt,
            float(clip_single_reward) * self._cfg.ctrl_dt,
        )
        return np.asarray(clipped / (scale * self._cfg.ctrl_dt), dtype=get_global_dtype())

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())

    def _reward_track_lin_vel_x(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 0] - ctx.linvel[:, 0])
        reward = np.asarray(np.exp(-error / (ctx.tracking_sigma**2)), dtype=get_global_dtype())
        return self._clip_lingzu_reward("track_lin_vel_x", reward)

    def _reward_track_lin_vel_x_enhance(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 0] - ctx.linvel[:, 0])
        reward = np.asarray(np.exp(-error / (ctx.tracking_sigma**2 * 10.0)) - 1.0, dtype=get_global_dtype())
        return self._clip_lingzu_reward("track_lin_vel_x_enhance", reward)

    def _reward_track_ang_vel_z(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 1] - ctx.gyro[:, 2])
        reward = np.asarray(np.exp(-error / (ctx.tracking_sigma**2)), dtype=get_global_dtype())
        return self._clip_lingzu_reward("track_ang_vel_z", reward)

    def _reward_lin_vel_z_lingzu(self, ctx: RewardContext) -> np.ndarray:
        reward = np.asarray(np.square(ctx.linvel[:, 2]), dtype=get_global_dtype())
        return self._clip_lingzu_reward("lin_vel_z", reward)

    def _reward_ang_vel_xy_lingzu(self, ctx: RewardContext) -> np.ndarray:
        reward = np.asarray(np.sum(np.square(ctx.gyro[:, :2]), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("ang_vel_xy", reward)

    def _reward_orientation_lingzu(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        reward = np.asarray(np.square(ctx.gravity[:, 0]) + np.square(ctx.gravity[:, 1]), dtype=get_global_dtype())
        return self._clip_lingzu_reward("orientation", reward)

    def _reward_base_height_cmd(self, ctx: RewardContext) -> np.ndarray:
        target = ctx.info["commands"][:, 2]
        std = max(float(self._reward_cfg.base_height_std), 1.0e-6)
        reward = np.asarray(np.square(ctx.base_height - target) / (std * std), dtype=get_global_dtype())
        return self._clip_lingzu_reward(
            "base_height",
            reward,
            clip_single_reward=float(self._reward_cfg.base_height_clip),
        )

    def _reward_joint_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))), dtype=get_global_dtype())
        reward = np.asarray(np.sum(np.square(torques[:, LEG_ACTION_INDICES]), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("joint_torques_l2", reward)

    def _reward_joint_torques_wheel_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))), dtype=get_global_dtype())
        reward = np.asarray(np.sum(np.square(torques[:, WHEEL_ACTION_INDICES]), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("joint_torques_wheel_l2", reward)

    def _reward_joint_vel_l2(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        reward = np.asarray(np.sum(np.square(ctx.dof_vel[:, LEG_ACTION_INDICES]), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("joint_vel_l2", reward)

    def _reward_joint_acc_l2(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_DR002_ACTIONS))), dtype=get_global_dtype())
        reward = np.asarray(np.sum(np.square(qacc[:, LEG_ACTION_INDICES]), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("joint_acc_l2", reward)

    def _reward_joint_acc_wheel_l2(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_DR002_ACTIONS))), dtype=get_global_dtype())
        reward = np.asarray(np.sum(np.square(qacc[:, WHEEL_ACTION_INDICES]), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("joint_acc_wheel_l2", reward)

    def _reward_joint_pos_limits(self, ctx: RewardContext) -> np.ndarray:
        if self._joint_range is None:
            return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        leg = LEG_ACTION_INDICES
        joint_range = np.asarray(self._joint_range, dtype=get_global_dtype())
        lower = joint_range[leg, 0]
        upper = joint_range[leg, 1]
        low_error = np.clip(lower - ctx.dof_pos[:, leg], 0.0, None)
        high_error = np.clip(ctx.dof_pos[:, leg] - upper, 0.0, None)
        reward = np.asarray(np.sum(low_error + high_error, axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("joint_pos_limits", reward)

    def _reward_nominal_state_lingzu(self, ctx: RewardContext) -> np.ndarray:
        theta_left = ctx.dof_pos[:, 0] + 0.5 * ctx.dof_pos[:, 1]
        theta_right = ctx.dof_pos[:, 3] + 0.5 * ctx.dof_pos[:, 4]
        reward = np.asarray(np.square(theta_left - theta_right), dtype=get_global_dtype())
        return self._clip_lingzu_reward("nominal_state_lingzu", reward)

    def _reward_action_rate_lingzu(self, ctx: RewardContext) -> np.ndarray:
        current = np.asarray(ctx.info["current_actions"], dtype=get_global_dtype())
        last = np.asarray(ctx.info["last_actions"], dtype=get_global_dtype())
        reward = np.asarray(np.sum(np.square(current - last), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("action_rate_l2", reward)

    def _reward_action_smooth_lingzu(self, ctx: RewardContext) -> np.ndarray:
        current = np.asarray(ctx.info["current_actions"], dtype=get_global_dtype())
        leg = LEG_ACTION_INDICES
        diff = current[:, leg] - 2.0 * self._lingzu_prev_action[:, leg] + self._lingzu_prev_prev_action[:, leg]
        reward = np.asarray(np.sum(np.square(diff), axis=1), dtype=get_global_dtype())
        reward *= self._lingzu_action_history_count >= 2
        self._lingzu_prev_prev_action[:] = self._lingzu_prev_action
        self._lingzu_prev_action[:] = current
        self._lingzu_action_history_count += 1
        return self._clip_lingzu_reward("action_smooth_lingzu", reward)

    def _reward_undesired_contacts(self, ctx: RewardContext) -> np.ndarray:
        contacts = self._undesired_contact_values(ctx.num_envs)
        reward = np.asarray(
            np.sum(contacts > self._reward_cfg.undesired_contact_threshold, axis=1),
            dtype=get_global_dtype(),
        )
        return self._clip_lingzu_reward("undesired_contacts", reward)


registry.register_env("DR002JoystickFlat", DR002JoystickEnv, sim_backend="motrix")
