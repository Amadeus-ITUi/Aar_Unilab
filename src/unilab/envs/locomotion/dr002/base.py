from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)

JOINT_SENSOR_PREFIXES: tuple[str, ...] = (
    "left_thigh",
    "left_calf",
    "left_foot",
    "right_thigh",
    "right_calf",
    "right_foot",
)
LEG_ACTION_INDICES = np.asarray([0, 1, 3, 4], dtype=np.int32)
WHEEL_ACTION_INDICES = np.asarray([2, 5], dtype=np.int32)
# IsaacLab trains joint-velocity observations in leg-then-wheel order:
# [left_thigh, left_calf, right_thigh, right_calf, left_foot, right_foot].
JOINT_VEL_OBSERVATION_INDICES = np.asarray([0, 1, 3, 4, 2, 5], dtype=np.int32)

NUM_DR002_ACTIONS = len(JOINT_SENSOR_PREFIXES)
NUM_LEG_ACTIONS = len(LEG_ACTION_INDICES)
NUM_WHEEL_ACTIONS = len(WHEEL_ACTION_INDICES)

DEFAULT_DR002_ANGLES = np.asarray([0.8, -1.6, 0.0, 0.8, -1.6, 0.0], dtype=np.float64)


@dataclass
class NoiseConfig(BaseNoiseConfig):
    scale_joint_angle: float = 0.08
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.2
    scale_wheel_vel: float = 0.5


@dataclass
class ControlConfig(PdControlConfig):
    action_scale: float = 0.5
    wheel_action_scale: float = 10.0
    clip_actions: float = 100.0
    simulate_action_latency: bool = True
    action_delay_min_steps: int = 4
    action_delay_max_steps: int = 8
    resample_action_delay: bool = True
    Kp: list[float] = field(default_factory=lambda: [2.0, 8.0, 0.0, 2.0, 8.0, 0.0])  # noqa: N815
    Kd: list[float] = field(default_factory=lambda: [0.1, 0.8, 0.05, 0.1, 0.8, 0.05])  # noqa: N815


@dataclass
class Asset:
    base_name = "base_link"
    ground = "floor"


@dataclass
class DR002BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.0025
    ctrl_dt: float = 0.02


def stack_joint_sensors(backend, suffix: str, *, dtype: np.dtype | type) -> np.ndarray:
    names = tuple(f"{prefix}_{suffix}" for prefix in JOINT_SENSOR_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1)[:, :NUM_DR002_ACTIONS], dtype=dtype)


def compute_dr002_motor_ctrl(
    policy_ctrl: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    ctrl_lower: np.ndarray,
    ctrl_upper: np.ndarray,
    out: np.ndarray,
    torque_scale: np.ndarray | None = None,
) -> np.ndarray:
    out.fill(0.0)
    leg = LEG_ACTION_INDICES
    wheel = WHEEL_ACTION_INDICES
    out[:, leg] = kp[:, leg] * (policy_ctrl[:, leg] - joint_pos[:, leg]) - kd[:, leg] * joint_vel[:, leg]
    out[:, wheel] = kd[:, wheel] * (policy_ctrl[:, wheel] - joint_vel[:, wheel])
    if torque_scale is not None:
        out *= torque_scale
    np.clip(out, ctrl_lower, ctrl_upper, out=out)
    return out


class DR002BaseEnv(LocomotionBaseEnv):
    _cfg: DR002BaseCfg

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_DR002_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_DR002_ANGLES, dtype=self.default_angles.dtype)
        if self._init_qpos.shape[0] >= 7 + NUM_DR002_ACTIONS:
            self._init_qpos[-NUM_DR002_ACTIONS:] = self.default_angles
        if self._init_qvel.shape[0] >= 6 + NUM_DR002_ACTIONS:
            self._init_qvel[-NUM_DR002_ACTIONS:] = 0.0

    def get_dof_pos(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)
