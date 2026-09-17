"""PE02-owned observations, rewards, reset and control contract."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from unilab.base.backend.mujoco.single_robot import SingleRobotSimulation
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe02.config import ROOT, load_legacy_config, validate_config


@dataclass(frozen=True)
class PE02Observation:
    actor: np.ndarray
    critic: np.ndarray


class PE02Env:
    def __init__(
        self, model_path: str | Path | None = None, *, config: DictConfig | None = None
    ) -> None:
        self.config = OmegaConf.merge(config if config is not None else load_legacy_config())
        validate_config(self.config)
        if self.config.observation != "pe02_v1":
            raise ValueError("PE02Env is the v1 compatibility adapter; use PE02VectorEnv for v2")
        self.frame_size = int(self.config.env.frame_size)
        self.history_length = int(self.config.env.history_length)
        self.action_size = len(self.config.env.joint_order)
        self.physics_hz = int(self.config.control.physics_hz)
        self.policy_hz = int(self.config.control.policy_hz)
        self.physics_steps_per_action = self.physics_hz // self.policy_hz
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else repository_path(str(self.config.env.model_path), ROOT)
        )
        self._backend = SingleRobotSimulation(
            self.model_path,
            physics_hz=self.physics_hz,
            joint_order=tuple(self.config.env.joint_order),
            reset_keyframe=self.config.env.get("reset_keyframe"),
        )
        self.model = self._backend.model
        self.data = self._backend.data
        self._history = np.zeros((self.history_length, self.frame_size), dtype=np.float32)

    def reset(self) -> PE02Observation:
        height = self.config.env.initial_height
        self._backend.reset(float(height) if height is not None else None)
        self._history.fill(0.0)
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[PE02Observation, float, bool, dict[str, float]]:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (self.action_size,):
            raise ValueError(
                f"PE02 action must have shape ({self.action_size},), got {value.shape}"
            )
        clip = float(self.config.control.action_clip)
        self._backend.step(
            np.clip(value, -clip, clip) * float(self.config.control.action_scale),
            self.physics_steps_per_action,
        )
        observation = self._observation()
        height = self._backend.base_height
        reward = float(self.config.reward.base_height) * height + float(
            self.config.reward.action_l2
        ) * float(np.square(value).sum())
        terminated = height < float(self.config.env.termination_height)
        return observation, reward, terminated, {"base_height": height}

    def _observation(self) -> PE02Observation:
        frame = np.zeros(self.frame_size, dtype=np.float32)
        count = self.action_size
        frame[:count] = self._backend.joint_positions
        frame[count : 2 * count] = self._backend.joint_velocities
        frame[2 * count : 2 * count + 3] = self._backend.angular_velocity
        frame[2 * count + 3 : 2 * count + 6] = self._backend.linear_velocity
        frame[2 * count + 6 : 3 * count + 6] = self._backend.control
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame
        critic = np.concatenate(
            (frame, np.asarray(self._backend.linear_velocity, dtype=np.float32))
        )
        return PE02Observation(actor=self._history.reshape(-1).copy(), critic=critic)
