"""Small MuJoCo protocol adapter used to validate the PE01 migration boundary.

The 30-value frame and ten-frame actor history match the predecessor contract.
Reward ownership and the custom PPO implementation remain independently
versioned in the PE01 profile instead of being folded into WE11.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PE01Observation:
    actor: np.ndarray
    critic: np.ndarray


class PE01Env:
    frame_size = 30
    history_length = 10
    action_size = 6

    def __init__(self, model_path: str | Path | None = None) -> None:
        import mujoco

        default = Path(__file__).resolve().parents[3] / "assets/robots/pe01/scene.xml"
        self.model_path = Path(model_path or default)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self._history = np.zeros((self.history_length, self.frame_size), dtype=np.float32)

    def reset(self) -> PE01Observation:
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.38
        self._mujoco.mj_forward(self.model, self.data)
        self._history.fill(0.0)
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[PE01Observation, float, bool, dict[str, float]]:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (self.action_size,):
            raise ValueError(f"PE01 action must have shape ({self.action_size},), got {value.shape}")
        self.data.ctrl[:] = np.clip(value, -1.0, 1.0)
        self._mujoco.mj_step(self.model, self.data)
        observation = self._observation()
        height = float(self.data.qpos[2])
        reward = height - 0.001 * float(np.square(value).sum())
        terminated = height < 0.08
        return observation, reward, terminated, {"base_height": height}

    def _observation(self) -> PE01Observation:
        frame = np.zeros(self.frame_size, dtype=np.float32)
        qpos = np.asarray(self.data.qpos[7:13], dtype=np.float32)
        qvel = np.asarray(self.data.qvel[6:12], dtype=np.float32)
        frame[:6] = qpos
        frame[6:12] = qvel
        frame[12:15] = np.asarray(self.data.qvel[3:6], dtype=np.float32)
        frame[15:18] = np.asarray(self.data.qvel[:3], dtype=np.float32)
        frame[18:24] = np.asarray(self.data.ctrl, dtype=np.float32)
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame
        # The predecessor contract adds three privileged values to its 30-value frame.
        critic = np.concatenate((frame, np.asarray(self.data.qvel[:3], dtype=np.float32)))
        return PE01Observation(actor=self._history.reshape(-1).copy(), critic=critic)
