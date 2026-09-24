"""PE04 TRON1-style deterministic playback over its own training environment."""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from unilab.envs.locomotion.pe04.observation import PE04Observation
from unilab.envs.locomotion.pe04.vector_env import PE04VectorEnv


class PE04PlayEnv:
    action_size = 6

    def __init__(self, config: DictConfig, model_path: Path | None = None) -> None:
        self.vector = PE04VectorEnv(
            config,
            evaluation=True,
            num_envs=1,
            visual=True,
            model_path=model_path,
            auto_reset=False,
        )
        self.model = self.vector.backend.model
        self.data = self.vector.backend.create_visual_data()
        self.config = config
        self.frame_size = int(config.env.frame_size)
        self.history_length = int(config.env.history_length)
        self.physics_hz = int(config.control.physics_hz)
        self.policy_hz = int(config.control.policy_hz)

    def reset(self) -> PE04Observation:
        state = self.vector.reset()
        self.data.xfrc_applied[:] = 0
        self.vector.backend.sync_visual_data(self.data)
        return PE04Observation(state.obs["actor"][0], state.obs["critic"][0])

    def set_command(self, command: np.ndarray) -> np.ndarray:
        return self.vector.set_command(command, self.config.play.gait)[0]

    def step(self, action: np.ndarray) -> tuple[PE04Observation, float, bool, dict[str, float]]:
        self.vector.backend.stage_visual_forces(self.data)
        state = self.vector.step(np.asarray(action)[None])
        self.vector.backend.sync_visual_data(self.data)
        obs = PE04Observation(state.obs["actor"][0], state.obs["critic"][0])
        # Like the native player, playback continues after falls; training uses autoreset.
        return obs, float(state.reward[0]), False, {"base_height": float(self.data.qpos[2])}

    def close(self) -> None:
        self.vector.close()


def make_play_env(config: DictConfig, model_path: Path | None = None) -> PE04PlayEnv:
    return PE04PlayEnv(config, model_path)
