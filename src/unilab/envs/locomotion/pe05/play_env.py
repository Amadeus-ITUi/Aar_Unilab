"""PE05 v1 deterministic single-robot playback over the same training backend."""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from unilab.envs.locomotion.pe05.env import PE05Observation
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


class PE05PlayEnv:
    frame_size = 30
    action_size = 6

    def __init__(self, config: DictConfig, model_path: Path | None = None) -> None:
        self.vector = PE05VectorEnv(
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
        self.history_length = int(config.env.history_length)
        self.physics_hz = int(config.control.physics_hz)
        self.policy_hz = int(config.control.policy_hz)

    def reset(self) -> PE05Observation:
        state = self.vector.reset()
        self.data.xfrc_applied[:] = 0
        self.vector.backend.sync_visual_data(self.data)
        return PE05Observation(state.obs["actor"][0], state.obs["critic"][0])

    def set_command(self, command: np.ndarray) -> np.ndarray:
        self.vector.cfg["play"]["command"] = np.asarray(command).tolist()
        self.vector.commands[0] = command
        scale = self.vector.cfg["normalization"]
        return np.asarray(command, dtype=np.float32) * np.array(
            [scale["lin_vel"], scale["lin_vel"], scale["ang_vel"]], dtype=np.float32
        )

    def step(self, action: np.ndarray) -> tuple[PE05Observation, float, bool, dict[str, float]]:
        self.vector.backend.stage_visual_forces(self.data)
        state = self.vector.step(np.asarray(action)[None])
        self.vector.backend.sync_visual_data(self.data)
        obs = PE05Observation(state.obs["actor"][0], state.obs["critic"][0])
        # Like the native player, playback continues after falls; training uses autoreset.
        return obs, float(state.reward[0]), False, {"base_height": float(self.data.qpos[2])}

    def close(self) -> None:
        self.vector.close()


def make_play_env(config: DictConfig, model_path: Path | None = None) -> PE05PlayEnv:
    return PE05PlayEnv(config, model_path)
