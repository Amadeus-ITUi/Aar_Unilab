"""PE02 v2 deterministic single-robot playback over the same training backend."""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from unilab.envs.locomotion.pe02.env import PE02Env, PE02Observation
from unilab.envs.locomotion.pe02.vector_env import PE02VectorEnv


class PE02PlayEnv:
    action_size = 6

    def __init__(self, config: DictConfig, model_path: Path | None = None) -> None:
        self.vector = PE02VectorEnv(
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

    def reset(self) -> PE02Observation:
        state = self.vector.reset()
        self.data.xfrc_applied[:] = 0
        self.vector.backend.sync_visual_data(self.data)
        return PE02Observation(state.obs["actor"][0], state.obs["critic"][0])

    def set_command(self, command: np.ndarray) -> np.ndarray:
        self.vector.cfg["play"]["command"] = np.asarray(command).tolist()
        self.vector.commands[0] = command
        scale = self.vector.cfg["normalization"]
        return np.asarray(command, dtype=np.float32) * np.array(
            [scale["lin_vel"], scale["lin_vel"], scale["ang_vel"]], dtype=np.float32
        )

    def step(self, action: np.ndarray) -> tuple[PE02Observation, float, bool, dict[str, float]]:
        self.vector.backend.stage_visual_forces(self.data)
        state = self.vector.step(np.asarray(action)[None])
        self.vector.backend.sync_visual_data(self.data)
        obs = PE02Observation(state.obs["actor"][0], state.obs["critic"][0])
        # Like the native player, playback continues after falls; training uses autoreset.
        return obs, float(state.reward[0]), False, {"base_height": float(self.data.qpos[2])}

    def close(self) -> None:
        self.vector.close()


def make_play_env(config: DictConfig, model_path: Path | None = None) -> PE02Env | PE02PlayEnv:
    if config.observation == "pe02_v1":
        return PE02Env(model_path, config=config)
    return PE02PlayEnv(config, model_path)
