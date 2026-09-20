"""Select an independently owned environment by its versioned task contract."""

from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.vector_env import PE03VectorEnv


def make_env(config, **kwargs):
    return (PE03GaitEnv if config.observation == "pe03_v4" else PE03VectorEnv)(config, **kwargs)
