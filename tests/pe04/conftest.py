import pytest

from unilab.envs.locomotion.pe04.config import load_config


@pytest.fixture
def config():
    return load_config(
        [
            "algo.num_envs=2",
            "algo.num_steps_per_env=4",
            "algo.max_iterations=1",
            "algo.num_learning_epochs=1",
            "algo.num_mini_batches=1",
            "training.logger=none",
            "training.device=cpu",
            "training.evaluation_interval=0",
            "training.mujoco_threads=1",
        ]
    )
