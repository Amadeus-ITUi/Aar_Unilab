"""Rollout ownership and terminal bootstrapping across PE03 observation layouts."""

import numpy as np
import pytest
import torch

from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.envs.locomotion.pe03.config import load_config


@pytest.mark.parametrize("experiment", ["standing", "walking", "gait_fixed"])
@pytest.mark.parametrize("ending", [[], [0], [0, 1, 2, 3]])
def test_rollout_values_match_observations_and_storage_is_owned(tmp_path, experiment, ending):
    config = load_config(
        [
            f"+experiment={experiment}",
            "algo.num_envs=4",
            "algo.num_steps_per_env=3",
            "training.device=cpu",
            "training.mujoco_threads=1",
            "training.logger=none",
            "training.export=false",
            "training.evaluation_interval=0",
        ]
    )
    runner = PE03Runner(config, tmp_path)
    try:
        runner.env.episode_steps[ending] = runner.env.max_episode_steps
        original_step = runner.env.step
        observations, next_values = [], []

        def record_step(action):
            observations.append({key: value.copy() for key, value in runner.env.state.obs.items()})
            state = original_step(action)
            # Independently evaluate the full batch with terminal observations
            # substituted before the critic, including sparse and dense resets.
            bootstrap = {key: value.copy() for key, value in state.obs.items()}
            if state.final_observation is not None:
                done = state.terminated | state.truncated
                for key in bootstrap:
                    bootstrap[key][done] = state.final_observation[key][done]
            with torch.no_grad():
                next_values.append(runner.algorithm.value(runner._tensors(bootstrap)))
            return state

        runner.env.step = record_step
        rollout, _ = runner.collect()
        expected_done = torch.zeros(4, dtype=torch.bool)
        expected_done[ending] = True
        torch.testing.assert_close(rollout["truncated"][0], expected_done)
        torch.testing.assert_close(rollout["next_value"], torch.stack(next_values))
        for key in observations[0]:
            np.testing.assert_array_equal(
                rollout[key].numpy(), np.stack([o[key] for o in observations])
            )
        with torch.no_grad():
            expected_values = runner.algorithm.value(
                {key: value.flatten(0, 1) for key, value in rollout.items()}
            )
        torch.testing.assert_close(rollout["value"].flatten(), expected_values)

        runner.env.step = original_step
        retained = {key: value.clone() for key, value in rollout.items()}
        runner.collect()
        for key in retained:
            torch.testing.assert_close(rollout[key], retained[key], rtol=0, atol=0)
    finally:
        runner.close()
