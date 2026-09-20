"""Caching must preserve PPO updates, command replacement and encoder supervision."""

import copy

import pytest
import torch

from unilab.algos.torch.pe03.ppo import PE03PPO, matmul_precision
from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.envs.locomotion.pe03.config import load_config


@pytest.mark.parametrize("experiment", ["standing", "walking", "gait_fixed"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_updates_equal_recomputation_across_rollouts(tmp_path, experiment, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = load_config(
        [
            f"+experiment={experiment}",
            "algo.num_envs=4",
            "algo.num_steps_per_env=3",
            "algo.num_learning_epochs=3",
            "algo.num_mini_batches=2",
            f"training.device={device}",
            "training.matmul_precision=highest",
            "training.mujoco_threads=1",
            "training.logger=none",
            "training.export=false",
            "training.evaluation_interval=0",
        ]
    )
    runner = PE03Runner(config, tmp_path)
    try:
        reference_policy = copy.deepcopy(runner.policy)
        config.training.cache_update_inputs = False
        reference = PE03PPO(reference_policy, config)
        for _ in range(2):
            rollout, _ = runner.collect()
            # In v4 the supplied command replaces the last history frame's
            # command fields. Caching must preserve this even when they differ.
            rollout["command"] = rollout["command"] + 0.13
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
            actual = runner.algorithm.update(rollout)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            expected = reference.update(rollout)
            assert actual == expected
            for key, value in runner.policy.state_dict().items():
                torch.testing.assert_close(
                    value, reference_policy.state_dict()[key], rtol=0, atol=0
                )
        assert runner.algorithm.updates == reference.updates == 12
        assert runner.algorithm.encoder_updates == reference.encoder_updates == 12
    finally:
        runner.close()


def test_matmul_precision_restored_on_success_and_failure():
    original = torch.get_float32_matmul_precision()
    alternate = "high" if original == "highest" else "highest"
    with matmul_precision(alternate):
        assert torch.get_float32_matmul_precision() == alternate
    assert torch.get_float32_matmul_precision() == original
    with pytest.raises(RuntimeError, match="update failed"):
        with matmul_precision(alternate):
            raise RuntimeError("update failed")
    assert torch.get_float32_matmul_precision() == original


@pytest.mark.parametrize("experiment", ["walking", "gait_fixed"])
def test_update_runtime_settings_validate_and_support_old_configs(experiment):
    config = load_config([f"+experiment={experiment}"])
    assert config.training.matmul_precision == "high"
    assert config.training.cache_update_inputs is True
    with pytest.raises(ValueError, match="matmul_precision"):
        load_config([f"+experiment={experiment}", "training.matmul_precision=medium"])
    with pytest.raises(ValueError, match="cache_update_inputs"):
        load_config([f"+experiment={experiment}", "training.cache_update_inputs=invalid"])
    from unilab.algos.torch.pe03.policy import PE03EncoderPolicy

    del config.training.matmul_precision
    del config.training.cache_update_inputs
    algorithm = PE03PPO(PE03EncoderPolicy(config), config)
    assert algorithm.matmul_precision == "highest"
    assert algorithm.cache_update_inputs is True
