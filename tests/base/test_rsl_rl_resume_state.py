from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from unilab.training.experiment import patch_rsl_rl_resume_state
from unilab.training.rsl_rl import RslRlVecEnvWrapper


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.load_calls: list[tuple[dict[str, Any], dict[str, Any] | None, bool]] = []

    def save(self) -> dict[str, Any]:
        return {"actor_state_dict": {"weight": torch.asarray([1.0])}}

    def load(
        self,
        loaded_dict: dict[str, Any],
        load_cfg: dict[str, Any] | None,
        strict: bool,
    ) -> bool:
        self.load_calls.append((loaded_dict, load_cfg, strict))
        return True if load_cfg is None else bool(load_cfg.get("iteration", False))


class _FakeEnvironmentAdapter:
    def __init__(self, state: dict[str, Any] | None) -> None:
        self.saved_state = state
        self.loaded_states: list[dict[str, Any]] = []

    def training_state_dict(self) -> dict[str, Any] | None:
        return self.saved_state

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded_states.append(state)


class _FakeLogger:
    def __init__(self) -> None:
        self.tot_time = 12.5
        self.tot_timesteps = 3456
        self.saved_models: list[tuple[str, int]] = []

    def save_model(self, path: str, iteration: int) -> None:
        self.saved_models.append((path, iteration))


def _fake_runner(env: _FakeEnvironmentAdapter):
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    patch_rsl_rl_resume_state()
    runner = object.__new__(OnPolicyRunner)
    runner.alg = _FakeAlgorithm()
    runner.env = env
    runner.logger = _FakeLogger()
    runner.current_learning_iteration = 42
    return runner


def test_rsl_rl_checkpoint_round_trips_environment_training_state(tmp_path) -> None:
    env_state = {
        "kind": "test.curriculum",
        "version": 1,
        "command": {"scale": 2.0},
        "force": {"level": 3, "paths": ["1hz.csv", "2hz.csv", "3hz.csv"]},
    }
    env = _FakeEnvironmentAdapter(env_state)
    runner = _fake_runner(env)
    checkpoint = tmp_path / "model.pt"

    runner.save(str(checkpoint), infos={"source": "unit-test"})

    safe_payload = torch.load(checkpoint, weights_only=True)
    assert safe_payload["unilab_env_training_state"] == env_state
    assert safe_payload["unilab_logger_state"] == {
        "tot_time": pytest.approx(12.5),
        "tot_timesteps": 3456,
    }
    assert runner.logger.saved_models == [(str(checkpoint), 42)]

    runner.current_learning_iteration = 0
    runner.logger.tot_time = 0.0
    runner.logger.tot_timesteps = 0
    infos = runner.load(str(checkpoint))

    assert infos == {"source": "unit-test"}
    assert runner.current_learning_iteration == 42
    assert runner.logger.tot_time == pytest.approx(12.5)
    assert runner.logger.tot_timesteps == 3456
    assert env.loaded_states == [env_state]


def test_rsl_rl_actor_only_load_does_not_restore_environment_state(tmp_path) -> None:
    env = _FakeEnvironmentAdapter({"kind": "test.curriculum", "version": 1})
    runner = _fake_runner(env)
    checkpoint = tmp_path / "actor_only.pt"
    runner.save(str(checkpoint))

    runner.load(str(checkpoint), load_cfg={"actor": True, "iteration": False})

    assert env.loaded_states == []


def test_rsl_rl_new_stateless_checkpoint_resumes_without_legacy_warning(tmp_path) -> None:
    env = _FakeEnvironmentAdapter(None)
    runner = _fake_runner(env)
    checkpoint = tmp_path / "stateless.pt"
    runner.save(str(checkpoint))

    payload = torch.load(checkpoint, weights_only=True)
    assert "unilab_env_training_state" in payload
    assert payload["unilab_env_training_state"] is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.load(str(checkpoint))

    assert caught == []
    assert env.loaded_states == []


def test_rsl_rl_stateless_checkpoint_warns_when_current_environment_is_stateful(
    tmp_path,
) -> None:
    env = _FakeEnvironmentAdapter(None)
    runner = _fake_runner(env)
    checkpoint = tmp_path / "stateless_to_stateful.pt"
    runner.save(str(checkpoint))
    env.saved_state = {"fresh": True}

    with pytest.warns(RuntimeWarning, match="contains no environment curriculum state"):
        runner.load(str(checkpoint))

    assert env.loaded_states == []


def test_rsl_rl_legacy_stateless_checkpoint_does_not_emit_curriculum_warning(
    tmp_path,
) -> None:
    env = _FakeEnvironmentAdapter(None)
    runner = _fake_runner(env)
    checkpoint = tmp_path / "legacy_stateless.pt"
    torch.save(
        {
            "actor_state_dict": {"weight": torch.asarray([1.0])},
            "iter": 5,
            "infos": None,
        },
        checkpoint,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.load(str(checkpoint))

    assert caught == []


def test_rsl_rl_legacy_checkpoint_warns_and_leaves_environment_fresh(tmp_path) -> None:
    env = _FakeEnvironmentAdapter({"fresh": True})
    runner = _fake_runner(env)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "actor_state_dict": {"weight": torch.asarray([1.0])},
            "iter": 17,
            "infos": None,
            "unilab_logger_state": {"tot_time": 3.0, "tot_timesteps": 128},
        },
        checkpoint,
    )

    with pytest.warns(RuntimeWarning, match="predates environment curriculum persistence"):
        runner.load(str(checkpoint))

    assert runner.current_learning_iteration == 17
    assert env.loaded_states == []


def test_rsl_rl_wrapper_forwards_formal_environment_training_state_contract() -> None:
    concrete_env = _FakeEnvironmentAdapter({"level": 3})
    wrapper = object.__new__(RslRlVecEnvWrapper)
    wrapper.env = concrete_env

    assert wrapper.training_state_dict() == {"level": 3}
    wrapper.load_training_state_dict({"level": 2})
    assert concrete_env.loaded_states == [{"level": 2}]
