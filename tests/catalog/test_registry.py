from pathlib import Path

import pytest

from unilab.catalog import catalog
from unilab.catalog.cli import build_legacy_command, parse_selectors

WE11 = [
    "robot=we11",
    "task=flat",
    "observation=we11_default",
    "policy=we11_mlp",
    "algorithm=rsl_rl_ppo",
    "simulator=mujoco",
]


def test_selectors_are_order_independent_and_preserve_hydra_overrides():
    values, passthrough = parse_selectors([*reversed(WE11), "training.num_envs=4"])
    assert values["robot"] == "we11"
    assert passthrough == ["training.num_envs=4"]


def test_we11_routes_to_existing_owner_config():
    command = build_legacy_command("play", WE11, Path("/repo"))
    assert command[1:] == [
        "/repo/scripts/train_rsl_rl.py",
        "task=dr002_joystick_flat_we11/mujoco",
        "training.play_only=true",
    ]


def test_legacy_we11_observation_adds_only_compatibility_overrides():
    legacy = [
        value.replace("observation=we11_default", "observation=we11_legacy_135")
        for value in WE11
    ]
    command = build_legacy_command("play", legacy, Path("/repo"))
    assert "+env.actor_observation_contract=we11_legacy_135" in command
    assert "algo.actor.history_term_dims=[3,3,4,6,6,2,3]" in command
    assert "training.play_only=true" in command


def test_robot_and_observation_must_match():
    with pytest.raises(ValueError, match="observation=.*does not support"):
        catalog.resolve(
            {
                "robot": "we11",
                "task": "flat",
                "observation": "pe01_legacy",
                "policy": "we11_mlp",
                "algorithm": "rsl_rl_ppo",
                "simulator": "mujoco",
            }
        )


def test_cross_robot_task_is_rejected():
    with pytest.raises(ValueError, match="does not support"):
        catalog.resolve(
            {
                "robot": "pe01",
                "task": "flat",
                "observation": "pe01_legacy",
                "policy": "pe01_encoder_mlp",
                "algorithm": "pe01_custom_ppo",
                "simulator": "mujoco",
            }
        )
