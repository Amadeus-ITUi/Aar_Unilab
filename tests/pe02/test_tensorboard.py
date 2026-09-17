"""Check TensorBoard event values, axes and lossless historical regrouping."""

import json
import zipfile

import pytest
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tools.rebuild_pe02_tensorboard import rebuild
from torch.utils.tensorboard import SummaryWriter

from unilab.algos.torch.pe02.runner import PE02Runner
from unilab.algos.torch.pe02.tensorboard import tensorboard_metrics
from unilab.envs.locomotion.pe02.config import load_config


def test_metric_groups_preserve_values_and_convert_episode_seconds_to_steps():
    metrics = {
        "policy_loss": -0.2,
        "value_loss": 1.2,
        "encoder_loss": 0.3,
        "action_std": 0.9,
        "samples_per_second": 1000.0,
        "episode/return": 8.0,
        "episode/seconds": 2.0,
        "reward/keep_balance": 0.02,
        "evaluation/failure_rate": 0.25,
    }
    actual = tensorboard_metrics(metrics, policy_dt=0.02)
    assert actual == {
        "Loss/surrogate": -0.2,
        "Loss/value": 1.2,
        "Loss/encoder": 0.3,
        "Policy/mean_std": 0.9,
        "Perf/total_fps": 1000.0,
        "Train/mean_reward": 8.0,
        "Train/mean_episode_time": 2.0,
        "Train/mean_episode_length": 100.0,
        "reward/keep_balance": 0.02,
        "Eval/failure_rate": 0.25,
    }
    assert "Train/mean_episode_length" not in tensorboard_metrics({}, policy_dt=0.02)


def test_training_writes_grouped_events_and_keeps_original_jsonl_and_policy(tmp_path):
    cfg = load_config(
        [
            "algo.num_envs=2",
            "algo.num_steps_per_env=4",
            "algo.max_iterations=1",
            "algo.num_learning_epochs=1",
            "algo.num_mini_batches=1",
            "env.episode_length_s=0.02",
            "training.device=cpu",
            "training.mujoco_threads=1",
            "training.evaluation_interval=1",
            "training.evaluation_episodes=2",
            "training.export=false",
        ]
    )
    states = []
    for name, logger in (("logged", "tensorboard"), ("unlogged", "none")):
        cfg.training.logger = logger
        runner = PE02Runner(cfg, tmp_path / name)
        try:
            runner.learn()
            states.append({key: value.clone() for key, value in runner.policy.state_dict().items()})
        finally:
            runner.close()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    row = json.loads((tmp_path / "logged/metrics.jsonl").read_text())
    assert "policy_loss" in row and "Loss/surrogate" not in row
    events = EventAccumulator(str(tmp_path / "logged/tensorboard")).Reload()
    expected = tensorboard_metrics(row, policy_dt=0.02)
    assert set(events.Tags()["scalars"]) == set(expected)
    for tag, value in expected.items():
        (point,) = events.Scalars(tag)
        assert point.step == 1
        assert point.value == pytest.approx(value)


def test_rebuild_archives_old_events_and_preserves_iterations_and_wall_times(tmp_path):
    run = tmp_path / "logs/run"
    run.mkdir(parents=True)
    cfg = load_config()
    OmegaConf.save(cfg, run / "training_config.yaml")
    rows = [
        {"iteration": i, "policy_loss": -i / 10, "episode/seconds": i, "reward/keep_balance": 0.02}
        for i in (1, 2)
    ]
    original_metrics = "\n".join(json.dumps(row) for row in rows) + "\n"
    (run / "metrics.jsonl").write_text(original_metrics)
    with SummaryWriter(str(run / "tensorboard")) as writer:
        for row in rows:
            for index, (key, value) in enumerate(row.items()):
                writer.add_scalar(
                    key, value, row["iteration"], walltime=1000 + row["iteration"] + index * 0.01
                )
    original = {
        str(path.relative_to(run)): path.read_bytes() for path in (run / "tensorboard").iterdir()
    }
    backup, count = rebuild(run)
    assert count == 2
    with zipfile.ZipFile(backup) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == original
    assert (run / "metrics.jsonl").read_text() == original_metrics
    events = EventAccumulator(str(run / "tensorboard")).Reload()
    assert "policy_loss" not in events.Tags()["scalars"]
    points = events.Scalars("Loss/surrogate")
    assert [point.step for point in points] == [1, 2]
    assert [point.wall_time for point in points] == [1001.01, 1002.01]
    assert [point.value for point in points] == pytest.approx([-0.1, -0.2])
    assert [point.value for point in events.Scalars("Train/mean_episode_length")] == [50, 100]
