"""PE03-style presentation preserves PE05 metrics and training behavior."""

import json

import pytest
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tools.sync_pe05_tensorboard import EventMirror
from torch.utils.tensorboard import SummaryWriter

from unilab.algos.torch.pe05.runner import PE05Runner
from unilab.algos.torch.pe05.tensorboard import tensorboard_metrics
from unilab.envs.locomotion.pe05.config import load_config


def test_groups_keep_pe05_diagnostics_and_reward_units():
    metrics = {
        "policy_loss": -0.2,
        "action_std": 0.8,
        "samples_per_second": 1000.0,
        "episode/seconds": 2.0,
        "reward/foot_clearance": -0.02,
        "raw_reward/foot_clearance": 2.0,
        "command_curriculum/vx_max": 0.3,
        "contact/L_calf_Link/ground_fraction": 0.01,
        "evaluation/forward/success_rate": 0.5,
    }
    assert tensorboard_metrics(metrics, policy_dt=0.02) == {
        "Loss/surrogate": -0.2,
        "Policy/mean_std": 0.8,
        "Perf/total_fps": 1000.0,
        "Train/mean_episode_time": 2.0,
        "Train/mean_episode_length": 100.0,
        "reward/foot_clearance": -0.02,
        "raw_reward/foot_clearance": 2.0,
        "command_curriculum/vx_max": 0.3,
        "contact/L_calf_Link/ground_fraction": 0.01,
        "Eval/forward/success_rate": 0.5,
    }


def test_runner_grouping_does_not_change_jsonl_or_policy(tmp_path):
    cfg = load_config(
        [
            "algo.num_envs=2",
            "algo.num_steps_per_env=4",
            "algo.max_iterations=1",
            "algo.num_learning_epochs=1",
            "algo.num_mini_batches=1",
            "env.episode_length_s=0.04",
            "training.device=cpu",
            "training.mujoco_threads=1",
            "training.evaluation_interval=0",
            "training.export=false",
        ]
    )
    states = []
    for name, logger in (("logged", "tensorboard"), ("unlogged", "none")):
        cfg.training.logger = logger
        runner = PE05Runner(cfg, tmp_path / name)
        try:
            runner.learn()
            states.append({k: v.clone() for k, v in runner.policy.state_dict().items()})
        finally:
            runner.close()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    run = tmp_path / "logged"
    row = json.loads((run / "metrics.jsonl").read_text())
    assert "policy_loss" in row and "Loss/surrogate" not in row
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["experiment"] == "gait_fixed"
    assert metadata["run_name"].startswith("PE05__pe05_flat-mujoco__gait_fixed__")
    link = tmp_path / "tensorboard_runs" / metadata["run_name"]
    assert link.resolve() == run / "tensorboard"
    events = EventAccumulator(str(link)).Reload()
    expected = tensorboard_metrics(row, policy_dt=0.02)
    assert set(events.Tags()["scalars"]) == set(expected)
    for key, value in expected.items():
        (point,) = events.Scalars(key)
        assert point.step == 1 and point.value == pytest.approx(value)


def test_live_mirror_preserves_sources_timestamps_and_handles_appends(tmp_path):
    run = tmp_path / "2026-09-28_13-38-25_381588_mujoco"
    run.mkdir()
    OmegaConf.save(load_config(), run / "training_config.yaml")
    (run / "model_100.pt").write_bytes(b"unchanged checkpoint")
    (run / "metrics.jsonl").write_text('{"iteration": 100}\n')
    sources = {p: p.read_bytes() for p in run.iterdir()}
    with SummaryWriter(str(run / "tensorboard")) as writer:
        writer.add_scalar("policy_loss", -0.1, 100, walltime=1000)
        writer.add_scalar("episode/seconds", 2.0, 100, walltime=1001)
        writer.flush()
        original = {p: p.read_bytes() for p in (run / "tensorboard").iterdir()}
        mirror = EventMirror(run)
        try:
            assert mirror.sync() == 3
            assert mirror.sync() == 0
            assert all(p.read_bytes() == data for p, data in original.items())
            writer.add_scalar("policy_loss", -0.2, 101, walltime=1002)
            writer.flush()
            assert mirror.sync() == 1
        finally:
            mirror.close()
    assert all(p.read_bytes() == data for p, data in sources.items())
    events = EventAccumulator(str(mirror.link)).Reload()
    assert "policy_loss" not in events.Tags()["scalars"]
    assert [p.step for p in events.Scalars("Loss/surrogate")] == [100, 101]
    assert [p.wall_time for p in events.Scalars("Loss/surrogate")] == [1000, 1002]
    assert events.Scalars("Train/mean_episode_length")[0].value == 100
    # A restarted mirror replays into the same named run without duplicate visible steps.
    second = EventMirror(run)
    try:
        second.sync()
        assert second.link == mirror.link
    finally:
        second.close()
    events.Reload()
    assert [p.step for p in events.Scalars("Loss/surrogate")] == [100, 101]


def test_index_existing_runs_preserves_old_identity_and_artifacts(tmp_path):
    from unilab.algos.torch.pe05.run_logging import index_existing_runs

    run = tmp_path / "2026-09-28_13-00-00_000001_mujoco"
    run.mkdir()
    cfg = load_config()
    del cfg.training.experiment_name  # Saved before explicit experiment naming existed.
    OmegaConf.save(cfg, run / "training_config.yaml")
    (run / "model_100.pt").write_bytes(b"unchanged-model")
    with SummaryWriter(str(run / "tensorboard")) as writer:
        writer.add_scalar("Loss/surrogate", -0.2, 100)
    originals = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    (tmp_path / "validation").mkdir()
    (tmp_path / "2026-09-29_13-00-00_mujoco").mkdir()  # No events yet.
    links = index_existing_runs(tmp_path)
    assert len(links) == 1
    assert "__velocity_bins__" in links[0].name
    assert links[0].resolve() == run / "tensorboard"
    assert index_existing_runs(tmp_path) == links
    assert all(p.read_bytes() == value for p, value in originals.items())
