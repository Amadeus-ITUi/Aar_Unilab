"""Run aliases must identify experiments and preserve source events/checkpoint selection."""

import json
import os
from datetime import datetime

import pytest
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from unilab.adapters.pe03_ppo import resolve_play_checkpoint
from unilab.algos.torch.pe03.run_logging import index_existing_runs, index_tensorboard_run
from unilab.envs.locomotion.pe03.config import load_config


@pytest.mark.parametrize("experiment", ["walking", "standing"])
def test_alias_names_experiment_and_original_start_time_without_changing_events(
    tmp_path, experiment
):
    cfg = load_config([f"+experiment={experiment}"])
    run = tmp_path / "2026-09-17_17-46-59_339378_mujoco"
    events = run / "tensorboard"
    with SummaryWriter(str(events)) as writer:
        writer.add_scalar("reward/lin_vel_z", -0.123, 500, walltime=1789638419.5)
    original = {path.name: path.read_bytes() for path in events.iterdir()}
    os.utime(run, (2_000_000_000, 2_000_000_000))
    link = index_tensorboard_run(run, cfg)
    assert link.name == f"PE03__pe03_flat-mujoco__{experiment}__2026-09-17_17-46-59_339378"
    assert link.resolve() == events
    assert not os.path.isabs(os.readlink(link))
    assert {path.name: path.read_bytes() for path in events.iterdir()} == original
    (point,) = EventAccumulator(str(link)).Reload().Scalars("reward/lin_vel_z")
    assert point.step == 500 and point.wall_time == 1789638419.5
    assert point.value == pytest.approx(-0.123)
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["experiment"] == experiment
    assert metadata["observation"] == cfg.observation
    assert index_tensorboard_run(run, cfg, started_at=datetime(2099, 1, 1)) == link


def test_legacy_index_excludes_validation_and_preserves_latest_checkpoint(tmp_path):
    cfg = load_config(["+experiment=walking"])
    del cfg.training.experiment_name
    names = ("2026-09-16_21-45-06_473367_mujoco", "2026-09-17_17-46-59_339378_mujoco")
    for name in (*names, "validation/v3"):
        run = tmp_path / name
        (run / "tensorboard").mkdir(parents=True)
        (run / "tensorboard/events.out.tfevents.fixture").write_bytes(b"unchanged")
        OmegaConf.save(cfg, run / "training_config.yaml")
        (run / "model_500.pt").write_bytes(b"unchanged checkpoint")
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    expected = resolve_play_checkpoint(-1, log_root=tmp_path)
    links = index_existing_runs(tmp_path)
    assert len(links) == 2
    assert all("__walking__" in link.name for link in links)
    assert all("validation" not in str(link.resolve()) for link in links)
    assert index_existing_runs(tmp_path) == links
    assert resolve_play_checkpoint(-1, log_root=tmp_path) == expected
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content


def test_direct_runner_start_is_stable_and_collision_cannot_replace_another_run(tmp_path):
    cfg = load_config(["training.experiment_name='turn / test'"])
    for folder in ("first", "second"):
        (tmp_path / folder / "tensorboard").mkdir(parents=True)
    first = index_tensorboard_run(tmp_path / "first", cfg, started_at=datetime(2026, 9, 17, 18))
    assert "__turn_test__" in first.name
    assert index_tensorboard_run(tmp_path / "first", cfg) == first
    with pytest.raises(FileExistsError, match="already belongs"):
        index_tensorboard_run(tmp_path / "second", cfg, started_at=datetime(2026, 9, 17, 18))
    assert first.resolve() == tmp_path / "first/tensorboard"
