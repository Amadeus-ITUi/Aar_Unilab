"""Latest PE03 playback must select a run before selecting a model iteration."""

import os

import pytest

from unilab.adapters.pe03_ppo import resolve_play_checkpoint
from unilab.envs.locomotion.pe03.config import load_config


@pytest.mark.parametrize("selector", [-1, "-1"])
def test_latest_run_then_numeric_iteration_ignores_mtime_and_temporary_files(tmp_path, selector):
    older = tmp_path / "2026-09-16_21-45-06_473367_mujoco"
    latest = tmp_path / "2026-09-17_16-04-08_275498_mujoco"
    older.mkdir()
    latest.mkdir()
    (older / "model_10000.pt").touch()
    for name in ("model_9.pt", "model_100.pt", "model_99.pt", "model_200.tmp", "model_latest.pt"):
        (latest / name).touch()
    (latest / "model_999.pt").mkdir()
    unrelated = tmp_path / "validation"
    unrelated.mkdir()
    (unrelated / "model_20000.pt").touch()
    os.utime(older, (2_000_000_000, 2_000_000_000))
    os.utime(latest, (1_000_000_000, 1_000_000_000))
    assert resolve_play_checkpoint(selector, log_root=tmp_path) == latest / "model_100.pt"


def test_timestamp_sort_accepts_runs_without_microseconds(tmp_path):
    for name in ("2026-09-17_16-04-08_mujoco", "2026-09-17_16-04-08_000001_mujoco"):
        run = tmp_path / name
        run.mkdir()
        (run / "model_1.pt").touch()
    assert resolve_play_checkpoint(-1, log_root=tmp_path).parent.name.endswith("000001_mujoco")


def test_explicit_path_does_not_require_a_log_root(tmp_path):
    checkpoint = tmp_path / "chosen.pt"
    assert resolve_play_checkpoint(checkpoint, log_root=tmp_path / "absent") == checkpoint


def test_missing_root_or_run_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_play_checkpoint(-1, log_root=tmp_path / "absent")
    (tmp_path / "tensorboard").mkdir()
    with pytest.raises(FileNotFoundError, match="No timestamped"):
        resolve_play_checkpoint(-1, log_root=tmp_path)


def test_newest_run_without_checkpoint_does_not_fall_back(tmp_path):
    older = tmp_path / "2026-09-16_21-45-06_473367_mujoco"
    latest = tmp_path / "2026-09-17_16-04-08_275498_mujoco"
    older.mkdir()
    latest.mkdir()
    (older / "model_100.pt").touch()
    (latest / "model_200.tmp").touch()
    with pytest.raises(FileNotFoundError, match="Latest PE03 run has no saved"):
        resolve_play_checkpoint(-1, log_root=tmp_path)


def test_walking_checkpoint_selector_composes_as_integer():
    cfg = load_config(["+experiment=walking", "mode=play", "checkpoint=-1"])
    assert cfg.checkpoint == -1
    assert cfg.training.log_root == "logs/pe03_walking"


def test_walking_play_defaults_to_latest_checkpoint_and_can_start_paused():
    cfg = load_config(["+experiment=walking", "mode=play", "play.paused=true"])
    assert cfg.checkpoint == -1
    assert cfg.play.paused
