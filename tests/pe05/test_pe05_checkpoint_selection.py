"""Playback selects training start time and numeric iteration, not filesystem mtime."""

import os

import pytest

from unilab.envs.locomotion.pe05.config import load_config
from unilab.envs.locomotion.pe05.contracts import resolve_checkpoint


@pytest.mark.parametrize("selector", [-1, "-1"])
def test_run_then_iteration_ignores_mtime_and_nontraining_files(tmp_path, selector):
    old = tmp_path / "2026-09-27_23-59-59_mujoco"
    latest = tmp_path / "2026-09-28_00-00-00_000001_mujoco"
    old.mkdir()
    latest.mkdir()
    (old / "model_20000.pt").touch()
    for name in ("model_9.pt", "model_100.pt", "model_99.pt", "model_200.tmp", "model_latest.pt"):
        (latest / name).touch()
    (latest / "model_999.pt").mkdir()
    unrelated = tmp_path / "validation"
    unrelated.mkdir()
    (unrelated / "model_30000.pt").touch()
    os.utime(old / "model_20000.pt", (2_000_000_000, 2_000_000_000))
    os.utime(latest / "model_9.pt", (2_000_000_000, 2_000_000_000))
    assert resolve_checkpoint(selector, tmp_path) == latest / "model_100.pt"


def test_explicit_path_does_not_search_roots(tmp_path):
    chosen = tmp_path / "model_5.pt"
    assert resolve_checkpoint(chosen, tmp_path / "absent") == chosen


def test_errors_do_not_silently_select_an_older_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_checkpoint(-1, tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="No timestamped"):
        resolve_checkpoint(-1, tmp_path)
    old = tmp_path / "2026-09-28_00-00-00_mujoco"
    latest = tmp_path / "2026-09-28_00-00-00_000001_mujoco"
    old.mkdir()
    latest.mkdir()
    (old / "model_100.pt").touch()
    (latest / "model_200.tmp").touch()
    with pytest.raises(FileNotFoundError, match="Latest PE05 run has no saved"):
        resolve_checkpoint(-1, tmp_path)


def test_play_defaults_match_pe03_ui_without_changing_pe05_gait():
    cfg = load_config(["mode=play"])
    assert cfg.checkpoint == -1
    assert cfg.play.steps == -1 and cfg.play.render == "interactive"
    assert cfg.play.command_source == "gamepad"
    assert not cfg.play.paused and not cfg.play.plot
    assert list(cfg.play.gait) == [2, 0.5, 0.5, 0.03]


def test_fixed_profile_alias_has_identical_training_and_play_config():
    from unilab.catalog.launch import DEFAULTS, build_launch
    from unilab.envs.locomotion.pe05.contracts import resume_contract

    configs = []
    for profile in ("pe05", "pe05_gait_fixed"):
        command, _, _ = build_launch(DEFAULTS, profile, [], environment={})
        config = load_config(command[2:])
        assert config.training.log_root == "logs/pe05_gait_fixed"
        assert config.training.experiment_name == "gait_fixed"
        assert config.play.telemetry == "logs/play/pe05/pe05_flat/fixed"
        configs.append(config)
    assert resume_contract(configs[0]) == resume_contract(configs[1])
