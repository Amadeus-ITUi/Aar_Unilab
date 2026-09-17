"""Launch presets must preserve owner configs and set process state before Python starts."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from unilab.catalog.launch import DEFAULTS, ROOT, build_launch, main
from unilab.envs.locomotion.pe01.config import load_config as load_pe01
from unilab.envs.locomotion.pe02.config import load_config as load_pe02


@pytest.mark.parametrize("profile", ["we11_getup", "pe01", "pe02_walking"])
def test_profiles_compose_the_existing_owner_configs(profile):
    command, environment, displayed = build_launch(DEFAULTS, profile, [], environment={})
    if profile == "we11_getup":
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/ppo")):
            cfg = compose(config_name="config", overrides=command[2:])
        assert cfg.training.task_name == "DR002JoystickGetupWE11"
        assert cfg.env.reset_pose.mode == "getup"
        assert cfg.training.no_play
    else:
        cfg = (load_pe01 if profile == "pe01" else load_pe02)(command[2:])
        assert cfg.training.mujoco_threads == 32
        assert cfg.training.cpu_threads == 1
        if profile == "pe02_walking":
            assert not cfg.domain_rand.enabled and not cfg.noise.enabled
            assert list(cfg.env.joint_reset_range) == [0, 0]
            assert list(cfg.play.gait) == [2, 0.5, 0.5, 0.06]
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.num_steps_per_env == 24
    assert cfg.algo.max_iterations == 1000
    assert cfg.algo.save_interval == 100
    assert cfg.training.logger == "tensorboard"
    assert environment == displayed
    assert environment["UNILAB_MUJOCO_NTHREADS"] == "32"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_shell_and_cli_overrides_take_precedence_without_mutating_the_parent():
    incoming = {
        "CUDA_VISIBLE_DEVICES": "",
        "UNILAB_MUJOCO_NTHREADS": "24",
        "OMP_NUM_THREADS": "2",
        "PYTHONPATH": "/ros/python3.10",
    }
    before = incoming.copy()
    command, environment, _ = build_launch(DEFAULTS, "pe02_walking", [], environment=incoming)
    cfg = load_pe02(command[2:])
    assert cfg.training.mujoco_threads == 24 and cfg.training.cpu_threads == 2
    assert environment["CUDA_VISIBLE_DEVICES"] == ""  # An explicit CPU-only run stays CPU-only.
    assert "PYTHONPATH" not in environment
    assert incoming == before

    command, _, _ = build_launch(
        DEFAULTS,
        "pe02_walking",
        [
            "algo.num_envs=8",
            "algo.max_iterations=5",
            "training.mujoco_threads=7",
            "+experiment=standing",
        ],
        environment=incoming,
    )
    cfg = load_pe02(command[2:])
    assert cfg.algo.num_envs == 8 and cfg.algo.max_iterations == 5
    assert cfg.training.mujoco_threads == 7
    assert list(cfg.play.gait) == [0, 0, 0.5, 0]


def test_preview_never_executes_and_does_not_print_unrelated_environment(monkeypatch, capsys):
    monkeypatch.setenv("UNRELATED_PRIVATE_VALUE", "do-not-print-this")

    def no_exec(*args):
        raise AssertionError("preview attempted to start a training process")

    monkeypatch.setattr(os, "execve", no_exec)
    assert main(["pe01", "--dry-run", "algo.num_envs=8"]) == 0
    output = capsys.readouterr().out
    assert "train_pe01.py" in output and "algo.num_envs=8" in output
    assert "do-not-print-this" not in output
    assert "UNRELATED_PRIVATE_VALUE" not in output


def test_child_receives_clean_environment_and_literal_arguments(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_pe02.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'cuda': os.environ['CUDA_VISIBLE_DEVICES'], "
        "'threads': os.environ['UNILAB_MUJOCO_NTHREADS'], "
        "'pythonpath': os.environ.get('PYTHONPATH'), 'argv': sys.argv[1:]}))\n"
    )
    literal = "training.log_root='a path with spaces and $(not-a-shell-command)'"
    command, environment, _ = build_launch(
        DEFAULTS,
        "pe02_walking",
        [literal],
        root=tmp_path,
        environment={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "UNILAB_MUJOCO_NTHREADS": "16",
            "PYTHONPATH": "/ros/python3.10",
        },
    )
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
    child = json.loads(result.stdout)
    assert child["cuda"] == "" and child["threads"] == "16"
    assert child["pythonpath"] is None and literal in child["argv"]


def test_shell_wrapper_ignores_ros_pythonpath_before_loading_the_launcher(tmp_path):
    marker = tmp_path / "unexpected-pythonpath-import"
    (tmp_path / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    result = subprocess.run(
        ["bash", str(ROOT / "tools/train.sh"), "pe02_walking", "--dry-run"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "train_pe02.py" in result.stdout
    assert not marker.exists()


def test_invalid_profile_or_thread_setting_fails_before_launch():
    with pytest.raises(ValueError, match="unknown profile"):
        build_launch(DEFAULTS, "missing", [], environment={})
    with pytest.raises(ValueError, match="UNILAB_MUJOCO_NTHREADS"):
        build_launch(DEFAULTS, "pe01", [], environment={"UNILAB_MUJOCO_NTHREADS": "oops"})
