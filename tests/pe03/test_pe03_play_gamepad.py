"""PE03 playback routes live velocity commands into every checkpoint layout."""

import importlib.util
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.algos.torch.pe03 import PE03EncoderPolicy
from unilab.envs.locomotion.pe03.config import ROOT, load_config
from unilab.envs.locomotion.pe03.play_env import make_play_env
from unilab.playback import InteractiveSession


@pytest.mark.parametrize("experiment", [None, "walking", "gait_fixed"])
@pytest.mark.parametrize("source", ["gamepad", "fixed"])
def test_play_command_reaches_environment_and_policy(tmp_path, monkeypatch, experiment, source):
    spec = importlib.util.spec_from_file_location("train_pe03", ROOT / "scripts/train_pe03.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    overrides = [
        "mode=play",
        "checkpoint=unused.pt",
        "training.device=cpu",
        "training.mujoco_threads=1",
        "play.render=none",
        "play.plot=false",
        "play.steps=2",
        f"play.telemetry={tmp_path}",
        "play.command=[0.8,0.4,0.2]",
        *([f"+experiment={experiment}"] if experiment else []),
        "normalization.lin_vel=2.0",
        "normalization.ang_vel=0.25",
    ]
    if source == "fixed":
        overrides.append("play.command_source=fixed")
    config = load_config(overrides)
    assert config.play.command_source == source
    # Older checkpoints do not contain the new input-source settings.
    checkpoint_config = OmegaConf.merge(config)
    del checkpoint_config.play.command_source
    del checkpoint_config.play.gamepad
    policy = PE03EncoderPolicy(checkpoint_config)
    monkeypatch.setattr(module, "load_policy", lambda *args, **kwargs: policy)
    monkeypatch.setattr(module.sys, "argv", ["train_pe03.py", *overrides])
    live = iter([np.array([0.2, -0.1, 0.3]), np.zeros(3)])

    def gamepad(self):
        assert source == "gamepad"
        return next(live)

    monkeypatch.setattr(InteractiveSession, "gamepad", gamepad)
    environments = []
    original_make_env = module.make_play_env

    def make_env(play_config):
        env = original_make_env(play_config)
        environments.append(env)
        return env

    monkeypatch.setattr(module, "make_play_env", make_env)
    seen = []
    original_action = policy.action_mean

    def action_mean(history, frame, command):
        seen.append(command.cpu().numpy()[0].copy())
        np.testing.assert_allclose(
            command.cpu().numpy()[0, :3],
            environments[0].vector.commands[0] * [2, 2, 0.25],
        )
        return original_action(history, frame, command)

    monkeypatch.setattr(policy, "action_mean", action_mean)
    try:
        assert module.main() == 0
        expected = (
            np.array([[0.2, -0.1, 0.3], [0, 0, 0]])
            if source == "gamepad"
            else np.array([[0.8, 0.4, 0.2]] * 2)
        )
        scale = np.array([config.normalization.lin_vel] * 2 + [config.normalization.ang_vel])
        np.testing.assert_allclose(np.array(seen)[:, :3], expected * scale)
        np.testing.assert_allclose(environments[0].vector.commands[0], expected[-1])
        if experiment == "gait_fixed":
            assert np.array(seen).shape == (2, 6)
            np.testing.assert_allclose(seen[0][3:], seen[1][3:])
        rows = [
            json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
        ]
        np.testing.assert_allclose(
            [[row[key] for key in ("command_x", "command_y", "command_yaw")] for row in rows],
            expected,
        )
    finally:
        for env in environments:
            env.close()


@pytest.mark.parametrize("experiment", ["walking", "gait_fixed"])
def test_pause_and_reset_restore_real_backend_history_and_control(
    tmp_path, monkeypatch, experiment
):
    config = load_config(
        [
            f"+experiment={experiment}",
            "training.mujoco_threads=1",
            "play.render=none",
            "play.plot=false",
            "play.delay_ms=25" if experiment == "walking" else "play.delay_ms=0",
        ]
    )
    env = make_play_env(config)
    monkeypatch.setattr("unilab.playback.interactive.time.sleep", lambda _: None)
    snapshots = []
    observations = []

    def action(observation, command):
        observations.append(observation.actor.copy())
        env.set_command(command)
        return np.full(6, 0.1)

    try:
        with InteractiveSession(
            env.model, env.data, telemetry_dir=tmp_path, render=False, plot=False
        ) as session:
            monkeypatch.setattr(session, "gamepad", lambda: np.array([0.2, -0.1, 0.3]))
            original_tick = session.tick

            def tick(command, *, record=True):
                snapshots.append((env.vector.backend.state.copy(), env.vector.history.copy()))
                index = len(snapshots)
                if index == 1:
                    session.key_callback(32)
                elif index == 2:
                    np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
                    np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
                    env.data.xfrc_applied[:] = 10
                    session.key_callback(ord("R"))
                elif index == 3:
                    assert env.data.time == 0 and session.paused
                    np.testing.assert_allclose(env.vector.backend.qpos[0], env.vector.home)
                    assert not env.vector.backend.qvel.any()
                    assert not env.vector.backend.delay_buffer.any()
                    assert not env.vector.actions.any()
                    assert not env.data.xfrc_applied.any()
                    assert not env.vector.commands.any()
                    np.testing.assert_array_equal(
                        env.vector.history,
                        np.repeat(env.vector.history[:, :1], env.history_length, axis=1),
                    )
                    session.key_callback(ord("N"))
                elif index == 4:
                    assert session.paused and env.data.time == pytest.approx(1 / env.policy_hz)
                    session.key_callback(ord("Q"))
                else:
                    pytest.fail("unexpected playback iteration")
                return original_tick(command, record=record)

            monkeypatch.setattr(session, "tick", tick)
            session.run(-1, action, env)
        assert len(observations) == 2
        np.testing.assert_allclose(observations[0], observations[1])
    finally:
        env.close()
