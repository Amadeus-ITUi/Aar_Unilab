"""Playback controls operate on the live environment at policy boundaries."""

from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import pytest

from unilab.playback import InteractiveSession


@pytest.fixture
def play(tmp_path, monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 0.3"><freejoint/>'
        '<geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    viewer = SimpleNamespace(
        cam=mujoco.MjvCamera(),
        perturb=mujoco.MjvPerturb(),
        lock=lambda: nullcontext(),
        is_running=lambda: True,
        sync=lambda: None,
        set_texts=lambda texts: None,
    )
    callbacks = []

    def launch(model, data, *, key_callback):
        callbacks.append(key_callback)
        return nullcontext(viewer)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr("unilab.playback.interactive.time.monotonic", lambda: 0)
    sleeps = []
    monkeypatch.setattr("unilab.playback.interactive.time.sleep", sleeps.append)
    resets, steps, observations = [], [], []

    def reset():
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        resets.append(len(steps))
        return len(resets), 0

    def step(action):
        steps.append(data.time)
        data.time += 0.02
        return (len(resets), len(steps)), 0, False, {}

    env = SimpleNamespace(model=model, policy_hz=50, reset=reset, step=step)

    def action(observation, command):
        observations.append(observation)
        return np.zeros(6)

    with InteractiveSession(
        model, data, telemetry_dir=tmp_path, render=True, plot=False
    ) as session:
        monkeypatch.setattr(session, "gamepad", lambda: np.zeros(3))
        yield SimpleNamespace(
            session=session,
            viewer=viewer,
            data=data,
            env=env,
            action=action,
            key=callbacks[0],
            steps=steps,
            resets=resets,
            observations=observations,
            sleeps=sleeps,
            telemetry=tmp_path / "telemetry.jsonl",
        )


def test_pause_step_reset_resume_and_quit_keep_ui_alive(play):
    play.session.paused = True
    frames = []

    def sync():
        frames.append((len(play.steps), float(play.data.time), play.session.paused))
        index = len(frames) - 1
        if index == 0:
            play.key(ord("N"))
        elif index == 2:
            play.key(ord("R"))
        elif index == 3:
            play.key(32)
        elif index == 4:
            play.key(ord("P"))
        elif index == 5:
            play.key(ord("Q"))
        assert index < 6, "paused viewer stopped processing keys"

    play.viewer.sync = sync
    play.session.run(3, play.action, play.env)
    assert frames == [
        (0, 0, True),
        (1, 0.02, True),
        (1, 0.02, True),
        (1, 0, True),
        (2, 0.02, False),
        (2, 0.02, True),
    ]
    assert play.resets == [0, 1]
    assert play.observations == [(1, 0), (2, 0)]
    assert play.sleeps == pytest.approx([0.02] * 6)
    play.session._jsonl_stream.flush()
    assert len(play.telemetry.read_text().splitlines()) == 2


def test_running_reset_preserves_camera_and_policy_step_budget(play):
    frames = []

    def sync():
        frames.append((len(play.steps), float(play.data.time)))
        if len(frames) == 1:
            play.viewer.cam.lookat[:] = [4, 5, 6]
            play.viewer.perturb.active = 1
            play.viewer.perturb.active2 = 1
            play.key(259)  # Backspace
        else:
            np.testing.assert_array_equal(play.viewer.cam.lookat, [4, 5, 6])
            assert not play.viewer.perturb.active and not play.viewer.perturb.active2
        assert len(frames) <= 4

    play.viewer.sync = sync
    play.session.run(3, play.action, play.env)
    assert frames == [(1, 0.02), (1, 0), (2, 0.02), (3, 0.04)]
    assert play.resets == [0, 1]
    assert play.observations[:2] == [(1, 0), (2, 0)]
    assert not play.session.paused


def test_recenter_and_window_close_while_paused(play):
    play.session.paused = True
    frames = []

    def sync():
        frames.append(play.viewer.cam.lookat.copy())
        if len(frames) == 1:
            play.viewer.cam.lookat[:] = [8, 9, 10]
            play.key(ord("C"))
        else:
            play.viewer.is_running = lambda: False

    play.viewer.sync = sync
    play.session.run(-1, play.action, play.env)
    assert len(frames) == 2
    np.testing.assert_allclose(frames[1], play.data.xpos[1])
    assert not play.steps and play.resets == [0]


def test_key_callback_only_queues_reset_and_single_step_needs_pause(play):
    play.key(ord("R"))
    play.key(ord("N"))
    assert not play.resets and not play.steps
    play.session.run(1, play.action, play.env)
    assert play.resets == [0, 0]
    assert len(play.steps) == 1


def test_paused_overlay_shows_commands_and_controls(play):
    texts = []
    play.viewer.set_texts = texts.append
    play.session.paused = True
    play.session.tick(np.array([0.3, -0.2, 0.1]), record=False)
    assert "Space / P" in texts[0][2] and "R / Backspace" in texts[0][2]
    assert "PAUSED (gamepad)" in texts[0][3]
    assert "+0.30 / -0.20 / +0.10" in texts[0][3]
    play.session._jsonl_stream.flush()
    assert not play.telemetry.read_text()


def test_headless_start_paused_is_rejected_instead_of_waiting_without_a_viewer(tmp_path):
    with pytest.raises(ValueError, match="play.render=interactive"):
        InteractiveSession(
            None, None, telemetry_dir=tmp_path, render=False, plot=False, paused=True
        )
