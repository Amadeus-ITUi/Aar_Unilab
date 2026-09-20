"""Gamepad mapping, release-to-stop and explicit fixed-command playback."""

import json
from queue import Empty
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from unilab.playback import InteractiveSession, joystick_command


@pytest.mark.parametrize(
    "axes, expected",
    [
        ([0, -1, 0, 0, -1, -1], [1, 0, 0]),
        ([-1, 0, 0, 0, -1, -1], [0, 1, 0]),
        ([0, 0, -1, 0, -1, -1], [0, 0, 1]),
        ([1, 1, 1, 0, -1, -1], [-1, -1, -1]),
        ([0, 0, 0, 1, 1, 1], [0, 0, 0]),
        ([0.1, -0.1, 0.1, -1], [0, 0, 0]),
        ([2, -2, 2, 0], [1, -1, -1]),
        ([0, float("nan"), 0, 0], [0, 0, 0]),
        ([0, 0], [0, 0, 0]),
        (None, [0, 0, 0]),
    ],
)
def test_stick_directions_deadzone_and_unused_axes(axes, expected):
    command = joystick_command(None if axes is None else np.array(axes))
    np.testing.assert_allclose(command, expected)
    assert command.dtype == np.float32


@pytest.fixture
def session(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    with InteractiveSession(
        model,
        data,
        telemetry_dir=tmp_path,
        render=False,
        plot=False,
        fixed_command=[0.8, 0.4, 0.2],
        gamepad_index=2,
        gamepad_deadzone=0.2,
        gamepad_scale=[0.6, 0.3, 1.5],
    ) as play:
        yield play


def test_standard_gamepad_polling_disconnect_and_reconnect(session, monkeypatch):
    import glfw

    session.render = True
    states = iter(
        [
            SimpleNamespace(axes=[-1, -1, -1, 0, -1, -1]),
            SimpleNamespace(axes=[0.1, -0.1, 0.1, 1, 1, 1]),
            None,
            SimpleNamespace(axes=[1, 1, 1, 0, -1, -1]),
        ]
    )

    def read(index):
        assert index == glfw.JOYSTICK_1 + 2
        return next(states)

    monkeypatch.setattr(glfw, "get_gamepad_state", read)
    np.testing.assert_allclose(session.gamepad(), [0.6, 0.3, 1.5])
    np.testing.assert_array_equal(session.gamepad(), [0, 0, 0])
    np.testing.assert_array_equal(session.gamepad(), [0, 0, 0])
    np.testing.assert_allclose(session.gamepad(), [-0.6, -0.3, -1.5])


@pytest.mark.parametrize("source", ["gamepad", "fixed"])
def test_selected_command_reaches_policy_and_telemetry(session, tmp_path, monkeypatch, source):
    session.command_source = source
    commands = iter([np.array([0.2, -0.1, 0.3]), np.zeros(3)])

    def gamepad():
        assert source == "gamepad", "fixed playback must not read the gamepad"
        return next(commands)

    monkeypatch.setattr(session, "gamepad", gamepad)
    seen = []

    def action(observation, command):
        seen.append(command.copy())
        return np.zeros(6)

    env = SimpleNamespace(
        model=session.model,
        reset=lambda: None,
        step=lambda action: (None, 0, False, {}),
    )
    session.run(2, action, env)
    expected = [[0.2, -0.1, 0.3], [0, 0, 0]] if source == "gamepad" else [[0.8, 0.4, 0.2]] * 2
    np.testing.assert_allclose(seen, expected)
    session._jsonl_stream.flush()
    rows = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    np.testing.assert_allclose(
        [[row[key] for key in ("command_x", "command_y", "command_yaw")] for row in rows],
        expected,
    )


def test_headless_gamepad_is_zero(session, monkeypatch):
    import glfw

    monkeypatch.setattr(
        glfw, "get_gamepad_state", lambda _: pytest.fail("headless playback polled GLFW")
    )
    np.testing.assert_array_equal(session.gamepad(), [0, 0, 0])


def test_gamepad_buttons_are_edge_triggered_and_disconnect_releases_them(session, monkeypatch):
    import glfw

    session.render = True

    def state(*pressed):
        buttons = [0] * 15
        for button in pressed:
            buttons[button] = 1
        return SimpleNamespace(axes=np.zeros(6), buttons=buttons)

    start = state(glfw.GAMEPAD_BUTTON_START)
    reset = state(glfw.GAMEPAD_BUTTON_RIGHT_BUMPER, glfw.GAMEPAD_BUTTON_Y)
    states = iter([start, start, state(), reset, reset, None, start])
    monkeypatch.setattr(glfw, "get_gamepad_state", lambda _: next(states))
    for _ in range(7):
        session.gamepad()
    assert [session._events.get_nowait() for _ in range(3)] == ["pause", "reset", "pause"]
    with pytest.raises(Empty):
        session._events.get_nowait()
