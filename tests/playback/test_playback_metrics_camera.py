"""Playback measurements, camera ownership and policy-rate pacing regressions."""

import csv
import json
from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from unilab.base.backend.mujoco.play_metrics import base_motion
from unilab.playback import InteractiveSession


@pytest.fixture
def robot():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.0025"/><worldbody><body pos="0 0 0.35">'
        '<freejoint/><inertial pos="0.4 0 0.1" mass="1" diaginertia="0.1 0.1 0.1"/>'
        '<geom type="sphere" size="0.05"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    data.qvel[:] = [0, 1, 0, 0, 0, 0.7]
    mujoco.mj_forward(model, data)
    return model, data


class Viewer:
    def __init__(self):
        self.cam = mujoco.MjvCamera()
        self.texts = None
        self.lookats = []

    def is_running(self):
        return True

    def lock(self):
        return nullcontext()

    def set_texts(self, texts):
        self.texts = texts

    def sync(self):
        self.lookats.append(self.cam.lookat.copy())


def test_measurements_use_body_axes_and_link_origin_not_offset_com(robot):
    model, data = robot
    # At yaw=90 degrees, world +Y is body +X. The rotating, offset COM
    # has a different linear velocity and height from the requested link origin.
    assert base_motion(model, data, 1) == pytest.approx(
        {
            "base_height": 0.35,
            "base_velocity_x": 1.0,
            "base_velocity_y": 0.0,
            "base_yaw_rate": 0.7,
        }
    )


@pytest.mark.parametrize(
    "camera_type", [mujoco.mjtCamera.mjCAMERA_FREE, mujoco.mjtCamera.mjCAMERA_TRACKING]
)
def test_tick_preserves_user_camera_and_records_live_measurements(robot, tmp_path, camera_type):
    model, data = robot
    viewer = Viewer()
    viewer.cam.type = camera_type
    viewer.cam.trackbodyid = 1
    viewer.cam.lookat[:] = [4, 5, 6]
    viewer.cam.distance = 2.5
    with InteractiveSession(
        model, data, telemetry_dir=tmp_path, render=False, plot=False
    ) as session:
        session.viewer = viewer
        session.tick(np.zeros(3))
        data.qpos[:3] = [0.5, 0.7, 0.42]
        data.qvel[:3] = [0, 2, 0]
        mujoco.mj_forward(model, data)
        session.tick(np.zeros(3))
        np.testing.assert_array_equal(viewer.lookats, [[4, 5, 6], [4, 5, 6]])
        assert viewer.cam.type == camera_type
        assert viewer.cam.trackbodyid == 1 and viewer.cam.distance == 2.5
        assert "2.000 m/s" in viewer.texts[3]
        assert "0.420 m" in viewer.texts[3]
    rows = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert rows[-1]["base_velocity_x"] == pytest.approx(2)
    assert rows[-1]["base_height"] == pytest.approx(0.42)
    with (tmp_path / "telemetry.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert float(csv_rows[-1]["base_velocity_x"]) == pytest.approx(rows[-1]["base_velocity_x"])


def test_run_frames_camera_once_and_paces_policy_steps(robot, tmp_path, monkeypatch):
    model, data = robot
    viewer = Viewer()
    sleeps = []
    monkeypatch.setattr("unilab.playback.interactive.time.monotonic", lambda: 0.0)
    monkeypatch.setattr("unilab.playback.interactive.time.sleep", sleeps.append)

    def step(action):
        data.qpos[2] += 0.01
        mujoco.mj_forward(model, data)
        return None, 0, False, {}

    env = SimpleNamespace(model=model, policy_hz=50, reset=lambda: None, step=step)
    with InteractiveSession(
        model, data, telemetry_dir=tmp_path, render=False, plot=False
    ) as session:
        session.viewer = viewer
        session.render = True
        monkeypatch.setattr(session, "gamepad", lambda: np.zeros(3))
        session.run(3, lambda observation, command: np.zeros(6), env)
    assert sleeps == pytest.approx([0.02] * 3)
    np.testing.assert_allclose(viewer.lookats, [[0, 0, 0.35]] * 3)


@pytest.mark.parametrize("steps, expected_steps", [(0, 0), (3, 3), (-1, 5)])
def test_run_step_limit_or_viewer_close(robot, tmp_path, monkeypatch, steps, expected_steps):
    model, data = robot
    viewer = Viewer()
    completed = 0

    def step(action):
        nonlocal completed
        completed += 1
        assert completed <= 5, "playback continued after the viewer closed"
        return None, 0, False, {}

    monkeypatch.setattr(viewer, "is_running", lambda: completed < 5)
    env = SimpleNamespace(model=model, policy_hz=50, reset=lambda: None, step=step)
    with InteractiveSession(
        model, data, telemetry_dir=tmp_path, render=False, plot=False
    ) as session:
        session.viewer = viewer
        session.run(steps, lambda observation, command: np.zeros(6), env)
    assert completed == expected_steps
    assert session._csv_stream.closed and session._jsonl_stream.closed


def test_unlimited_playback_interrupt_closes_telemetry(robot, tmp_path):
    model, data = robot

    def step(action):
        raise KeyboardInterrupt

    env = SimpleNamespace(model=model, policy_hz=50, reset=lambda: None, step=step)
    with pytest.raises(KeyboardInterrupt):
        with InteractiveSession(
            model, data, telemetry_dir=tmp_path, render=False, plot=False
        ) as session:
            session.run(-1, lambda observation, command: np.zeros(6), env)
    assert session._csv_stream.closed and session._jsonl_stream.closed
