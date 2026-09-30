"""Numerical checks for the isolated PE identification replay, not live hardware."""

import ctypes
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pe_fit", ROOT / "scripts/identification/fit_pe_sweeps.py"
)
pe_fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe_fit)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pe_native")
    dll = pe_fit.native_library(folder)
    bodies = "".join(
        f'<body pos="{i} 0 0"><joint name="j{i}" type="hinge" axis="0 1 0"/>'
        '<inertial pos="0 0 -0.1" mass="1" diaginertia="0.01 0.01 0.01"/></body>'
        for i in range(6)
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002" integrator="implicitfast" gravity="0 0 0"/>'
        f"<worldbody>{bodies}</worldbody></mujoco>"
    )
    filename = folder / "model.mjb"
    mujoco.mj_saveModel(model, str(filename), None)
    return dll, filename, model


@pytest.mark.parametrize("dt", [0.001, 0.002])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_native_matches_explicit_python_zoh_scaled_pd(fixture, dt, legacy, independent):
    dll, filename, original = fixture
    model = mujoco.MjModel.from_binary_path(str(filename))
    par = np.array([0.01] * 3 + [0.1] * 3 + [0.02] * 3 + [0.004] * 3)
    model.dof_armature[:] = par[:3].tolist() * 2
    model.dof_damping[:] = par[3:6].tolist() * 2
    model.dof_frictionloss[:] = par[6:9].tolist() * 2
    delays = np.repeat(par[9], 6)
    if independent:
        # Deliberately asymmetric values catch accidental left/right sharing.
        par = np.r_[
            np.linspace(0.006, 0.025, 6),
            np.linspace(0.01, 0.25, 6),
            np.linspace(0.02, 0.3, 6),
            np.arange(6) * 0.002,
        ]
        model.dof_armature[:], model.dof_damping[:], model.dof_frictionloss[:] = np.split(
            par[:18], 3
        )
        delays = par[18:]
    model.opt.timestep = dt
    q0, dq0 = np.full(6, 0.2), np.full(6, 5.0)
    times = np.array([0.0, 0.01, 0.02])
    command = np.zeros((3, 30))
    command[:, :6] = np.array([1.0, -0.2, 0.6])[:, None]
    command[:, 12:18] = 100
    command[:, 18:24] = 5
    samples = np.arange(0, 0.052, 0.002)
    actual = np.zeros((len(samples), 25))
    handle = dll.pe_create(str(filename).encode())
    assert handle
    try:
        run = dll.pe_run_independent if independent else dll.pe_run
        code = run(
            handle,
            par,
            dt,
            0.002,
            len(times),
            times,
            command,
            q0,
            dq0,
            len(samples),
            samples,
            actual,
            int(legacy),
            1,
        )
        assert code == 0
    finally:
        dll.pe_destroy(handle)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q0, dq0
    mujoco.mj_forward(model, data)
    limit = np.array([5.5, 5.5, 14] * 2)
    expected = []
    idx = np.zeros(6, dtype=int)
    scale = np.ones(6)
    tau = np.zeros(6)
    for step in range(round(samples[-1] / dt) + 1):
        if step % round(0.002 / dt) == 0:
            expected.append(np.r_[data.qpos, data.qvel, tau, scale, data.ncon])
            for j in range(6):
                while idx[j] + 1 < len(times) and times[idx[j] + 1] <= data.time - delays[j] + 1e-9:
                    idx[j] += 1
            selected = command[idx]
            joints = np.arange(6)
            p = selected[joints, 12 + joints] * (selected[joints, joints] - data.qpos)
            d = selected[joints, 18 + joints] * (selected[joints, 6 + joints] - data.qvel)
            scale = np.minimum(1, limit / np.maximum(abs(p) + abs(d), 1e-30))
        p = selected[joints, 12 + joints] * (selected[joints, joints] - data.qpos)
        d = selected[joints, 18 + joints] * (selected[joints, 6 + joints] - data.qvel)
        tau = np.clip(p + d, -limit, limit) if legacy else scale * (p + d)
        data.qfrc_applied[:] = tau
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[1, 12] > 0
    if not legacy:
        assert actual[1, 12] < 5.5  # P and D oppose: sum-of-absolute scaling is not net clipping.
    assert original.nq == 6
