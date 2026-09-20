#!/usr/bin/env python3
"""Compare the WE11 C++ actor/control trajectory with a Python MuJoCo reference."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

DEFAULTS = np.array([0.8, -1.6, 0.0, 0.8, -1.6, 0.0])
KP = np.array([2.0, 7.59, 0.0, 2.0, 7.59, 0.0])
KD = np.array([0.08, 0.682, 0.05, 0.08, 0.682, 0.05])
SCALE = np.array([0.5, 0.5, 10.0, 0.5, 0.5, 10.0])
LIMIT = np.array([5.5, 14.0, 5.5, 5.5, 14.0, 5.5])


def sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    address = model.sensor_adr[sensor_id]
    dimension = model.sensor_dim[sensor_id]
    return data.sensordata[address : address + dimension]


def observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    histories: list[np.ndarray],
    previous_action: np.ndarray,
) -> np.ndarray:
    joint_ids = model.actuator_trnid[:6, 0]
    positions = data.qpos[model.jnt_qposadr[joint_ids]]
    velocities = data.qvel[model.jnt_dofadr[joint_ids]]
    x_axis = sensor(model, data, "xvector")
    y_axis = sensor(model, data, "yvector")
    z_axis = sensor(model, data, "upvector")
    terms = [
        sensor(model, data, "imu_gyro"),
        np.array([-x_axis[2], -y_axis[2], -z_axis[2]]),
        positions[[0, 1, 3, 4]] - DEFAULTS[[0, 1, 3, 4]],
        velocities[[0, 1, 3, 4, 2, 5]] * 0.1,
        previous_action,
        np.array(
            [sensor(model, data, "left_wing_pos")[0], sensor(model, data, "right_wing_pos")[0]]
        )
        / np.pi,
        np.array(
            [
                sensor(model, data, "left_wing_vel_sensor")[0],
                sensor(model, data, "right_wing_vel_sensor")[0],
            ]
        )
        * 0.1,
        np.zeros(3),
    ]
    for index, term in enumerate(terms):
        histories[index] = np.concatenate((histories[index][term.size :], term))
    return np.concatenate(histories).astype(np.float32)[None, :]


def run_python(release: Path, steps: int) -> list[dict[str, float]]:
    model = mujoco.MjModel.from_xml_path(str(release / "robot/scene_flat_we11.xml"))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    session = ort.InferenceSession(str(release / "policy.onnx"), providers=["CPUExecutionProvider"])
    term_sizes = [3, 3, 4, 6, 6, 2, 2, 3]
    histories = [np.zeros(size * 5) for size in term_sizes]
    previous_action = np.zeros(6, dtype=np.float32)
    result = []
    for _ in range(steps):
        obs = observation(model, data, histories, previous_action)
        action = session.run(["act"], {"obs": obs})[0][0]
        for substep in range(8):
            if substep % 2 == 0:
                joint_ids = model.actuator_trnid[:6, 0]
                positions = data.qpos[model.jnt_qposadr[joint_ids]]
                velocities = data.qvel[model.jnt_dofadr[joint_ids]]
                selected = np.clip(action.astype(np.float64), -100.0, 100.0)
                selected[[2, 5]] = np.clip(selected[[2, 5]], -3.5, 3.5)
                torque = KP * (DEFAULTS + selected * SCALE - positions) - KD * velocities
                torque[[2, 5]] = KD[[2, 5]] * (
                    selected[[2, 5]] * SCALE[[2, 5]] - velocities[[2, 5]]
                )
                data.ctrl[:6] = np.clip(torque, -LIMIT, LIMIT)
                wing_joints = model.actuator_trnid[6:8, 0]
                data.ctrl[6:8] = np.clip(
                    -10.0 * data.qpos[model.jnt_qposadr[wing_joints]]
                    - 0.5 * data.qvel[model.jnt_dofadr[wing_joints]],
                    -5.5,
                    5.5,
                )
            mujoco.mj_step(model, data)
        row = {"base_height": float(data.qpos[2])}
        row.update({f"action_{i}": float(value) for i, value in enumerate(action)})
        row.update({f"ctrl_{i}": float(value) for i, value in enumerate(data.ctrl[:6])})
        result.append(row)
        previous_action = action
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    for steps in (1, 10, 100):
        expected = run_python(args.release, steps)
        with tempfile.TemporaryDirectory(prefix="aar-we11-") as temporary:
            telemetry = Path(temporary)
            subprocess.run(
                [
                    str(args.binary),
                    str(args.release),
                    "--steps",
                    str(steps),
                    "--telemetry",
                    str(telemetry),
                ],
                check=True,
            )
            with (telemetry / "sim2sim.csv").open(newline="", encoding="utf-8") as stream:
                actual = [
                    {key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(stream)
                ]
        keys = ["base_height", *(f"action_{i}" for i in range(6)), *(f"ctrl_{i}" for i in range(6))]
        error = max(abs(actual[i][key] - expected[i][key]) for i in range(steps) for key in keys)
        if error > 1.0e-9:
            raise AssertionError(f"WE11 {steps}-step Python/C++ mismatch: {error}")
        print(f"WE11 Python/C++ trajectory steps={steps} max_error={error:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
