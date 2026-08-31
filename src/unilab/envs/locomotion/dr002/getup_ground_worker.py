"""Isolated MuJoCo kinematics worker for WE11 get-up reset heights."""

from __future__ import annotations

import struct
import sys

import mujoco
import numpy as np


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(sys.argv[1])
    data = mujoco.MjData(model)
    wheel_geom_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"),
        ],
        dtype=np.int32,
    )
    radius_and_half_length = model.geom_size[wheel_geom_ids, :2]
    source = sys.stdin.buffer
    sink = sys.stdout.buffer
    while True:
        try:
            num_rows, nq, clearance = struct.unpack("<IId", _read_exact(source, 16))
        except EOFError:
            return
        qpos = np.frombuffer(_read_exact(source, int(num_rows) * int(nq) * 8), dtype="<f8").reshape(
            num_rows, nq
        )
        shared = np.frombuffer(_read_exact(source, int(num_rows) * 8), dtype="<f8")
        result = np.empty((num_rows,), dtype="<f8")
        for row, pose in enumerate(qpos):
            data.qpos[:] = pose
            data.qpos[2] = 0.0
            mujoco.mj_kinematics(model, data)
            bottoms: list[float] = []
            for geom_id, (radius, half_length) in zip(
                wheel_geom_ids, radius_and_half_length, strict=True
            ):
                axis_z = float(data.geom_xmat[geom_id].reshape(3, 3)[2, 2])
                vertical_extent = float(radius) * np.sqrt(max(0.0, 1.0 - axis_z * axis_z))
                vertical_extent += float(half_length) * abs(axis_z)
                bottoms.append(float(data.geom_xpos[geom_id, 2]) - vertical_extent)
            result[row] = -min(bottoms) + float(clearance) * (1.0 - shared[row])
        sink.write(result.tobytes())
        sink.flush()


if __name__ == "__main__":
    main()
