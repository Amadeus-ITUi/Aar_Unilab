"""Offline, collision-checked foot workspaces; runtime only queries cached samples."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from unilab.base.backend.mujoco.batched_robot import compile_robot_scene, robot_xml_path
from unilab.base.backend.mujoco.collision_geometry import ground_clearances


def workspace_fingerprint(scene: Path) -> str:
    digest = hashlib.sha256()
    paths = [
        scene,
        robot_xml_path(scene),
        *sorted((scene.parent / "collision_meshes").glob("*.STL")),
    ]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_workspace(scene, output, body_names, foot_names, points, *, resolution=25):
    model = compile_robot_scene(scene, tuple(body_names), visual=False)
    data = mujoco.MjData(model)
    home = model.key("home").qpos.copy()
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    feet = [model.body(name).id for name in foot_names]
    nominal = (
        np.array(
            [data.xpos[idx] + data.xmat[idx].reshape(3, 3) @ p for idx, p in zip(feet, points)]
        )[:, :2]
        - home[:2]
    )
    payload = {
        "fingerprint": np.array(workspace_fingerprint(scene)),
        "nominal": nominal,
        "reference_points": np.asarray(points),
        "home": home,
        "resolution": np.array(resolution),
    }
    for side, foot in enumerate(feet):
        bounds = model.jnt_range[1 + side * 3 : 4 + side * 3]
        grid = np.stack(
            np.meshgrid(*[np.linspace(a, b, resolution) for a, b in bounds], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        grid = np.vstack((home[7 + side * 3 : 10 + side * 3], grid))
        samples, joints = [], []
        for angles in grid:
            data.qpos[:] = home
            data.qpos[7 + side * 3 : 10 + side * 3] = angles
            mujoco.mj_forward(model, data)
            if any(c.dist < -1e-6 for c in data.contact):
                continue
            clearance = ground_clearances(model, data)
            name = foot_names[side] + "_collision_sole_0"
            if clearance[name] < -1e-6:
                continue
            reference = data.xpos[foot] + data.xmat[foot].reshape(3, 3) @ points[side]
            if reference[1] * (1 if side == 0 else -1) < 0.025:
                continue
            samples.append(
                [reference[0] - home[0], reference[1] - home[1], max(0, clearance[name])]
            )
            joints.append(angles)
        if not samples:
            raise ValueError("no collision-free workspace samples")
        payload[f"points_{side}"] = np.asarray(samples)
        payload[f"joints_{side}"] = np.asarray(joints)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return {key: len(payload[key]) for key in ("points_0", "points_1")}


class FootWorkspace:
    def __init__(self, path: Path, scene: Path, reference_points):
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["fingerprint"]) != workspace_fingerprint(scene):
                raise ValueError("workspace assets changed; rebuild the offline gait workspace")
            np.testing.assert_allclose(payload["reference_points"], reference_points, atol=1e-12)
            self.nominal = payload["nominal"].copy()
            self.points = [payload[f"points_{i}"].copy() for i in range(2)]
        self.trees = [cKDTree(points) for points in self.points]

    def project(self, xy, clearance):
        requested = np.concatenate((xy, clearance[..., None]), axis=-1)
        projected = requested.copy()
        error = np.zeros(requested.shape[:2])
        for foot, tree in enumerate(self.trees):
            distance, indices = tree.query(requested[:, foot])
            projected[:, foot] = self.points[foot][indices]
            error[:, foot] = distance
        return projected[..., :2], projected[..., 2], error
