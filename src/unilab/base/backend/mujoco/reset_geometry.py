"""Cached, batched signed sole support geometry for flat-ground resets.

Only the world-z row of each link transform is needed. No simulation, collision
query, asset access or per-environment Python loop is used by ``heights``.
"""

from __future__ import annotations

import mujoco
import numpy as np


class ResetSoleGeometry:
    def __init__(self, model, foot_names, vertices, spheres):
        self.feet = []
        for name, mesh, balls in zip(foot_names, vertices, spheres, strict=True):
            chain = []
            body = model.body(name).id
            while model.body_parentid[body] != 0:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.body_quat[body])
                joints = []
                for joint in range(
                    model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]
                ):
                    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise ValueError("reset sole geometry requires hinge leg joints")
                    address = int(model.jnt_qposadr[joint])
                    joints.append(
                        (
                            address,
                            model.jnt_axis[joint].copy(),
                            model.jnt_pos[joint].copy(),
                            float(model.qpos0[address]),
                        )
                    )
                chain.append((model.body_pos[body].copy(), rotation.reshape(3, 3), joints))
                body = int(model.body_parentid[body])
            self.feet.append((list(reversed(chain)), mesh.copy(), balls.copy()))

    def heights(self, qpos: np.ndarray) -> np.ndarray:
        """Signed lowest collision height for each sole, without clipping."""
        w, x, y, z = qpos[:, 3:7].T
        root_row = np.column_stack(
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
        )
        output = np.empty((len(qpos), len(self.feet)))
        for foot, (chain, vertices, spheres) in enumerate(self.feet):
            row, height = root_row.copy(), qpos[:, 2].copy()
            for position, rotation, joints in chain:
                height += row @ position
                row = row @ rotation
                for address, axis, pivot, reference in joints:
                    angle = qpos[:, address] - reference
                    cosine, sine = np.cos(angle)[:, None], np.sin(angle)[:, None]
                    updated = (
                        cosine * row
                        + sine * np.cross(row, axis)
                        + (1 - cosine) * (row @ axis)[:, None] * axis
                    )
                    height += (row - updated) @ pivot
                    row = updated
            support = np.full(len(qpos), np.inf)
            # Bound temporary storage for thousands of simultaneous resets.
            for start in range(0, len(qpos), 512):
                stop = start + 512
                if len(vertices):
                    support[start:stop] = (row[start:stop] @ vertices.T).min(axis=1)
                if len(spheres):
                    support[start:stop] = np.minimum(
                        support[start:stop],
                        (row[start:stop] @ spheres[:, :3].T - spheres[:, 3]).min(axis=1),
                    )
            output[:, foot] = height + support
        return output
