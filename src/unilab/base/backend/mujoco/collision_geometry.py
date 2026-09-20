"""Cold-path collision geometry inspection for asset validation and offline tools."""

import mujoco
import numpy as np


def geom_projection_bounds(model, data, geom_id: int, direction: np.ndarray) -> tuple[float, float]:
    """Exact projection bounds of a compiled primitive or convex mesh, in world coordinates."""
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1):
        raise ValueError("projection direction must be a unit 3-vector")
    local = data.geom_xmat[geom_id].reshape(3, 3).T @ direction
    center = float(data.geom_xpos[geom_id] @ direction)
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        projected = model.mesh_vert[start : start + count] @ local
        return center + float(projected.min()), center + float(projected.max())
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = float(np.abs(local) @ size)
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = float(size[0] * np.linalg.norm(local[:2]) + size[1] * abs(local[2]))
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = float(size[0] + size[1] * abs(local[2]))
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = float(size[0])
    else:
        raise ValueError(f"unsupported finite collision geom: {kind}")
    return center - extent, center + extent


def ground_clearances(model, data) -> dict[str, float]:
    return {
        model.geom(geom).name: geom_projection_bounds(model, data, geom, np.array([0.0, 0.0, 1.0]))[
            0
        ]
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    }
