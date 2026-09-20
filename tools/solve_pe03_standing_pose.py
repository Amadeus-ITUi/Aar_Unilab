#!/usr/bin/env python3
"""Offline PE03 standing-pose fit; keep mesh analysis out of environment hot paths.

The curved soles have no large planar contact face. Rank lower convex-hull facets
by the cross-section area 0.5 mm above their supporting plane (a geometric proxy,
not a rubber deformation model). Constrain the knee-to-support angle to 45–56°,
keep the torso level, then solve both legs and height using the exported joint
frames, including their small left/right assembly differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from unilab.base.backend.mujoco.collision_geometry import ground_clearances

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/unilab/assets/robots/pe03"


def stl_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"]
    return np.unique(triangles.reshape(-1, 3).astype(float), axis=0)


def section_area(vertices: np.ndarray, edges: np.ndarray, normal: np.ndarray, depth: float):
    """Convex-hull cross section parallel to its supporting plane, in square metres."""
    cut = (vertices @ normal).max() - depth
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    da, db = a @ normal - cut, b @ normal - cut
    crossing = da * db < 0
    points = (
        a[crossing]
        + (b[crossing] - a[crossing]) * (da[crossing] / (da[crossing] - db[crossing]))[:, None]
    )
    u = np.cross(normal, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    hull = ConvexHull(np.column_stack([points @ u, points @ v]))
    return float(hull.volume), points[hull.vertices]


def solve(asset: Path = ASSET, patch_depth: float = 0.0005):
    if not 0 < patch_depth <= 0.002:
        raise ValueError("patch depth must be in (0, 2] mm")
    model = mujoco.MjModel.from_xml_path(str(asset / "scene.xml"))
    data = mujoco.MjData(model)
    vertices = [stl_vertices(asset / "meshes" / f"{side}_foot_Link.STL") for side in ("L", "R")]
    left = vertices[0]
    hull = ConvexHull(left)
    triangles = hull.simplices
    edges = np.unique(
        np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])),
        axis=0,
    )
    feet = [model.body(f"{side}_foot_Link").id for side in ("L", "R")]
    knees = [model.body(name).id for name in ("L_calf_Link", "R_calf_link")]
    candidates = []
    for normal, triangle in zip(hull.equations[:, :3], triangles, strict=True):
        # Lower rounded cap, with at most 10 degrees of lateral opening.
        if normal[0] > -np.cos(np.deg2rad(20)) or not -np.sin(np.deg2rad(10)) <= normal[1] <= 0:
            continue
        center = left[triangle].mean(axis=0)
        hip = np.arcsin(-normal[1])
        pitch = -np.pi / 2 + np.arctan2(-normal[2], -normal[0])
        rotation = Rotation.from_euler("x", hip) * Rotation.from_euler("y", pitch)
        line = rotation.apply(model.body_pos[feet[0]] + center)
        angle = np.rad2deg(np.arctan2(-line[2], np.linalg.norm(line[:2])))
        if 45 <= angle <= 56:
            area, _ = section_area(left, edges, normal, patch_depth)
            candidates.append((area, normal, center, triangle))
    if not candidates:
        raise ValueError("no lower sole facet satisfies the 45–56 degree geometric constraint")
    _, normal, center, triangle = max(candidates, key=lambda item: item[0])
    mirror = np.array([1, -1, 1])
    normals, centers = np.array([normal, normal * mirror]), np.array([center, center * mirror])
    for v, n, c in zip(vertices, normals, centers, strict=True):
        if abs((v @ n).max() - c @ n) > 1e-7:
            raise ValueError("the selected mirrored plane does not support both foot meshes")

    def forward(x):
        data.qpos[:] = np.r_[0, 0, x[-1], 1, 0, 0, 0, x[:6]]
        mujoco.mj_forward(model, data)
        rotations = data.xmat[feet].reshape(2, 3, 3)
        world_centers = np.einsum("bij,bj->bi", rotations, centers) + data.xpos[feet]
        world_normals = np.einsum("bij,bj->bi", rotations, normals)
        return world_centers, world_normals

    def residual(x):
        contacts, world_normals = forward(x)
        return np.r_[
            world_normals[:, :2].ravel() * 0.2,
            contacts[:, 2],
            data.subtree_com[1, 0] - contacts[:, 0].mean(),
        ]

    lower = np.r_[model.jnt_range[1:, 0], 0.1]
    upper = np.r_[model.jnt_range[1:, 1], 0.5]
    result = least_squares(
        residual,
        [0.025, -0.33, -1.24, -0.025, 0.33, 1.24, 0.294],
        bounds=(lower, upper),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    if not result.success or np.max(np.abs(residual(result.x))) > 1e-8:
        raise ValueError(f"standing-pose fit failed: {result.message}")
    contacts, world_normals = forward(result.x)
    com = data.subtree_com[1].copy()
    offset = com[:2] - contacts[:, :2].mean(axis=0)
    if np.linalg.norm(offset) > 0.0005:
        raise ValueError(f"COM midpoint error exceeds 0.5 mm: {offset}")
    if np.any(data.qpos[7:] < model.jnt_range[1:, 0]) or np.any(
        data.qpos[7:] > model.jnt_range[1:, 1]
    ):
        raise ValueError("home exceeds a mechanical joint limit")

    # Check the actual compiled runtime geoms, including mesh recentering by MuJoCo.
    clearances = ground_clearances(model, data)
    if min(clearances.values()) < -1e-7:
        raise ValueError("a runtime collision mesh penetrates the ground")
    self_contacts = [
        [model.geom(contact.geom1).name, model.geom(contact.geom2).name, float(contact.dist)]
        for contact in data.contact
        if model.geom_bodyid[contact.geom1]
        and model.geom_bodyid[contact.geom2]
        and contact.dist < 0
    ]
    if self_contacts:
        raise ValueError(f"home has self collisions: {self_contacts}")
    angles = []
    for contact, knee in zip(contacts, knees, strict=True):
        line = contact - data.xpos[knee]
        angles.append(float(np.rad2deg(np.arctan2(-line[2], np.linalg.norm(line[:2])))))
    if not all(45 <= angle <= 56 for angle in angles):
        raise ValueError(f"knee-to-support angles outside requested band: {angles}")
    tri = left[triangle]
    rigid_area = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2
    # Static inverse dynamics with vertical reactions at the two supporting centres.
    weight = model.body_mass.sum() * -model.opt.gravity[2]
    left_force = weight * (com[1] - contacts[1, 1]) / (contacts[0, 1] - contacts[1, 1])
    support = np.zeros(model.nv)
    for foot, contact, force in zip(feet, contacts, [left_force, weight - left_force], strict=True):
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, contact, foot)
        support += jac[2] * force
    required = data.qfrc_bias - support
    report = {
        "method": "level torso, independently solved joint angles in original assembly frames; largest near-plane area among lower hull facets in the 45–56 deg knee-to-contact band",
        "near_plane_depth_m": patch_depth,
        "near_plane_area_is_not_physical_contact_area": True,
        "mass_kg": float(model.body_mass.sum()),
        "joint_order": [model.joint(i).name for i in range(1, model.njnt)],
        "joint_angles_rad": data.qpos[7:].tolist(),
        "joint_angles_deg": np.rad2deg(data.qpos[7:]).tolist(),
        "qpos": data.qpos.tolist(),
        "com_world_m": com.tolist(),
        "support_centers_world_m": contacts.tolist(),
        "com_minus_support_midpoint_xy_m": offset.tolist(),
        "sole_normals_local": normals.tolist(),
        "sole_centers_local_m": centers.tolist(),
        "sole_normals_world": world_normals.tolist(),
        "knee_to_support_angle_deg": angles,
        "rigid_support_triangle_area_mm2_per_foot": float(rigid_area * 1e6),
        "near_plane_area_mm2_per_foot": {
            str(depth * 1000): section_area(left, edges, normal, depth)[0] * 1e6
            for depth in (0.0001, 0.00025, 0.0005, 0.001, 0.002)
        },
        "runtime_collision_ground_clearance_m": clearances,
        "self_contacts": self_contacts,
        "estimated_static_joint_torque_nm": required[6:].tolist(),
        "estimated_vertical_support_force_n": [float(left_force), float(weight - left_force)],
        "static_floating_base_residual": required[:6].tolist(),
        "actuator_torque_limits_nm": np.abs(model.actuator_gear[:, 0]).tolist(),
        "static_torque_limit_fraction": (np.abs(required[6:] / model.actuator_gear[:, 0])).tolist(),
        "estimated_static_torque_is_not_a_controller": True,
        "source_sha256": {
            path.relative_to(asset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                asset / "urdf/pe03.urdf",
                asset / "pe03.xml",
                *sorted((asset / "meshes").glob("*foot*")),
            )
        },
    }
    return report, model, data


def render(path: Path, model, data, report):
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.035, 0, 0.16]
    camera.distance, camera.azimuth, camera.elevation = 0.68, 90, -5
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        pixels = renderer.render().copy()
    fig, (side, top) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    side.imshow(pixels)
    side.axis("off")
    angle = report["knee_to_support_angle_deg"][0]
    side.set_title(
        f"PE03 computed home | torso level\nKnee to contact: {angle:.2f} deg | base z: {data.qpos[2] * 1000:.2f} mm"
    )
    for index, name in enumerate(("L_foot_Link", "R_foot_Link")):
        geom = next(
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body(name).id and model.geom_contype[i]
        )
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = (
            model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
        )
        hull = ConvexHull(points[:, :2])
        outline = points[hull.vertices, :2] * 1000
        top.fill(outline[:, 0], outline[:, 1], alpha=0.15, color="steelblue")
        faces = ConvexHull(points).simplices
        edges = np.unique(
            np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])), axis=0
        )
        _, close = section_area(points, edges, np.array([0, 0, -1]), 0.0005)
        close = close[:, :2] * 1000
        top.fill(
            close[:, 0],
            close[:, 1],
            alpha=0.5,
            color="steelblue",
            label="Within 0.5 mm of floor" if index == 0 else None,
        )
    centers = np.array(report["support_centers_world_m"])[:, :2] * 1000
    com = np.array(report["com_world_m"])[:2] * 1000
    top.plot(centers[:, 0], centers[:, 1], "k+--", label="Contact centres")
    top.scatter(*com, color="crimson", zorder=5, label="COM projection")
    error = np.linalg.norm(report["com_minus_support_midpoint_xy_m"]) * 1000
    top.set_title(f"Top view | COM midpoint error: {error:.3f} mm")
    top.set_xlabel("x (mm)")
    top.set_ylabel("y (mm)")
    top.set_aspect("equal")
    top.grid(alpha=0.3)
    top.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--patch-depth-mm", type=float, default=0.5)
    parser.add_argument("--write", action="store_true", help="write scene home and analysis JSON")
    parser.add_argument(
        "--render", action="store_true", help="save an annotated standing-pose image"
    )
    args = parser.parse_args()
    report, model, data = solve(args.asset, args.patch_depth_mm / 1000)
    print(json.dumps(report, indent=2))
    output = args.asset / "analysis"
    if args.write or args.render:
        output.mkdir(exist_ok=True)
    if args.write:
        scene = ET.parse(args.asset / "scene.xml")
        keyframes = scene.find("keyframe")
        if keyframes is None:
            keyframes = ET.SubElement(scene.getroot(), "keyframe")
        home = next((key for key in keyframes if key.get("name") == "home"), None)
        if home is None:
            home = ET.SubElement(keyframes, "key", name="home")
        home.set("qpos", " ".join(format(value, ".16g") for value in report["qpos"]))
        home.set("qvel", " ".join(["0"] * model.nv))
        home.set("ctrl", " ".join(["0"] * model.nu))
        ET.indent(scene, space="  ")
        scene.write(args.asset / "scene.xml", encoding="unicode")
        (output / "standing_pose.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.render:
        render(output / "standing_pose.png", model, data, report)


if __name__ == "__main__":
    main()
