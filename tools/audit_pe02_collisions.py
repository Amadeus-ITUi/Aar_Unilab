#!/usr/bin/env python3
"""Offline PE02 collision fit, cavity, symmetry and motion checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from unilab.base.backend.mujoco.collision_geometry import geom_projection_bounds, ground_clearances

ASSET = Path(__file__).resolve().parents[1] / "src/unilab/assets/robots/pe02"
PAIRS = [
    ("L_hip_Link", "R_hip_Link"),
    ("L_thigh_Link", "R_thigh_Link"),
    ("L_calf_Link", "R_calf_link"),
    ("L_foot_Link", "R_foot_Link"),
]


def surface_fit(asset: Path, name: str, samples: int = 5000) -> dict:
    """Point-to-union distances; source surface samples include internal CAD surfaces."""
    urdf = ET.parse(asset / "urdf/pe02.urdf")
    cad = trimesh.load_mesh(asset / "meshes" / f"{name}.STL")
    points, _ = trimesh.sample.sample_surface(cad, samples, seed=42)
    proxies, inside = [], np.zeros(len(points), dtype=bool)
    for element in urdf.find(f"link[@name='{name}']").findall("collision"):
        origin = element.find("origin")
        position = np.fromstring(origin.get("xyz"), sep=" ")
        rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
        local = (points - position) @ rotation
        shape = list(element.find("geometry"))[0]
        if shape.tag == "mesh":
            value = trimesh.load_mesh(
                asset / shape.get("filename").removeprefix("package://pe02/")
            ).convex_hull
            planes = ConvexHull(value.vertices).equations
            distance = np.max(local @ planes[:, :3].T + planes[:, 3], axis=1)
        elif shape.tag == "box":
            value = np.fromstring(shape.get("size"), sep=" ") / 2
            distance = np.max(np.abs(local) - value, axis=1)
        elif shape.tag == "cylinder":
            value = [float(shape.get("radius")), float(shape.get("length")) / 2]
            distance = np.maximum(
                np.linalg.norm(local[:, :2], axis=1) - value[0], np.abs(local[:, 2]) - value[1]
            )
        else:
            raise ValueError(shape.tag)
        inside |= distance < 1e-7
        proxies.append((shape.tag, position, rotation, value))
    outside = points[~inside]
    distance = np.full(len(outside), np.inf)
    for kind, position, rotation, value in proxies:
        local = (outside - position) @ rotation
        if kind == "box":
            candidate = np.linalg.norm(np.maximum(np.abs(local) - value, 0), axis=1)
        elif kind == "cylinder":
            candidate = np.linalg.norm(
                np.maximum(
                    np.column_stack(
                        [
                            np.linalg.norm(local[:, :2], axis=1) - value[0],
                            np.abs(local[:, 2]) - value[1],
                        ]
                    ),
                    0,
                ),
                axis=1,
            )
        else:
            chunks = []
            triangles = value.triangles
            count = len(triangles)
            for start in range(0, len(local), 32):
                batch = local[start : start + 32]
                repeated = np.repeat(batch, count, axis=0)
                closest = trimesh.triangles.closest_point(
                    np.tile(triangles, (len(batch), 1, 1)), repeated
                )
                chunks.extend(
                    np.linalg.norm(repeated - closest, axis=1)
                    .reshape(len(batch), count)
                    .min(axis=1)
                )
            candidate = np.array(chunks)
        distance = np.minimum(distance, candidate)
    full = np.zeros(len(points))
    full[~inside] = distance
    return {
        "samples": samples,
        "inside_fraction": float(inside.mean()),
        "outside_distance_p99_mm": float(np.quantile(full, 0.99) * 1000),
        "outside_distance_max_mm": float(full.max() * 1000),
        "worst_point_local_m": points[np.argmax(full)].tolist(),
    }


def legacy_model(asset: Path):
    """Reconstruct the previous collision rule: use each decimated visual as one convex hull."""
    tree = ET.parse(asset / "pe02.xml")
    tree.find("compiler").set("meshdir", str((asset / "runtime_meshes").resolve()))
    for body in tree.findall(".//body"):
        for geom in list(body.findall("geom")):
            if geom.get("group") == "3":
                body.remove(geom)
        visual = next(g for g in body.findall("geom") if g.get("group") == "2")
        collision = copy.deepcopy(visual)
        collision.attrib.update(
            name=f"{body.get('name')}_legacy_collision", contype="1", conaffinity="1", group="3"
        )
        body.append(collision)
    for element in ET.parse(asset / "scene.xml").findall("worldbody/*"):
        tree.find("worldbody").append(copy.deepcopy(element))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))


def motion_check(model, home, joints) -> dict:
    data = mujoco.MjData(model)
    poses_with_contact = 0
    worst = {"depth_m": 0.0}
    for angles in joints:
        data.qpos[:] = home
        data.qpos[2] = 1.0  # Remove floor contact from the self-collision sweep.
        data.qpos[7:] = angles
        mujoco.mj_forward(model, data)
        contacts = [
            c
            for c in data.contact
            if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2] and c.dist < -1e-6
        ]
        poses_with_contact += bool(contacts)
        for contact in contacts:
            if -contact.dist > worst["depth_m"]:
                worst = {
                    "depth_m": float(-contact.dist),
                    "bodies": [
                        model.body(model.geom_bodyid[c]).name
                        for c in (contact.geom1, contact.geom2)
                    ],
                    "joint_angles_rad": angles.tolist(),
                }
    return {"poses": len(joints), "poses_with_self_contact": poses_with_contact, "worst": worst}


def audit(asset: Path = ASSET):
    model = mujoco.MjModel.from_xml_path(str(asset / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    baseline = legacy_model(asset)
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_axis",
        "jnt_range",
        "dof_armature",
        "actuator_gear",
    ):
        np.testing.assert_array_equal(getattr(model, field), getattr(baseline, field))
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    symmetry = {}
    for left, right in PAIRS:
        errors = []
        for direction in directions:
            projections = []
            for name, vector in ((left, direction), (right, direction * [1, -1, 1])):
                body_id = model.body(name).id
                world = data.xmat[body_id].reshape(3, 3) @ vector
                offset = data.xpos[body_id] @ world
                projections.append(
                    max(
                        geom_projection_bounds(model, data, g, world)[1] - offset
                        for g in range(model.ngeom)
                        if model.geom_bodyid[g] == body_id and model.geom_contype[g]
                    )
                )
            errors.append(abs(projections[0] - projections[1]))
        symmetry[left] = max(errors)
        if symmetry[left] > 1e-7:
            raise ValueError(f"mirrored collision mismatch: {left}")
    home = data.qpos.copy()
    clearances = ground_clearances(model, data)
    home_self = [
        c
        for c in data.contact
        if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2] and c.dist < -1e-7
    ]
    if home_self or min(clearances.values()) < -1e-7:
        raise ValueError("home has self contact or ground penetration")
    near = np.clip(
        home[7:] + rng.uniform(-np.deg2rad(5), np.deg2rad(5), (500, 6)),
        model.jnt_range[1:, 0],
        model.jnt_range[1:, 1],
    )
    broad = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], (1000, 6))
    cad = trimesh.load_mesh(asset / "meshes/L_calf_Link.STL").convex_hull
    calf_pieces = [
        trimesh.load_mesh(p).convex_hull
        for p in sorted((asset / "collision_meshes").glob("L_calf_Link_curve_*.STL"))
    ]
    new_volume = sum(p.volume for p in calf_pieces)  # Disjoint slabs, shared boundary only.
    x, z = np.meshgrid(np.linspace(-0.12, 0, 101), np.linspace(-0.14, 0, 101))
    points = np.column_stack([x.ravel(), np.full(x.size, -0.015), z.ravel()])
    hull = ConvexHull(cad.vertices)
    inside = (points @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1) < 0
    holes = points[inside]
    distances = []
    for part in calf_pieces:
        planes = ConvexHull(part.vertices).equations
        distances.append((holes @ planes[:, :3].T + planes[:, 3]).max(axis=1))
    gaps = np.min(distances, axis=0)
    gap_point = holes[np.argmax(gaps)]
    report = {
        "model_sha256": hashlib.sha256((asset / "pe02.xml").read_bytes()).hexdigest(),
        "mass_kg": float(model.body_mass.sum()),
        "collision_geom_count": sum(
            bool(model.geom_bodyid[g] and model.geom_contype[g]) for g in range(model.ngeom)
        ),
        "mirrored_support_error_m": symmetry,
        "home_clearance_m": clearances,
        "calf_old_convex_volume_m3": float(cad.volume),
        "calf_split_volume_m3": float(new_volume),
        "calf_convex_filled_volume_removed_fraction": float(1 - new_volume / cad.volume),
        "calf_preserved_gap_example_local_m": gap_point.tolist(),
        "gap_distance_lower_bound_m": float(gaps.max()),
        "near_home_new": motion_check(model, home, near),
        "near_home_old": motion_check(baseline, home, near),
        "legal_range_new": motion_check(model, home, broad),
        "legal_range_old": motion_check(baseline, home, broad),
        "source_surface_coverage": {
            name: surface_fit(asset, name)
            for name in ("body_link", *[n for pair in PAIRS for n in pair])
        },
        "coverage_note": "Area-weighted source surface samples, including internal CAD surfaces; coverage is not an outer overfill error bound. Broad-range contacts may be physically valid and are not excluded.",
    }
    return report, model, baseline, home


def render(path: Path, model, baseline, home, *, before_title="Before: one hull per link"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, selected, collision, title in zip(
        axes,
        (baseline, model, model),
        (True, True, False),
        (before_title, "After: separate collision parts", "Visual mesh"),
        strict=True,
    ):
        selected.vis.global_.offwidth, selected.vis.global_.offheight = 800, 600
        data = mujoco.MjData(selected)
        data.qpos[:] = home
        mujoco.mj_forward(selected, data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [-0.035, 0, 0.16]
        cam.distance, cam.azimuth, cam.elevation = 0.75, 110, -8
        options = mujoco.MjvOption()
        options.geomgroup[2], options.geomgroup[3] = int(not collision), int(collision)
        options.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = int(collision)
        colors = plt.get_cmap("tab10")
        for g in range(selected.ngeom):
            if selected.geom_group[g] == 3:
                selected.geom_rgba[g] = colors(int(selected.geom_bodyid[g]) % 10)
        with mujoco.Renderer(selected, height=600, width=800) as renderer:
            renderer.update_scene(data, camera=cam, scene_option=options)
            ax.imshow(renderer.render().copy())
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    report, model, baseline, home = audit(args.asset)
    directory = args.asset / "analysis"
    directory.mkdir(exist_ok=True)
    (directory / "collision_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.render:
        render(directory / "collision_comparison.png", model, baseline, home)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
