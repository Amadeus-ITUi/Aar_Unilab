import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from unilab.base.backend.mujoco.batched_robot import compile_robot_scene
from unilab.base.backend.mujoco.collision_geometry import geom_projection_bounds, ground_clearances
from unilab.envs.locomotion.pe02 import PE02Env

ASSET = Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/pe02"
PAIRS = [
    ("L_hip_Link", "R_hip_Link"),
    ("L_thigh_Link", "R_thigh_Link"),
    ("L_calf_Link", "R_calf_link"),
    ("L_foot_Link", "R_foot_Link"),
]


def vertices(path):
    dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.unique(
        np.frombuffer(path.read_bytes(), dtype=dtype, offset=84)["v"].reshape(-1, 3), axis=0
    ).astype(float)


def test_collision_assets_are_separate_and_inertias_stay_explicit():
    tree = ET.parse(ASSET / "pe02.xml")
    assert tree.find("compiler").get("inertiafromgeom") == "false"
    assets = {m.get("name"): m.get("file") for m in tree.findall("asset/mesh")}
    for body in tree.findall(".//body"):
        assert body.find("inertial") is not None
        collisions = [g for g in body.findall("geom") if g.get("group") == "3"]
        assert collisions
        for geom in collisions:
            assert geom.get("density") == "0"
            if geom.get("type") == "mesh":
                assert assets[geom.get("mesh")].startswith("../collision_meshes/")
        for geom in body.findall("geom"):
            if geom.get("group") == "2":
                assert not assets[geom.get("mesh")].startswith("../collision_meshes/")
                assert geom.get("contype") == geom.get("conaffinity") == "0"
    manifest = json.loads((ASSET / "asset_manifest.json").read_text())["collision"]
    assert (
        hashlib.sha256((ASSET / "collision_recipe.json").read_bytes()).hexdigest()
        == manifest["recipe_sha256"]
    )
    for link in manifest["links"].values():
        assert (
            hashlib.sha256((ASSET / link["source"]).read_bytes()).hexdigest()
            == link["source_sha256"]
        )
        for part in link["parts"]:
            if part["kind"] == "hull":
                assert (
                    hashlib.sha256((ASSET / part["file"]).read_bytes()).hexdigest()
                    == part["sha256"]
                )


def test_all_masses_centers_and_full_inertia_tensors_match_urdf():
    env = PE02Env()
    urdf = ET.parse(ASSET / "urdf/pe02.urdf")
    for link in urdf.findall("link"):
        body = env.model.body(link.get("name"))
        inertial = link.find("inertial")
        assert body.mass[0] == float(inertial.find("mass").get("value"))
        np.testing.assert_allclose(
            body.ipos, np.fromstring(inertial.find("origin").get("xyz"), sep=" "), atol=1e-14
        )
        inertia = inertial.find("inertia").attrib
        source = np.array(
            [[float(inertia[f"i{min(a, b)}{max(a, b)}"]) for b in "xyz"] for a in "xyz"]
        )
        source_rotation = Rotation.from_euler(
            "xyz", np.fromstring(inertial.find("origin").get("rpy"), sep=" ")
        ).as_matrix()
        source = source_rotation @ source @ source_rotation.T
        rotation = Rotation.from_quat(body.iquat, scalar_first=True).as_matrix()
        np.testing.assert_allclose(
            rotation @ np.diag(body.inertia) @ rotation.T, source, atol=2e-10, rtol=1e-6
        )
    assert np.isclose(env.model.body_mass.sum(), 3.22689)


def test_compiled_collision_surfaces_are_mirrored_and_home_is_ground_aligned():
    env = PE02Env()
    env.reset()
    model, data = env.model, env.data
    rng = np.random.default_rng(17)
    directions = rng.normal(size=(64, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for left, right in PAIRS:
        left_geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == model.body(left).id and model.geom_contype[g]
        ]
        right_geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == model.body(right).id and model.geom_contype[g]
        ]
        assert len(left_geoms) == len(right_geoms)
        for lg, rg in zip(left_geoms, right_geoms, strict=True):
            for axis in directions:
                projections = []
                for name, geom, local in [(left, lg, axis), (right, rg, axis * [1, -1, 1])]:
                    body = model.body(name).id
                    world = data.xmat[body].reshape(3, 3) @ local
                    projection = geom_projection_bounds(model, data, geom, world)
                    projections.append(np.array(projection) - data.xpos[body] @ world)
                np.testing.assert_allclose(*projections, atol=1e-7, rtol=0)
    clearance = ground_clearances(model, data)
    assert min(clearance.values()) > -1e-7
    for name, value in clearance.items():
        if "foot" in name:
            assert abs(value) < 1e-7
        else:
            assert value > 0.03


def test_split_calf_preserves_a_gap_filled_by_the_old_single_hull():
    point = np.array([-0.1044, -0.015, -0.0868])
    old = ConvexHull(vertices(ASSET / "meshes/L_calf_Link.STL"))
    assert np.max(old.equations[:, :3] @ point + old.equations[:, 3]) < 0
    pieces = sorted((ASSET / "collision_meshes").glob("L_calf_Link_curve_*.STL"))
    assert len(pieces) >= 3
    total = 0.0
    for path in pieces:
        hull = ConvexHull(vertices(path))
        total += hull.volume
        assert np.max(hull.equations[:, :3] @ point + hull.equations[:, 3]) > 0.01
    assert total < old.volume * 0.8


def test_training_collision_budget_and_original_sole_contact_cap():
    model = compile_robot_scene(ASSET / "scene.xml", (), visual=False)
    assert model.nbuffer < 750_000
    assert model.ngeom <= 30  # Includes the floor; guards against accidental CAD collision use.
    report = json.loads((ASSET / "analysis/standing_pose.json").read_text())
    rng = np.random.default_rng(91)
    directions = rng.normal(size=(512, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for side, normal in zip(("L", "R"), report["sole_normals_local"], strict=True):
        # The collision recipe deliberately mirrors the left CAD on the right;
        # the two original CAD exports have a small tessellation difference.
        source = vertices(ASSET / "meshes/L_foot_Link.STL")
        if side == "R":
            source *= [1, -1, 1]
        source = source[ConvexHull(source).vertices]
        proxy = vertices(ASSET / f"collision_meshes/{side}_foot_Link_sole_0.STL")
        gap = (source @ directions.T).max(axis=0) - (proxy @ directions.T).max(axis=0)
        assert gap.min() >= -1e-7
        assert gap.max() <= 0.0005 + 1e-7
        projection = source @ normal
        cap = source[projection.max() - projection <= 0.001]
        distances = np.linalg.norm(cap[:, None] - proxy[None], axis=-1).min(axis=1)
        assert distances.max() < 1e-7
