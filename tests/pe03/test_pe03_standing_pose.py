import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSET = Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/pe03"


def test_home_matches_source_urdf_mass_geometry_and_ground_contact():
    env = home_model()
    report = json.loads((ASSET / "analysis/standing_pose.json").read_text())
    np.testing.assert_allclose(env.data.qpos, report["qpos"], atol=1e-12)
    # Independent forward kinematics from original URDF transforms and mechanical-zero angles.
    urdf = ET.parse(ASSET / "urdf/pe03.urdf")
    transforms = {"body_link": (np.eye(3), env.data.qpos[:3])}
    angles = dict(zip(report["joint_order"], env.data.qpos[7:], strict=True))
    remaining = list(urdf.findall("joint"))
    while remaining:
        available = [j for j in remaining if j.find("parent").get("link") in transforms]
        assert available, "URDF is not a connected tree"
        for joint in available:
            parent_rotation, parent_position = transforms[joint.find("parent").get("link")]
            origin = joint.find("origin")
            local_rotation = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy"), sep=" ")
            ).as_matrix()
            local_position = np.fromstring(origin.get("xyz"), sep=" ")
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                local_rotation = (
                    local_rotation
                    @ Rotation.from_rotvec(axis * angles[joint.get("name")]).as_matrix()
                )
                limit = joint.find("limit")
                assert (
                    float(limit.get("lower"))
                    <= angles[joint.get("name")]
                    <= float(limit.get("upper"))
                )
            transforms[joint.find("child").get("link")] = (
                parent_rotation @ local_rotation,
                parent_position + parent_rotation @ local_position,
            )
            remaining.remove(joint)
    weighted_com = np.zeros(3)
    total_mass = 0.0
    for link in urdf.findall("link"):
        rotation, position = transforms[link.get("name")]
        np.testing.assert_allclose(env.data.body(link.get("name")).xpos, position, atol=1e-12)
        mass = float(link.find("inertial/mass").get("value"))
        local_com = np.fromstring(link.find("inertial/origin").get("xyz"), sep=" ")
        weighted_com += mass * (rotation @ local_com + position)
        total_mass += mass
    com = weighted_com / total_mass
    np.testing.assert_allclose(com, env.data.subtree_com[1], atol=1e-12)
    centers = []
    for index, side in enumerate(("L", "R")):
        name = f"{side}_foot_Link"
        rotation, position = transforms[name]
        center = rotation @ np.array(report["sole_centers_local_m"][index]) + position
        centers.append(center)
        normal = rotation @ report["sole_normals_local"][index]
        np.testing.assert_allclose(normal, [0, 0, -1], atol=1e-9)
        assert abs(center[2]) < 1e-9
        dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        vertices = np.frombuffer(
            (ASSET / "meshes" / f"{name}.STL").read_bytes(), dtype=dtype, offset=84
        )["v"].reshape(-1, 3)
        world = vertices @ rotation.T + position
        assert abs(world[:, 2].min()) < 1e-8
        knee = transforms["L_calf_Link" if side == "L" else "R_calf_link"][1]
        line = center - knee
        angle = np.rad2deg(np.arctan2(-line[2], np.linalg.norm(line[:2])))
        assert 45 <= angle <= 56
        assert abs(angle - 51) < 0.5
    error = com[:2] - np.mean(centers, axis=0)[:2]
    assert abs(error[0]) < 1e-9
    assert np.linalg.norm(error) < 0.0005
    for contact in env.data.contact:
        assert not (env.model.geom_bodyid[contact.geom1] and env.model.geom_bodyid[contact.geom2])
        assert contact.dist > -1e-7


def home_model():
    model = mujoco.MjModel.from_xml_path(str(ASSET / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)
