"""Offline CAD URDF conversion using an existing model's control defaults.

Run with ``python -m unilab.base.backend.mujoco.urdf_import --help``.
Rebuilding needs trimesh; optional decimation also needs fast-simplification.
Training does not depend on either offline tool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _numbers(values) -> str:
    return " ".join(f"{value:.16g}" for value in values)


def _required(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise ValueError(f"missing {path!r} in {parent.tag} {parent.get('name', '')!r}")
    return element


def _pose(origin: ET.Element | None) -> dict[str, str]:
    if origin is None:
        return {}
    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    x, y, z, w = rotation.as_quat()
    return {"pos": origin.get("xyz", "0 0 0"), "quat": _numbers([w, x, y, z])}


def build_model(
    urdf: Path,
    control_template: Path,
    *,
    max_faces: int | None = 20_000,
    source_name: str | None = None,
) -> Path:
    """Preserve URDF mechanics; use lossless OBJ visuals when max_faces is None."""
    import trimesh

    if max_faces is not None and not 4 <= max_faces <= 200_000:
        raise ValueError("max_faces must be between 4 and MuJoCo's 200000-face STL limit")
    robot = ET.parse(urdf).getroot()
    template = ET.parse(control_template).getroot()
    name = robot.attrib["name"]
    root = urdf.parent.parent
    output = root / f"{name}.xml"
    meshes_dir = root / "runtime_meshes"
    meshes_dir.mkdir(exist_ok=True)
    model = ET.Element("mujoco", model=name)
    ET.SubElement(
        model, "compiler", angle="radian", meshdir="runtime_meshes", inertiafromgeom="false"
    )
    for tag in ("option", "default"):
        element = template.find(tag)
        if element is not None:
            model.append(copy.deepcopy(element))
    assets = ET.SubElement(model, "asset")
    mesh_names: dict[str, str] = {}
    provenance: list[dict[str, object]] = []
    for mesh in robot.findall(".//mesh"):
        uri = mesh.attrib["filename"]
        if uri in mesh_names:
            continue
        prefix = f"package://{name}/"
        if not uri.startswith(prefix):
            raise ValueError(f"expected a mesh inside package {name!r}: {uri}")
        source = root / uri.removeprefix(prefix)
        if source.relative_to(root).parts[0] == "collision_meshes":
            # Dedicated convex proxies are already fitted to the original CAD.
            # Visual decimation must not change their contact surface.
            mesh_names[uri] = f"collision_{source.stem}"
            ET.SubElement(
                assets, "mesh", name=mesh_names[uri], file=f"../collision_meshes/{source.name}"
            )
            continue
        geometry = trimesh.load_mesh(source, process=max_faces is not None)
        original_count = len(geometry.faces)
        if max_faces is None:
            # OBJ has no STL decoder's 200000-face cap. Merge only exactly equal
            # positions; retain every source triangle, its winding and coordinates.
            vertices, inverse = np.unique(geometry.vertices, axis=0, return_inverse=True)
            faces = inverse[geometry.faces]
            target = meshes_dir / f"{source.stem}.obj"
            with target.open("w", encoding="utf-8") as stream:
                stream.write("# Original visual geometry, without decimation. Units: metres.\n")
                np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
                np.savetxt(stream, faces + 1, fmt="f %d %d %d")
        else:
            if original_count > max_faces:
                geometry = geometry.simplify_quadric_decimation(face_count=max_faces)
            target = meshes_dir / source.name
            geometry.export(target, file_type="stl")
        mesh_names[uri] = source.stem
        ET.SubElement(assets, "mesh", name=source.stem, file=target.name)
        provenance.append(
            {
                "source": source.relative_to(root).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_faces": original_count,
                "runtime": target.relative_to(root).as_posix(),
                "runtime_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "runtime_faces": len(geometry.faces),
            }
        )

    links = {link.attrib["name"]: link for link in robot.findall("link")}
    joints = robot.findall("joint")
    child_names = {_required(joint, "child").attrib["link"] for joint in joints}
    roots = set(links) - child_names
    if len(roots) != 1:
        raise ValueError("URDF must have exactly one root link")
    actuated = [joint for joint in joints if joint.attrib["type"] != "fixed"]
    motors = template.findall("actuator/motor")
    if len(actuated) != len(motors):
        raise ValueError("URDF and control template must have the same actuator count")
    template_joints = {joint.get("name"): joint for joint in template.findall(".//joint")}
    motor_by_joint = dict(zip((joint.attrib["name"] for joint in actuated), motors, strict=True))

    def add_link(parent: ET.Element, link_name: str, joint: ET.Element | None = None) -> None:
        link = links[link_name]
        pose = (
            _pose(joint.find("origin"))
            if joint is not None
            else {"pos": _required(template, "worldbody/body").get("pos", "0 0 0")}
        )
        body = ET.SubElement(parent, "body", {"name": link_name, **pose})
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        elif joint.attrib["type"] != "fixed":
            if joint.attrib["type"] != "revolute":
                raise ValueError("only fixed/revolute joints are supported")
            joint_name = joint.attrib["name"]
            limit = _required(joint, "limit").attrib
            attributes = {
                "name": joint_name,
                "axis": _required(joint, "axis").attrib["xyz"],
                "range": f"{limit['lower']} {limit['upper']}",
            }
            source_joint = template_joints[motor_by_joint[joint_name].attrib["joint"]]
            for key in ("damping", "armature", "frictionloss"):
                if key in source_joint.attrib:
                    attributes[key] = source_joint.attrib[key]
            ET.SubElement(body, "joint", attributes)
        inertial = _required(link, "inertial")
        inertia = _required(inertial, "inertia").attrib
        tensor = np.array(
            [[float(inertia[f"i{min(a, b)}{max(a, b)}"]) for b in "xyz"] for a in "xyz"]
        )
        origin = _required(inertial, "origin")
        rotation = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        tensor = rotation @ tensor @ rotation.T
        ET.SubElement(
            body,
            "inertial",
            pos=origin.get("xyz", "0 0 0"),
            mass=_required(inertial, "mass").attrib["value"],
            fullinertia=_numbers(tensor[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]),
        )
        for kind in ("visual", "collision"):
            for index, item in enumerate(link.findall(kind)):
                geometry = _required(item, "geometry")
                shape = list(geometry)[0]
                attributes = {
                    "name": item.get("name", f"{link_name}_{kind}_{index}"),
                    "group": "2" if kind == "visual" else "3",
                    "density": "0",
                    **_pose(item.find("origin")),
                }
                if shape.tag == "mesh":
                    if shape.get("scale", "1 1 1") != "1 1 1":
                        raise ValueError("expected unscaled URDF mesh geometry")
                    attributes.update(type="mesh", mesh=mesh_names[shape.attrib["filename"]])
                elif shape.tag == "box":
                    attributes.update(
                        type="box", size=_numbers(np.fromstring(shape.attrib["size"], sep=" ") / 2)
                    )
                elif shape.tag == "cylinder":
                    attributes.update(
                        type="cylinder",
                        size=_numbers(
                            [float(shape.attrib["radius"]), float(shape.attrib["length"]) / 2]
                        ),
                    )
                elif shape.tag == "sphere":
                    attributes.update(type="sphere", size=shape.attrib["radius"])
                else:
                    raise ValueError(f"unsupported URDF geometry: {shape.tag}")
                if kind == "visual":
                    attributes.update(contype="0", conaffinity="0")
                    color = item.find("material/color")
                    if color is not None:
                        attributes["rgba"] = color.attrib["rgba"]
                ET.SubElement(body, "geom", attributes)
        for child_joint in joints:
            if _required(child_joint, "parent").attrib["link"] == link_name:
                add_link(body, _required(child_joint, "child").attrib["link"], child_joint)

    add_link(ET.SubElement(model, "worldbody"), roots.pop())
    actuator = ET.SubElement(model, "actuator")
    for joint in actuated:
        joint_name = joint.attrib["name"]
        motor = copy.deepcopy(motor_by_joint[joint_name])
        motor.set("name", joint_name)
        motor.set("joint", joint_name)
        actuator.append(motor)
    ET.indent(model, space="  ")
    ET.ElementTree(model).write(output, encoding="unicode")
    scene = ET.parse(control_template.with_name("scene.xml")).getroot()
    scene.set("model", f"{name}_scene")
    _required(scene, "include").set("file", output.name)
    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(root / "scene.xml", encoding="unicode")
    manifest_path = root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        robot=name,
        source_asset_name=source_name or manifest.get("source_asset_name", name),
        visual_mode="original_obj" if max_faces is None else "decimated_stl",
        meshes=provenance,
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--control-template", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=20_000)
    parser.add_argument(
        "--full-resolution-visuals",
        action="store_true",
        help="preserve all original visual triangles as OBJ instead of decimating to STL",
    )
    parser.add_argument("--source-name")
    args = parser.parse_args()
    print(
        build_model(
            args.urdf,
            args.control_template,
            max_faces=None if args.full_resolution_visuals else args.max_faces,
            source_name=args.source_name,
        )
    )
