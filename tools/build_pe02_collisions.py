#!/usr/bin/env python3
"""Build PE02 collision proxies from original CAD, independent of visual decimation.

Requires the same offline trimesh dependency as the URDF importer. This never
estimates mass/inertia from proxies. Recipe component indices refer to connected
CAD surfaces sorted by descending surface area, not arbitrary triangle ranges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/unilab/assets/robots/pe02"
MIRROR = np.array([1, -1, 1])


def numbers(value) -> str:
    return " ".join(format(float(x), ".16g") for x in value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pieces(mesh, spec):
    cuts = spec.get("cuts", [])
    direction = np.asarray(spec.get("cut_axis", [0, 0, 1]), dtype=float)
    direction /= np.linalg.norm(direction)
    vertices = mesh.vertices
    a, b = vertices[mesh.edges_unique[:, 0]], vertices[mesh.edges_unique[:, 1]]
    for low, high in zip([-np.inf, *cuts], [*cuts, np.inf], strict=True):
        if not cuts:
            yield mesh
            continue
        projection = vertices @ direction
        points = [vertices[(projection >= low) & (projection <= high)]]
        # Include triangle-edge intersections so adjacent slabs share exactly the
        # same cut surface. Selecting only existing vertices would leave gaps.
        for cut in (low, high):
            if np.isfinite(cut):
                da, db = a @ direction - cut, b @ direction - cut
                crossing = da * db < 0
                points.append(
                    a[crossing]
                    + (b[crossing] - a[crossing])
                    * (da[crossing] / (da[crossing] - db[crossing]))[:, None]
                )
        part = trimesh.Trimesh(
            vertices=np.unique(np.concatenate(points), axis=0), faces=[], process=False
        )
        if len(part.vertices) < 4:
            raise ValueError(f"empty/degenerate cut: {spec['name']} {low} {high}")
        yield part


def simplify_hull(hull, tolerance: float, spec: dict):
    """Inscribed convex approximation with a checked Euclidean error bound.

    Distance to a convex set is convex: checking every original hull vertex
    bounds the distance for its entire interior and surface. Retained vertices
    also ensure the new hull cannot expand beyond the original convex shape.
    The optional contact cap retains the original support facet and nearby CAD
    vertices instead of flattening the curved foot into a box or sphere.
    """
    if tolerance <= 0:
        raise ValueError("hull_max_error_m must be positive")
    vertices = hull.vertices
    selected = set(np.r_[vertices.argmax(axis=0), vertices.argmin(axis=0)].tolist())
    normal = spec.get("protected_normal")
    protected = np.array([], dtype=int)
    if normal is not None:
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        depth = float(spec["protected_depth_m"])
        if depth < 0:
            raise ValueError("protected_depth_m must be nonnegative")
        projection = vertices @ normal
        protected = np.flatnonzero(projection.max() - projection <= depth + 1e-10)
        selected.update(protected.tolist())
    while True:
        indices = sorted(selected)
        candidate = ConvexHull(vertices[indices])
        violation = (vertices @ candidate.equations[:, :3].T + candidate.equations[:, 3]).max(
            axis=1
        )
        worst = int(violation.argmax())
        if violation[worst] > tolerance:
            selected.add(worst)
            continue
        result = trimesh.Trimesh(
            vertices=vertices[indices], faces=candidate.simplices, process=False
        ).convex_hull
        # Plane violation is only a lower bound near edges. Verify actual
        # point-to-surface distance before accepting the approximation.
        outside = np.flatnonzero(violation > 1e-10)
        distances = np.zeros(len(vertices))
        for start in range(0, len(outside), 32):
            batch = outside[start : start + 32]
            _, distance, _ = trimesh.proximity.closest_point_naive(result, vertices[batch])
            distances[batch] = distance
        worst = int(distances.argmax())
        if distances[worst] <= tolerance + 1e-12:
            return result, {
                "original_vertices": len(vertices),
                "maximum_surface_error_m": float(distances.max()),
                "error_limit_m": tolerance,
                "protected_vertices": len(protected),
                "inscribed": True,
            }
        selected.add(worst)


def build(asset: Path = ASSET) -> dict:
    recipe_path = asset / "collision_recipe.json"
    recipe = json.loads(recipe_path.read_text())
    source_manifest = json.loads((asset / "asset_manifest.json").read_text())
    previous_files = {
        part["file"]
        for link in source_manifest.get("collision", {}).get("links", {}).values()
        for part in link["parts"]
        if "file" in part
    }
    source_hashes = {mesh["source"]: mesh["source_sha256"] for mesh in source_manifest["meshes"]}
    for source, expected in source_hashes.items():
        if sha256(asset / source) != expected:
            raise ValueError(
                f"source CAD changed; review the collision recipe and source manifest: {source}"
            )
    model = ET.parse(asset / "pe02.xml")
    urdf = ET.parse(asset / "urdf/pe02.urdf")
    assets = model.find("asset")
    model.getroot().find("compiler").set("inertiafromgeom", "false")
    for mesh in list(assets):
        if mesh.get("name", "").startswith("collision_"):
            assets.remove(mesh)
    directory = asset / "collision_meshes"
    directory.mkdir(exist_ok=True)
    report = {"version": recipe["version"], "recipe_sha256": sha256(recipe_path), "links": {}}
    for source_name, entry in recipe["links"].items():
        source = asset / "meshes" / f"{source_name}.STL"
        cad = trimesh.load_mesh(source, process=True)
        components = sorted(cad.split(only_watertight=False), key=lambda m: -m.area)
        fitted = []
        for spec in entry["parts"]:
            if spec.get("preserve_contact") and (
                spec["kind"] != "hull" or spec.get("cuts") or "components" in spec
            ):
                raise ValueError(
                    "contact-preserving parts must start from the unpartitioned original mesh"
                )
            selected = (
                trimesh.util.concatenate([components[i] for i in spec["components"]])
                if "components" in spec
                else cad
            )
            for index, part in enumerate(pieces(selected, spec)):
                fitted.append((f"{spec['name']}_{index}", spec, part))
        for target, reflect in [
            (source_name, False),
            *([(entry["mirror_to"], True)] if "mirror_to" in entry else []),
        ]:
            body = model.find(f".//body[@name='{target}']")
            link = urdf.find(f"link[@name='{target}']")
            if body is None or link is None:
                raise ValueError(f"missing model/URDF link: {target}")
            for geom in list(body.findall("geom")):
                if geom.get("group") == "3":
                    body.remove(geom)
            for collision in list(link.findall("collision")):
                link.remove(collision)
            part_reports = []
            for label, spec, part in fitted:
                name = f"{target}_collision_{label}"
                kind = spec["kind"]
                position, rpy = np.zeros(3), np.zeros(3)
                attributes = {
                    "name": name,
                    "group": "3",
                    "density": "0",
                    "rgba": "0.2 0.65 0.9 0.35",
                }
                collision = ET.SubElement(link, "collision", name=name)
                origin = ET.SubElement(collision, "origin")
                geometry = ET.SubElement(collision, "geometry")
                record = {
                    "name": name,
                    "kind": kind,
                    "source_components": spec.get("components", "all"),
                }
                if kind == "hull":
                    hull = part.convex_hull
                    tolerance = spec.get("hull_max_error_m", recipe.get("hull_max_error_m"))
                    if tolerance is not None:
                        if spec.get("preserve_contact") and "protected_normal" not in spec:
                            raise ValueError("simplifying a contact hull requires a protected cap")
                        hull, simplification = simplify_hull(hull, float(tolerance), spec)
                        record["simplification"] = simplification
                    if reflect:
                        transform = np.eye(4)
                        transform[:3, :3] = np.diag(MIRROR)
                        hull.apply_transform(transform)
                    path = directory / f"{target}_{label}.STL"
                    hull.export(path, file_type="stl")
                    mesh_name = f"collision_{target}_{label}"
                    ET.SubElement(
                        assets, "mesh", name=mesh_name, file=f"../collision_meshes/{path.name}"
                    )
                    attributes.update(type="mesh", mesh=mesh_name)
                    ET.SubElement(
                        geometry, "mesh", filename=f"package://pe02/collision_meshes/{path.name}"
                    )
                    record.update(
                        file=path.relative_to(asset).as_posix(),
                        sha256=sha256(path),
                        faces=len(hull.faces),
                        vertices=len(hull.vertices),
                        volume_m3=float(hull.volume),
                    )
                elif kind in ("box", "cylinder"):
                    bounds = part.bounds
                    position = bounds.mean(axis=0)
                    if kind == "box":
                        half_size = np.diff(bounds, axis=0)[0] / 2
                        attributes.update(type="box", size=numbers(half_size))
                        ET.SubElement(geometry, "box", size=numbers(2 * half_size))
                    else:
                        axis = int(spec["axis"])
                        radial = [i for i in range(3) if i != axis]
                        radius = np.linalg.norm((part.vertices - position)[:, radial], axis=1).max()
                        half_length = np.diff(bounds, axis=0)[0, axis] / 2
                        rpy = np.array([[0, np.pi / 2, 0], [-np.pi / 2, 0, 0], [0, 0, 0]])[axis]
                        if reflect and axis == 1:
                            rpy *= -1
                        attributes.update(type="cylinder", size=numbers([radius, half_length]))
                        ET.SubElement(
                            geometry, "cylinder", radius=str(radius), length=str(2 * half_length)
                        )
                    if reflect:
                        position *= MIRROR
                    record.update(position_m=position.tolist(), size=attributes["size"])
                else:
                    raise ValueError(f"unsupported proxy: {kind}")
                origin.set("xyz", numbers(position))
                origin.set("rpy", numbers(rpy))
                attributes.update(
                    pos=numbers(position),
                    quat=numbers(Rotation.from_euler("xyz", rpy).as_quat(scalar_first=True)),
                )
                # Keep geoms before child bodies for readable MJCF.
                body.insert(
                    list(body).index(body.find("body"))
                    if body.find("body") is not None
                    else len(body),
                    ET.Element("geom", attributes),
                )
                part_reports.append(record)
            report["links"][target] = {
                "source": source.relative_to(asset).as_posix(),
                "source_sha256": sha256(source),
                "mirrored": reflect,
                "parts": part_reports,
            }
    # Mass and transforms are copied unchanged from the existing models.
    for tree, path in ((model, asset / "pe02.xml"), (urdf, asset / "urdf/pe02.urdf")):
        ET.indent(tree, space="  ")
        tree.write(path, encoding="unicode")
    manifest_path = asset / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["collision"] = report
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    current_files = {
        part["file"]
        for link in report["links"].values()
        for part in link["parts"]
        if "file" in part
    }
    for stale in previous_files - current_files:
        path = asset / stale
        if path.parent.resolve() != directory.resolve():
            raise ValueError(f"generated collision path escapes its directory: {stale}")
        path.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ASSET)
    args = parser.parse_args()
    report = build(args.asset)
    print(
        json.dumps({name: len(link["parts"]) for name, link in report["links"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
