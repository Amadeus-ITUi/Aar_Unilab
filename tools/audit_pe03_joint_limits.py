#!/usr/bin/env python3
"""Offline hip/leg collision sweeps of the PE03 mechanical joint envelope.

These are configuration slices, not a certified six-dimensional collision-free
controller. The other leg is held at home or moved in mirror symmetry explicitly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from unilab.base.backend.mujoco.batched_robot import compile_robot_scene
from unilab.base.backend.mujoco.foot_workspace import workspace_fingerprint
from unilab.envs.locomotion.pe03.config import ROOT, load_config


class HipSweep:
    def __init__(self, scene: Path, bodies: tuple[str, ...]):
        self.model = compile_robot_scene(scene, bodies, visual=False)
        self.data = mujoco.MjData(self.model)
        self.home = self.model.key("home").qpos.copy()

    def contact(self, hip: float, thigh: float, calf: float, mode: str):
        m, d = self.model, self.data
        d.qpos[:] = self.home
        d.qpos[2] = 1.0  # Exclude the floor; preserve all enabled robot contact pairs.
        if mode in {"left", "mirrored"}:
            d.qpos[7:10] = [hip, thigh, calf]
        if mode in {"right", "mirrored"}:
            d.qpos[10:13] = [-hip, -thigh, -calf]
        mujoco.mj_fwdPosition(m, d)
        contacts = [
            c
            for c in d.contact
            if m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2] and c.dist < -1e-6
        ]
        if not contacts:
            return None
        c = min(contacts, key=lambda x: x.dist)
        return {
            "bodies": [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)],
            "geoms": [m.geom(g).name for g in (c.geom1, c.geom2)],
            "penetration_m": float(-c.dist),
        }

    def boundary(self, thigh: float, calf: float, mode: str):
        if contact := self.contact(0, thigh, calf, mode):
            return {"free_at_zero": False, "free_hip": None, "contact": contact}
        free = 0.0
        # First boundary connected to zero: do not jump across colliding intervals.
        for hip in np.linspace(0, -0.785, 158)[1:]:
            if contact := self.contact(float(hip), thigh, calf, mode):
                blocked = float(hip)
                for _ in range(10):
                    middle = (free + blocked) / 2
                    if collision := self.contact(middle, thigh, calf, mode):
                        blocked, contact = middle, collision
                    else:
                        free = middle
                return {
                    "free_at_zero": True,
                    "free_hip": float(free),
                    "colliding_hip": blocked,
                    "mechanical_endpoint_reached": False,
                    "contact": contact,
                }
            free = float(hip)
        return {
            "free_at_zero": True,
            "free_hip": free,
            "mechanical_endpoint_reached": True,
            "contact": None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=25)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/unilab/assets/robots/pe03/analysis/joint_limits"
    )
    args = parser.parse_args()
    cfg = load_config(["+experiment=gait_fixed"])
    scene = ROOT / cfg.env.model_path
    sweep = HipSweep(scene, tuple(cfg.env.body_names))
    modes = ("left", "right", "mirrored")
    thighs = np.linspace(-1.658, 0, args.resolution)
    calves = np.linspace(-1.99, 0, args.resolution)
    tables = np.full((3, len(thighs), len(calves)), np.nan)
    counts = {}
    for index, mode in enumerate(modes):
        pairs: Counter[str] = Counter()
        for i, thigh in enumerate(thighs):
            for j, calf in enumerate(calves):
                result = sweep.boundary(float(thigh), float(calf), mode)
                if result["free_at_zero"]:
                    tables[index, i, j] = result["free_hip"]
                if result["contact"]:
                    pairs[" / ".join(result["contact"]["bodies"])] += 1
        counts[mode] = dict(pairs)
        print(mode, "zero-clear slices", np.isfinite(tables[index]).sum(), flush=True)
    examples = []
    for thigh, calf in [tuple(sweep.home[8:10]), (-0.1, -0.2), (-0.8, -1.4), (-1.0, -0.2)]:
        examples.append(
            {
                "left_thigh_calf_rad": [float(thigh), float(calf)],
                **{mode: sweep.boundary(float(thigh), float(calf), mode) for mode in modes},
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scene": str(scene.relative_to(ROOT)),
        "fingerprint": workspace_fingerprint(scene),
        "joint_ranges": sweep.model.jnt_range[1:].tolist(),
        "home": sweep.home.tolist(),
        "resolution": args.resolution,
        "penetration_threshold_m": 1e-6,
        "scan_step_rad": 0.005,
        "boundary_bisections": 10,
        "scope": "Static enabled collision proxies; floor removed. Right angles use mirrored signs. Other leg at home except mirrored mode. No dynamic or full 6D guarantee.",
        "examples": examples,
        "first_contact_pairs": counts,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "slices.npz", thighs=thighs, calves=calves, inward=tables)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    color = plt.colormaps["viridis"].copy()
    color.set_bad("#cccccc")
    for index, (ax, mode) in enumerate(zip(axes, modes, strict=True)):
        plot = ax.pcolormesh(
            np.rad2deg(calves),
            np.rad2deg(thighs),
            np.rad2deg(tables[index]),
            shading="nearest",
            cmap=color,
            vmin=-45,
            vmax=0,
        )
        ax.plot(*np.rad2deg(sweep.home[[9, 8]]), "r+", markersize=12)
        ax.set(title=mode, xlabel="Calf angle, left convention (deg)", ylabel="Thigh angle (deg)")
    fig.colorbar(plot, ax=axes, label="Inward hip boundary (deg, left convention)")
    fig.suptitle("PE03 hip collision slices: gray = collision already at hip 0; + = home leg shape")
    fig.savefig(args.output / "hip_collision_slices.png", dpi=160)
    fig.savefig(args.output / "hip_collision_slices.svg")
    print(json.dumps(examples, indent=2), flush=True)


if __name__ == "__main__":
    main()
