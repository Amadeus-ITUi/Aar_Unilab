#!/usr/bin/env python3
"""Build the asset-bound, collision-checked PE03 gait workspace offline."""

import argparse
from pathlib import Path

from unilab.base.backend.mujoco.foot_workspace import build_workspace
from unilab.envs.locomotion.pe03.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=41)
    parser.add_argument("--variant", choices=("all", "gait", "flat"), default="all")
    args = parser.parse_args()
    cfg = load_config(["+experiment=gait_fixed"])
    variants = {
        "gait": (ROOT / cfg.env.model_path, ROOT / cfg.env.workspace_path),
        "flat": (
            ROOT / "src/unilab/assets/robots/pe03/scene.xml",
            ROOT / "src/unilab/assets/robots/pe03/analysis/gait_workspace.npz",
        ),
    }
    for name, (scene, output) in variants.items():
        if args.variant not in ("all", name):
            continue
        print(
            name,
            build_workspace(
                scene,
                output,
                cfg.env.body_names,
                cfg.env.foot_names,
                cfg.env.reference_points,
                resolution=args.resolution,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
