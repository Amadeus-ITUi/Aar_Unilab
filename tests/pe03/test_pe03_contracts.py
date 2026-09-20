"""CNC source preservation and independent formal PE03 contracts."""

import ast
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

from unilab.adapters.pe03_ppo import load_policy
from unilab.algos.torch.pe03.runner import PE03Runner
from unilab.catalog import catalog
from unilab.catalog.cli import build_legacy_command
from unilab.envs.locomotion.pe03.config import ROOT, load_config
from unilab.envs.locomotion.pe03.play_env import make_play_env

ASSET = ROOT / "src/unilab/assets/robots/pe03"


def test_playback_keeps_full_visuals_and_uses_local_appearance():
    env = make_play_env(load_config(["+experiment=walking", "training.mujoco_threads=1"]))
    try:
        env.reset()
        model = env.model
        assert model.nmeshface > 377070
        floor = model.geom("floor").id
        assert model.mat(model.geom_matid[floor]).name == "pe03_play_ground"
        assert model.texture("pe03_play_sky").id >= 0
    finally:
        env.close()


@pytest.mark.parametrize("version,experiment,frame", [(2, "standing", 30), (3, "walking", 24)])
def test_formal_catalog_and_independent_configuration(version, experiment, frame):
    selectors = {
        "robot": "pe03",
        "task": "pe03_flat",
        "observation": f"pe03_v{version}",
        "policy": "pe03_encoder_mlp",
        "algorithm": "pe03_custom_ppo",
        "simulator": "mujoco",
    }
    selected = catalog.resolve(selectors)
    assert selected.algorithm.adapter == "unilab.adapters.pe03_ppo"
    assert selected.task.owner_config == "pe03/task/pe03_flat"
    command = build_legacy_command("train", [f"{k}={v}" for k, v in selectors.items()], ROOT)
    assert command[1] == str(ROOT / "scripts/train_pe03.py")
    config = load_config([f"+experiment={experiment}"])
    assert config.env.frame_size == frame and config.env.history_length == 10
    assert config.control.physics_hz == config.control.motor_hz == 400
    assert config.control.policy_hz == 50
    assert config.reward.base_height_target is None
    assert not config.domain_rand.enabled and not config.noise.enabled
    assert list(config.env.joint_reset_range) == [0, 0]
    assert "pe03_v1" not in catalog.observations
    with pytest.raises(ValueError, match="observation"):
        catalog.resolve({**selectors, "observation": "pe02_v3"})


def test_production_modules_do_not_import_other_prototypes():
    paths = [ROOT / "scripts/train_pe03.py", ROOT / "src/unilab/adapters/pe03_ppo.py"]
    for folder in ("algos/torch/pe03", "envs/locomotion/pe03"):
        paths.extend((ROOT / "src/unilab" / folder).glob("*.py"))
    for path in paths:
        assert not path.is_symlink()
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            assert not any(
                token in module for module in modules for token in ("pe01", "pe02", "references")
            ), path


@pytest.mark.parametrize("robot", ["pe01", "pe02"])
def test_foreign_checkpoints_rejected_by_play_and_resume(tmp_path, robot):
    path = tmp_path / "foreign.pt"
    torch.save({"robot_id": robot, "schema": f"{robot}.training.v2"}, path)
    with pytest.raises(ValueError, match="robot_id"):
        load_policy(path)
    cfg = load_config(
        [
            "+experiment=standing",
            "algo.num_envs=2",
            "training.device=cpu",
            "training.logger=none",
            "training.mujoco_threads=1",
            f"training.resume={path}",
        ]
    )
    with pytest.raises(ValueError, match="formal PE03 resume"):
        PE03Runner(cfg, tmp_path / "rejected")


def test_cnc_source_preserved_except_authorized_mass_and_names():
    manifest = json.loads((ASSET / "asset_manifest.json").read_text())
    assert manifest["source_asset_name"] == "点足CNC"
    original_path = ASSET / "source/original.urdf"
    assert (
        hashlib.sha256(original_path.read_bytes()).hexdigest()
        == (manifest["source_files_sha256"]["urdf/点足CNC .urdf"])
    )
    original = ET.parse(original_path)
    current = ET.parse(ASSET / "urdf/pe03.urdf")
    mass = 0.0
    for link in original.findall("link"):
        updated = current.find(f"link[@name='{link.get('name')}']")
        for tag in ("origin", "inertia"):
            assert link.find(f"inertial/{tag}").attrib == updated.find(f"inertial/{tag}").attrib
        value = float(updated.find("inertial/mass").get("value"))
        if link.get("name") in ("L_calf_Link", "R_calf_link"):
            assert value == 0.118
        else:
            assert value == float(link.find("inertial/mass").get("value"))
        mass += value
    assert mass == pytest.approx(3.26689, abs=1e-12)
    for old, new in zip(original.findall("joint"), current.findall("joint"), strict=True):
        assert ET.tostring(old).strip() == ET.tostring(new).strip()
    assert manifest["visual_mode"] == "original_obj"
    assert len(manifest["meshes"]) == 9
    for mesh in manifest["meshes"]:
        for kind in ("source", "runtime"):
            path = ASSET / mesh[kind]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == mesh[f"{kind}_sha256"]
        assert mesh["runtime_faces"] == mesh["source_faces"] > 0
        assert Path(mesh["runtime"]).suffix == ".obj"
    for mesh in current.findall(".//mesh"):
        uri = mesh.get("filename")
        assert uri.startswith("package://pe03/")
        assert (ASSET / uri.removeprefix("package://pe03/")).is_file()


def test_cnc_collision_reduction_and_support_torque_budget():
    manifest = json.loads((ASSET / "asset_manifest.json").read_text())
    parts = [part for link in manifest["collision"]["links"].values() for part in link["parts"]]
    assert len(parts) == 26
    for part in parts:
        if part["kind"] == "hull":
            # This bounds decimation relative to each component hull, not CAD concavity.
            assert part["simplification"]["maximum_surface_error_m"] <= 0.0005
    report = json.loads((ASSET / "analysis/standing_pose.json").read_text())
    assert np.linalg.norm(report["com_minus_support_midpoint_xy_m"]) <= 0.0005
    assert np.max(np.abs(np.array(report["support_centers_world_m"])[:, 2])) < 1e-6
    cfg = load_config(["+experiment=standing"])
    assert np.all(np.abs(report["estimated_static_joint_torque_nm"]) < cfg.control.torque_limits)
