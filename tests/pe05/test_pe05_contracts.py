"""Independent ownership, source provenance and launch contracts."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from unilab.adapters.pe05_ppo import load_policy
from unilab.algos.torch.pe05.runner import PE05Runner
from unilab.catalog import catalog
from unilab.catalog.launch import DEFAULTS, build_launch
from unilab.envs.locomotion.pe05.config import ROOT, load_config
from unilab.envs.locomotion.pe05.contracts import resolve_checkpoint


def test_no_predecessor_runtime_dependencies():
    paths = list((ROOT / "src/unilab/envs/locomotion/pe05").glob("*.py"))
    paths += list((ROOT / "src/unilab/algos/torch/pe05").glob("*.py"))
    paths += [ROOT / "scripts/train_pe05.py", ROOT / "src/unilab/adapters/pe05_ppo.py"]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert not {"pe01", "pe02", "pe03", "pe04", "references"}.intersection(
                    name.split(".")
                )
    for path in (ROOT / "conf/pe05").rglob("*.yaml"):
        assert not any(f"/pe0{i}/" in path.read_text() for i in range(1, 5))
    cpp = (ROOT / "sim2sim/include/aar/pe05_runtime.hpp").read_text()
    assert not any(f"pe0{i}_runtime" in cpp for i in range(1, 5))
    assert "references/" not in cpp


def test_assets_are_frozen_from_latest_pe03():
    assets = ROOT / "src/unilab/assets/robots/pe05"
    source = ROOT / "src/unilab/assets/robots/pe03"
    manifest = json.loads((assets / "provenance.json").read_text())
    rename = {
        "pe03_joint_limits.xml": "pe05.xml",
        "scene_joint_limits.xml": "scene.xml",
        "urdf/pe03_joint_limits.urdf": "urdf/pe05.urdf",
    }
    for name, digest in manifest["sha256"].items():
        original = source / name
        frozen = assets / rename.get(name, name)
        assert not frozen.is_symlink()
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
        if original.suffix in {".xml", ".urdf"}:
            assert frozen.read_text() == original.read_text().replace(
                "pe03_joint_limits", "pe05"
            ).replace("pe03", "pe05")
        else:
            assert frozen.read_bytes() == original.read_bytes()
    provenance = json.loads((ROOT / "docs/PE05_SOURCE_MANIFEST.json").read_text())
    assert provenance["commit"] == "17a4b93a7480233de07d52ccb5c790cfdb1d0286"
    assert not provenance["files"]["leggedgym/legged_gym/envs/dragon_flat/dragon_flat.py"][
        "modified"
    ]


def test_catalog_and_launch_profiles():
    for profile, iterations, count, save in (("pe05", 15000, 4096, 400), ("pe05_smoke", 10, 32, 5)):
        command, _, _ = build_launch(DEFAULTS, profile, [], environment={})
        assert Path(command[1]).name == "train_pe05.py"
        cfg = load_config(command[2:])
        assert (cfg.algo.max_iterations, cfg.algo.num_envs, cfg.algo.save_interval) == (
            iterations,
            count,
            save,
        )
        catalog.resolve(
            dict(
                robot="pe05",
                task="pe05_flat",
                observation="pe05_v1",
                policy="pe05_encoder_mlp",
                algorithm="pe05_custom_ppo",
                simulator="mujoco",
            )
        )
    assert load_config().algo.num_envs == 8192
    with pytest.raises(ValueError, match="history"):
        load_config(["env.history_length=9"])


def test_checkpoint_load_rejects_asset_layout_and_config_changes(tmp_path):
    cfg = load_config(
        [
            "algo.num_envs=2",
            "algo.num_steps_per_env=2",
            "algo.max_iterations=1",
            "training.device=cpu",
            "training.logger=none",
            "training.mujoco_threads=1",
        ]
    )
    runner = PE05Runner(cfg, tmp_path / "run")
    checkpoint = tmp_path / "run/model_0.pt"
    try:
        runner.save(checkpoint)
    finally:
        runner.close()
    state = torch.load(checkpoint, weights_only=True)
    for key, value, error in (
        ("robot_id", "pe01", "robot_id"),
        ("asset_sha256", "changed", "assets"),
        ("observation_layout", {}, "layout"),
        ("training_contract", {}, "contract"),
    ):
        altered = {**state, key: value}
        torch.save(altered, tmp_path / "invalid.pt")
        with pytest.raises(ValueError, match=error):
            load_policy(tmp_path / "invalid.pt")
    state["training_config"]["control"]["action_scale"] = 0.5
    torch.save(state, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="contract"):
        load_policy(tmp_path / "invalid.pt")
    assert resolve_checkpoint(-1, tmp_path) == checkpoint
    np.testing.assert_equal(load_policy(checkpoint).critic[0].in_features, 39)
