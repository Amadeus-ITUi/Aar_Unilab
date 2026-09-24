import ast
from pathlib import Path

from unilab.catalog import catalog
from unilab.catalog.launch import DEFAULTS, build_launch
from unilab.envs.locomotion.pe04.config import ROOT, load_config


def test_independent_runtime_sources_and_configs():
    paths = list((ROOT / "src/unilab/envs/locomotion/pe04").glob("*.py"))
    paths += list((ROOT / "src/unilab/algos/torch/pe04").glob("*.py"))
    paths += [ROOT / "scripts/train_pe04.py", ROOT / "src/unilab/adapters/pe04_ppo.py"]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [n.name for n in node.names]
            if isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            assert all(
                not any(part in name.split(".") for part in ("pe01", "pe02", "pe03", "references"))
                for name in imports
            ), path
    for path in (ROOT / "conf/pe04").rglob("*.yaml"):
        assert "conf/pe03" not in path.read_text() and "/pe03/" not in path.read_text()
    cpp = (ROOT / "sim2sim/include/aar/pe04_runtime.hpp").read_text()
    assert "pe03_runtime" not in cpp and "references/" not in cpp
    assets = ROOT / "src/unilab/assets/robots/pe04"
    assert not any(p.is_symlink() for p in assets.rglob("*"))
    assert '<include file="pe04.xml"' in (assets / "scene.xml").read_text()


def test_profiles_catalog_and_owner_defaults():
    for name, iterations in [("pe04_tron1", 2000), ("pe04_smoke", 10)]:
        command, _, _ = build_launch(DEFAULTS, name, [], environment={})
        cfg = load_config(command[2:])
        assert Path(command[1]).name == "train_pe04.py"
        assert cfg.algo.max_iterations == iterations
        assert cfg.algo.num_steps_per_env == 24
    result = catalog.resolve(
        dict(
            robot="pe04",
            task="pe04_flat",
            observation="pe04_tron1_v1",
            policy="pe04_encoder_mlp",
            algorithm="pe04_custom_ppo",
            simulator="mujoco",
        )
    )
    assert result.observation.history == 10 and result.observation.critic_dim == 267
