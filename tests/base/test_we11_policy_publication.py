from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.publish_we11_to_play import (
    validate_or_adopt_export,
    validate_play_config,
    validate_play_target_drift,
)

from unilab.training import policy_export

EXPECTED_CONTRACT = {
    "input": {"name": "obs", "shape": [1, 145], "dtype": "tensor(float)"},
    "output": {"name": "act", "shape": [1, 6], "dtype": "tensor(float)"},
    "history_layout": "term-major",
    "history_length": 5,
    "history_term_dims": [3, 3, 4, 6, 6, 2, 2, 3],
}


def _make_run(tmp_path: Path, task: str = "DR002JoystickFlatWE11") -> tuple[Path, Path, Path]:
    run = tmp_path / "2026-08-21_17-11-53_mujoco"
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps({"run": {"task": task, "git": {"commit": "abc", "dirty": False}}}),
        encoding="utf-8",
    )
    (run / "run_summary.json").write_text('{"iteration": 499}\n', encoding="utf-8")
    checkpoint = run / "model_499.pt"
    checkpoint.write_bytes(b"checkpoint")
    onnx = run / "policy.onnx"
    onnx.write_bytes(b"onnx")
    return run, checkpoint, onnx


def test_export_manifest_records_explicit_checkpoint(monkeypatch, tmp_path: Path) -> None:
    run, checkpoint, onnx = _make_run(tmp_path)
    monkeypatch.setattr(policy_export, "inspect_onnx_contract", lambda _: EXPECTED_CONTRACT)

    manifest = policy_export.build_policy_export_manifest(
        run_dir=run,
        checkpoint=checkpoint,
        onnx_path=onnx,
    )

    assert manifest["source"]["checkpoint"] == "model_499.pt"
    assert manifest["source"]["checkpoint_iteration"] == 499
    assert manifest["source"]["checkpoint_sha256"] == policy_export.sha256_file(checkpoint)
    assert manifest["artifact"]["sha256"] == policy_export.sha256_file(onnx)
    assert manifest["contract"] == EXPECTED_CONTRACT


def test_export_manifest_accepts_getup_shared_policy(monkeypatch, tmp_path: Path) -> None:
    run, checkpoint, onnx = _make_run(tmp_path, task="DR002JoystickGetupWE11")
    monkeypatch.setattr(policy_export, "inspect_onnx_contract", lambda _: EXPECTED_CONTRACT)

    manifest = policy_export.build_policy_export_manifest(
        run_dir=run, checkpoint=checkpoint, onnx_path=onnx
    )
    assert manifest["source"]["task"] == "DR002JoystickGetupWE11"


def test_export_manifest_rejects_rough_task(monkeypatch, tmp_path: Path) -> None:
    run, checkpoint, onnx = _make_run(tmp_path, task="DR002JoystickRoughWE11")
    monkeypatch.setattr(policy_export, "inspect_onnx_contract", lambda _: EXPECTED_CONTRACT)

    with pytest.raises(ValueError, match="DR002JoystickGetupWE11"):
        policy_export.build_policy_export_manifest(
            run_dir=run,
            checkpoint=checkpoint,
            onnx_path=onnx,
        )


def test_onnx_contract_rejects_135_input(monkeypatch, tmp_path: Path) -> None:
    class FakeSession:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="obs", shape=[1, 135], type="tensor(float)")]

        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="act", shape=[1, 6], type="tensor(float)")]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported WE11 ONNX contract"):
        policy_export.inspect_onnx_contract(onnx)


def test_existing_export_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    run, checkpoint, onnx = _make_run(tmp_path)
    manifest = {
        "source": {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": "0" * 64,
        },
        "artifact": {"sha256": policy_export.sha256_file(onnx)},
    }
    (run / "policy_export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint hash no longer matches"):
        validate_or_adopt_export(run, checkpoint)


def test_play_target_drift_rejects_manual_replacement(tmp_path: Path) -> None:
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"manual")
    old_manifest = {"artifacts": {"onnx": {"sha256": "0" * 64}}}
    with pytest.raises(RuntimeError, match="untracked manual replacement"):
        validate_play_target_drift(policy, "1" * 64, old_manifest)


def test_play_config_contract_accepts_145_and_rejects_135(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """num_observations: 145
observations: ["ang_vel", "projected_gravity", "dof_pos_without_wheel_compact", "dof_vel_mapped", "actions", "wing_angle", "wing_vel", "commands"]
observations_history: [4, 3, 2, 1, 0]
observations_history_priority: term
""",
        encoding="utf-8",
    )
    validate_play_config(config)

    config.write_text(config.read_text(encoding="utf-8").replace("145", "135"), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 145"):
        validate_play_config(config)
