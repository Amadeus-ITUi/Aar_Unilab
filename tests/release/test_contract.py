import json

import numpy as np
import pytest

from unilab.release_contract import create_release, load_manifest


def test_example_contract_supports_dynamic_tensor_lists():
    manifest = load_manifest("releases/examples/we11_flat/deployment_manifest.json")
    assert manifest["schema"] == "aar-unilab.actor.v1"
    assert len(manifest["policy"]["inputs"]) >= 1


def test_unknown_contract_fails_closed(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "aar-unilab.actor.v999"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_manifest(path)


def test_absolute_artifact_path_is_rejected(tmp_path):
    payload = json.loads(
        open("releases/examples/we11_flat/deployment_manifest.json", encoding="utf-8").read()
    )
    payload["artifacts"]["policy_path"] = "/tmp/policy.onnx"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="release-relative"):
        load_manifest(path)


def test_release_contains_golden_tensors_and_hashes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.onnx").write_bytes(b"test-policy")
    (source / "runtime.yaml").write_text("robot: test\n", encoding="utf-8")
    manifest = json.loads(
        open("releases/examples/we11_flat/deployment_manifest.json", encoding="utf-8").read()
    )
    release = create_release(
        tmp_path / "release",
        onnx=source / "policy.onnx",
        runtime_config=source / "runtime.yaml",
        manifest=manifest,
        golden_inputs={"obs": np.zeros((1, 135), dtype=np.float32)},
        golden_outputs={"act": np.zeros((1, 6), dtype=np.float32)},
    )
    assert (release / "golden_inputs/obs.npy").is_file()
    assert "golden_outputs/act.npy" in (release / "SHA256SUMS").read_text()
