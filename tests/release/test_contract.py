import json

import pytest

from unilab.release_contract import load_manifest


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
