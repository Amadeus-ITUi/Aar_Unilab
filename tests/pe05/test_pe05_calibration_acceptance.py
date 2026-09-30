"""Reject a bad command group even when an aggregate metric looks successful."""

import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "tools/evaluate_pe05_calibration.py"
spec = importlib.util.spec_from_file_location("pe05_calibration_evaluation", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def healthy_metrics():
    metrics = {}
    for command in module.COMMANDS:
        for key in module.LIMITS:
            metrics[f"evaluation/{command}/{key}"] = 0.0
        metrics[f"evaluation/{command}/episode_seconds"] = 20.0
    return metrics


def test_every_command_must_pass():
    metrics = healthy_metrics()
    assert module.acceptance(metrics)["passed"]
    metrics["evaluation/left/height_error_m"] = 0.021
    metrics["evaluation/height_error_m"] = 0.003
    result = module.acceptance(metrics)
    assert not result["passed"]
    assert result["groups"]["left"]["failed_metrics"] == ["height_error_m"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_missing_or_invalid_evidence_is_not_a_pass(value):
    metrics = healthy_metrics()
    metrics["evaluation/standing/episode_seconds"] = value
    assert not module.acceptance(metrics)["passed"]


def test_survival_alone_does_not_pass_tracking():
    metrics = healthy_metrics()
    metrics["evaluation/forward/tracking_error_mps"] = 0.3
    assert not module.acceptance(metrics)["passed"]
