"""Cold-path provenance helpers for exported WE11 policies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WE11_HISTORY_LENGTH = 5
WE11_HISTORY_TERM_DIMS = (3, 3, 4, 6, 6, 2, 2, 3)
WE11_ACTOR_INPUT_DIM = WE11_HISTORY_LENGTH * sum(WE11_HISTORY_TERM_DIMS)
WE11_ACTION_DIM = 6
WE11_PUBLISHABLE_TASKS = frozenset({"DR002JoystickFlatWE11", "DR002JoystickGetupWE11"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_iteration(path: str | Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(path).name)
    if match is None:
        raise ValueError(f"Checkpoint must be named model_<iteration>.pt, got {path}")
    return int(match.group(1))


def inspect_onnx_contract(path: str | Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on supported runtime installation
        raise RuntimeError(
            "onnxruntime is required; run this command in the supported UniLab Conda environment"
        ) from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"WE11 policy must have one input and one output, got {len(inputs)} and {len(outputs)}"
        )
    model_input = inputs[0]
    model_output = outputs[0]
    contract = {
        "input": {
            "name": model_input.name,
            "shape": list(model_input.shape),
            "dtype": model_input.type,
        },
        "output": {
            "name": model_output.name,
            "shape": list(model_output.shape),
            "dtype": model_output.type,
        },
        "history_layout": "term-major",
        "history_length": WE11_HISTORY_LENGTH,
        "history_term_dims": list(WE11_HISTORY_TERM_DIMS),
    }
    expected_input = {
        "name": "obs",
        "shape": [1, WE11_ACTOR_INPUT_DIM],
        "dtype": "tensor(float)",
    }
    expected_output = {
        "name": "act",
        "shape": [1, WE11_ACTION_DIM],
        "dtype": "tensor(float)",
    }
    if contract["input"] != expected_input or contract["output"] != expected_output:
        raise ValueError(
            "Unsupported WE11 ONNX contract: "
            f"input={contract['input']} output={contract['output']}; "
            f"expected input={expected_input} output={expected_output}"
        )
    return contract


def build_policy_export_manifest(
    *,
    run_dir: str | Path,
    checkpoint: str | Path,
    onnx_path: str | Path,
    provenance_mode: str = "export",
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    policy_path = Path(onnx_path).resolve()
    if checkpoint_path.parent != run_path:
        raise ValueError(f"Checkpoint {checkpoint_path} is not inside run directory {run_path}")
    if policy_path.parent != run_path:
        raise ValueError(f"ONNX {policy_path} is not inside run directory {run_path}")
    for required in (checkpoint_path, policy_path, run_path / "run_config.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    run_config = read_json(run_path / "run_config.json")
    run_info = run_config.get("run", {})
    if not isinstance(run_info, dict):
        raise ValueError("run_config.json has no run object")
    task = str(run_info.get("task", ""))
    if task not in WE11_PUBLISHABLE_TASKS:
        raise ValueError(
            f"Only DR002JoystickFlatWE11 and DR002JoystickGetupWE11 are publishable, got {task!r}"
        )

    summary_path = run_path / "run_summary.json"
    summary_hash = sha256_file(summary_path) if summary_path.is_file() else None
    return {
        "schema_version": 1,
        "kind": "we11_policy_export",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_mode": provenance_mode,
        "source": {
            "task": task,
            "run": run_path.name,
            "run_path": str(run_path),
            "checkpoint": checkpoint_path.name,
            "checkpoint_iteration": checkpoint_iteration(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "run_config_sha256": sha256_file(run_path / "run_config.json"),
            "run_summary_sha256": summary_hash,
            "training_git": run_info.get("git"),
        },
        "contract": inspect_onnx_contract(policy_path),
        "artifact": {
            "file": policy_path.name,
            "sha256": sha256_file(policy_path),
            "size_bytes": policy_path.stat().st_size,
        },
    }


def write_policy_export_manifest(
    *, run_dir: str | Path, checkpoint: str | Path, onnx_path: str | Path
) -> Path:
    target = Path(run_dir) / "policy_export_manifest.json"
    atomic_write_json(
        target,
        build_policy_export_manifest(
            run_dir=run_dir,
            checkpoint=checkpoint,
            onnx_path=onnx_path,
        ),
    )
    return target
