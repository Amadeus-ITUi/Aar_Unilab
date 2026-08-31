#!/usr/bin/env python3
"""Validate and publish one WE11 Flat/Getup ONNX into their shared Play slot."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.training.policy_export import (  # noqa: E402
    WE11_ACTION_DIM,
    WE11_ACTOR_INPUT_DIM,
    WE11_HISTORY_LENGTH,
    WE11_HISTORY_TERM_DIMS,
    atomic_write_json,
    build_policy_export_manifest,
    read_json,
    sha256_file,
)

PLAY_OBSERVATIONS = [
    "ang_vel",
    "projected_gravity",
    "dof_pos_without_wheel_compact",
    "dof_vel_mapped",
    "actions",
    "wing_angle",
    "wing_vel",
    "commands",
]


def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Play config is missing {key}")
    return match.group(1)


def validate_play_config(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    num_observations = int(_yaml_scalar(text, "num_observations"))
    observations = ast.literal_eval(_yaml_scalar(text, "observations"))
    history = ast.literal_eval(_yaml_scalar(text, "observations_history"))
    history_priority = _yaml_scalar(text, "observations_history_priority").strip("\"'")
    if num_observations != WE11_ACTOR_INPUT_DIM:
        raise ValueError(
            f"Play num_observations={num_observations}, expected {WE11_ACTOR_INPUT_DIM}"
        )
    if observations != PLAY_OBSERVATIONS:
        raise ValueError(f"Play observations={observations}, expected {PLAY_OBSERVATIONS}")
    if history != list(range(WE11_HISTORY_LENGTH - 1, -1, -1)):
        raise ValueError(f"Play observations_history={history}, expected [4, 3, 2, 1, 0]")
    if history_priority != "term":
        raise ValueError(
            f"Play observations_history_priority={history_priority!r}, expected 'term'"
        )


def resolve_run(run: str, task: str) -> Path:
    candidate = Path(run).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    resolved = ROOT_DIR / "logs" / "rsl_rl_ppo" / task / run
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved}")
    return resolved.resolve()


def resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is None:
        export_manifest = run_dir / "policy_export_manifest.json"
        if not export_manifest.is_file():
            raise ValueError(
                "This legacy export has no policy_export_manifest.json; pass --checkpoint explicitly"
            )
        source = read_json(export_manifest).get("source", {})
        if not isinstance(source, dict) or not source.get("checkpoint"):
            raise ValueError(f"Invalid source in {export_manifest}")
        checkpoint = str(source["checkpoint"])
    path = Path(checkpoint)
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_or_adopt_export(run_dir: Path, checkpoint: Path) -> tuple[dict[str, Any], bool]:
    onnx_path = run_dir / "policy.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    sidecar_path = run_dir / "policy_export_manifest.json"
    legacy_adoption = not sidecar_path.is_file()
    if legacy_adoption:
        if onnx_path.stat().st_mtime_ns < checkpoint.stat().st_mtime_ns:
            raise ValueError(
                f"Legacy ONNX {onnx_path} predates checkpoint {checkpoint}; re-export it first"
            )
        manifest = build_policy_export_manifest(
            run_dir=run_dir,
            checkpoint=checkpoint,
            onnx_path=onnx_path,
            provenance_mode="explicit_legacy_adoption",
        )
    else:
        manifest = read_json(sidecar_path)
        source = manifest.get("source", {})
        artifact = manifest.get("artifact", {})
        if not isinstance(source, dict) or not isinstance(artifact, dict):
            raise ValueError(f"Invalid export manifest: {sidecar_path}")
        if source.get("checkpoint") != checkpoint.name:
            raise ValueError(
                f"Export sidecar names checkpoint {source.get('checkpoint')!r}, "
                f"but {checkpoint.name!r} was selected"
            )
        if source.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError("Checkpoint hash no longer matches policy_export_manifest.json")
        if artifact.get("sha256") != sha256_file(onnx_path):
            raise ValueError("ONNX hash no longer matches policy_export_manifest.json")
        fresh = build_policy_export_manifest(
            run_dir=run_dir,
            checkpoint=checkpoint,
            onnx_path=onnx_path,
            provenance_mode=str(manifest.get("provenance_mode", "export")),
        )
        if fresh["contract"] != manifest.get("contract"):
            raise ValueError("ONNX contract no longer matches policy_export_manifest.json")
    return manifest, legacy_adoption


def build_play_manifest(export: dict[str, Any], old: dict[str, Any] | None) -> dict[str, Any]:
    source = export["source"]
    contract = export["contract"]
    artifact = export["artifact"]
    replay = (old or {}).get(
        "replay",
        {
            "physics_hz": 400,
            "motor_hz": 200,
            "policy_hz": 50,
            "flat_scene": "scene_flat_we11",
            "level9_scenes": ["flat", "uphill", "downhill", "rough"],
            "level9_scene_prefix": "scene_level9_noise006_",
            "default_level9_scene": "rough",
            "supported_replay_hz": [0, 1, 2, 3],
        },
    )
    return {
        "schema_version": 2,
        "selector": "we11",
        "source_task": source["task"],
        "source_run": source["run"],
        "source_checkpoint": source["checkpoint"],
        "source_checkpoint_iteration": source["checkpoint_iteration"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_run_config_sha256": source["run_config_sha256"],
        "source_run_summary_sha256": source.get("run_summary_sha256"),
        "source_training_git": source.get("training_git"),
        "source_provenance_mode": export["provenance_mode"],
        "asset_selector": "we11",
        "policy_contract": contract,
        "artifacts": {
            "onnx": {
                "file": "policy.onnx",
                "sha256": artifact["sha256"],
                "size_bytes": artifact["size_bytes"],
                "input": f"obs[1,{WE11_ACTOR_INPUT_DIM}]",
                "output": f"act[1,{WE11_ACTION_DIM}]",
            }
        },
        "replay": replay,
    }


def render_source_md(manifest: dict[str, Any]) -> str:
    source_git = manifest.get("source_training_git") or {}
    return f"""# Pheonix WE11 current policy

- Source task: `{manifest["source_task"]}`
- Training run: `{manifest["source_run"]}`
- Checkpoint: `{manifest["source_checkpoint"]}`
- Checkpoint iteration: `{manifest["source_checkpoint_iteration"]}`
- Checkpoint SHA256: `{manifest["source_checkpoint_sha256"]}`
- Run-config SHA256: `{manifest["source_run_config_sha256"]}`
- Training Git: `{source_git.get("commit", "unknown")}` on `{source_git.get("branch", "unknown")}` (dirty={source_git.get("dirty", "unknown")})
- Provenance: `{manifest["source_provenance_mode"]}`
- ONNX: `policy.onnx`
- ONNX SHA256: `{manifest["artifacts"]["onnx"]["sha256"]}`
- Interface: `obs[1,{WE11_ACTOR_INPUT_DIM}] -> act[1,{WE11_ACTION_DIM}]`
- History: term-major, `{WE11_HISTORY_LENGTH}` frames, term dims `{list(WE11_HISTORY_TERM_DIMS)}`
- Play selector: `we11`

`deployment_manifest.json` is the machine-readable source of truth for this policy.
"""


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def update_checksums(text: str, replacements: dict[str, str]) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1] in replacements:
            path = parts[1]
            result.append(f"{replacements[path]}  {path}")
            seen.add(path)
        else:
            result.append(line)
    for path, digest in replacements.items():
        if path not in seen:
            result.append(f"{digest}  {path}")
    return "\n".join(result) + "\n"


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def validate_play_target_drift(
    play_policy: Path,
    source_hash: str,
    old_manifest: dict[str, Any] | None,
) -> None:
    if not play_policy.is_file():
        return
    target_hash = sha256_file(play_policy)
    recorded_hash = ((old_manifest or {}).get("artifacts", {}).get("onnx", {})).get("sha256")
    if target_hash != source_hash and recorded_hash != target_hash:
        raise RuntimeError(
            "Play policy differs from both the source and its recorded manifest; "
            "refusing to overwrite an untracked manual replacement"
        )


def commit_files(replacements: dict[Path, bytes], *, backup_dir: Path, label_root: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    originals: dict[Path, bytes | None] = {}
    for target in replacements:
        originals[target] = target.read_bytes() if target.exists() else None
        if target.exists():
            relative = target.resolve().relative_to(label_root.resolve())
            backup = backup_dir / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
    try:
        for target, content in replacements.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    except Exception:
        for target, original in originals.items():
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(original)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory name or absolute path")
    parser.add_argument(
        "--checkpoint", help="Checkpoint filename/path; required for legacy exports"
    )
    parser.add_argument("--task", default="DR002JoystickFlatWE11")
    parser.add_argument("--play-root", type=Path, default=ROOT_DIR.parent / "Play")
    parser.add_argument("--apply", action="store_true", help="Apply changes; default is dry-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_run(args.run, args.task)
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    export, legacy_adoption = validate_or_adopt_export(run_dir, checkpoint)
    play_root = args.play_root.expanduser().resolve()
    policy_dir = play_root / "policy" / "dr002" / "we11"
    play_policy = policy_dir / "policy.onnx"
    play_manifest_path = policy_dir / "deployment_manifest.json"
    play_source_path = policy_dir / "SOURCE.md"
    checksums_path = play_root / "SHA256SUMS"
    for required in (policy_dir / "config.yaml", checksums_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    validate_play_config(policy_dir / "config.yaml")

    source_policy = run_dir / "policy.onnx"
    source_hash = sha256_file(source_policy)
    old_manifest = read_json(play_manifest_path) if play_manifest_path.is_file() else None
    validate_play_target_drift(play_policy, source_hash, old_manifest)

    play_manifest = build_play_manifest(export, old_manifest)
    manifest_content = json_bytes(play_manifest)
    source_content = render_source_md(play_manifest).encode()
    checksums_content = update_checksums(
        checksums_path.read_text(encoding="utf-8"),
        {
            "policy/dr002/we11/policy.onnx": source_hash,
            "policy/dr002/we11/deployment_manifest.json": sha256_bytes(manifest_content),
        },
    ).encode()

    release_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{export['source']['run']}_model{export['source']['checkpoint_iteration']}"
    )
    print(f"mode={'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"run={run_dir}")
    print(f"checkpoint={checkpoint.name} sha256={export['source']['checkpoint_sha256']}")
    print(f"onnx={source_policy} sha256={source_hash}")
    print(f"contract=obs[1,{WE11_ACTOR_INPUT_DIM}] -> act[1,{WE11_ACTION_DIM}]")
    print(f"provenance={export['provenance_mode']}")
    print(f"target={play_policy}")
    if not args.apply:
        print("No files changed. Re-run with --apply to publish.")
        return 0

    workspace_root = play_root.parent
    backup_dir = workspace_root / "artifacts" / "we11-policy-backups" / release_id / "Play"
    replacements = {
        play_policy: source_policy.read_bytes(),
        play_manifest_path: manifest_content,
        play_source_path: source_content,
        checksums_path: checksums_content,
    }
    commit_files(replacements, backup_dir=backup_dir, label_root=play_root)
    if legacy_adoption:
        atomic_write_json(run_dir / "policy_export_manifest.json", export)
    print(f"Published Play policy; backup={backup_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
