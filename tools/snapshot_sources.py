#!/usr/bin/env python3
"""Create a deterministic, read-only provenance snapshot of migration sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "unilab": Path("/ssd/Pheonix/UniLab"),
    "play": Path("/ssd/Pheonix/Play"),
    "deploy_host_only": Path("/ssd/Pheonix/DeployHostOnly/Deploy"),
    "deploy_host_device": Path("/ssd/Pheonix/DeployHostDevice/Deploy"),
    "pe01_predecessor": Path("/home/angela/ssd/Point_Phoenix/point-legged"),
}
EXCLUDED_PARTS = frozenset(
    {".git", ".venv", "build", "install", "log", "logs", "__pycache__", ".pytest_cache"}
)
SOURCE_SUFFIXES = frozenset(
    {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".xml", ".urdf", ".yaml", ".yml", ".json", ".md", ".txt", ".sh"}
)


def run_git(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def selected_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES or path.name in {"CMakeLists.txt", "LICENSE"}:
            yield path


def tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in selected_files(root):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    return digest.hexdigest(), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "sources": {}}
    for name, root in SOURCES.items():
        digest, count = tree_digest(root)
        payload["sources"][name] = {
            "path": str(root),
            "git_commit": run_git(root, "rev-parse", "HEAD"),
            "git_branch": run_git(root, "branch", "--show-current"),
            "git_status_short": run_git(root, "status", "--short"),
            "selected_file_count": count,
            "selected_tree_sha256": digest,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
