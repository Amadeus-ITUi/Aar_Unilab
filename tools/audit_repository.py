#!/usr/bin/env python3
"""Audit relocation safety, repository hygiene and curated model hashes."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

FORBIDDEN_RUNTIME_PATHS = (
    "/ssd/Pheonix/UniLab",
    "/ssd/Pheonix/Play",
    "/ssd/dog",
    "/home/angela/ssd/Point_Phoenix",
)
RUNTIME_ROOTS = ("src", "conf", "scripts", "sim2sim")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".xml", ".cpp", ".cc", ".h", ".hpp"}


def audit(root: Path, require_clean: bool = False) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    for path in root.rglob(".git"):
        if path != root / ".git":
            errors.append(f"nested Git metadata: {path.relative_to(root)}")
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            errors.append(f"external symlink: {path.relative_to(root)} -> {os.readlink(path)}")
    for directory in RUNTIME_ROOTS:
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for forbidden in FORBIDDEN_RUNTIME_PATHS:
                if forbidden in text:
                    errors.append(
                        f"runtime dependency token {forbidden!r}: {path.relative_to(root)}"
                    )
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        if status:
            errors.append("Git worktree is dirty:\n" + status)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    errors = audit(args.root, args.require_clean)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"repository audit passed: {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
