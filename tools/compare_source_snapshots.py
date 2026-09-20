#!/usr/bin/env python3
"""Fail if a migration source changed between two provenance snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = (
    "path",
    "git_commit",
    "git_branch",
    "git_status_short",
    "selected_file_count",
    "selected_tree_sha256",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before = json.loads(args.before.read_text(encoding="utf-8"))["sources"]
    after = json.loads(args.after.read_text(encoding="utf-8"))["sources"]
    differences: list[str] = []
    if before.keys() != after.keys():
        differences.append("source set differs")
    for source in sorted(before.keys() & after.keys()):
        for field in FIELDS:
            if before[source].get(field) != after[source].get(field):
                differences.append(f"{source}.{field} differs")
    if differences:
        print("migration source protection check failed:")
        for difference in differences:
            print(f"- {difference}")
        return 1
    print(f"migration source protection check passed: {len(before)} sources unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
