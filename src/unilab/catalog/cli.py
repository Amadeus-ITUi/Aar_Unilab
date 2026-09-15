"""Parser and compatibility routing for the generic Train/Play entrypoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from unilab.catalog import catalog

SELECTORS = ("robot", "task", "observation", "policy", "algorithm", "simulator")


def parse_selectors(argv: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    selected: dict[str, str] = {}
    passthrough: list[str] = []
    for value in argv:
        key, separator, item = value.partition("=")
        if separator and key in SELECTORS:
            if not item:
                raise ValueError(f"{key} cannot be empty")
            selected[key] = item
        else:
            passthrough.append(value)
    missing = [key for key in SELECTORS if key not in selected]
    if missing:
        raise ValueError("missing selectors: " + ", ".join(missing))
    return selected, passthrough


def build_legacy_command(mode: str, argv: Sequence[str], root: Path) -> list[str]:
    values, passthrough = parse_selectors(argv)
    selection = catalog.resolve(values)
    if selection.algorithm.id == "pe01_custom_ppo":
        command = [sys.executable, str(root / "scripts" / "train_pe01.py")]
        if mode == "play":
            command.append("mode=play")
        command.extend(passthrough)
        return command
    if selection.algorithm.id != "rsl_rl_ppo":
        raise ValueError(f"no executable adapter for algorithm={selection.algorithm.id!r}")
    command = [sys.executable, str(root / "scripts" / "train_rsl_rl.py")]
    command.append(f"task={selection.task.owner_config}/{selection.simulator.id}")
    command.extend(selection.observation.adapter_overrides)
    if mode == "play":
        command.append("training.play_only=true")
    command.extend(passthrough)
    return command


def run(mode: str, argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[3]
    try:
        command = build_legacy_command(mode, list(argv or sys.argv[1:]), root)
    except ValueError as exc:
        print(f"selection error: {exc}", file=sys.stderr)
        return 2
    return subprocess.run(command, check=False, cwd=root).returncode
