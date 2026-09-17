"""Local training presets, applied before importing numerical libraries in the child."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from unilab.catalog.cli import SELECTORS, build_legacy_command

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = ROOT / "conf/train_defaults.yaml"


def build_launch(
    path: Path,
    profile: str,
    overrides: Sequence[str],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Resolve a catalog selection and child environment without starting training."""
    config = OmegaConf.load(path)
    if profile not in config.profiles:
        raise ValueError(f"unknown profile {profile!r}; choose: {', '.join(config.profiles)}")
    child_env = dict(os.environ if environment is None else environment)
    for key in config.get("unset_environment", []):
        child_env.pop(key, None)
    displayed_env: dict[str, str] = {}
    for key, default in config.environment.items():
        value = child_env.get(key, str(default))
        # Retain typed integer interpolation into Hydra (PE thread counts, for example).
        thread_count = key.endswith("_NUM_THREADS") or key == "UNILAB_MUJOCO_NTHREADS"
        if type(default) is int or thread_count:
            try:
                number = int(value)
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer, got {value!r}") from exc
            if number < 1 and thread_count:
                raise ValueError(f"{key} must be positive")
            config.environment[key] = number
        else:
            config.environment[key] = value
        child_env[key] = displayed_env[key] = value

    selected = config.profiles[profile]
    selectors = cast(dict[str, Any], OmegaConf.to_container(selected.selectors, resolve=True))
    defaults = cast(dict[str, Any], OmegaConf.to_container(config.common, resolve=True))
    defaults.update(cast(dict[str, Any], OmegaConf.to_container(selected.overrides, resolve=True)))
    arguments = {
        key: f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in defaults.items()
    }
    for argument in overrides:
        key, separator, value = argument.partition("=")
        if key in SELECTORS and separator:
            selectors[key] = value
            continue
        if not separator and not argument.startswith("~"):
            raise ValueError(f"expected a Hydra key=value override, got {argument!r}")
        # A CLI +experiment override replaces the preset instead of adding the group twice.
        canonical = key.lstrip("+~")
        arguments = {
            existing: text
            for existing, text in arguments.items()
            if existing.lstrip("+~") != canonical
        }
        arguments[key] = argument
    selection_args = [f"{key}={value}" for key, value in selectors.items()]
    command = build_legacy_command("train", [*selection_args, *arguments.values()], root)
    return command, child_env, displayed_env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train from conf/train_defaults.yaml presets.")
    parser.add_argument(
        "profile", nargs="?", help="training profile, e.g. we11_getup or pe02_walking"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides (highest priority)")
    parser.add_argument("--config", type=Path, default=DEFAULTS, help="alternative defaults YAML")
    parser.add_argument("--list", action="store_true", help="list profiles without training")
    parser.add_argument(
        "--dry-run", action="store_true", help="show effective command without training"
    )
    args = parser.parse_intermixed_args(argv)
    try:
        if args.list:
            print("\n".join(OmegaConf.load(args.config).profiles))
            return 0
        if args.profile is None:
            parser.error("profile is required; use --list to see available profiles")
        command, environment, displayed_env = build_launch(
            args.config, args.profile, args.overrides
        )
    except (OSError, ValueError, OmegaConfBaseException) as exc:
        parser.error(str(exc))
    print(f"Training defaults: {args.config.resolve()} | profile: {args.profile}", flush=True)
    print(
        "Environment: " + shlex.join(f"{key}={value}" for key, value in displayed_env.items()),
        flush=True,
    )
    print("Command: " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    os.chdir(ROOT)
    os.execve(command[0], command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
