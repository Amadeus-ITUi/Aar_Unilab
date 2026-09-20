"""Shared helpers for training entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.base.registry import ensure_registries as _ensure_registries


def ensure_registries() -> None:
    """Import env modules so registry-based entrypoints can instantiate tasks."""
    _ensure_registries()


def setup_logger(
    log_dir: str | Path,
    algo_name: str,
    *,
    echo: bool = True,
    filename: str = "train.log",
) -> logging.Logger:
    """Create a simple file-backed logger for script-local progress messages."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger_name = f"unilab.training.{algo_name}.{path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(path / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if echo:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def create_env(
    cfg: DictConfig,
    *,
    num_envs: int,
    env_cfg_override: dict[str, Any] | None = None,
    sim_backend: str | None = None,
    task_name: str | None = None,
):
    """Construct an environment via the registry using the current Hydra config."""
    from unilab.base import registry

    return registry.make(
        task_name or str(OmegaConf.select(cfg, "training.task_name")),
        num_envs=num_envs,
        sim_backend=sim_backend or str(OmegaConf.select(cfg, "training.sim_backend")),
        env_cfg_override=env_cfg_override,
    )
