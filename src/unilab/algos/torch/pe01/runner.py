# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# See LICENSE.pe01 for terms; PE01 migration changes are maintained independently.
"""PE01-owned iterative rollout, evaluation, resumable checkpoints and logging."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.pe01.console import format_iteration
from unilab.algos.torch.pe01.policy import PE01EncoderPolicy
from unilab.algos.torch.pe01.ppo import PE01PPO
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe01.config import ROOT, validate_config
from unilab.envs.locomotion.pe01.vector_env import PE01VectorEnv


def tensor_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_tree(item) for item in value]
    return value


def numpy_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numpy_tree(item) for item in value]
    return value


def asset_fingerprint(config: DictConfig) -> str:
    folder = repository_path(str(config.env.model_path), ROOT).parent
    files = [folder / "pe01.xml", repository_path(str(config.env.model_path), ROOT)]
    files.extend(sorted((folder / "collision_meshes").glob("*.STL")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(folder)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resume_contract(config: DictConfig) -> dict[str, Any]:
    value = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    for key in ("mode", "checkpoint", "play"):
        value.pop(key, None)
    value.pop("training", None)
    value["algo"].pop("max_iterations")
    value["algo"].pop("save_interval")
    return value


class PE01Runner:
    def __init__(self, config: DictConfig, output_dir: Path) -> None:
        self.config = OmegaConf.merge(config)
        validate_config(self.config)
        torch.set_num_threads(int(config.training.cpu_threads))
        torch.manual_seed(int(config.training.seed))
        device = str(config.training.device)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.policy = PE01EncoderPolicy(config).to(self.device)
        self.algorithm = PE01PPO(self.policy, config)
        self.env = PE01VectorEnv(config)
        self.observation = self.env.state.obs
        self.iteration = 0
        self.total_samples = 0
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_hash = asset_fingerprint(config)
        self.writer = None
        self.returns: deque[float] = deque(maxlen=100)
        self.lengths: deque[float] = deque(maxlen=100)
        try:
            if config.training.resume:
                self.load(Path(str(config.training.resume)))
            if config.training.logger == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(
                    str(output_dir / "tensorboard"), purge_step=self.iteration
                )
            elif config.training.logger != "none":
                raise ValueError("PE01 supports training.logger=tensorboard or none")
            OmegaConf.save(config, output_dir / "training_config.yaml", resolve=True)
        except Exception:
            self.close()
            raise

    def _tensors(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in obs.items()
        }

    def collect(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        frames: dict[str, list[torch.Tensor]] = {}
        reward_terms: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(int(self.config.algo.num_steps_per_env)):
                obs = self._tensors(self.observation)
                distribution = self.algorithm.distribution(obs)
                action = distribution.sample()
                value = self.algorithm.value(obs)
                result = self.env.step(action.cpu().numpy())
                next_obs = self._tensors(result.obs)
                # Autoreset observations cannot bootstrap a timed-out episode.
                if result.final_observation is not None:
                    final = self._tensors(result.final_observation)
                    mask = torch.as_tensor(result.terminated | result.truncated, device=self.device)
                    for key in next_obs:
                        next_obs[key] = torch.where(mask[:, None], final[key], next_obs[key])
                entry = {
                    **obs,
                    "action": action,
                    "value": value,
                    "next_value": self.algorithm.value(next_obs),
                    "log_prob": distribution.log_prob(action).sum(-1),
                    "mean": distribution.mean,
                    "sigma": distribution.stddev,
                    "reward": torch.as_tensor(result.reward, device=self.device),
                    "terminated": torch.as_tensor(result.terminated, device=self.device),
                    "truncated": torch.as_tensor(result.truncated, device=self.device),
                }
                for key, val in entry.items():
                    frames.setdefault(key, []).append(val.clone())
                for key, val in result.info["reward_terms"].items():
                    reward_terms.setdefault(key, []).append(val)
                self.returns.extend(result.info["episode_returns"].tolist())
                self.lengths.extend(result.info["episode_lengths"].tolist())
                self.observation = result.obs
        rollout = {key: torch.stack(values) for key, values in frames.items()}
        return rollout, {
            f"reward/{key}": float(np.mean(values)) for key, values in reward_terms.items()
        }

    def save(self, path: Path) -> None:
        payload = {
            "schema": "pe01.training.v2",
            "robot_id": "pe01",
            "iteration": self.iteration,
            "total_samples": self.total_samples,
            "asset_sha256": self.asset_hash,
            "training_config": OmegaConf.to_container(self.config, resolve=True),
            "actor_state_dict": self.policy.state_dict(),
            "optimizer": self.algorithm.optimizer.state_dict(),
            "encoder_optimizer": self.algorithm.encoder_optimizer.state_dict(),
            "learning_rate": self.algorithm.learning_rate,
            "optimizer_updates": self.algorithm.updates,
            "encoder_updates": self.algorithm.encoder_updates,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "environment": tensor_tree(self.env.snapshot()),
            "recent_returns": list(self.returns),
            "recent_lengths": list(self.lengths),
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("schema") != "pe01.training.v2" or payload.get("robot_id") != "pe01":
            raise ValueError("formal PE01 resume requires a pe01.training.v2 checkpoint")
        previous = OmegaConf.create(payload["training_config"])
        if resume_contract(previous) != resume_contract(self.config):
            raise ValueError(
                "resume changes model, environment, network or PPO contract; start a new run"
            )
        if payload["asset_sha256"] != self.asset_hash:
            raise ValueError("PE01 assets changed since checkpoint")
        self.policy.load_state_dict(payload["actor_state_dict"])
        self.algorithm.optimizer.load_state_dict(payload["optimizer"])
        self.algorithm.encoder_optimizer.load_state_dict(payload["encoder_optimizer"])
        self.algorithm.learning_rate = payload["learning_rate"]
        self.algorithm.updates = payload["optimizer_updates"]
        self.algorithm.encoder_updates = payload["encoder_updates"]
        self.iteration, self.total_samples = payload["iteration"], payload["total_samples"]
        self.observation = self.env.restore(numpy_tree(payload["environment"])).obs
        self.returns.extend(payload["recent_returns"])
        self.lengths.extend(payload["recent_lengths"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if self.device.type == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])

    def evaluate(self) -> dict[str, float]:
        env = PE01VectorEnv(
            self.config,
            evaluation=True,
            num_envs=int(self.config.training.evaluation_episodes),
            evaluation_reset_noise=True,
        )
        try:
            state = env.state
            live = np.ones(env.num_envs, bool)
            returns = np.zeros(env.num_envs)
            lengths = np.zeros(env.num_envs)
            failures = np.zeros(env.num_envs, bool)
            sums = {
                key: 0.0
                for key in ("base_height", "base_tilt_deg", "nonfoot_contact", "standing_fraction")
            }
            measured = 0
            with torch.no_grad():
                for _ in range(env.max_episode_steps + 1):
                    obs = self._tensors(state.obs)
                    action = self.policy.action_mean(obs["actor"], obs["frame"], obs["command"])
                    state = env.step(action.cpu().numpy())
                    measured += int(live.sum())
                    for key in ("base_height", "base_tilt_deg", "nonfoot_contact"):
                        sums[key] += float(state.info[key][live].sum())
                    standing = np.abs(
                        state.info["base_height"] - env.height_target
                    ) < self.config.training.get("evaluation_height_tolerance", 0.03)
                    standing &= state.info["base_tilt_deg"] < self.config.training.get(
                        "evaluation_tilt_deg", 10
                    )
                    standing &= state.info["base_speed"] < self.config.training.get(
                        "evaluation_speed_tolerance", 0.1
                    )
                    standing &= ~state.info["nonfoot_contact"]
                    sums["standing_fraction"] += float(standing[live].sum())
                    returns[live] += state.reward[live]
                    lengths[live] += 1
                    failures |= live & state.terminated
                    live &= ~(state.terminated | state.truncated)
                    if not live.any():
                        break
            return {
                "evaluation/return": float(returns.mean()),
                "evaluation/episode_seconds": float(lengths.mean() * env.dt),
                "evaluation/failure_rate": float(failures.mean()),
                **{f"evaluation/{key}": value / max(1, measured) for key, value in sums.items()},
            }
        finally:
            env.close()

    def learn(self) -> Path:
        last_path = self.output_dir / f"model_{self.iteration}.pt"
        log_path = self.output_dir / "metrics.jsonl"
        start_iteration = self.iteration
        target_iteration = start_iteration + int(self.config.algo.max_iterations)
        session_start = time.perf_counter()
        for _ in range(int(self.config.algo.max_iterations)):
            start = time.perf_counter()
            rollout, metrics = self.collect()
            collection_seconds = time.perf_counter() - start
            update_start = time.perf_counter()
            metrics.update(self.algorithm.update(rollout))
            update_seconds = time.perf_counter() - update_start
            self.iteration += 1
            self.total_samples += int(
                self.config.algo.num_envs * self.config.algo.num_steps_per_env
            )
            metrics.update(
                iteration=self.iteration,
                samples=self.total_samples,
                mean_step_reward=float(rollout["reward"].mean()),
                collection_seconds=collection_seconds,
                update_seconds=update_seconds,
                samples_per_second=float(
                    rollout["reward"].numel() / (collection_seconds + update_seconds)
                ),
            )
            if self.returns:
                metrics["episode/return"] = float(np.mean(self.returns))
                metrics["episode/seconds"] = float(np.mean(self.lengths) * self.env.dt)
            interval = int(self.config.training.evaluation_interval)
            if interval > 0 and self.iteration % interval == 0:
                metrics.update(self.evaluate())
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("non-finite training metrics")
            with log_path.open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, self.iteration)
            if self.iteration % int(self.config.algo.save_interval) == 0:
                last_path = self.output_dir / f"model_{self.iteration}.pt"
                self.save(last_path)
            now = time.perf_counter()
            print(
                format_iteration(
                    metrics,
                    target_iteration=target_iteration,
                    completed_iterations=self.iteration - start_iteration,
                    iteration_seconds=now - start,
                    elapsed_seconds=now - session_start,
                    policy_dt=self.env.dt,
                ),
                flush=True,
            )
        last_path = self.output_dir / f"model_{self.iteration}.pt"
        self.save(last_path)
        return last_path

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.env.close()
