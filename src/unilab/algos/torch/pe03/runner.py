# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# See LICENSE.pe01 for terms; PE03 migration changes are maintained independently.
"""PE03-owned iterative rollout, evaluation, resumable checkpoints and logging."""

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

from unilab.algos.torch.pe03.console import format_iteration
from unilab.algos.torch.pe03.policy import PE03EncoderPolicy
from unilab.algos.torch.pe03.ppo import PE03PPO
from unilab.algos.torch.pe03.run_logging import index_tensorboard_run
from unilab.algos.torch.pe03.tensorboard import tensorboard_metrics
from unilab.base.backend.mujoco.batched_robot import robot_xml_path
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe03.config import ROOT, validate_config
from unilab.envs.locomotion.pe03.factory import make_env
from unilab.envs.locomotion.pe03.vector_env import PE03VectorEnv


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
    scene = repository_path(str(config.env.model_path), ROOT)
    folder = scene.parent
    files = [robot_xml_path(scene), scene]
    files.extend(sorted((folder / "collision_meshes").glob("*.STL")))
    if config.observation == "pe03_v4":
        files.append(ROOT / config.env.workspace_path)
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
    if config.observation == "pe03_v4":
        value["evaluation_conditions"] = config.training.get("evaluation_conditions", "nominal")
    value["algo"].pop("max_iterations")
    value["algo"].pop("save_interval")
    return value


class PE03Runner:
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
        self.policy = PE03EncoderPolicy(config).to(self.device)
        self.algorithm = PE03PPO(self.policy, config)
        self.env = make_env(config)
        self.observation = self.env.state.obs
        self.iteration = 0
        self.total_samples = 0
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_hash = asset_fingerprint(config)
        self.writer = None
        self.returns: deque[float] = deque(maxlen=100)
        self.lengths: deque[float] = deque(maxlen=100)
        self.acceptance_records: list[dict[str, Any]] = []
        self.stage_source: str | None = None
        try:
            if config.training.resume:
                self.load(Path(str(config.training.resume)))
            elif config.observation == "pe03_v4" and config.training.stage_from:
                self.load_stage(Path(str(config.training.stage_from)))
            elif config.observation == "pe03_v4" and config.gait.stage == "variable":
                raise ValueError("variable-stage training requires stage_from or same-stage resume")
            if config.training.logger == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(
                    str(output_dir / "tensorboard"), purge_step=self.iteration, flush_secs=10
                )
                link = index_tensorboard_run(output_dir, config)
                print(
                    f"TensorBoard run: {link.name}\nTensorBoard logdir: {link.parent}", flush=True
                )
            elif config.training.logger != "none":
                raise ValueError("PE03 supports training.logger=tensorboard or none")
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
        rollout: dict[str, torch.Tensor] = {}
        reward_terms: dict[str, list[float]] = {}
        steps = int(self.config.algo.num_steps_per_env)

        def store(step: int, entry: dict[str, torch.Tensor]) -> None:
            for key, val in entry.items():
                if key not in rollout:
                    rollout[key] = torch.empty(
                        (steps, *val.shape), dtype=val.dtype, device=val.device
                    )
                rollout[key][step].copy_(val)

        with torch.no_grad():
            obs = self._tensors(self.observation)
            value = self.algorithm.value(obs)
            for step in range(steps):
                distribution = self.algorithm.distribution(obs)
                action = distribution.sample()
                # Own each frame before stepping an environment that may reuse
                # NumPy buffers. Allocate once instead of clone-then-stack.
                store(
                    step,
                    {
                        **obs,
                        "action": action,
                        "value": value,
                        "log_prob": distribution.log_prob(action).sum(-1),
                        "mean": distribution.mean,
                        "sigma": distribution.stddev,
                    },
                )
                result = self.env.step(action.cpu().numpy())
                next_obs = self._tensors(result.obs)
                value = self.algorithm.value(next_obs)
                next_value = value
                # Timeouts and curriculum cuts bootstrap from pre-reset observations.
                # Reuse the next frame's value; only ended episodes need a
                # separate terminal value evaluation.
                if result.final_observation is not None:
                    done = result.terminated | result.truncated
                    final = self._tensors(
                        {
                            key: result.final_observation[key][done]
                            for key in ("actor", "critic", "command")
                        }
                    )
                    mask = torch.as_tensor(done, device=self.device)
                    next_value = value.clone()
                    next_value[mask] = self.algorithm.value(final)
                store(
                    step,
                    {
                        "next_value": next_value,
                        "reward": torch.as_tensor(result.reward, device=self.device),
                        "terminated": torch.as_tensor(result.terminated, device=self.device),
                        "truncated": torch.as_tensor(result.truncated, device=self.device),
                    },
                )
                for key, val in result.info["reward_terms"].items():
                    reward_terms.setdefault(key, []).append(val)
                for key, val in result.info.get("diagnostics", {}).items():
                    reward_terms.setdefault("diagnostics/" + key, []).append(val)
                self.returns.extend(result.info["episode_returns"].tolist())
                self.lengths.extend(result.info["episode_lengths"].tolist())
                self.observation = result.obs
                obs = next_obs
        metrics = {
            (
                key.removeprefix("diagnostics/")
                if key.startswith("diagnostics/")
                else f"reward/{key}"
            ): float(np.mean(values))
            for key, values in reward_terms.items()
        }
        if self.config.observation == "pe03_v4":
            # Course counters are state, not per-step rates; don't average a promotion.
            metrics.update(self.env.robustness_curriculum.metrics())
            for suffix in ("min", "mean", "max"):
                key = f"randomization/applied_level_{suffix}"
                metrics[key] = reward_terms["diagnostics/" + key][-1]
        return rollout, metrics

    def save(self, path: Path) -> None:
        payload = {
            "schema": "pe03.training.v2",
            "robot_id": "pe03",
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
            "acceptance_records": self.acceptance_records,
            "stage_source": self.stage_source,
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("schema") != "pe03.training.v2" or payload.get("robot_id") != "pe03":
            raise ValueError("formal PE03 resume requires a pe03.training.v2 checkpoint")
        previous = OmegaConf.create(payload["training_config"])
        if resume_contract(previous) != resume_contract(self.config):
            raise ValueError(
                "resume changes model, environment, network or PPO contract; start a new run"
            )
        if payload["asset_sha256"] != self.asset_hash:
            raise ValueError("PE03 assets changed since checkpoint")
        self.policy.load_state_dict(payload["actor_state_dict"])
        self.algorithm.optimizer.load_state_dict(payload["optimizer"])
        self.algorithm.encoder_optimizer.load_state_dict(payload["encoder_optimizer"])
        self.algorithm.learning_rate = payload["learning_rate"]
        self.algorithm.updates = payload["optimizer_updates"]
        self.algorithm.encoder_updates = payload["encoder_updates"]
        self.iteration, self.total_samples = payload["iteration"], payload["total_samples"]
        self.acceptance_records = payload.get("acceptance_records", [])
        self.stage_source = payload.get("stage_source")
        self.observation = self.env.restore(numpy_tree(payload["environment"])).obs
        self.returns.extend(payload["recent_returns"])
        self.lengths.extend(payload["recent_lengths"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if self.device.type == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])

    def load_stage(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        source = OmegaConf.create(payload["training_config"])
        from unilab.envs.locomotion.pe03.gait_randomization import evaluation_protocol

        protocol = evaluation_protocol(self.config)
        if evaluation_protocol(source) != protocol:
            raise ValueError("stage transition changes evaluation/randomization protocol")
        if (
            payload.get("robot_id") != "pe03"
            or source.observation != "pe03_v4"
            or source.gait.stage != "fixed"
        ):
            raise ValueError("stage_from requires a fixed-stage PE03 v4 checkpoint")
        if payload["asset_sha256"] != self.asset_hash:
            raise ValueError("stage transition assets differ")
        for section in ("env", "control", "normalization", "network", "reward"):
            if OmegaConf.to_container(source[section]) != OmegaConf.to_container(
                self.config[section]
            ):
                raise ValueError(f"stage transition changes {section} contract")
        required = int(self.config.training.acceptance.consecutive)
        records = payload.get("acceptance_records", [])[-required:]
        if protocol["conditions"] == "randomized" and any(
            record.get("protocol") != protocol["fingerprint"] for record in records
        ):
            raise ValueError("stage transition requires current randomized acceptance records")
        if (
            len(records) != required
            or not all(record["passed"] for record in records)
            or records[-1]["iteration"] != payload["iteration"]
            or any(b["iteration"] - a["iteration"] != 100 for a, b in zip(records, records[1:]))
            or source.training.evaluation_episodes < 64
            or source.training.evaluation_interval != 100
            or list(source.gait.fixed) != list(self.config.gait.fixed)
        ):
            raise ValueError(
                "fixed checkpoint has not passed three consecutive current evaluations"
            )
        review_path = self.config.training.visual_review
        if not review_path:
            raise ValueError("stage transition requires a checkpoint-bound visual_review JSON")
        review = json.loads(Path(str(review_path)).read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checks = ("no_dragging", "no_crossing", "no_nonfoot_support", "no_frequent_limits")
        if (
            review.get("checkpoint_sha256") != digest
            or not review.get("reviewer")
            or not all(review.get(k) is True for k in checks)
        ):
            raise ValueError("visual review must confirm all checks for this exact checkpoint")
        self.policy.load_state_dict(payload["actor_state_dict"])
        self.stage_source = str(path.resolve())

    def evaluate(self) -> dict[str, float]:
        if self.config.observation == "pe03_v4":
            from unilab.algos.torch.pe03.gait_evaluation import evaluate_gait

            metrics, report = evaluate_gait(
                self.policy,
                self.config,
                self.device,
                randomization_level=self.env.robustness_curriculum.level,
            )
            self.acceptance_records.append(
                dict(
                    iteration=self.iteration,
                    passed=report["acceptance"]["passed"],
                    protocol=report["protocol"]["fingerprint"],
                    randomization_level=report["randomization"]["level"],
                )
            )
            self.acceptance_records = self.acceptance_records[
                -int(self.config.training.acceptance.consecutive) :
            ]
            report["iteration"] = self.iteration
            (self.output_dir / f"evaluation_{self.iteration}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            return cast(dict[str, float], metrics)
        env = PE03VectorEnv(
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
                    standing = np.ones(env.num_envs, dtype=bool)
                    if self.config.reward.scales.base_height != 0:
                        standing &= np.abs(
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
                for key, value in tensorboard_metrics(metrics, policy_dt=self.env.dt).items():
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
