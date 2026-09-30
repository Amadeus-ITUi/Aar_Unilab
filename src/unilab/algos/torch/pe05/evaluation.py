"""Command-stratified evaluation for PE05's weighted task (no policy inputs added)."""

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.pe05.gait_metrics import GaitMetrics
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


def evaluate_commands(runner) -> dict[str, float]:
    metrics = {}
    summaries = []
    cfg = runner.config.evaluation
    for name, command in cfg.commands.items():
        config = OmegaConf.merge(runner.config)
        config.play.command = command
        config.play.delay_ms = cfg.delay_ms
        env = PE05VectorEnv(
            config,
            evaluation=True,
            num_envs=int(config.training.evaluation_episodes),
            evaluation_reset_noise=True,
            auto_reset=False,
        )
        try:
            gait_metrics = GaitMetrics(env.dt)
            live = np.ones(env.num_envs, bool)
            counts = np.zeros(env.num_envs)
            returns = np.zeros(env.num_envs)
            failures = np.zeros(env.num_envs, bool)
            errors = np.zeros((env.num_envs, 4))
            height_samples = []
            sums = dict(
                base_height=0.0,
                base_tilt_deg=0.0,
                nonfoot_contact=0.0,
                standing_fraction=0.0,
                height_error_m=0.0,
                tracking_error_mps=0.0,
                yaw_error_radps=0.0,
                slip_speed_mps=0.0,
            )
            for body in config.env.penalized_bodies:
                sums[f"contact/{body}/ground_fraction"] = 0.0
                sums[f"contact/{body}/self_fraction"] = 0.0
            swing_sum, swing_count = 0.0, 0
            state = env.state
            with torch.inference_mode():
                for _ in range(env.max_episode_steps):
                    obs = runner._tensors(state.obs)
                    actions = runner.policy.action_mean(obs["actor"], obs["frame"], obs["command"])
                    state = env.step(actions.cpu().numpy())
                    info, behavior = state.info, state.info["behavior"]
                    gait_metrics.update(behavior, live)
                    for key in ("base_height", "base_tilt_deg", "nonfoot_contact"):
                        sums[key] += float(info[key][live].sum())
                    for out, key in (
                        ("height_error_m", "height_error"),
                        ("tracking_error_mps", "tracking_error"),
                        ("yaw_error_radps", "yaw_error"),
                    ):
                        sums[out] += float(behavior[key][live].sum())
                    standing = (
                        (behavior["height_error"] < config.training.evaluation_height_tolerance)
                        & (info["base_tilt_deg"] < config.training.evaluation_tilt_deg)
                        & (info["base_speed"] < config.training.evaluation_speed_tolerance)
                        & ~info["nonfoot_contact"]
                    )
                    sums["standing_fraction"] += float(standing[live].sum())
                    sums["slip_speed_mps"] += float(behavior["slip_speed"][live].mean(1).sum())
                    swing = behavior["swing"][live]
                    swing_sum += float(
                        (
                            np.abs(
                                behavior["foot_clearance"][live]
                                - behavior["target_clearance"][live]
                            )
                            * swing
                        ).sum()
                    )
                    swing_count += int(swing.sum())
                    for i, body in enumerate(config.env.penalized_bodies):
                        sums[f"contact/{body}/ground_fraction"] += float(
                            behavior["ground_contact"][live, i].sum()
                        )
                        sums[f"contact/{body}/self_fraction"] += float(
                            behavior["self_contact"][live, i].sum()
                        )
                    height_samples.extend(behavior["height_error"][live].tolist())
                    errors[live] += np.column_stack(
                        (
                            info["nonfoot_contact"],
                            behavior["height_error"],
                            behavior["tracking_error"],
                            behavior["yaw_error"],
                        )
                    )[live]
                    counts[live] += 1
                    returns[live] += state.reward[live]
                    failures |= live & state.terminated
                    live &= ~(state.terminated | state.truncated)
                    if not live.any():
                        break
            means = errors / np.maximum(counts[:, None], 1)
            success = (~failures) & (counts >= env.max_episode_steps)
            success &= (
                means
                <= [
                    cfg.max_nonfoot_fraction,
                    cfg.max_height_error,
                    cfg.max_tracking_error,
                    cfg.max_yaw_error,
                ]
            ).all(1)
            result = {key: value / max(1, counts.sum()) for key, value in sums.items()}
            result.update(return_=float(returns.mean()))
            result["return"] = result.pop("return_")
            result.update(
                episode_seconds=float(counts.mean() * env.dt),
                failure_rate=float(failures.mean()),
                success_rate=float(success.mean()),
                height_error_p95_m=float(np.percentile(height_samples, 95)),
                clearance_error_m=swing_sum / max(1, swing_count),
            )
            result.update(gait_metrics.result())
            metrics.update({f"evaluation/{name}/{key}": value for key, value in result.items()})
            summaries.append(result)
        finally:
            env.close()
    # Aggregate gives equal weight to every requested command, independent of early failures.
    metrics.update(
        {f"evaluation/{key}": float(np.mean([s[key] for s in summaries])) for key in summaries[0]}
    )
    return metrics
