"""Deterministic full-range gait evaluation, separate from short engineering checks."""

import numpy as np
import torch

from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.gait_randomization import evaluation_protocol


def evaluate_gait(policy, config, device, *, randomization_level=None):
    protocol = evaluation_protocol(config)
    env = PE03GaitEnv(
        config,
        evaluation=True,
        num_envs=config.training.evaluation_episodes,
        robustness=protocol["conditions"] == "randomized",
        randomization_level=randomization_level,
    )
    try:
        n, steps = env.num_envs, env.max_episode_steps
        rng = np.random.default_rng(int(config.training.seed) + 12345)
        commands = rng.uniform(config.commands.low, config.commands.high, (n, 3))
        zero = np.zeros(n, bool)
        zero[: max(1, round(n * config.commands.zero_probability))] = True
        rng.shuffle(zero)
        commands[zero] = 0
        target_gait = np.tile(config.gait.fixed, (n, 1)).astype(float)
        if config.gait.stage == "variable":
            variable = rng.random(n) >= config.gait.fixed_probability
            target_gait[variable] = rng.uniform(
                config.gait.low, config.gait.high, (variable.sum(), 3)
            )
        env.set_command(commands, target_gait)
        env.history[:] = env._frame()[:, None]
        state = env.state
        state.obs = env._observe()
        live, failed = np.ones(n, bool), np.zeros(n, bool)
        lengths, returns = np.zeros(n), np.zeros(n)
        scores = np.zeros((n, 4))
        correct, contact_samples = np.zeros(n), np.zeros(n)
        heights = np.full((steps, n), np.nan)
        sums = {
            key: np.zeros(n)
            for key in (
                "nonfoot_contact",
                "joint_limit",
                "slip_speed",
                "crossing",
                "torque_saturation",
            )
        }
        peak, duty, frequency = (
            [[] for _ in range(n)],
            [[] for _ in range(n)],
            [[] for _ in range(n)],
        )
        estimator = np.zeros((n, 3))
        failure_reasons = np.zeros((n, 3))
        with torch.no_grad():
            for tick in range(steps):
                obs = {
                    key: torch.as_tensor(value, dtype=torch.float32, device=device)
                    for key, value in state.obs.items()
                }
                estimate = policy.encode(obs["actor"]).cpu().numpy()
                estimator[live] += (estimate[live] - state.obs["critic"][live, :3]) ** 2
                action = policy.action_mean(obs["actor"], obs["frame"], obs["command"])
                state = env.step(action.cpu().numpy())
                info = state.info
                returns[live] += state.reward[live]
                lengths[live] += 1
                scores[live] += info["tracking_scores"][live]
                correct[live] += info["contact_correct"][live]
                contact_samples[live] += info["contact_samples"][live]
                heights[tick, live] = info["base_height_error"][live]
                for key in sums:
                    sums[key][live] += info[key][live]
                failure_reasons[live] += info["failure_reasons"][live]
                for idx in np.flatnonzero(live):
                    events = info["cycle_event"][idx]
                    peak[idx].extend(info["peak_error"][idx, events])
                    duty[idx].extend(info["duty_error"][idx, events])
                    frequency[idx].extend(info["actual_frequency"][idx, info["touch_event"][idx]])
                failed |= live & state.terminated
                live &= ~(state.terminated | state.truncated)
                if not live.any():
                    break
        groups = {
            "all": np.ones(n, bool),
            "zero": zero,
            "nonzero": ~zero,
            "forward": commands[:, 0] > 0.1,
            "backward": commands[:, 0] < -0.1,
            "low_speed": (~zero) & (np.linalg.norm(commands[:, :2], axis=1) <= 0.5),
            "high_speed": np.linalg.norm(commands[:, :2], axis=1) > 0.5,
        }
        for axis, name in enumerate(("vx", "vy", "yaw")):
            idx = np.floor(
                (commands[:, axis] - config.commands.low[axis]) / config.commands.bin_width[axis]
            ).astype(int)
            for cell in np.unique(idx[~zero]):
                groups[f"{name}_bin_{cell}"] = (idx == cell) & ~zero
        if config.gait.stage == "variable":
            joint_cells = np.rint(
                (target_gait - np.asarray(config.gait.low)) / np.asarray(config.gait.bin_width)
            ).astype(int)
            for cell in np.unique(joint_cells, axis=0):
                groups["gait_cell_" + "_".join(map(str, cell))] = (joint_cells == cell).all(1)
            for axis, name in enumerate(("frequency", "support", "clearance")):
                idx = np.rint(
                    (target_gait[:, axis] - config.gait.low[axis]) / config.gait.bin_width[axis]
                ).astype(int)
                for cell in np.unique(idx):
                    groups[f"{name}_bin_{cell}"] = idx == cell
        report = {}
        for name, selected in groups.items():
            ids = np.flatnonzero(selected)
            if not len(ids):
                continue
            peaks = [x for idx in ids for x in peak[idx]]
            duties = [x for idx in ids for x in duty[idx]]
            frequencies = [x for idx in ids for x in frequency[idx]]
            result = dict(
                episodes=len(ids),
                success_rate=float((~failed[ids]).mean()),
                return_mean=float(returns[ids].mean()),
                episode_seconds=float(lengths[ids].mean() * env.dt),
                contact_accuracy=float(correct[ids].sum() / max(1, contact_samples[ids].sum())),
                height_p95=float(np.nanpercentile(heights[:, ids], 95)),
                peak_mae=float(np.mean(peaks)) if peaks else 1.0,
                duty_error=float(np.mean(duties)) if duties else 1.0,
                actual_frequency=float(np.mean(frequencies)) if frequencies else 0.0,
                completed_cycles=len(peaks),
            )
            for index, metric in enumerate(
                ("tracking_lin_vel", "tracking_ang_vel", "contact_force", "contact_velocity")
            ):
                result[metric] = float((scores[ids, index] / steps).mean())
            for key, value in sums.items():
                result[key] = float(value[ids].sum() / max(1, lengths[ids].sum()))
            for axis, label in enumerate(("x", "y", "z")):
                result[f"velocity_rmse_{label}"] = float(
                    np.sqrt(estimator[ids, axis].sum() / max(1, lengths[ids].sum()))
                )
            for axis, label in enumerate(("body_contact", "tilt", "height")):
                result[f"failure_{label}"] = float(
                    failure_reasons[ids, axis].sum() / max(1, lengths[ids].sum())
                )
            report[name] = result
        a, result = config.training.acceptance, report["all"]
        tolerance = (
            a.fixed_clearance_mae if config.gait.stage == "fixed" else a.variable_clearance_mae
        )
        checks = dict(
            success=result["success_rate"] >= a.success_rate,
            tracking=all(
                result[key] >= threshold
                for key, threshold in zip(
                    ("tracking_lin_vel", "tracking_ang_vel", "contact_force", "contact_velocity"),
                    config.curriculum.thresholds,
                )
            ),
            contact=result["contact_accuracy"] >= a.contact_accuracy,
            height=result["height_p95"] <= a.height_p95,
            clearance=result["completed_cycles"] > 0 and result["peak_mae"] <= tolerance,
            duty=result["duty_error"] <= a.duty_error,
            nonfoot=result["nonfoot_contact"] <= a.max_nonfoot_fraction,
            limits=result["joint_limit"] <= a.max_limit_fraction,
            slip=result["slip_speed"] <= a.max_slip_speed,
            crossing=result["crossing"] == 0,
        )
        checks["evaluation_protocol"] = (
            n >= 64 and steps * env.dt >= 20 and config.training.evaluation_interval == 100
        )
        if env.robustness_curriculum.enabled:
            checks["randomization_level"] = (
                env.robustness_curriculum.level >= env.robustness_curriculum.maximum
            )
        report["acceptance"] = {
            "checks": checks,
            "passed": all(checks.values()),
            "visual_review_required": True,
        }
        metrics = {
            f"evaluation/{group}/{key}": float(value)
            for group, values in report.items()
            if group != "acceptance"
            for key, value in values.items()
        }
        metrics["evaluation/acceptance_passed"] = float(all(checks.values()))
        metrics["evaluation/randomized_conditions"] = float(env.robustness)
        metrics["evaluation/randomization_level"] = env.robustness_curriculum.level
        report["protocol"] = protocol
        report["randomization"] = {
            "level": env.robustness_curriculum.level,
            "friction": env.friction.tolist(),
            "delay_ms": (env.backend.delay_steps * env.backend.dt * 1000).tolist(),
            "total_mass": (env.weight / env.backend.gravity_acceleration).tolist(),
        }
        return metrics, report
    finally:
        env.close()
