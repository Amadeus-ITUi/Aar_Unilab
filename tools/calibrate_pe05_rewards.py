"""Offline PE03 trajectory / PE05 reward-scale audit; never imports into training.

Run with the supported SSD Python from the repository root. Outputs go to logs.
The source policy only controls PE03. PE05 rewards are evaluated on a read-only
view of its physical samples with PE05's own phase/targets and contact history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.pe03.policy import PE03EncoderPolicy
from unilab.algos.torch.pe03.runner import asset_fingerprint
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe05.config import load_config
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFERENCE = Path("logs/pe03_gait_fixed/2026-09-21_21-43-13_055597_mujoco/model_20000.pt")
COMMANDS = np.array(
    [[0, 0, 0], [0.3, 0, 0], [-0.3, 0, 0], [0, 0.1, 0], [0, -0.1, 0], [0, 0, 0.4], [0, 0, -0.4]]
)


def percentiles(values):
    values = np.asarray(values)
    return dict(zip(("p50", "p90", "p95"), np.percentile(values, [50, 90, 95]).tolist()))


def verify_substeps(config):
    """Validate that collecting every sensor sample does not change nominal PD."""
    first = PE03GaitEnv(config, evaluation=True, robustness=False, num_envs=1)
    second = PE03GaitEnv(config, evaluation=True, robustness=False, num_envs=1)
    maxima = dict.fromkeys(("qpos", "qvel", "joint_velocity", "torque"), 0.0)
    rng = np.random.default_rng(123)
    try:
        for _ in range(25):
            actions = rng.uniform(-0.1, 0.1, (1, 6))
            first.backend.step(actions, 8)
            for _ in range(8):
                second.backend.step(actions, 1)
            for name in maxima:
                left, right = getattr(first.backend, name), getattr(second.backend, name)
                maxima[name] = max(maxima[name], float(np.abs(left - right).max()))
                np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-10)
        return maxima
    finally:
        first.close()
        second.close()


def scenario_rewards(config):
    """Isolated counterfactual inputs to the production reward, not simulated gaits."""
    env = PE05VectorEnv(config, evaluation=True, num_envs=1)
    try:
        result = {}

        def record(name, failure=False):
            reward, terms = env._rewards()
            if failure:
                reward, terms = env.adaptation.apply_failure_cost(reward, terms, np.array([True]))
            result[name] = dict(
                total=float(reward[0]), **{k: float(v[0]) for k, v in terms.items()}
            )

        record("home_initial")
        feet = env.backend.gait_foot_state(np.array(env.cfg["env"]["reference_points"]))
        env.backend.gait_foot_state = lambda points: feet
        env.phase[:] = 0.75
        env.commands[:] = [0.3, 0, 0]
        env.base_velocity[:] = [0.3, 0, 0]
        feet["ground_force"][:] = 0
        feet["ground_force"][:, 1, 2] = 10
        feet["clearance"][:] = [0.03, 0]
        feet["contact_velocity"][:] = 0
        record("ideal_tracking_inputs")
        feet["clearance"][:, 0] = 0
        feet["ground_force"][:, 0, 2] = 10
        feet["contact_velocity"][:, 0, 0] = 0.2
        record("swing_drag_inputs")
        env.backend.contact_history[:, 0, env.penalized[0], 2] = 10
        record("nonfoot_contact_inputs")
        record("failure_inputs", failure=True)
        assert result["ideal_tracking_inputs"]["total"] > result["swing_drag_inputs"]["total"]
        assert result["swing_drag_inputs"]["total"] > result["nonfoot_contact_inputs"]["total"]
        assert result["nonfoot_contact_inputs"]["total"] > result["failure_inputs"]["total"]
        return result
    finally:
        env.close()


def curves(source, old, new, output, *, dt, action_scale):
    p0 = sum(source["scales"][k] for k in ("tracking_lin_vel", "tracking_ang_vel")) * dt
    factor = p0 / source["sigma_negative"]
    normalized = {
        "base_height": "base_height_std",
        "foot_clearance": "foot_clearance_std",
        "feet_slip": "slip_velocity_std",
        "orientation": "orientation_std",
    }
    calculated = {
        name: factor * source["scales"][name] * new[std] ** 2 for name, std in normalized.items()
    }
    for name in ("lin_vel_z", "ang_vel_xy", "torques", "dof_acc"):
        calculated[name] = factor * source["scales"][name]
    calculated.update(
        action_rate=factor
        * (
            source["scales"]["action_rate"]
            + source["scales"]["target_smoothness_1"] * action_scale**2
        ),
        action_smooth=factor * source["scales"]["target_smoothness_2"] * action_scale**2,
        dof_pos_limits=factor * source["scales"]["dof_pos_limits"] * new["joint_limit_std"],
        collision=source["scales"]["collision"],
        tracking_lin_vel=source["scales"]["tracking_lin_vel"],
        tracking_ang_vel=source["scales"]["tracking_ang_vel"],
    )
    for name, value in calculated.items():
        np.testing.assert_allclose(new["scales"][name], value, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(new["tracking_velocity_std"] ** 2, source["tracking_sigma"])
    np.testing.assert_allclose(new["tracking_yaw_std"] ** 2, source["yaw_tracking_sigma"])
    specs = [
        ("base_height", "base_height_std", [0.01, 0.02, 0.03], 0.06, "height error (m)"),
        (
            "foot_clearance",
            "foot_clearance_std",
            [0.005, 0.01, 0.02],
            0.04,
            "one swing foot error (m)",
        ),
        ("feet_slip", "slip_velocity_std", [0.05, 0.1, 0.2], 0.4, "one contact slip speed (m/s)"),
        ("orientation", "orientation_std", [5, 10, 20], 30, "tilt (degrees)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    report = {
        "dt": dt,
        "p0": p0,
        "local_factor": factor,
        "calculated_weights": calculated,
        "curves": {},
    }
    for ax, (name, std, anchors, maximum, label) in zip(axes.flat, specs):

        def evaluate(x, name=name, std=std):
            e = np.sin(np.deg2rad(x)) if name == "orientation" else x
            cost = abs(source["scales"][name]) * e**2 * dt
            return {
                "pe03_isolated_loss": p0 * np.expm1(-cost / source["sigma_negative"]),
                "pe05_before": old["scales"][name] * (e / old[std]) ** 2 * dt,
                "pe05_calibrated": new["scales"][name] * (e / new[std]) ** 2 * dt,
            }

        x = np.linspace(0, maximum, 200)
        for key, y in evaluate(x).items():
            ax.plot(x, y, label=key)
        ax.set(xlabel=label, ylabel="reward contribution / step")
        ax.grid(alpha=0.3)
        report["curves"][name] = {
            "anchors": anchors,
            "contributions": {k: v.tolist() for k, v in evaluate(np.array(anchors)).items()},
            "old_coefficient": abs(old["scales"][name]) / old[std] ** 2,
            "new_coefficient": abs(new["scales"][name]) / new[std] ** 2,
        }
    for ax, name, std in zip(
        list(axes.flat)[4:],
        ("tracking_lin_vel", "tracking_ang_vel"),
        ("tracking_velocity_std", "tracking_yaw_std"),
    ):
        x = np.linspace(0, 1, 200)
        sigma = source["tracking_sigma" if name == "tracking_lin_vel" else "yaw_tracking_sigma"]
        weight = source["scales"][name]
        ax.plot(x, weight * dt * np.exp(-(x**2) / sigma), label="PE03")
        for label, cfg in (("before", old), ("calibrated", new)):
            ax.plot(x, cfg["scales"][name] * dt * (1 - (x / cfg[std]) ** 2), label=label)
        ax.set(xlabel=name + " error", ylabel="reward contribution / step")
        ax.grid(alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "reward_curves.png", dpi=160)
    plt.close(fig)
    return report


def audit(checkpoint, old_task, output, steps, calibrated_task):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(payload["training_config"])
    saved_source = OmegaConf.to_container(cfg, resolve=True)
    source_asset_hash = asset_fingerprint(cfg)
    if source_asset_hash != payload["asset_sha256"]:
        raise ValueError("PE03 assets differ from the reference checkpoint")
    cfg.training.mujoco_threads = 1
    cfg.noise.enabled = False
    cfg.domain_rand.enabled = False
    cfg.domain_rand.push_enabled = False
    substep_errors = verify_substeps(cfg)
    policy = PE03EncoderPolicy(cfg).eval()
    policy.load_state_dict(payload["actor_state_dict"])
    old = OmegaConf.to_container(OmegaConf.load(old_task), resolve=True)["reward"]
    target_cfg = load_config(["training.mujoco_threads=1", "commands.curriculum=false"])
    target_cfg.reward = OmegaConf.load(calibrated_task).reward
    source = PE03GaitEnv(cfg, evaluation=True, robustness=False, num_envs=7, auto_reset=False)
    target = PE05VectorEnv(target_cfg, evaluation=True, num_envs=7, auto_reset=False)
    target_backend = target.backend
    new = target.cfg["reward"]
    assert source.dt == target.dt
    assert cfg.control.action_scale == target_cfg.control.action_scale
    result = curves(
        saved_source["reward"],
        old,
        new,
        output,
        dt=source.dt,
        action_scale=cfg.control.action_scale,
    )
    result.update(
        isolated_scenarios=scenario_rewards(target_cfg),
        substep_equivalence_max_error=substep_errors,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_asset_sha256=source_asset_hash,
        source_reward=saved_source["reward"],
        target_reward=new,
        old_reward=old,
        source_randomization=False,
        commands=COMMANDS.tolist(),
        limitations=[
            "Source contact sensors sampled at each physics step via eight equivalent 1-step PD calls.",
            "PE05 phase/50% support/sine-squared targets differ from PE03 60%/triangular targets.",
            "PE05 history-any collision and instantaneous slip mask differ from PE03 current collision and contact OR previous.",
            "PE05 smoothness lacks PE03 startup masks; scalar action_rate merges two PE03 terms only after startup.",
            "PE03 soft limits protect home with a capped margin; PE05 uses symmetric margins. Both homes are legal.",
            "Offline totals exclude PE05 failure cost: source termination is not a PE05 termination replay.",
            "Discrete contact_schedule, feet_distance and termination are retained PE05 budgets, not exact translations.",
        ],
    )
    errors, rewards, artifacts = {}, {}, {}
    live = np.ones(7, bool)
    lengths = np.zeros(7, int)
    contacts, grounds = [], []
    original_step = source.backend.step

    def sampled_step(actions, substeps, **kwargs):
        if kwargs.get("push_forces") is not None:
            raise ValueError("calibration must be nominal")
        contacts.clear()
        grounds.clear()
        for _ in range(substeps):
            original_step(actions, 1)
            contacts.append(-source.backend.contact_forces.copy())
            grounds.append(
                source.backend.gait_foot_state(source.reference_points)["body_ground_force"].copy()
            )

    source.backend.step = sampled_step
    try:
        source.set_command(COMMANDS, np.tile(cfg.gait.fixed, (7, 1)))
        source.history[:] = source._frame()[:, None]
        state = source._observe()
        with torch.inference_mode():
            for _ in range(steps):
                obs = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in state.items()}
                action = policy.action_mean(obs["actor"], obs["frame"], obs["command"]).numpy()
                previous_velocity = source.backend.joint_velocity.copy()
                transition = source.step(action)
                b = source.backend
                feet = b.gait_foot_state(source.reference_points)
                view = SimpleNamespace(
                    qpos=b.qpos,
                    qvel=b.qvel,
                    joint_velocity=b.joint_velocity,
                    torque=b.torque,
                    contact_history=np.stack(contacts[::-1], axis=1),
                    ground_contact_history=np.stack(grounds[::-1], axis=1),
                    gait_foot_state=lambda points: feet,
                )
                target.backend = view
                target.commands[:] = COMMANDS
                target.actions[:] = source.actions
                target.last_actions[:, 0] = source.last_actions
                target.last_actions[:, 1] = source.old_actions
                target.previous_velocity[:] = previous_velocity
                target.base_velocity[:] = source.base_velocity
                target.phase[:] = source.episode_steps * source.dt * 2 % 1
                target.episode_steps[:] = source.episode_steps
                for name, reward_cfg in (("before", old), ("calibrated", new)):
                    target.cfg["reward"] = reward_cfg
                    mid = target_backend.joint_range.mean(1)
                    half = (
                        np.diff(target_backend.joint_range, axis=1)[:, 0]
                        * 0.5
                        * reward_cfg["soft_joint_limit"]
                    )
                    target.soft_limits[:] = np.column_stack((mid - half, mid + half))
                    total, terms = target._rewards()
                    for term, value in dict(total=total, **terms).items():
                        rewards.setdefault(name + "/" + term, []).extend(value[live].tolist())
                info = transition.info
                physical = dict(
                    height_m=info["base_height_error"],
                    velocity_mps=np.linalg.norm(
                        source.commands[:, :2] - source.base_velocity[:, :2], axis=1
                    ),
                    yaw_radps=np.abs(source.commands[:, 2] - b.qvel[:, 5]),
                    tilt_deg=info["base_tilt_deg"],
                    slip_mps=info["slip_speed"],
                    clearance_m=target.adaptation.latest["foot_clearance"],
                    clearance_error_m=np.abs(
                        target.adaptation.latest["foot_clearance"]
                        - target.adaptation.latest["target_clearance"]
                    ),
                    source_attenuation=info["attenuation"],
                    source_effective_factor=info["positive_reward"]
                    * info["attenuation"]
                    / cfg.reward.sigma_negative,
                )
                for name, value in physical.items():
                    if name == "clearance_error_m":
                        value = value[live][target.adaptation.latest["swing"][live]]
                    else:
                        value = value[live].ravel()
                    errors.setdefault(name, []).extend(value.tolist())
                for name, value in dict(
                    qpos=b.qpos,
                    qvel=b.qvel,
                    action=source.actions,
                    command=COMMANDS,
                    phase=target.phase,
                    contact_history=view.contact_history,
                    ground_history=view.ground_contact_history,
                    source_reward=transition.reward,
                    live=live,
                ).items():
                    artifacts.setdefault(name, []).append(value.copy())
                lengths += live
                live &= ~(transition.terminated | transition.truncated)
                state = transition.obs
                if not live.any():
                    break
        result["source_episode_seconds"] = (lengths * source.dt).tolist()
        result["physical_error_percentiles"] = {k: percentiles(v) for k, v in errors.items()}
        result["reward_contributions"] = {
            k: dict(mean=float(np.mean(v)), **percentiles(v)) for k, v in rewards.items()
        }
        np.savez_compressed(
            output / "source_trajectory.npz", **{k: np.array(v) for k, v in artifacts.items()}
        )
        (output / "calibration.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "checkpoint_sha256",
                        "source_episode_seconds",
                        "physical_error_percentiles",
                    )
                },
                indent=2,
            )
        )
    finally:
        target.backend = target_backend
        target.close()
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=Path("logs/reports/pe05_scale_calibration"))
    parser.add_argument("--old-task", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--calibrated-task",
        type=Path,
        default=Path("docs/assets/pe05_pe01_gait/before_task.yaml"),
        help="Frozen 17-term PE03-calibrated task, independent of current training defaults",
    )
    args = parser.parse_args()
    audit(args.checkpoint, args.old_task, args.output, args.steps, args.calibrated_task)
