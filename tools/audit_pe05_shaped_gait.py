"""Reproduce synthetic full-cycle PE05 reward budgets; not a dynamics rollout."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from unilab.envs.locomotion.pe05.config import load_config
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


def audit(output):
    cfg = load_config(["algo.num_envs=5", "training.mujoco_threads=1"])
    env = PE05VectorEnv(cfg, evaluation=True)
    names = ("static", "alternating", "wrong_phase", "dragging", "fast_landing")
    rows = {name: [] for name in names}
    try:
        feet = env.backend.gait_foot_state(np.array(env.cfg["env"]["reference_points"]))
        env.backend.gait_foot_state = lambda points: feet
        env.commands[:] = env.base_velocity[:] = 0
        env.backend.qvel[:] = env.backend.torque[:] = env.previous_velocity[:] = 0
        env.backend.qpos[:, 2] = env.height_target
        for tick in range(100):
            phase = (tick * env.dt * 2) % 1
            env.phase[:] = phase
            foot_phase = (phase + np.array([0, 0.5])) % 1
            swing = foot_phase >= 0.5
            travel = np.clip((foot_phase - 0.5) / 0.5, 0, 1)
            height = 0.03 * np.sin(np.pi * travel) ** 2
            vz = swing * 0.03 * np.pi * 4 * np.sin(2 * np.pi * travel)
            feet["ground_force"][:] = [0, 0, 20]
            feet["clearance"][:] = 0
            feet["reference_velocity"][:] = feet["contact_velocity"][:] = 0
            for i in (1, 2, 4):
                actual = ~swing if i == 2 else swing
                feet["ground_force"][i, actual] = 0
                feet["clearance"][i] = height[::-1] if i == 2 else height
                feet["reference_velocity"][i, :, 2] = vz[::-1] if i == 2 else vz
                feet["reference_velocity"][i, actual, 0] = 0.2
            feet["reference_velocity"][3, swing, 0] = 0.2
            feet["reference_velocity"][4, swing & (vz < 0), 2] = -2
            # Keep body/torque terms fixed; isolate explicit, smoothly varying action costs.
            for lag in range(3):
                angle = 2 * np.pi * 2 * (tick - lag) * env.dt
                actions = 0.2 * np.sin(angle + np.array([0, 0, 0, np.pi, np.pi, np.pi]))
                target = env.actions if lag == 0 else env.last_actions[:, lag - 1]
                target[:] = actions
                target[0] = 0
            total, terms = env._rewards()
            for i, name in enumerate(names):
                rows[name].append(
                    dict(
                        time_s=tick * env.dt,
                        total=float(total[i]),
                        **{k: float(v[i]) for k, v in terms.items()},
                    )
                )
        report = dict(
            kind="imposed_kinematic_scenarios_not_dynamics_or_policy_acceptance",
            dt=env.dt,
            configuration=OmegaConf.to_container(cfg, resolve=True),
            scenarios={
                name: {
                    k: float(np.mean([r[k] for r in values])) for k in values[0] if k != "time_s"
                }
                for name, values in rows.items()
            },
            traces=rows,
            notes=[
                "Body, torque, acceleration, collision and termination states are held fixed.",
                "Motion scenarios share one imposed action trace; static actions are constant.",
                "No claim that every imposed kinematic/action combination is dynamically realizable.",
                "Negative clearance is clamped only in feet_regulation, not in clearance tracking.",
            ],
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "cycle_audit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(report["scenarios"], indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/pe05_pe01_gait"))
    audit(parser.parse_args().output)
