"""PE05-owned velocity cells and full-window scoring, adapted from PE03's course.

The four promotion scores follow PE03's tracking/contact formulas; they never
replace PE05's weighted PPO rewards. No runtime dependency on another robot.
"""

import itertools

import numpy as np
from scipy.special import ndtr


class VelocityCommandCurriculum:
    def __init__(self, config, num_envs, interval):
        self.config = config
        self.interval = interval
        ranges = np.array([config["ranges"][k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")])
        self.low, self.high = ranges.T
        self.width = np.asarray(config["bin_width"])
        axes = [np.arange(a + w / 2, b, w) for a, b, w in zip(self.low, self.high, self.width)]
        self.shape = tuple(map(len, axes))
        self.grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.weights = np.zeros(len(self.grid))
        active = (
            (self.grid >= np.asarray(config["initial_low"]) - 1e-8)
            & (self.grid <= np.asarray(config["initial_high"]) + 1e-8)
        ).all(1)
        if not active.any():
            raise ValueError("command curriculum initial domain contains no cells")
        self.weights[active] = 1
        self.offsets = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        self.bins = np.full(num_envs, -1, dtype=np.int64)
        self.zero = np.zeros(num_envs, dtype=bool)
        self.scores = np.zeros((num_envs, 4))
        self.ticks = np.zeros(num_envs, dtype=np.int64)
        self.completed_windows = 0
        self.successful_windows = 0

    def sample(self, rng, ids):
        bins = rng.choice(len(self.grid), len(ids), p=self.weights / self.weights.sum())
        commands = self.grid[bins] + rng.uniform(-0.5, 0.5, (len(ids), 3)) * self.width
        commands = np.clip(commands, self.low, self.high)
        zero = rng.random(len(ids)) < self.config["zero_probability"]
        commands[zero] = 0
        self.bins[ids], self.zero[ids] = bins, zero
        self.scores[ids], self.ticks[ids] = 0, 0
        return commands

    def tracking_scores(self, commands, velocity, yaw_rate, phase, gaits, feet, weight):
        cfg = self.config
        foot_phase = (phase[:, None] + gaits[:, 1:2] * [0, 1]) % 1
        support = gaits[:, 2:3]
        mapped = np.where(
            foot_phase < support,
            foot_phase * 0.5 / support,
            0.5 + (foot_phase - support) * 0.5 / (1 - support),
        )
        kappa = cfg["score_contact_kappa"]
        desired = ndtr(mapped / kappa) * (1 - ndtr((mapped - 0.5) / kappa))
        desired += ndtr((mapped - 1) / kappa) * (1 - ndtr((mapped - 1.5) / kappa))
        force_square = np.square(feet["ground_force"]).sum(2)
        speed_square = np.square(feet["contact_velocity"]).sum(2)
        return np.column_stack(
            (
                np.exp(
                    -np.square(commands[:, :2] - velocity[:, :2]).sum(1)
                    / cfg["score_tracking_sigma"]
                ),
                np.exp(-np.square(commands[:, 2] - yaw_rate) / cfg["score_yaw_sigma"]),
                1
                - (
                    (1 - desired)
                    * (
                        1
                        - np.exp(
                            -force_square
                            / (cfg["score_force_weight_fraction"] * weight[:, None]) ** 2
                        )
                    )
                ).mean(1),
                1
                - (
                    desired * (1 - np.exp(-speed_square / cfg["score_contact_velocity_sigma"]))
                ).mean(1),
            )
        )

    def observe(self, scores, terminated, truncated):
        self.scores += scores
        self.ticks += 1
        ids = np.flatnonzero((self.ticks >= self.interval) | terminated | truncated)
        success = (self.scores[ids] / self.interval >= self.config["thresholds"]).all(1)
        success &= ~terminated[ids] & ~self.zero[ids]
        selected = np.unique(self.bins[ids][success & (self.bins[ids] >= 0)])
        if len(selected):
            indices = np.array(np.unravel_index(selected, self.shape)).T
            neighbors = (indices[:, None] + self.offsets).reshape(-1, 3)
            neighbors = neighbors[((neighbors >= 0) & (neighbors < self.shape)).all(1)]
            flat = np.unique(np.ravel_multi_index(neighbors.T, self.shape))
            self.weights[flat] = np.minimum(1, self.weights[flat] + self.config["weight_increment"])
        self.completed_windows += len(ids)
        self.successful_windows += int(success.sum())
        self.scores[ids], self.ticks[ids] = 0, 0
        return ids

    def metrics(self):
        active = self.grid[self.weights > 0]
        result = {
            "command_curriculum/active_fraction": float((self.weights > 0).mean()),
            "command_curriculum/completed_windows": float(self.completed_windows),
            "command_curriculum/successful_windows": float(self.successful_windows),
        }
        for i, name in enumerate(("vx", "vy", "yaw")):
            result[f"command_curriculum/{name}_min"] = float(
                max(self.low[i], active[:, i].min() - self.width[i] / 2)
            )
            result[f"command_curriculum/{name}_max"] = float(
                min(self.high[i], active[:, i].max() + self.width[i] / 2)
            )
        return result

    def snapshot(self):
        return {
            **{
                name: getattr(self, name).copy()
                for name in ("weights", "bins", "zero", "scores", "ticks")
            },
            "completed_windows": self.completed_windows,
            "successful_windows": self.successful_windows,
        }

    def restore(self, state):
        for name in ("weights", "bins", "zero", "scores", "ticks"):
            if state[name].shape != getattr(self, name).shape:
                raise ValueError("command curriculum checkpoint dimensions differ")
            getattr(self, name)[:] = state[name]
        self.completed_windows = state["completed_windows"]
        self.successful_windows = state["successful_windows"]
