"""Shared biped phase targets and success-driven command distributions."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import ndtr


def phase_targets(phase: np.ndarray, gait: np.ndarray, kappa: float):
    foot_phase = (phase[:, None] + np.array([0.0, 0.5])) % 1.0
    beta = gait[:, 1:2]
    stance = foot_phase < beta
    swing = np.clip((foot_phase - beta) / (1 - beta), 0, 1)
    mapped = np.where(stance, foot_phase * 0.5 / beta, 0.5 + 0.5 * swing)
    contact = ndtr(mapped / kappa) * (1 - ndtr((mapped - 0.5) / kappa))
    contact += ndtr((mapped - 1) / kappa) * (1 - ndtr((mapped - 1.5) / kappa))
    height = gait[:, 2:3] * (1 - np.abs(2 * swing - 1))
    travel = np.where(stance, 0.5 - foot_phase / beta, swing - 0.5)
    return contact, height, travel, stance


def placement_targets(commands, gait, travel, nominal):
    twist = np.stack((-nominal[:, 1], nominal[:, 0]), axis=-1)
    speed = commands[:, None, :2] + commands[:, None, 2:3] * twist
    return (
        nominal[None, :, :2] + travel[..., None] * speed * (gait[:, 1] / gait[:, 0])[:, None, None]
    )


class CommandCurriculum:
    """WTW-style weighted cells; failures are scored against the full window.

    Separate velocity and gait grids avoid a dense six-dimensional allocation.
    Joint velocity/gait success is still required before either grid expands.
    """

    def __init__(
        self, low, high, width, initial_low, initial_high, increment=0.2, *, centers=False
    ):
        self.low, self.high, self.width = map(np.asarray, (low, high, width))
        if centers:
            axes = [np.arange(a, b + w / 2, w) for a, b, w in zip(self.low, self.high, self.width)]
        else:
            axes = [np.arange(a + w / 2, b, w) for a, b, w in zip(self.low, self.high, self.width)]
        self.shape = tuple(map(len, axes))
        self.grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.weights = np.zeros(len(self.grid))
        active = (
            (self.grid >= np.asarray(initial_low) - 1e-8)
            & (self.grid <= np.asarray(initial_high) + 1e-8)
        ).all(1)
        if not active.any():
            raise ValueError("curriculum initial domain contains no cells")
        self.weights[active] = 1.0
        self.increment = increment
        self.offsets = np.array(list(itertools.product((-1, 0, 1), repeat=3)))

    def sample(self, rng, count):
        ids = rng.choice(len(self.grid), count, p=self.weights / self.weights.sum())
        values = self.grid[ids] + rng.uniform(-0.5, 0.5, (count, 3)) * self.width
        return np.clip(values, self.low, self.high), ids

    def update(self, ids, success):
        selected = np.unique(np.asarray(ids)[np.asarray(success) & (np.asarray(ids) >= 0)])
        if not len(selected):
            return
        indices = np.array(np.unravel_index(selected, self.shape)).T
        neighbors = (indices[:, None] + self.offsets).reshape(-1, 3)
        neighbors = neighbors[((neighbors >= 0) & (neighbors < self.shape)).all(1)]
        flat = np.unique(np.ravel_multi_index(neighbors.T, self.shape))
        self.weights[flat] = np.minimum(1, self.weights[flat] + self.increment)

    def snapshot(self):
        return self.weights.copy()

    def restore(self, weights):
        if weights.shape != self.weights.shape:
            raise ValueError("curriculum checkpoint dimensions differ")
        self.weights[:] = weights
