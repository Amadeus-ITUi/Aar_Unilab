"""Monotone robustness curriculum using completed episodes at the current level."""

from collections import deque

import numpy as np


class RobustnessCurriculum:
    def __init__(self, config, *, enabled, evaluation=False, level=None, num_envs=None):
        self.enabled = bool(enabled and config.get("enabled", False))
        self.evaluation = evaluation
        self.maximum = float(config.get("max_level", 1.0)) if self.enabled else 1.0
        self.increment = float(config.get("increment", 0.1))
        # Missing strategy identifies checkpoints with the original window-mean rule.
        self.strategy = config.get("strategy", "window_mean")
        self.num_envs = int(num_envs or 0)
        if self.strategy == "per_env_mean" and self.num_envs < 1:
            raise ValueError("per-env mean curriculum requires num_envs")
        # -1 means this environment has not yet contributed in this round.
        self.env_lengths = np.full(self.num_envs, -1, dtype=np.int64)
        self.promotion_truncations = 0
        fraction = self.strategy == "episode_fraction"
        self.window_size = int(
            config.get("recent_episodes", 100) if fraction else config.get("window_episodes", 4096)
        )
        self.threshold = float(
            config.get("min_episode_length", 950)
            if fraction
            else config.get("mean_episode_length", 950 if self.strategy == "per_env_mean" else 1000)
        )
        self.success_fraction = float(config.get("success_fraction", 0.5))
        self.recent_lengths = deque(maxlen=self.window_size)
        self.last_success_fraction = 0.0
        self.required_windows = int(config.get("consecutive_windows", 2))
        self.level = float(config.get("initial_level", 0.0)) if self.enabled else float(enabled)
        if self.enabled and evaluation:
            self.level = self.maximum if level is None else float(level)
        if not np.isfinite(self.level) or not 0 <= self.level <= self.maximum:
            raise ValueError("robustness level must be within the curriculum bounds")
        self.count = 0
        self.length_sum = 0
        self.successful_windows = 0
        self.completed_windows = 0
        self.promotions = 0
        self.last_window_level = self.level
        self.last_window_mean = 0.0

    def observe(self, lengths, levels, env_ids=None):
        """Observe natural episode ends; return whether this batch promoted the level."""
        if not self.enabled or self.evaluation:
            return False
        # A completed old-level episode is never evidence for the new level.
        current = np.asarray(levels) == self.level
        lengths = np.asarray(lengths)[current]
        if self.strategy == "per_env_mean":
            if env_ids is None:
                raise ValueError("per-env mean curriculum requires environment identities")
            return self._observe_per_env(lengths, np.asarray(env_ids, dtype=int)[current])
        previous_level = self.level
        if self.strategy == "episode_fraction":
            self._observe_fraction(lengths)
            return self.level > previous_level
        cursor = 0
        while cursor < len(lengths):
            take = min(self.window_size - self.count, len(lengths) - cursor)
            self.length_sum += int(lengths[cursor : cursor + take].sum())
            self.count += take
            cursor += take
            if self.count < self.window_size:
                break
            self.last_window_level = self.level
            self.last_window_mean = self.length_sum / self.count
            self.completed_windows += 1
            self.successful_windows = (
                self.successful_windows + 1 if self.last_window_mean >= self.threshold else 0
            )
            self.count = self.length_sum = 0
            if self.successful_windows >= self.required_windows and self.level < self.maximum:
                self.level = round(min(self.maximum, self.level + self.increment), 10)
                self.promotions += 1
                self.successful_windows = 0
                # Remaining episodes in this batch also began at the OLD level.
                break
        return self.level > previous_level

    def _observe_per_env(self, lengths, ids):
        if not len(ids):
            return False
        # Use each environment's first completed episode, including short failures.
        # Frequent failures/resets cannot crowd out the slower environments.
        ids, first = np.unique(ids, return_index=True)
        pending = self.env_lengths[ids] < 0
        lengths, ids = lengths[first][pending], ids[pending]
        self.env_lengths[ids] = lengths
        self.count += len(ids)
        self.length_sum += int(lengths.sum())
        if self.count < self.num_envs:
            return False
        self.last_window_level = self.level
        self.last_window_mean = self.length_sum / self.num_envs
        self.completed_windows += 1
        promoted = self.last_window_mean >= self.threshold and self.level < self.maximum
        if promoted:
            self.level = round(min(self.maximum, self.level + self.increment), 10)
            self.promotions += 1
        # Each round is disjoint; never repeatedly check an overlapping sample.
        self.env_lengths.fill(-1)
        self.count = self.length_sum = 0
        return promoted

    def _observe_fraction(self, lengths):
        if not len(lengths):
            return
        # Match a recent-episode statistic; process this completed batch once.
        self.recent_lengths.extend(int(x) for x in lengths[-self.window_size :])
        self.count = len(self.recent_lengths)
        self.length_sum = sum(self.recent_lengths)
        if self.count < self.window_size:
            return
        self.completed_windows += 1
        self.last_window_level = self.level
        self.last_window_mean = self.length_sum / self.count
        self.last_success_fraction = (
            sum(x >= self.threshold for x in self.recent_lengths) / self.count
        )
        if self.last_success_fraction >= self.success_fraction and self.level < self.maximum:
            self.level = round(min(self.maximum, self.level + self.increment), 10)
            self.promotions += 1
            self.recent_lengths.clear()
            self.count = self.length_sum = 0

    def metrics(self):
        values = {
            "randomization/level": self.level,
            "randomization/curriculum_enabled": float(self.enabled),
            "randomization/promotions": float(self.promotions),
        }
        if self.strategy == "per_env_mean":
            values.update(
                {
                    "randomization/episode_count": float(self.count),
                    "randomization/required_envs": float(self.num_envs),
                    "randomization/round_coverage": self.count / self.num_envs,
                    "randomization/mean_episode_length": self.length_sum / max(1, self.count),
                    "randomization/episode_length_threshold": self.threshold,
                    "randomization/last_check_level": self.last_window_level,
                    "randomization/last_check_mean_episode_length": self.last_window_mean,
                    "randomization/checks": float(self.completed_windows),
                    "randomization/promotion_truncations": float(self.promotion_truncations),
                }
            )
            return values
        if self.strategy == "episode_fraction":
            values.update(
                {
                    "randomization/episode_count": float(self.count),
                    "randomization/mean_episode_length": self.length_sum / max(1, self.count),
                    "randomization/success_fraction": sum(
                        x >= self.threshold for x in self.recent_lengths
                    )
                    / max(1, self.count),
                    "randomization/required_success_fraction": self.success_fraction,
                    "randomization/episode_length_threshold": self.threshold,
                    "randomization/last_check_level": self.last_window_level,
                    "randomization/last_check_success_fraction": self.last_success_fraction,
                    "randomization/last_check_mean_episode_length": self.last_window_mean,
                    "randomization/checks": float(self.completed_windows),
                }
            )
            return values
        values.update(
            {
                "randomization/window_episodes": float(self.count),
                "randomization/window_mean_episode_length": self.length_sum / max(1, self.count),
                "randomization/last_window_level": self.last_window_level,
                "randomization/last_window_mean_episode_length": self.last_window_mean,
                "randomization/successful_windows": float(self.successful_windows),
                "randomization/completed_windows": float(self.completed_windows),
            }
        )
        return values

    def snapshot(self):
        state = {
            name: getattr(self, name)
            for name in (
                "level",
                "count",
                "length_sum",
                "successful_windows",
                "completed_windows",
                "promotions",
                "last_window_level",
                "last_window_mean",
            )
        }
        if self.strategy == "episode_fraction":
            state.update(
                recent_lengths=list(self.recent_lengths),
                last_success_fraction=self.last_success_fraction,
            )
        if self.strategy == "per_env_mean":
            state.update(
                env_lengths=self.env_lengths.tolist(),
                promotion_truncations=self.promotion_truncations,
            )
        return state

    def restore(self, state):
        if self.strategy == "episode_fraction" and "recent_lengths" not in state:
            raise ValueError("episode-fraction resume requires saved recent episode lengths")
        if self.strategy == "per_env_mean" and len(state.get("env_lengths", [])) != self.num_envs:
            raise ValueError("per-env mean resume requires saved results for every environment")
        for name, value in state.items():
            if name == "recent_lengths":
                self.recent_lengths = deque(value, maxlen=self.window_size)
            elif name == "env_lengths":
                self.env_lengths = np.array(value, dtype=np.int64)
            else:
                setattr(self, name, value)
