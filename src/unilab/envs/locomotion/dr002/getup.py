"""WE11 stationary get-up curriculum task."""

from __future__ import annotations

import struct
import subprocess
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.dr002.joystick import (
    DR002JoystickEnv,
    DR002JoystickFlatWE11Cfg,
    ResetPoseConfig,
)

_DEFAULT_POSE_BANK = Path(__file__).parents[3] / "assets/robots/dr002/we11/getup_pose_bank_v2.npz"

_STAGE_ORIGINAL = 0
_STAGE_FRONT = 1
_STAGE_BACK = 2
_STAGE_MIXED = 3
_STAGE_NAMES = ("original", "front", "back", "mixed")

_RESET_ORIGINAL = 0
_RESET_FRONT = 1
_RESET_BACK = 2


@dataclass
class GetupPoseCurriculumConfig:
    easy_keyframe: str = "home"
    hard_keyframe: str = "getup_start_v2"
    initial_difficulty: float = 0.0
    difficulty_step: float = 0.05
    frontier_width: float = 0.05
    replay_fraction: float = 0.20
    component_progress_jitter: float = 0.025
    hard_anchor_fraction: float = 0.10
    window_episodes: int = 2048
    promote_success_rate: float = 0.80
    demote_success_rate: float = 0.50
    pose_bank_file: str = str(_DEFAULT_POSE_BANK)
    front_stage_home_fraction: float = 0.30
    front_stage_hard_fraction: float = 0.20
    front_stage_interpolated_fraction: float = 0.10
    front_stage_front_fraction: float = 0.40
    back_stage_home_fraction: float = 0.30
    back_stage_hard_fraction: float = 0.10
    back_stage_interpolated_fraction: float = 0.10
    back_stage_front_fraction: float = 0.10
    back_stage_back_fraction: float = 0.40
    mixed_home_fraction: float = 0.30
    mixed_hard_fraction: float = 0.20
    mixed_interpolated_fraction: float = 0.10
    mixed_front_fraction: float = 0.20
    mixed_back_fraction: float = 0.20
    back_thigh_group_edges: tuple[float, ...] = (-0.13, 0.0, 0.20, 0.40, 0.61)
    ground_clearance_easy_m: float = 0.015
    forced_difficulty: float | None = None


class _GroundHeightWorker:
    def __init__(self, model_file: str) -> None:
        self._process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("getup_ground_worker.py")),
                model_file,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._finalizer = weakref.finalize(self, self._close_process, self._process)

    @staticmethod
    def _close_process(process: subprocess.Popen) -> None:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=1.0)

    def query(self, qpos: np.ndarray, shared: np.ndarray, clearance: float) -> np.ndarray:
        process = self._process
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("get-up ground-height worker has no communication pipes")
        poses = np.ascontiguousarray(qpos, dtype="<f8")
        phases = np.ascontiguousarray(shared, dtype="<f8")
        process.stdin.write(struct.pack("<IId", poses.shape[0], poses.shape[1], clearance))
        process.stdin.write(poses.tobytes())
        process.stdin.write(phases.tobytes())
        process.stdin.flush()
        size = poses.shape[0] * 8
        payload = process.stdout.read(size)
        if len(payload) != size:
            raise RuntimeError(
                f"get-up ground-height worker stopped early with code {process.poll()}"
            )
        return np.frombuffer(payload, dtype="<f8").copy()


@registry.envcfg("DR002JoystickGetupWE11")
@dataclass
class DR002JoystickGetupWE11Cfg(DR002JoystickFlatWE11Cfg):
    reset_pose: ResetPoseConfig = field(
        default_factory=lambda: ResetPoseConfig(
            mode="getup",
            getup_probability=1.0,
            getup_keyframe="getup_start_v2",
            getup_termination_grace_seconds=1.0,
        )
    )
    getup_curriculum: GetupPoseCurriculumConfig = field(default_factory=GetupPoseCurriculumConfig)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Shortest-path quaternion slerp for MuJoCo's wxyz convention."""
    qa = np.asarray(q0, dtype=np.float64)
    qb = np.asarray(q1, dtype=np.float64).copy()
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb *= -1.0
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    t = np.asarray(phase, dtype=np.float64).reshape(-1, 1)
    if dot > 0.9995:
        result = (1.0 - t) * qa + t * qb
    else:
        theta = float(np.arccos(dot))
        denom = float(np.sin(theta))
        result = np.sin((1.0 - t) * theta) / denom * qa + np.sin(t * theta) / denom * qb
    return result / np.linalg.norm(result, axis=1, keepdims=True)


@registry.env("DR002JoystickGetupWE11", sim_backend="mujoco")
class DR002JoystickGetupEnv(DR002JoystickEnv):
    """Flat behavior with a dedicated balance-to-get-up reset curriculum."""

    _cfg: DR002JoystickGetupWE11Cfg

    def __init__(self, cfg: DR002JoystickGetupWE11Cfg, num_envs=1, backend_type="mujoco"):
        self._validate_getup_cfg(cfg)
        # A second MjData in this process slows MuJoCo 3.8 native batched
        # stepping by roughly 4x. Keep reset-only kinematics isolated.
        self._ground_height_worker = _GroundHeightWorker(str(cfg.scene.model_file))
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

        dtype = get_global_dtype()
        self._easy_qpos = np.asarray(
            self._backend.get_keyframe_qpos(cfg.getup_curriculum.easy_keyframe), dtype=dtype
        )
        self._hard_qpos = np.asarray(
            self._backend.get_keyframe_qpos(cfg.getup_curriculum.hard_keyframe), dtype=dtype
        )
        self._getup_qpos = self._hard_qpos.copy()
        self._leg_qpos_indices = (
            self._backend.get_joint_dof_pos_indices(
                ("left_thigh_joint", "left_calf_joint", "right_thigh_joint", "right_calf_joint")
            )
            + 7
        )
        self._wing_dof_indices = self._backend.get_joint_dof_pos_indices(
            ("left_wing_joint", "right_wing_joint")
        )
        self._wing_qpos_indices = self._wing_dof_indices + 7
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            raise RuntimeError("Getup wing reset requires backend joint limits")
        joint_range = np.asarray(joint_range, dtype=np.float64)
        self._wing_joint_limits = joint_range[self._wing_dof_indices]
        curriculum = cfg.getup_curriculum
        self._load_pose_bank(Path(curriculum.pose_bank_file))

        self._getup_curriculum_difficulty = float(curriculum.initial_difficulty)
        self._getup_curriculum_promotions = 0
        self._getup_curriculum_demotions = 0
        self._getup_curriculum_mastered = False
        self._getup_curriculum_stage = _STAGE_ORIGINAL
        self._getup_curriculum_window_completed = 0
        self._getup_curriculum_window_successes = 0
        self._getup_curriculum_last_success_rate = np.nan
        self._pending_reset_difficulty = np.empty((0,), dtype=dtype)
        self._pending_reset_frontier = np.empty((0,), dtype=np.bool_)
        self._pending_reset_category = np.empty((0,), dtype=np.int8)
        self._episode_reset_difficulty = np.zeros((num_envs,), dtype=dtype)
        self._episode_frontier = np.zeros((num_envs,), dtype=np.bool_)
        self._episode_reset_category = np.full((num_envs,), _RESET_ORIGINAL, dtype=np.int8)
        self._episode_initialized = np.zeros((num_envs,), dtype=np.bool_)
        self._family_episode_counts = np.zeros((2,), dtype=np.int64)
        self._family_success_counts = np.zeros((2,), dtype=np.int64)
        self._family_last_window_success_rates = np.full((2,), np.nan, dtype=np.float64)

    def _load_pose_bank(self, bank_file: Path) -> None:
        """Load and index the offline-generated reset library on the cold path."""
        if not bank_file.is_absolute():
            bank_file = Path(__file__).parents[5] / bank_file
        if not bank_file.is_file():
            raise FileNotFoundError(f"WE11 get-up pose bank does not exist: {bank_file}")
        with np.load(bank_file, allow_pickle=False) as bank:
            qpos = np.asarray(bank["qpos"], dtype=get_global_dtype())
            family = np.asarray(bank["family"], dtype=np.int8)
            thigh = np.asarray(bank["thigh"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self._easy_qpos.size:
            raise ValueError(
                f"get-up pose bank qpos must have shape (N, {self._easy_qpos.size}), got {qpos.shape}"
            )
        if family.shape != (qpos.shape[0],) or thigh.shape != (qpos.shape[0],):
            raise ValueError("get-up pose bank metadata does not match qpos rows")
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(thigh)):
            raise ValueError("get-up pose bank contains non-finite values")
        self._pose_bank_qpos = qpos
        self._front_pose_indices = np.flatnonzero(family == 0).astype(np.int32)
        back_indices = np.flatnonzero(family == 1).astype(np.int32)
        if self._front_pose_indices.size == 0 or back_indices.size == 0:
            raise ValueError("get-up pose bank must contain both front and back families")

        edges = np.asarray(self._cfg.getup_curriculum.back_thigh_group_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 3 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("back_thigh_group_edges must be a strictly increasing sequence")
        back_thigh = thigh[back_indices]
        if np.any(back_thigh < edges[0]) or np.any(back_thigh >= edges[-1]):
            raise ValueError("back pose-bank thigh values fall outside back_thigh_group_edges")
        group_ids = np.searchsorted(edges[1:-1], back_thigh, side="right")
        self._back_pose_groups = tuple(
            back_indices[group_ids == group] for group in range(edges.size - 1)
        )
        if any(indices.size == 0 for indices in self._back_pose_groups):
            raise ValueError("every configured back thigh group must contain at least one pose")

    @staticmethod
    def _validate_getup_cfg(cfg: DR002JoystickGetupWE11Cfg) -> None:
        c = cfg.getup_curriculum
        values = (
            c.initial_difficulty,
            c.difficulty_step,
            c.frontier_width,
            c.replay_fraction,
            c.component_progress_jitter,
            c.hard_anchor_fraction,
            c.promote_success_rate,
            c.demote_success_rate,
            c.front_stage_home_fraction,
            c.front_stage_hard_fraction,
            c.front_stage_interpolated_fraction,
            c.front_stage_front_fraction,
            c.back_stage_home_fraction,
            c.back_stage_hard_fraction,
            c.back_stage_interpolated_fraction,
            c.back_stage_front_fraction,
            c.back_stage_back_fraction,
            c.mixed_home_fraction,
            c.mixed_hard_fraction,
            c.mixed_interpolated_fraction,
            c.mixed_front_fraction,
            c.mixed_back_fraction,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("getup curriculum values must be finite")
        if not 0.0 <= c.initial_difficulty <= 1.0:
            raise ValueError("getup initial_difficulty must be in [0, 1]")
        if c.difficulty_step <= 0.0 or c.frontier_width < 0.0:
            raise ValueError(
                "getup difficulty_step must be positive and frontier_width nonnegative"
            )
        if not 0.0 <= c.replay_fraction <= 1.0 or not 0.0 <= c.hard_anchor_fraction <= 1.0:
            raise ValueError("getup replay and hard-anchor fractions must be in [0, 1]")
        if c.window_episodes <= 0:
            raise ValueError("getup window_episodes must be positive")
        fraction_sets = (
            (
                c.front_stage_home_fraction,
                c.front_stage_hard_fraction,
                c.front_stage_interpolated_fraction,
                c.front_stage_front_fraction,
            ),
            (
                c.back_stage_home_fraction,
                c.back_stage_hard_fraction,
                c.back_stage_interpolated_fraction,
                c.back_stage_front_fraction,
                c.back_stage_back_fraction,
            ),
            (
                c.mixed_home_fraction,
                c.mixed_hard_fraction,
                c.mixed_interpolated_fraction,
                c.mixed_front_fraction,
                c.mixed_back_fraction,
            ),
        )
        for fractions in fraction_sets:
            values = np.asarray(fractions, dtype=np.float64)
            if np.any(values < 0.0) or not np.isclose(
                float(np.sum(values)), 1.0, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError("each getup stage's reset fractions must sum to 1")
        if c.forced_difficulty is not None and not 0.0 <= c.forced_difficulty <= 1.0:
            raise ValueError("getup forced_difficulty must be in [0, 1]")

    def _sample_getup_wing_qpos(self, num_reset: int) -> np.ndarray:
        """Sample both wings independently across their mechanical ranges."""
        lower = self._wing_joint_limits[:, 0]
        upper = self._wing_joint_limits[:, 1]
        values = np.random.uniform(lower, upper, size=(num_reset, lower.size))
        return np.asarray(values, dtype=get_global_dtype())

    def _apply_getup_wing_reset(self, qpos: np.ndarray) -> None:
        qpos[:, self._wing_qpos_indices] = self._sample_getup_wing_qpos(qpos.shape[0])

    def init_state(self) -> NpEnvState:
        state = super().init_state()
        # Do not count randomized initial episode ages as curriculum outcomes.
        state.info["steps"].fill(0)
        return state

    def _sample_episode_progress(self, num_reset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shared, frontier, exact_hard, _category = self._sample_reset_plan(num_reset)
        return shared, frontier, exact_hard

    def _sample_original_progress(
        self, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self._cfg.getup_curriculum
        forced = c.forced_difficulty
        if forced is not None:
            shared = np.full((num_reset,), float(forced), dtype=np.float64)
            return (
                shared,
                np.ones((num_reset,), dtype=np.bool_),
                np.full((num_reset,), forced >= 1.0, dtype=np.bool_),
            )

        difficulty = self._getup_curriculum_difficulty
        replay = np.random.uniform(size=num_reset) < c.replay_fraction
        if difficulty <= 0.0:
            shared = np.zeros((num_reset,), dtype=np.float64)
            frontier = np.ones((num_reset,), dtype=np.bool_)
        else:
            shared = np.empty((num_reset,), dtype=np.float64)
            shared[replay] = np.random.uniform(0.0, difficulty, size=int(np.count_nonzero(replay)))
            frontier = ~replay
            low = max(0.0, difficulty - c.frontier_width)
            shared[frontier] = np.random.uniform(
                low, difficulty, size=int(np.count_nonzero(frontier))
            )
        exact_hard = np.zeros((num_reset,), dtype=np.bool_)
        if difficulty >= 1.0:
            exact_hard = np.random.uniform(size=num_reset) < c.hard_anchor_fraction
            shared[exact_hard] = 1.0
            frontier[exact_hard] = True
        return shared, frontier, exact_hard

    def _sample_reset_plan(
        self, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c = self._cfg.getup_curriculum
        if c.forced_difficulty is not None or self._getup_curriculum_stage == _STAGE_ORIGINAL:
            shared, frontier, exact_hard = self._sample_original_progress(num_reset)
            category = np.full((num_reset,), _RESET_ORIGINAL, dtype=np.int8)
            return shared, frontier, exact_hard, category

        if self._getup_curriculum_stage == _STAGE_FRONT:
            fractions = (
                c.front_stage_home_fraction,
                c.front_stage_hard_fraction,
                c.front_stage_interpolated_fraction,
                c.front_stage_front_fraction,
                0.0,
            )
        elif self._getup_curriculum_stage == _STAGE_BACK:
            fractions = (
                c.back_stage_home_fraction,
                c.back_stage_hard_fraction,
                c.back_stage_interpolated_fraction,
                c.back_stage_front_fraction,
                c.back_stage_back_fraction,
            )
        else:
            fractions = (
                c.mixed_home_fraction,
                c.mixed_hard_fraction,
                c.mixed_interpolated_fraction,
                c.mixed_front_fraction,
                c.mixed_back_fraction,
            )

        cutoffs = np.cumsum(np.asarray(fractions, dtype=np.float64))
        draw = np.random.uniform(size=num_reset)
        bins = np.searchsorted(cutoffs, draw, side="right")
        # Bins: 0 home, 1 exact original D1, 2 original interpolation,
        # 3 front bank, 4 back bank.
        shared = np.zeros((num_reset,), dtype=np.float64)
        exact_hard = bins == 1
        shared[exact_hard] = 1.0
        interpolated = bins == 2
        shared[interpolated] = np.random.uniform(0.0, 1.0, size=int(np.count_nonzero(interpolated)))
        category = np.full((num_reset,), _RESET_ORIGINAL, dtype=np.int8)
        category[bins == 3] = _RESET_FRONT
        category[bins == 4] = _RESET_BACK
        return shared, np.zeros((num_reset,), dtype=np.bool_), exact_hard, category

    def _sample_pose_bank_indices(self, category: int, count: int) -> np.ndarray:
        if category == _RESET_FRONT:
            return np.random.choice(self._front_pose_indices, size=count, replace=True)
        if category != _RESET_BACK:
            raise ValueError(f"unsupported pose-bank category {category}")
        # Choose a thigh band first so dense bands do not dominate the reset
        # distribution, then choose one legal lower-body pose in that band.
        group_ids = np.random.randint(0, len(self._back_pose_groups), size=count)
        selected = np.empty((count,), dtype=np.int32)
        for group, indices in enumerate(self._back_pose_groups):
            mask = group_ids == group
            selected[mask] = np.random.choice(
                indices, size=int(np.count_nonzero(mask)), replace=True
            )
        return selected

    def sample_getup_reset_qpos(self, num_reset: int) -> np.ndarray:
        shared, frontier, exact_hard, category = self._sample_reset_plan(num_reset)
        c = self._cfg.getup_curriculum
        forced = c.forced_difficulty is not None
        qpos = np.broadcast_to(self._easy_qpos, (num_reset, self._easy_qpos.size)).copy()
        original = category == _RESET_ORIGINAL
        original_rows = np.flatnonzero(original)
        if original_rows.size:
            original_shared = shared[original]
            if forced:
                base_phase = thigh_phase = calf_phase = original_shared
            else:
                jitter = float(c.component_progress_jitter)
                upper = max(self._getup_curriculum_difficulty, float(np.max(original_shared)))
                base_phase = np.clip(
                    original_shared + np.random.uniform(-jitter, jitter, original_rows.size),
                    0.0,
                    upper,
                )
                thigh_phase = np.clip(
                    original_shared + np.random.uniform(-jitter, jitter, original_rows.size),
                    0.0,
                    upper,
                )
                calf_phase = np.clip(
                    original_shared + np.random.uniform(-jitter, jitter, original_rows.size),
                    0.0,
                    upper,
                )
            original_qpos = qpos[original].copy()
            original_qpos[:, 3:7] = _quat_slerp(
                self._easy_qpos[3:7], self._hard_qpos[3:7], base_phase
            )
            lt, lc, rt, rc = self._leg_qpos_indices
            original_qpos[:, [lt, rt]] = self._easy_qpos[lt] + thigh_phase[:, None] * (
                self._hard_qpos[lt] - self._easy_qpos[lt]
            )
            original_qpos[:, [lc, rc]] = self._easy_qpos[lc] + calf_phase[:, None] * (
                self._hard_qpos[lc] - self._easy_qpos[lc]
            )
            exact_easy = original_shared <= 1.0e-12
            grounded = ~exact_easy
            if np.any(grounded):
                original_qpos[grounded, 2] = self._grounded_root_height(
                    original_qpos[grounded], original_shared[grounded]
                )
            original_qpos[exact_easy] = self._easy_qpos
            original_qpos[exact_hard[original]] = self._hard_qpos
            qpos[original] = original_qpos

        for reset_category in (_RESET_FRONT, _RESET_BACK):
            rows = np.flatnonzero(category == reset_category)
            if rows.size:
                indices = self._sample_pose_bank_indices(reset_category, rows.size)
                qpos[rows] = self._pose_bank_qpos[indices]

        self._apply_getup_wing_reset(qpos)

        self._pending_reset_difficulty = np.asarray(shared, dtype=get_global_dtype())
        self._pending_reset_frontier = frontier
        self._pending_reset_category = category
        return np.asarray(qpos, dtype=get_global_dtype())

    def _grounded_root_height(self, qpos: np.ndarray, shared: np.ndarray) -> np.ndarray:
        return self._ground_height_worker.query(
            qpos,
            shared,
            self._cfg.getup_curriculum.ground_clearance_easy_m,
        )

    def record_getup_reset(self, env_ids: np.ndarray, _reset_xy: np.ndarray) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        if self._pending_reset_difficulty.shape != (rows.size,):
            raise RuntimeError("getup reset metadata does not match reset batch")
        if self._pending_reset_category.shape != (rows.size,):
            raise RuntimeError("getup reset category metadata does not match reset batch")
        self._episode_reset_difficulty[rows] = self._pending_reset_difficulty
        self._episode_frontier[rows] = self._pending_reset_frontier
        self._episode_reset_category[rows] = self._pending_reset_category
        self._episode_initialized[rows] = True

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        rows = np.asarray(env_indices, dtype=np.int32)
        # NpEnv clears state.info["steps"] before dispatching reset for done
        # environments. The Flat alive accumulator is cleared later inside
        # DR002JoystickEnv.reset, so it is the reliable completed-episode marker
        # at this point.
        episode_steps = np.asarray(
            getattr(self, "_episode_alive_steps", np.zeros((self._num_envs,), dtype=np.int32))
        )
        completed = rows[self._episode_initialized[rows] & (episode_steps[rows] > 0)]
        if completed.size:
            self._record_curriculum_outcomes(completed)
        return super().reset(rows)

    def _record_curriculum_outcomes(self, rows: np.ndarray) -> None:
        categories = self._episode_reset_category[rows]
        outcomes = np.asarray(self._getup_succeeded[rows], dtype=np.bool_)
        for category in (_RESET_FRONT, _RESET_BACK):
            family = category - _RESET_FRONT
            family_mask = categories == category
            self._family_episode_counts[family] += int(np.count_nonzero(family_mask))
            self._family_success_counts[family] += int(np.count_nonzero(outcomes[family_mask]))

        c = self._cfg.getup_curriculum
        if c.forced_difficulty is not None or self._getup_curriculum_stage == _STAGE_MIXED:
            return
        if self._getup_curriculum_stage == _STAGE_ORIGINAL:
            eligible = self._episode_frontier[rows]
        elif self._getup_curriculum_stage == _STAGE_FRONT:
            eligible = categories == _RESET_FRONT
        else:
            eligible = categories == _RESET_BACK
        eligible_outcomes = outcomes[eligible]
        if eligible_outcomes.size == 0:
            return

        offset = 0
        while offset < eligible_outcomes.size:
            capacity = c.window_episodes - self._getup_curriculum_window_completed
            take = min(capacity, int(eligible_outcomes.size - offset))
            window_slice = eligible_outcomes[offset : offset + take]
            self._getup_curriculum_window_completed += take
            self._getup_curriculum_window_successes += int(np.count_nonzero(window_slice))
            offset += take
            if self._getup_curriculum_window_completed < c.window_episodes:
                continue

            rate = self._getup_curriculum_window_successes / c.window_episodes
            self._getup_curriculum_last_success_rate = float(rate)
            old_difficulty = self._getup_curriculum_difficulty
            old_stage = self._getup_curriculum_stage
            if old_stage in (_STAGE_FRONT, _STAGE_BACK):
                self._family_last_window_success_rates[old_stage - _STAGE_FRONT] = rate
            if old_stage == _STAGE_ORIGINAL:
                if old_difficulty >= 1.0 and rate >= c.promote_success_rate:
                    self._getup_curriculum_mastered = True
                    self._getup_curriculum_stage = _STAGE_FRONT
                elif rate >= c.promote_success_rate:
                    self._getup_curriculum_difficulty = min(1.0, old_difficulty + c.difficulty_step)
                    if self._getup_curriculum_difficulty > old_difficulty:
                        self._getup_curriculum_promotions += 1
                elif rate < c.demote_success_rate:
                    self._getup_curriculum_difficulty = max(0.0, old_difficulty - c.difficulty_step)
                    if self._getup_curriculum_difficulty < old_difficulty:
                        self._getup_curriculum_demotions += 1
            elif rate >= c.promote_success_rate:
                self._getup_curriculum_stage += 1
            self._getup_curriculum_window_completed = 0
            self._getup_curriculum_window_successes = 0
            if self._getup_curriculum_stage != old_stage:
                break
            # Remaining rows were sampled at the old boundary and must not be
            # credited to a newly promoted/demoted level.
            if self._getup_curriculum_difficulty != old_difficulty:
                break

    def _log_getup_diagnostics(self, log: dict[str, Any], num_envs: int) -> None:
        super()._log_getup_diagnostics(log, num_envs)
        c = self._cfg.getup_curriculum
        difficulty = self._episode_reset_difficulty[:num_envs]
        log["getup_curriculum/max_difficulty"] = self._getup_curriculum_difficulty
        log["getup_curriculum/sample_mean"] = float(np.mean(difficulty))
        log["getup_curriculum/sample_min"] = float(np.min(difficulty))
        log["getup_curriculum/sample_max"] = float(np.max(difficulty))
        log["getup_curriculum/frontier_fraction"] = float(
            np.mean(self._episode_frontier[:num_envs])
        )
        log["getup_curriculum/frontier_success_rate"] = (
            float(self._getup_curriculum_last_success_rate)
            if np.isfinite(self._getup_curriculum_last_success_rate)
            else 0.0
        )
        log["getup_curriculum/window_completed"] = float(self._getup_curriculum_window_completed)
        log["getup_curriculum/window_target"] = float(c.window_episodes)
        log["getup_curriculum/promotions"] = float(self._getup_curriculum_promotions)
        log["getup_curriculum/demotions"] = float(self._getup_curriculum_demotions)
        log["getup_curriculum/mastered"] = float(self._getup_curriculum_mastered)
        log["getup_curriculum/stage"] = float(self._getup_curriculum_stage)
        for stage, name in enumerate(_STAGE_NAMES):
            log[f"getup_curriculum/stage_{name}"] = float(self._getup_curriculum_stage == stage)
        categories = self._episode_reset_category[:num_envs]
        log["getup_curriculum/front_reset_fraction"] = float(np.mean(categories == _RESET_FRONT))
        log["getup_curriculum/back_reset_fraction"] = float(np.mean(categories == _RESET_BACK))
        for family, name in enumerate(("front", "back")):
            episodes = int(self._family_episode_counts[family])
            successes = int(self._family_success_counts[family])
            log[f"getup_curriculum/{name}_episodes"] = float(episodes)
            log[f"getup_curriculum/{name}_successes"] = float(successes)
            log[f"getup_curriculum/{name}_success_rate"] = float(successes / max(episodes, 1))
            window_rate = self._family_last_window_success_rates[family]
            log[f"getup_curriculum/{name}_window_success_rate"] = (
                float(window_rate) if np.isfinite(window_rate) else 0.0
            )

    def training_state_dict(self) -> dict[str, Any]:
        state = super().training_state_dict()
        state["getup_pose_curriculum"] = {
            "difficulty": float(self._getup_curriculum_difficulty),
            "promotions": int(self._getup_curriculum_promotions),
            "demotions": int(self._getup_curriculum_demotions),
            "mastered": bool(self._getup_curriculum_mastered),
            "stage": int(self._getup_curriculum_stage),
            "window_completed": int(self._getup_curriculum_window_completed),
            "window_successes": int(self._getup_curriculum_window_successes),
            "last_success_rate": float(self._getup_curriculum_last_success_rate),
            "family_episode_counts": self._family_episode_counts.tolist(),
            "family_success_counts": self._family_success_counts.tolist(),
            "family_last_window_success_rates": self._family_last_window_success_rates.tolist(),
        }
        return state

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        super().load_training_state_dict(state)
        payload = state.get("getup_pose_curriculum")
        if not isinstance(payload, dict):
            raise ValueError("Getup training state is missing getup_pose_curriculum")
        difficulty = float(payload["difficulty"])
        if not np.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
            raise ValueError("saved getup difficulty must be finite and in [0, 1]")
        self._getup_curriculum_difficulty = difficulty
        self._getup_curriculum_promotions = max(int(payload["promotions"]), 0)
        self._getup_curriculum_demotions = max(int(payload["demotions"]), 0)
        self._getup_curriculum_mastered = bool(payload.get("mastered", False))
        default_stage = _STAGE_FRONT if self._getup_curriculum_mastered else _STAGE_ORIGINAL
        stage = int(payload.get("stage", default_stage))
        if not _STAGE_ORIGINAL <= stage <= _STAGE_MIXED:
            raise ValueError(f"saved getup curriculum stage is invalid: {stage}")
        self._getup_curriculum_stage = stage
        self._getup_curriculum_window_completed = max(int(payload["window_completed"]), 0)
        self._getup_curriculum_window_successes = max(int(payload["window_successes"]), 0)
        self._getup_curriculum_last_success_rate = float(payload["last_success_rate"])
        episode_counts = np.asarray(payload.get("family_episode_counts", [0, 0]), dtype=np.int64)
        success_counts = np.asarray(payload.get("family_success_counts", [0, 0]), dtype=np.int64)
        if episode_counts.shape != (2,) or success_counts.shape != (2,):
            raise ValueError("saved getup family counters must each contain front and back")
        if np.any(episode_counts < 0) or np.any(success_counts < 0):
            raise ValueError("saved getup family counters must be nonnegative")
        if np.any(success_counts > episode_counts):
            raise ValueError("saved getup successes cannot exceed completed family episodes")
        self._family_episode_counts[:] = episode_counts
        self._family_success_counts[:] = success_counts
        window_rates = np.asarray(
            payload.get("family_last_window_success_rates", [np.nan, np.nan]),
            dtype=np.float64,
        )
        if window_rates.shape != (2,) or np.any(
            np.isfinite(window_rates) & ((window_rates < 0.0) | (window_rates > 1.0))
        ):
            raise ValueError("saved getup family window rates must be NaN or in [0, 1]")
        self._family_last_window_success_rates[:] = window_rates
