"""WE11 stationary get-up curriculum task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply_batched, np_quat_from_euler_xyz, np_quat_mul
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.dr002.joystick import (
    DR002JoystickEnv,
    DR002JoystickFlatWE11Cfg,
    ResetPoseConfig,
)

_DEFAULT_POSE_BANK = Path(__file__).parents[3] / "assets/robots/dr002/we11/getup_pose_bank_v3.npz"

_STAGE_BALANCE_RECOVERY = 0
_STAGE_HOME_TO_GETUP = 1
_STAGE_GETUP_TO_FRONT = 2
_STAGE_FRONT = 3
_STAGE_BACK = 4
_STAGE_MIXED = 5
_STAGE_NAMES = (
    "balance_recovery",
    "home_to_getup",
    "getup_to_front",
    "front",
    "back",
    "mixed",
)

_RESET_HOME_TO_GETUP = 0
_RESET_FRONT = 1
_RESET_BACK = 2
_RESET_GETUP_TO_FRONT = 3
_RESET_BALANCE_RECOVERY = 4
# Compatibility name retained for tests and checkpoint-era terminology.
_RESET_ORIGINAL = _RESET_HOME_TO_GETUP


@dataclass
class GetupPoseCurriculumConfig:
    easy_keyframe: str = "home"
    hard_keyframe: str = "getup_start_v2"
    initial_difficulty: float = 0.0
    difficulty_step: float = 0.05
    frontier_width: float = 0.05
    replay_fraction: float = 0.20
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
    back_thigh_group_edges: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45, 0.61)
    balance_max_pitch_deg: float = 25.0
    balance_max_pitch_rate_rad_s: float = 0.2
    # Midpoint of the two wheel bodies relative to base_link in the home keyframe.
    balance_wheel_pivot_offset_body: tuple[float, float, float] = (
        0.02543588,
        0.0,
        -0.24231853,
    )
    forced_difficulty: float | None = None
    forced_stage: str | None = None


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


@registry.env("DR002JoystickGetupWE11", sim_backend="mujoco")
class DR002JoystickGetupEnv(DR002JoystickEnv):
    """Flat behavior with a dedicated balance-to-get-up reset curriculum."""

    _cfg: DR002JoystickGetupWE11Cfg

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["getup_workspace"] = self._reward_getup_workspace

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        balance_succeeded = (
            self._episode_reset_category == _RESET_BALANCE_RECOVERY
        ) & self._getup_succeeded
        np.logical_or(truncated, balance_succeeded, out=truncated)
        return truncated

    def __init__(self, cfg: DR002JoystickGetupWE11Cfg, num_envs=1, backend_type="mujoco"):
        self._validate_getup_cfg(cfg)
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
        self._getup_curriculum_stage = _STAGE_BALANCE_RECOVERY
        self._getup_curriculum_window_completed = 0
        self._getup_curriculum_window_successes = 0
        self._getup_curriculum_last_success_rate = np.nan
        self._balance_window_completed = np.zeros((2,), dtype=np.int64)
        self._balance_window_successes = np.zeros((2,), dtype=np.int64)
        self._balance_last_window_success_rates = np.full((2,), np.nan, dtype=np.float64)
        self._balance_episode_counts = np.zeros((2,), dtype=np.int64)
        self._balance_success_counts = np.zeros((2,), dtype=np.int64)
        self._pending_reset_difficulty = np.empty((0,), dtype=dtype)
        self._pending_reset_frontier = np.empty((0,), dtype=np.bool_)
        self._pending_reset_category = np.empty((0,), dtype=np.int8)
        self._pending_balance_direction = np.empty((0,), dtype=np.int8)
        self._pending_balance_pitch_rad = np.empty((0,), dtype=dtype)
        self._pending_balance_pitch_rate = np.empty((0,), dtype=dtype)
        self._episode_reset_difficulty = np.zeros((num_envs,), dtype=dtype)
        self._episode_frontier = np.zeros((num_envs,), dtype=np.bool_)
        self._episode_reset_category = np.full((num_envs,), _RESET_ORIGINAL, dtype=np.int8)
        self._episode_balance_direction = np.zeros((num_envs,), dtype=np.int8)
        self._episode_balance_pitch_rad = np.zeros((num_envs,), dtype=dtype)
        self._episode_balance_pitch_rate = np.zeros((num_envs,), dtype=dtype)
        self._episode_reset_xy = np.zeros((num_envs, 2), dtype=dtype)
        self._episode_max_abs_xy_displacement = np.zeros((num_envs,), dtype=dtype)
        self._episode_workspace_violated = np.zeros((num_envs,), dtype=np.bool_)
        self._episode_initialized = np.zeros((num_envs,), dtype=np.bool_)
        self._family_episode_counts = np.zeros((2,), dtype=np.int64)
        self._family_success_counts = np.zeros((2,), dtype=np.int64)
        self._family_last_window_success_rates = np.full((2,), np.nan, dtype=np.float64)
        self._workspace_completed_episodes = 0
        self._workspace_violation_episodes = 0
        self._workspace_max_displacement_sum = 0.0

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
            progress = np.asarray(bank["progress"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self._easy_qpos.size:
            raise ValueError(
                f"get-up pose bank qpos must have shape (N, {self._easy_qpos.size}), got {qpos.shape}"
            )
        if (
            family.shape != (qpos.shape[0],)
            or thigh.shape != (qpos.shape[0],)
            or progress.shape != (qpos.shape[0],)
        ):
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

        home_to_getup = np.flatnonzero(family == 5).astype(np.int32)
        getup_to_front = np.flatnonzero(family == 4).astype(np.int32)
        if home_to_getup.size == 0 or getup_to_front.size == 0:
            raise ValueError("get-up pose bank must contain both transition families")

        def build_path(
            indices: np.ndarray,
            start_qpos: np.ndarray,
            end_qpos: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            order = np.argsort(progress[indices])
            path_progress = progress[indices[order]]
            if (
                np.any(~np.isfinite(path_progress))
                or np.any(path_progress <= 0.0)
                or np.any(path_progress >= 1.0)
                or np.any(np.diff(path_progress) <= 0.0)
            ):
                raise ValueError("transition pose progress must be finite and strictly increasing")
            path_qpos = np.concatenate(
                [start_qpos[None, :], qpos[indices[order]], end_qpos[None, :]], axis=0
            )
            return (
                np.concatenate([[0.0], path_progress, [1.0]]),
                np.asarray(path_qpos, dtype=get_global_dtype()),
            )

        self._home_to_getup_progress, self._home_to_getup_qpos = build_path(
            home_to_getup, self._easy_qpos, self._hard_qpos
        )
        front_endpoint_candidates = self._front_pose_indices[
            np.isclose(thigh[self._front_pose_indices], 1.57)
            & np.isclose(
                qpos[self._front_pose_indices, self._leg_qpos_indices[1]],
                self._hard_qpos[self._leg_qpos_indices[1]],
            )
        ]
        if front_endpoint_candidates.size != 1:
            raise ValueError("pose bank must contain one exact getup_to_front endpoint")
        self._getup_to_front_progress, self._getup_to_front_qpos = build_path(
            getup_to_front,
            self._hard_qpos,
            qpos[int(front_endpoint_candidates[0])],
        )

    @staticmethod
    def _validate_getup_cfg(cfg: DR002JoystickGetupWE11Cfg) -> None:
        c = cfg.getup_curriculum
        values = (
            c.initial_difficulty,
            c.difficulty_step,
            c.frontier_width,
            c.replay_fraction,
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
            c.balance_max_pitch_deg,
            c.balance_max_pitch_rate_rad_s,
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
        if c.window_episodes % 2 != 0:
            raise ValueError("getup window_episodes must be even for balanced direction windows")
        if c.balance_max_pitch_deg <= 0.0 or c.balance_max_pitch_rate_rad_s <= 0.0:
            raise ValueError("balance pitch and pitch-rate limits must be positive")
        pivot = np.asarray(c.balance_wheel_pivot_offset_body, dtype=np.float64)
        if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
            raise ValueError("balance_wheel_pivot_offset_body must contain three finite values")
        if c.forced_stage not in {None, "balance_recovery", "home_to_getup"}:
            raise ValueError("getup forced_stage must be one of: balance_recovery, home_to_getup")
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
        if c.forced_difficulty is not None:
            shared, frontier, exact_hard = self._sample_original_progress(num_reset)
            forced_stage = c.forced_stage or "home_to_getup"
            reset_category = (
                _RESET_BALANCE_RECOVERY
                if forced_stage == "balance_recovery"
                else _RESET_HOME_TO_GETUP
            )
            category = np.full((num_reset,), reset_category, dtype=np.int8)
            return shared, frontier, exact_hard, category

        if self._getup_curriculum_stage in (
            _STAGE_BALANCE_RECOVERY,
            _STAGE_HOME_TO_GETUP,
            _STAGE_GETUP_TO_FRONT,
        ):
            shared, frontier, exact_hard = self._sample_original_progress(num_reset)
            if self._getup_curriculum_stage == _STAGE_BALANCE_RECOVERY:
                reset_category = _RESET_BALANCE_RECOVERY
            elif self._getup_curriculum_stage == _STAGE_GETUP_TO_FRONT:
                reset_category = _RESET_GETUP_TO_FRONT
            else:
                reset_category = _RESET_HOME_TO_GETUP
            category = np.full((num_reset,), reset_category, dtype=np.int8)
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
        shared, frontier, _exact_hard, category = self._sample_reset_plan(num_reset)
        qpos = np.broadcast_to(self._easy_qpos, (num_reset, self._easy_qpos.size)).copy()
        balance_direction = np.zeros((num_reset,), dtype=np.int8)
        balance_pitch = np.zeros((num_reset,), dtype=np.float64)
        balance_pitch_rate = np.zeros((num_reset,), dtype=np.float64)
        balance_rows = np.flatnonzero(category == _RESET_BALANCE_RECOVERY)
        if balance_rows.size:
            # Exact 50/50 direction balance for even batches; odd batches get
            # one independently sampled extra direction.
            directions = np.ones((balance_rows.size,), dtype=np.int8)
            directions[: balance_rows.size // 2] = -1
            if balance_rows.size % 2:
                directions[-1] = np.random.choice(np.asarray([-1, 1], dtype=np.int8))
            np.random.shuffle(directions)
            max_pitch = np.deg2rad(self._cfg.getup_curriculum.balance_max_pitch_deg)
            max_rate = self._cfg.getup_curriculum.balance_max_pitch_rate_rad_s
            sampled_progress = shared[balance_rows]
            disturbance_low = np.maximum(
                0.0,
                sampled_progress - self._cfg.getup_curriculum.frontier_width,
            )
            pitch_progress = np.random.uniform(disturbance_low, sampled_progress)
            rate_progress = np.random.uniform(disturbance_low, sampled_progress)
            pitch_magnitude = max_pitch * pitch_progress
            rate_magnitude = max_rate * rate_progress
            signed_pitch = directions * pitch_magnitude
            signed_rate = directions * rate_magnitude
            pitch_quat = np_quat_from_euler_xyz(
                np.zeros_like(signed_pitch),
                signed_pitch,
                np.zeros_like(signed_pitch),
            )
            qpos[balance_rows, 3:7] = np_quat_mul(
                pitch_quat,
                qpos[balance_rows, 3:7],
            )
            pivot_offset = np.asarray(
                self._cfg.getup_curriculum.balance_wheel_pivot_offset_body,
                dtype=np.float64,
            )
            root_from_pivot = np.broadcast_to(-pivot_offset, (balance_rows.size, 3))
            rotated_root = np_quat_apply_batched(pitch_quat, root_from_pivot)
            qpos[balance_rows, :3] += pivot_offset + rotated_root
            balance_direction[balance_rows] = directions
            balance_pitch[balance_rows] = signed_pitch
            balance_pitch_rate[balance_rows] = signed_rate
        for reset_category in (_RESET_HOME_TO_GETUP, _RESET_GETUP_TO_FRONT):
            rows = np.flatnonzero(category == reset_category)
            if not rows.size:
                continue
            if reset_category == _RESET_GETUP_TO_FRONT:
                path_progress = self._getup_to_front_progress
                path_qpos = self._getup_to_front_qpos
            else:
                path_progress = self._home_to_getup_progress
                path_qpos = self._home_to_getup_qpos
            insertion = np.searchsorted(path_progress, shared[rows], side="left")
            insertion = np.clip(insertion, 1, path_progress.size - 1)
            left = insertion - 1
            choose_right = np.abs(path_progress[insertion] - shared[rows]) < np.abs(
                shared[rows] - path_progress[left]
            )
            selected = np.where(choose_right, insertion, left)
            qpos[rows] = path_qpos[selected]

        for reset_category in (_RESET_FRONT, _RESET_BACK):
            rows = np.flatnonzero(category == reset_category)
            if rows.size:
                indices = self._sample_pose_bank_indices(reset_category, rows.size)
                qpos[rows] = self._pose_bank_qpos[indices]

        self._apply_getup_wing_reset(qpos)

        self._pending_reset_difficulty = np.asarray(shared, dtype=get_global_dtype())
        self._pending_reset_frontier = frontier
        self._pending_reset_category = category
        self._pending_balance_direction = balance_direction
        self._pending_balance_pitch_rad = np.asarray(balance_pitch, dtype=get_global_dtype())
        self._pending_balance_pitch_rate = np.asarray(balance_pitch_rate, dtype=get_global_dtype())
        return np.asarray(qpos, dtype=get_global_dtype())

    def sample_getup_reset_qvel(self, qvel: np.ndarray) -> np.ndarray:
        values = np.asarray(qvel, dtype=get_global_dtype()).copy()
        if self._pending_balance_pitch_rate.shape != (values.shape[0],):
            raise RuntimeError("getup reset velocity metadata does not match reset batch")
        balance = self._pending_reset_category == _RESET_BALANCE_RECOVERY
        values[balance, 4] = self._pending_balance_pitch_rate[balance]
        return values

    def record_getup_reset(self, env_ids: np.ndarray, _reset_xy: np.ndarray) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        if self._pending_reset_difficulty.shape != (rows.size,):
            raise RuntimeError("getup reset metadata does not match reset batch")
        if self._pending_reset_category.shape != (rows.size,):
            raise RuntimeError("getup reset category metadata does not match reset batch")
        if self._pending_balance_direction.shape != (rows.size,):
            raise RuntimeError("balance reset metadata does not match reset batch")
        self._episode_reset_difficulty[rows] = self._pending_reset_difficulty
        self._episode_frontier[rows] = self._pending_reset_frontier
        self._episode_reset_category[rows] = self._pending_reset_category
        self._episode_balance_direction[rows] = self._pending_balance_direction
        self._episode_balance_pitch_rad[rows] = self._pending_balance_pitch_rad
        self._episode_balance_pitch_rate[rows] = self._pending_balance_pitch_rate
        self._episode_reset_xy[rows] = np.asarray(_reset_xy, dtype=get_global_dtype())
        self._episode_max_abs_xy_displacement[rows] = 0.0
        self._episode_workspace_violated[rows] = False
        self._episode_initialized[rows] = True

    def sample_commands(
        self,
        num_samples: int,
        *,
        env_ids: np.ndarray | None = None,
        resample_standing: bool = False,
    ) -> np.ndarray:
        commands = super().sample_commands(
            num_samples,
            env_ids=env_ids,
            resample_standing=resample_standing,
        )
        if env_ids is None:
            return commands
        rows = np.asarray(env_ids, dtype=np.int32)
        balance = self._episode_reset_category[rows] == _RESET_BALANCE_RECOVERY
        if np.any(balance):
            commands[balance] = self.startup_commands(int(np.count_nonzero(balance)))
        return commands

    def _reward_getup_workspace(self, ctx: RewardContext) -> np.ndarray:
        half_extent = self._reward_cfg.getup_workspace_half_extent
        if half_extent is None:
            return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        base_xy = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[
            : ctx.num_envs, :2
        ]
        displacement = base_xy - self._episode_reset_xy[: ctx.num_envs]
        abs_displacement = np.abs(displacement)
        per_env_max = np.max(abs_displacement, axis=1)
        self._episode_max_abs_xy_displacement[: ctx.num_envs] = np.maximum(
            self._episode_max_abs_xy_displacement[: ctx.num_envs],
            per_env_max,
        )
        violated = np.any(abs_displacement > float(half_extent), axis=1)
        self._episode_workspace_violated[: ctx.num_envs] |= violated
        excess = np.maximum(abs_displacement - float(half_extent), 0.0) / float(half_extent)
        reward = np.sum(np.square(excess), axis=1)
        penalty_clip = self._reward_cfg.getup_workspace_penalty_clip
        if penalty_clip is not None:
            reward = np.minimum(reward, float(penalty_clip))
        return np.asarray(reward, dtype=get_global_dtype())

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
        self._workspace_completed_episodes += int(rows.size)
        self._workspace_violation_episodes += int(
            np.count_nonzero(self._episode_workspace_violated[rows])
        )
        self._workspace_max_displacement_sum += float(
            np.sum(self._episode_max_abs_xy_displacement[rows])
        )

        balance_mask = categories == _RESET_BALANCE_RECOVERY
        for index, direction in enumerate((-1, 1)):
            direction_mask = balance_mask & (self._episode_balance_direction[rows] == direction)
            self._balance_episode_counts[index] += int(np.count_nonzero(direction_mask))
            self._balance_success_counts[index] += int(np.count_nonzero(outcomes[direction_mask]))
        for category in (_RESET_FRONT, _RESET_BACK):
            family = category - _RESET_FRONT
            family_mask = categories == category
            self._family_episode_counts[family] += int(np.count_nonzero(family_mask))
            self._family_success_counts[family] += int(np.count_nonzero(outcomes[family_mask]))

        c = self._cfg.getup_curriculum
        if c.forced_difficulty is not None or self._getup_curriculum_stage == _STAGE_MIXED:
            return
        if self._getup_curriculum_stage == _STAGE_BALANCE_RECOVERY:
            self._record_balance_curriculum_outcomes(rows, outcomes)
            return
        if self._getup_curriculum_stage in (
            _STAGE_HOME_TO_GETUP,
            _STAGE_GETUP_TO_FRONT,
        ):
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
            if old_stage in (_STAGE_HOME_TO_GETUP, _STAGE_GETUP_TO_FRONT):
                if old_difficulty >= 1.0 and rate >= c.promote_success_rate:
                    if old_stage == _STAGE_HOME_TO_GETUP:
                        self._getup_curriculum_mastered = True
                        self._getup_curriculum_stage = _STAGE_GETUP_TO_FRONT
                        self._getup_curriculum_difficulty = 0.0
                    else:
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

    def _record_balance_curriculum_outcomes(self, rows: np.ndarray, outcomes: np.ndarray) -> None:
        c = self._cfg.getup_curriculum
        per_direction_target = c.window_episodes // 2
        for index, direction in enumerate((-1, 1)):
            eligible = (
                self._episode_frontier[rows]
                & (self._episode_reset_category[rows] == _RESET_BALANCE_RECOVERY)
                & (self._episode_balance_direction[rows] == direction)
            )
            values = outcomes[eligible]
            capacity = per_direction_target - int(self._balance_window_completed[index])
            take = min(capacity, int(values.size))
            if take > 0:
                self._balance_window_completed[index] += take
                self._balance_window_successes[index] += int(np.count_nonzero(values[:take]))

        if np.any(self._balance_window_completed < per_direction_target):
            return

        rates = self._balance_window_successes / self._balance_window_completed
        self._balance_last_window_success_rates[:] = rates
        self._getup_curriculum_last_success_rate = float(np.min(rates))
        old_difficulty = self._getup_curriculum_difficulty
        if np.all(rates >= c.promote_success_rate):
            if old_difficulty >= 1.0:
                self._getup_curriculum_stage = _STAGE_HOME_TO_GETUP
                self._getup_curriculum_difficulty = 0.0
            else:
                self._getup_curriculum_difficulty = min(1.0, old_difficulty + c.difficulty_step)
                self._getup_curriculum_promotions += int(
                    self._getup_curriculum_difficulty > old_difficulty
                )
        elif np.any(rates < c.demote_success_rate):
            self._getup_curriculum_difficulty = max(0.0, old_difficulty - c.difficulty_step)
            self._getup_curriculum_demotions += int(
                self._getup_curriculum_difficulty < old_difficulty
            )
        self._balance_window_completed.fill(0)
        self._balance_window_successes.fill(0)

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
        log["getup_curriculum/home_to_getup_reset_fraction"] = float(
            np.mean(categories == _RESET_HOME_TO_GETUP)
        )
        log["getup_curriculum/getup_to_front_reset_fraction"] = float(
            np.mean(categories == _RESET_GETUP_TO_FRONT)
        )
        balance = categories == _RESET_BALANCE_RECOVERY
        log["getup_curriculum/balance_reset_fraction"] = float(np.mean(balance))
        log["getup_curriculum/balance_pitch_abs_mean_deg"] = float(
            np.rad2deg(np.mean(np.abs(self._episode_balance_pitch_rad[:num_envs][balance])))
            if np.any(balance)
            else 0.0
        )
        log["getup_curriculum/balance_pitch_rate_abs_mean"] = float(
            np.mean(np.abs(self._episode_balance_pitch_rate[:num_envs][balance]))
            if np.any(balance)
            else 0.0
        )
        log["getup_curriculum/balance_window_completed"] = float(
            np.sum(self._balance_window_completed)
        )
        for index, name in enumerate(("backward", "forward")):
            episodes = int(self._balance_episode_counts[index])
            successes = int(self._balance_success_counts[index])
            rate = self._balance_last_window_success_rates[index]
            log[f"getup_curriculum/balance_{name}_episodes"] = float(episodes)
            log[f"getup_curriculum/balance_{name}_success_rate"] = float(
                successes / max(episodes, 1)
            )
            log[f"getup_curriculum/balance_{name}_window_success_rate"] = (
                float(rate) if np.isfinite(rate) else 0.0
            )
        completed = max(self._workspace_completed_episodes, 1)
        log["getup_workspace/violation_rate"] = float(
            self._workspace_violation_episodes / completed
        )
        log["getup_workspace/mean_episode_max_displacement"] = float(
            self._workspace_max_displacement_sum / completed
        )
        base_height = self._reward_base_height_values(num_envs)
        log["getup_base_height/below_range_fraction"] = float(np.mean(base_height < 0.20))
        log["getup_base_height/above_range_fraction"] = float(np.mean(base_height > 0.30))
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
            "layout_version": 3,
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
            "balance_window_completed": self._balance_window_completed.tolist(),
            "balance_window_successes": self._balance_window_successes.tolist(),
            "balance_last_window_success_rates": (self._balance_last_window_success_rates.tolist()),
            "balance_episode_counts": self._balance_episode_counts.tolist(),
            "balance_success_counts": self._balance_success_counts.tolist(),
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
        layout_version = int(payload.get("layout_version", 1))
        saved_stage = int(
            payload.get(
                "stage",
                1 if self._getup_curriculum_mastered else 0,
            )
        )
        if layout_version < 3:
            # Every pre-balance checkpoint keeps its policy/optimizer and
            # historical family counters, but must learn the newly inserted
            # dynamic recovery stage before continuing the pose curriculum.
            stage = _STAGE_BALANCE_RECOVERY
            self._getup_curriculum_difficulty = 0.0
            self._getup_curriculum_window_completed = 0
            self._getup_curriculum_window_successes = 0
        else:
            stage = saved_stage
        if not _STAGE_BALANCE_RECOVERY <= stage <= _STAGE_MIXED:
            raise ValueError(f"saved getup curriculum stage is invalid: {stage}")
        self._getup_curriculum_stage = stage
        if layout_version >= 3:
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
        if layout_version >= 3:
            self._load_pair_counter(
                payload, "balance_window_completed", self._balance_window_completed
            )
            self._load_pair_counter(
                payload, "balance_window_successes", self._balance_window_successes
            )
            self._load_pair_counter(payload, "balance_episode_counts", self._balance_episode_counts)
            self._load_pair_counter(payload, "balance_success_counts", self._balance_success_counts)
            if np.any(self._balance_window_successes > self._balance_window_completed):
                raise ValueError("saved balance window successes cannot exceed completions")
            if np.any(self._balance_success_counts > self._balance_episode_counts):
                raise ValueError("saved balance successes cannot exceed episodes")
            balance_rates = np.asarray(
                payload.get("balance_last_window_success_rates", [np.nan, np.nan]),
                dtype=np.float64,
            )
            if balance_rates.shape != (2,) or np.any(
                np.isfinite(balance_rates) & ((balance_rates < 0.0) | (balance_rates > 1.0))
            ):
                raise ValueError("saved balance window rates must be NaN or in [0, 1]")
            self._balance_last_window_success_rates[:] = balance_rates

    @staticmethod
    def _load_pair_counter(payload: dict[str, Any], name: str, target: np.ndarray) -> None:
        values = np.asarray(payload.get(name, [0, 0]), dtype=np.int64)
        if values.shape != (2,) or np.any(values < 0):
            raise ValueError(f"saved {name} must contain two nonnegative values")
        target[:] = values
