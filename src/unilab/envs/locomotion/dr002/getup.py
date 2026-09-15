"""WE11 tracking-aware get-up curriculum task."""

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
from unilab.envs.locomotion.dr002.base import LEG_ACTION_INDICES
from unilab.envs.locomotion.dr002.joystick import (
    DR002JoystickEnv,
    DR002JoystickFlatWE11Cfg,
    ResetPoseConfig,
)

_DEFAULT_POSE_BANK = Path(__file__).parents[3] / "assets/robots/dr002/we11/getup_pose_bank_v3.npz"

_STAGE_HOME_TO_GETUP = 0
_STAGE_EXACT_GETUP = 1
_STAGE_GETUP_WITH_HOME = 2
_STAGE_BALANCE = 3
_STAGE_MIXED = 4
_STAGE_ENHANCE = 5
_STAGE_NAMES = (
    "home_to_getup",
    "exact_getup",
    "getup_with_home",
    "balance",
    "mixed",
    "enhance",
)

_RESET_HOME_TO_GETUP = 0
_RESET_EXACT_GETUP = 1
_RESET_HOME = 2
_RESET_BALANCE = 3
# Compatibility name retained for tests and checkpoint-era terminology.
_RESET_ORIGINAL = _RESET_HOME_TO_GETUP
# Offline-only pose families retained for the pose-bank tools.
_RESET_FRONT = 10
_RESET_BACK = 11
_RESET_GETUP_TO_FRONT = 12
_RESET_BALANCE_RECOVERY = _RESET_BALANCE


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
    exact_getup_fraction: float = 0.80
    exact_getup_path_fraction: float = 0.20
    getup_with_home_getup_fraction: float = 0.60
    getup_with_home_home_fraction: float = 0.30
    getup_with_home_path_fraction: float = 0.10
    balance_getup_fraction: float = 0.40
    balance_home_fraction: float = 0.20
    balance_recovery_fraction: float = 0.30
    balance_path_fraction: float = 0.10
    balance_episode_seconds: float = 5.0
    mixed_getup_fraction: float = 0.50
    mixed_home_fraction: float = 0.20
    mixed_balance_fraction: float = 0.20
    mixed_path_fraction: float = 0.10
    # Final robustness stage: retain get-up/path replay while emphasizing
    # upright starts and balance recovery.
    enhance_getup_fraction: float = 0.20
    enhance_home_fraction: float = 0.30
    enhance_balance_fraction: float = 0.40
    enhance_path_fraction: float = 0.10
    # Begin each Enhance episode at zero command, then expose the policy to
    # acceleration, braking, and reversal transients more frequently.
    enhance_startup_stand_seconds: float = 1.0
    enhance_command_resampling_time: float = 2.5
    # The operator supplies a vertical velocity, while the policy continues to
    # receive an absolute base-height target. Enhance mirrors that interface by
    # integrating a held random velocity and clamping the target to this range.
    enhance_height_command_range: tuple[float, float] = (0.20, 0.30)
    enhance_height_command_initial: float = 0.25
    enhance_height_velocity_max_m_s: float = 0.03
    enhance_height_velocity_zero_fraction: float = 0.30
    # Only this fraction of Enhance environments receives the stronger,
    # multi-control-step interval push. Other stages retain the legacy push.
    enhance_push_episode_fraction: float = 0.20
    enhance_push_force_scale_range: tuple[float, float] = (0.5, 1.0)
    enhance_push_duration_s: tuple[float, float] = (0.04, 0.10)
    # When recovery is urgent, reduce tracking pressure so orientation/rate
    # penalties can prioritize catching the robot.
    enhance_tracking_tilt_start_deg: float = 10.0
    enhance_tracking_tilt_full_deg: float = 25.0
    enhance_tracking_min_multiplier: float = 0.20
    balance_max_pitch_deg: float = 16.25
    balance_max_pitch_rate_rad_s: float = 0.13
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
        self._reward_fns["getup_success_bonus"] = self._reward_getup_success_bonus

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        if self._getup_curriculum_stage <= _STAGE_GETUP_WITH_HOME:
            np.logical_or(truncated, self._getup_succeeded, out=truncated)
        elif self._getup_curriculum_stage == _STAGE_BALANCE:
            balance_steps = max(
                int(round(self._cfg.getup_curriculum.balance_episode_seconds / self._cfg.ctrl_dt)),
                1,
            )
            np.logical_or(truncated, self.episode_steps() >= balance_steps, out=truncated)
        return truncated

    def _getup_success_kinematics_clear(
        self,
        gravity: np.ndarray,
        base_height: np.ndarray,
        linvel: np.ndarray,
        gyro: np.ndarray,
    ) -> np.ndarray:
        # Tracking remains a reward and diagnostic target, but it must never
        # block pose-curriculum progression. All stages share this physical
        # get-up condition; contact clearance is applied by the caller.
        del linvel, gyro
        return np.asarray(
            (gravity[:, 2] > self._reward_cfg.getup_success_gravity_z_threshold)
            & (base_height > self._reward_cfg.getup_success_base_height_range[0]),
            dtype=np.bool_,
        )

    def _update_getup_success(self, gravity: np.ndarray) -> None:
        """Latch first success while keeping the live one-second streak current."""
        num_envs = gravity.shape[0]
        self._getup_just_succeeded[:num_envs] = False
        active = self._episode_getup_mask[:num_envs]
        if not np.any(active):
            return

        base_height = self._reward_base_height_values(num_envs)
        contacts = self._undesired_contact_values(num_envs)
        contacts_clear = (
            np.all(contacts < self._reward_cfg.undesired_contact_threshold, axis=1)
            if contacts.shape[1] > 0
            else np.ones((num_envs,), dtype=np.bool_)
        )
        workspace_clear = np.ones((num_envs,), dtype=np.bool_)
        workspace_half_extent = self._reward_cfg.getup_success_workspace_half_extent
        if workspace_half_extent is not None:
            displacement = np.abs(
                np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:num_envs, :2]
                - self._episode_reset_xy[:num_envs]
            )
            workspace_clear = np.all(displacement <= float(workspace_half_extent), axis=1)

        success_now = (
            active
            & self._getup_success_kinematics_clear(
                gravity,
                base_height,
                self.get_local_linvel(),
                self.get_gyro(),
            )
            & contacts_clear
            & workspace_clear
        )
        self._getup_success_hold_steps[:num_envs] = np.where(
            success_now,
            self._getup_success_hold_steps[:num_envs] + 1,
            0,
        )
        required_steps = max(
            int(round(self._reward_cfg.getup_success_hold_time_s / self._cfg.ctrl_dt)),
            1,
        )
        newly_succeeded = (
            active
            & ~self._getup_succeeded[:num_envs]
            & (self._getup_success_hold_steps[:num_envs] >= required_steps)
        )
        if not np.any(newly_succeeded):
            return
        rows = np.flatnonzero(newly_succeeded)
        self._getup_succeeded[rows] = True
        self._getup_just_succeeded[rows] = True
        self._getup_success_count += int(rows.size)
        self._getup_success_time_sum_s += float(
            np.sum((self.episode_steps()[rows].astype(np.float64) + 1.0) * self._cfg.ctrl_dt)
        )

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
        self._getup_curriculum_stage = _STAGE_HOME_TO_GETUP
        if curriculum.forced_difficulty is not None and curriculum.forced_stage is not None:
            forced_stage = (
                "balance"
                if curriculum.forced_stage == "balance_recovery"
                else curriculum.forced_stage
            )
            self._getup_curriculum_stage = _STAGE_NAMES.index(forced_stage)
        self._getup_curriculum_window_completed = 0
        self._getup_curriculum_window_successes = 0
        self._getup_curriculum_last_success_rate = np.nan
        self._balance_window_completed = np.zeros((2,), dtype=np.int64)
        self._balance_window_successes = np.zeros((2,), dtype=np.int64)
        self._balance_last_window_success_rates = np.full((2,), np.nan, dtype=np.float64)
        self._balance_episode_counts = np.zeros((2,), dtype=np.int64)
        self._balance_success_counts = np.zeros((2,), dtype=np.int64)
        self._consolidation_window_completed = np.zeros((2,), dtype=np.int64)
        self._consolidation_window_successes = np.zeros((2,), dtype=np.int64)
        self._consolidation_last_window_success_rates = np.full((2,), np.nan, dtype=np.float64)
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
        self._episode_enhance_push_mask = np.zeros((num_envs,), dtype=np.bool_)
        self._family_episode_counts = np.zeros((4,), dtype=np.int64)
        self._family_success_counts = np.zeros((4,), dtype=np.int64)
        self._command_episode_counts = np.zeros((2,), dtype=np.int64)
        self._command_success_counts = np.zeros((2,), dtype=np.int64)
        self._success_truncation_count = 0
        self._post_success_evaluation_count = 0
        self._success_then_failure_count = 0
        self._workspace_completed_episodes = 0
        self._workspace_violation_episodes = 0
        self._workspace_max_displacement_sum = 0.0
        self._getup_height_gate = np.zeros((num_envs,), dtype=dtype)
        self._getup_contact_gate = np.zeros((num_envs,), dtype=dtype)
        self._getup_tracking_gate = np.zeros((num_envs,), dtype=dtype)
        self._getup_lin_vel_z_multiplier = np.ones((num_envs,), dtype=dtype)
        self._getup_leg_regularization_multiplier = np.ones((num_envs,), dtype=dtype)
        self._getup_vx_error_normalized = np.zeros((num_envs,), dtype=dtype)
        self._getup_yaw_error_normalized = np.zeros((num_envs,), dtype=dtype)
        self._getup_current_commands = np.zeros((num_envs, 3), dtype=dtype)
        self._height_command_target = np.full(
            (num_envs,), curriculum.enhance_height_command_initial, dtype=dtype
        )
        self._height_velocity_command = np.zeros((num_envs,), dtype=dtype)

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

        # Retain the offline Front/Back indices for inspection utilities only.
        # The simplified training reset sampler never references either family.
        self._back_pose_indices = back_indices

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
            c.exact_getup_fraction,
            c.exact_getup_path_fraction,
            c.getup_with_home_getup_fraction,
            c.getup_with_home_home_fraction,
            c.getup_with_home_path_fraction,
            c.balance_getup_fraction,
            c.balance_home_fraction,
            c.balance_recovery_fraction,
            c.balance_path_fraction,
            c.balance_episode_seconds,
            c.mixed_getup_fraction,
            c.mixed_home_fraction,
            c.mixed_balance_fraction,
            c.mixed_path_fraction,
            c.enhance_getup_fraction,
            c.enhance_home_fraction,
            c.enhance_balance_fraction,
            c.enhance_path_fraction,
            c.enhance_startup_stand_seconds,
            c.enhance_command_resampling_time,
            *c.enhance_height_command_range,
            c.enhance_height_command_initial,
            c.enhance_height_velocity_max_m_s,
            c.enhance_height_velocity_zero_fraction,
            c.enhance_push_episode_fraction,
            *c.enhance_push_force_scale_range,
            *c.enhance_push_duration_s,
            c.enhance_tracking_tilt_start_deg,
            c.enhance_tracking_tilt_full_deg,
            c.enhance_tracking_min_multiplier,
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
        if c.balance_episode_seconds <= 0.0:
            raise ValueError("balance episode duration must be positive")
        if c.enhance_startup_stand_seconds < 0.0 or c.enhance_command_resampling_time <= 0.0:
            raise ValueError("enhance command timing must be non-negative/positive")
        height_low, height_high = c.enhance_height_command_range
        if (
            height_low >= height_high
            or not height_low <= c.enhance_height_command_initial <= height_high
            or c.enhance_height_velocity_max_m_s <= 0.0
            or not 0.0 <= c.enhance_height_velocity_zero_fraction <= 1.0
        ):
            raise ValueError("enhance height-command configuration is invalid")
        if not 0.0 <= c.enhance_push_episode_fraction <= 1.0:
            raise ValueError("enhance_push_episode_fraction must be in [0, 1]")
        push_scale = np.asarray(c.enhance_push_force_scale_range, dtype=np.float64)
        push_duration = np.asarray(c.enhance_push_duration_s, dtype=np.float64)
        if (
            push_scale.shape != (2,)
            or push_duration.shape != (2,)
            or np.any(push_scale < 0.0)
            or push_scale[0] > push_scale[1]
            or np.any(push_duration <= 0.0)
            or push_duration[0] > push_duration[1]
        ):
            raise ValueError("enhance push scale/duration ranges must be ordered and positive")
        if (
            not (0.0 <= c.enhance_tracking_tilt_start_deg < c.enhance_tracking_tilt_full_deg < 90.0)
            or not 0.0 <= c.enhance_tracking_min_multiplier <= 1.0
        ):
            raise ValueError("enhance tracking tilt gate configuration is invalid")
        pivot = np.asarray(c.balance_wheel_pivot_offset_body, dtype=np.float64)
        if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
            raise ValueError("balance_wheel_pivot_offset_body must contain three finite values")
        forced_stages = {
            None,
            "home_to_getup",
            "exact_getup",
            "getup_with_home",
            "balance",
            "balance_recovery",
            "mixed",
            "enhance",
        }
        if c.forced_stage not in forced_stages:
            raise ValueError("unsupported forced Getup stage")
        fraction_sets = (
            (c.exact_getup_fraction, c.exact_getup_path_fraction),
            (
                c.getup_with_home_getup_fraction,
                c.getup_with_home_home_fraction,
                c.getup_with_home_path_fraction,
            ),
            (
                c.balance_getup_fraction,
                c.balance_home_fraction,
                c.balance_recovery_fraction,
                c.balance_path_fraction,
            ),
            (
                c.mixed_getup_fraction,
                c.mixed_home_fraction,
                c.mixed_balance_fraction,
                c.mixed_path_fraction,
            ),
            (
                c.enhance_getup_fraction,
                c.enhance_home_fraction,
                c.enhance_balance_fraction,
                c.enhance_path_fraction,
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
        reward = cfg.reward_config
        if reward is not None:
            gate_low = reward.getup_tracking_gate_min_height
            gate_high = reward.getup_tracking_gate_full_height
            if gate_low is None or gate_high is None or not gate_low < gate_high:
                raise ValueError("Getup tracking gate heights must be increasing")
            for name, multiplier in (
                ("lin_vel_z", reward.getup_lin_vel_z_min_multiplier),
                ("leg_regularization", reward.getup_leg_regularization_min_multiplier),
            ):
                if not 0.0 <= multiplier <= 1.0:
                    raise ValueError(f"Getup {name} minimum multiplier must be in [0, 1]")

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

    def _sample_progress_at_difficulty(
        self, num_reset: int, difficulty: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self._cfg.getup_curriculum
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

    def _sample_original_progress(
        self, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forced = self._cfg.getup_curriculum.forced_difficulty
        if forced is not None:
            shared = np.full((num_reset,), float(forced), dtype=np.float64)
            return (
                shared,
                np.ones((num_reset,), dtype=np.bool_),
                np.full((num_reset,), forced >= 1.0, dtype=np.bool_),
            )
        return self._sample_progress_at_difficulty(num_reset, self._getup_curriculum_difficulty)

    def _sample_reset_plan(
        self, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c = self._cfg.getup_curriculum
        if c.forced_difficulty is not None:
            forced_stage = c.forced_stage or "home_to_getup"
            shared, frontier, exact_hard = self._sample_original_progress(num_reset)
            if forced_stage in {"balance", "balance_recovery"}:
                reset_category = _RESET_BALANCE
            elif forced_stage == "exact_getup":
                reset_category = _RESET_EXACT_GETUP
                shared.fill(1.0)
                exact_hard.fill(True)
            elif forced_stage == "getup_with_home":
                return self._sample_fixed_mixture(
                    num_reset,
                    (
                        c.getup_with_home_getup_fraction,
                        c.getup_with_home_home_fraction,
                        0.0,
                        c.getup_with_home_path_fraction,
                    ),
                )
            elif forced_stage == "mixed":
                return self._sample_fixed_mixture(
                    num_reset,
                    (
                        c.mixed_getup_fraction,
                        c.mixed_home_fraction,
                        c.mixed_balance_fraction,
                        c.mixed_path_fraction,
                    ),
                )
            elif forced_stage == "enhance":
                return self._sample_fixed_mixture(
                    num_reset,
                    (
                        c.enhance_getup_fraction,
                        c.enhance_home_fraction,
                        c.enhance_balance_fraction,
                        c.enhance_path_fraction,
                    ),
                )
            else:
                reset_category = _RESET_HOME_TO_GETUP
            category = np.full((num_reset,), reset_category, dtype=np.int8)
            return shared, frontier, exact_hard, category

        if self._getup_curriculum_stage == _STAGE_HOME_TO_GETUP:
            shared, frontier, exact_hard = self._sample_original_progress(num_reset)
            category = np.full((num_reset,), _RESET_HOME_TO_GETUP, dtype=np.int8)
            return shared, frontier, exact_hard, category
        if self._getup_curriculum_stage == _STAGE_EXACT_GETUP:
            fractions = (c.exact_getup_fraction, 0.0, 0.0, c.exact_getup_path_fraction)
        elif self._getup_curriculum_stage == _STAGE_GETUP_WITH_HOME:
            fractions = (
                c.getup_with_home_getup_fraction,
                c.getup_with_home_home_fraction,
                0.0,
                c.getup_with_home_path_fraction,
            )
        elif self._getup_curriculum_stage == _STAGE_BALANCE:
            fractions = (
                c.balance_getup_fraction,
                c.balance_home_fraction,
                c.balance_recovery_fraction,
                c.balance_path_fraction,
            )
        elif self._getup_curriculum_stage == _STAGE_MIXED:
            fractions = (
                c.mixed_getup_fraction,
                c.mixed_home_fraction,
                c.mixed_balance_fraction,
                c.mixed_path_fraction,
            )
        else:
            fractions = (
                c.enhance_getup_fraction,
                c.enhance_home_fraction,
                c.enhance_balance_fraction,
                c.enhance_path_fraction,
            )
        return self._sample_fixed_mixture(num_reset, fractions)

    def _sample_fixed_mixture(
        self,
        num_reset: int,
        fractions: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cutoffs = np.cumsum(np.asarray(fractions, dtype=np.float64))
        draw = np.random.uniform(size=num_reset)
        bins = np.searchsorted(cutoffs, draw, side="right")
        shared = np.zeros((num_reset,), dtype=np.float64)
        exact_getup = bins == 0
        home = bins == 1
        balance = bins == 2
        path = bins == 3
        shared[exact_getup] = 1.0
        if np.any(balance):
            shared[balance] = self._sample_progress_at_difficulty(
                int(np.count_nonzero(balance)), 1.0
            )[0]
        shared[path] = np.random.uniform(0.0, 1.0, size=int(np.count_nonzero(path)))
        category = np.full((num_reset,), _RESET_EXACT_GETUP, dtype=np.int8)
        category[home] = _RESET_HOME
        category[balance] = _RESET_BALANCE
        category[path] = _RESET_HOME_TO_GETUP
        return (
            shared,
            np.zeros((num_reset,), dtype=np.bool_),
            exact_getup,
            category,
        )

    def sample_getup_reset_qpos(self, num_reset: int) -> np.ndarray:
        shared, frontier, _exact_hard, category = self._sample_reset_plan(num_reset)
        qpos = np.broadcast_to(self._easy_qpos, (num_reset, self._easy_qpos.size)).copy()
        exact_rows = np.flatnonzero(category == _RESET_EXACT_GETUP)
        if exact_rows.size:
            qpos[exact_rows] = self._hard_qpos
        balance_direction = np.zeros((num_reset,), dtype=np.int8)
        balance_pitch = np.zeros((num_reset,), dtype=np.float64)
        balance_pitch_rate = np.zeros((num_reset,), dtype=np.float64)
        balance_rows = np.flatnonzero(category == _RESET_BALANCE)
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
        rows = np.flatnonzero(category == _RESET_HOME_TO_GETUP)
        if rows.size:
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
        balance = self._pending_reset_category == _RESET_BALANCE
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
        self._height_command_target[rows] = float(
            self._cfg.getup_curriculum.enhance_height_command_initial
        )
        self._height_velocity_command[rows] = 0.0
        if self._getup_curriculum_stage == _STAGE_ENHANCE:
            probability = float(self._cfg.getup_curriculum.enhance_push_episode_fraction)
            self._episode_enhance_push_mask[rows] = np.random.uniform(size=rows.size) < probability
        else:
            self._episode_enhance_push_mask[rows] = False

    def sample_commands(
        self,
        num_samples: int,
        *,
        env_ids: np.ndarray | None = None,
        resample_standing: bool = False,
    ) -> np.ndarray:
        if self._getup_curriculum_stage == _STAGE_ENHANCE:
            # Reset plans call this once before an episode starts. Enhance then
            # deliberately transitions away from rest in _update_commands.
            return self.startup_commands(num_samples)
        return super().sample_commands(
            num_samples,
            env_ids=env_ids,
            resample_standing=resample_standing,
        )

    def startup_commands(self, num_samples: int) -> np.ndarray:
        commands = super().startup_commands(num_samples)
        commands[:, 2] = float(self._cfg.getup_curriculum.enhance_height_command_initial)
        return commands

    def _sample_enhance_height_velocity(self, num_samples: int) -> np.ndarray:
        c = self._cfg.getup_curriculum
        velocity = np.random.uniform(
            -c.enhance_height_velocity_max_m_s,
            c.enhance_height_velocity_max_m_s,
            size=num_samples,
        )
        stopped = np.random.uniform(size=num_samples) < c.enhance_height_velocity_zero_fraction
        velocity[stopped] = 0.0
        return np.asarray(velocity, dtype=get_global_dtype())

    def _update_commands(self, info: dict) -> None:
        if self._getup_curriculum_stage != _STAGE_ENHANCE:
            super()._update_commands(info)
        else:
            commands = np.asarray(info.get("commands"), dtype=get_global_dtype())
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            c = self._cfg.getup_curriculum
            startup_steps = max(int(round(c.enhance_startup_stand_seconds / self._cfg.ctrl_dt)), 0)
            interval = max(int(round(c.enhance_command_resampling_time / self._cfg.ctrl_dt)), 1)
            startup = steps < startup_steps
            if np.any(startup):
                commands[startup, :2] = 0.0
                self._height_command_target[startup] = float(c.enhance_height_command_initial)
                self._height_velocity_command[startup] = 0.0
            elapsed = steps.astype(np.int64) - startup_steps
            resample = (steps == startup_steps) | (
                (steps > startup_steps) & ((elapsed % interval) == 0)
            )
            if np.any(resample):
                env_ids = np.flatnonzero(resample).astype(np.int32)
                commands[resample] = super().sample_commands(
                    len(env_ids), env_ids=env_ids, resample_standing=True
                )
                self._height_velocity_command[resample] = self._sample_enhance_height_velocity(
                    len(env_ids)
                )
            active = ~startup
            self._height_command_target[active] += (
                self._height_velocity_command[active] * self._cfg.ctrl_dt
            )
            height_low, height_high = c.enhance_height_command_range
            np.clip(
                self._height_command_target,
                float(height_low),
                float(height_high),
                out=self._height_command_target,
            )
            commands[:, 2] = self._height_command_target
            info["commands"] = commands
        commands = np.asarray(info.get("commands"), dtype=get_global_dtype())
        if commands.shape == self._getup_current_commands.shape:
            self._getup_current_commands[:] = commands

    def interval_push_episode_mask(self) -> np.ndarray:
        if self._getup_curriculum_stage != _STAGE_ENHANCE:
            return np.ones((self._num_envs,), dtype=np.bool_)
        return self._episode_enhance_push_mask.copy()

    def sample_interval_push_force_scale(self, num_push: int) -> np.ndarray:
        if self._getup_curriculum_stage != _STAGE_ENHANCE:
            return np.ones((num_push,), dtype=np.float64)
        low, high = self._cfg.getup_curriculum.enhance_push_force_scale_range
        return np.random.uniform(float(low), float(high), size=num_push)

    def sample_interval_push_duration_steps(self, num_push: int) -> np.ndarray:
        if self._getup_curriculum_stage != _STAGE_ENHANCE:
            return np.ones((num_push,), dtype=np.int32)
        low_s, high_s = self._cfg.getup_curriculum.enhance_push_duration_s
        low = max(int(np.ceil(float(low_s) / self._cfg.ctrl_dt)), 1)
        high = max(int(np.floor(float(high_s) / self._cfg.ctrl_dt)), low)
        return np.random.randint(low, high + 1, size=num_push, dtype=np.int32)

    def interval_push_startup_steps(self) -> int:
        if self._getup_curriculum_stage == _STAGE_ENHANCE:
            return max(
                int(
                    round(
                        self._cfg.getup_curriculum.enhance_startup_stand_seconds / self._cfg.ctrl_dt
                    )
                ),
                0,
            )
        return int(self._startup_stand_steps)

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        num_envs = linvel.shape[0]
        base_height = self._reward_base_height_values(num_envs)
        configured_low = self._reward_cfg.getup_tracking_gate_min_height
        configured_high = self._reward_cfg.getup_tracking_gate_full_height
        assert configured_low is not None and configured_high is not None
        low = float(configured_low)
        high = float(configured_high)
        height_gate = np.clip((base_height - low) / (high - low), 0.0, 1.0)
        contacts = self._undesired_contact_values(num_envs)
        contact_clear = (
            np.all(contacts < self._reward_cfg.undesired_contact_threshold, axis=1)
            if contacts.shape[1] > 0
            else np.ones((num_envs,), dtype=np.bool_)
        )
        contact_gate = contact_clear.astype(get_global_dtype())
        tracking_gate = np.asarray(height_gate * contact_gate, dtype=get_global_dtype())
        if self._getup_curriculum_stage == _STAGE_ENHANCE:
            c = self._cfg.getup_curriculum
            tilt_deg = np.rad2deg(
                np.arccos(np.clip(np.asarray(gravity[:, 2], dtype=np.float64), -1.0, 1.0))
            )
            tilt_progress = np.clip(
                (tilt_deg - c.enhance_tracking_tilt_start_deg)
                / (c.enhance_tracking_tilt_full_deg - c.enhance_tracking_tilt_start_deg),
                0.0,
                1.0,
            )
            tilt_gate = 1.0 - (1.0 - c.enhance_tracking_min_multiplier) * tilt_progress
            tracking_gate *= tilt_gate.astype(get_global_dtype())
        self._getup_height_gate[:num_envs] = height_gate
        self._getup_contact_gate[:num_envs] = contact_gate
        self._getup_tracking_gate[:num_envs] = tracking_gate
        self._getup_lin_vel_z_multiplier[:num_envs] = (
            self._reward_cfg.getup_lin_vel_z_min_multiplier
            + (1.0 - self._reward_cfg.getup_lin_vel_z_min_multiplier) * tracking_gate
        )
        self._getup_leg_regularization_multiplier[:num_envs] = (
            self._reward_cfg.getup_leg_regularization_min_multiplier
            + (1.0 - self._reward_cfg.getup_leg_regularization_min_multiplier) * tracking_gate
        )
        commands = np.asarray(info["commands"], dtype=get_global_dtype())[:num_envs]
        vx_full_scale = max(
            np.max(np.abs(np.asarray(self._cfg.commands.lin_vel_x, dtype=np.float64))),
            1.0e-6,
        )
        yaw_full_scale = max(
            np.max(np.abs(np.asarray(self._cfg.commands.ang_vel_z, dtype=np.float64))),
            1.0e-6,
        )
        self._getup_vx_error_normalized[:num_envs] = (
            np.abs(linvel[:, 0] - commands[:, 0]) / vx_full_scale
        )
        self._getup_yaw_error_normalized[:num_envs] = (
            np.abs(gyro[:, 2] - commands[:, 1]) / yaw_full_scale
        )
        return super()._compute_reward(info, linvel, gyro, gravity, dof_pos, dof_vel)

    def _reward_track_lin_vel_x(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 0] - ctx.linvel[:, 0])
        configured_std = self._reward_cfg.track_lin_vel_x_std
        std = ctx.tracking_sigma if configured_std is None else max(float(configured_std), 1.0e-6)
        reward = np.exp(-error / (std * std)) * self._getup_tracking_gate[: ctx.num_envs]
        return self._clip_lingzu_reward(
            "track_lin_vel_x",
            reward,
            clip_single_reward=float(self._reward_cfg.track_lin_vel_x_term_clip),
        )

    def _reward_base_height_cmd(self, ctx: RewardContext) -> np.ndarray:
        """Clear the ground while continuously taking over target tracking."""
        std = max(float(self._reward_cfg.base_height_std), 1.0e-6)
        min_height = float(self._reward_cfg.getup_success_base_height_range[0])
        floor_error = np.maximum(min_height - ctx.base_height, 0.0)
        tracking_error = (
            ctx.base_height - np.asarray(ctx.info["commands"], dtype=get_global_dtype())[:, 2]
        )
        # This gate ramps from zero at the minimum recoverable height to one at
        # 0.20 m. Multiplying the squared error by the gate avoids the former
        # discontinuity that made settling just below 0.20 m advantageous.
        tracking_weight = self._getup_tracking_gate[: ctx.num_envs]
        reward = np.asarray(
            (np.square(floor_error) + tracking_weight * np.square(tracking_error)) / (std * std),
            dtype=get_global_dtype(),
        )
        return self._clip_lingzu_reward(
            "base_height",
            reward,
            clip_single_reward=float(self._reward_cfg.base_height_clip),
        )

    def _reward_track_ang_vel_z(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 1] - ctx.gyro[:, 2])
        reward = np.exp(-error / (ctx.tracking_sigma**2))
        reward *= self._getup_tracking_gate[: ctx.num_envs]
        return self._clip_lingzu_reward(
            "track_ang_vel_z",
            reward,
            clip_single_reward=float(self._reward_cfg.track_ang_vel_z_term_clip),
        )

    def _reward_track_lin_vel_x_enhance(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 0] - ctx.linvel[:, 0])
        std = max(float(self._reward_cfg.track_lin_vel_x_enhance_std), 1.0e-6)
        reward = (np.exp(-error / (std * std)) - 1.0) * self._getup_tracking_gate[: ctx.num_envs]
        return self._clip_lingzu_reward(
            "track_lin_vel_x_enhance",
            reward,
            clip_single_reward=float(self._reward_cfg.track_lin_vel_x_term_clip),
        )

    def _reward_lin_vel_z_lingzu(self, ctx: RewardContext) -> np.ndarray:
        reward = np.square(ctx.linvel[:, 2]) * self._getup_lin_vel_z_multiplier[: ctx.num_envs]
        return self._clip_lingzu_reward("lin_vel_z", reward)

    def _reward_joint_acc_l2(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(
            ctx.info.get("qacc", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        reward = np.sum(np.square(qacc[:, LEG_ACTION_INDICES]), axis=1)
        reward *= self._getup_leg_regularization_multiplier[: ctx.num_envs]
        return self._clip_lingzu_reward("joint_acc_l2", reward)

    def _reward_action_smooth_lingzu(self, ctx: RewardContext) -> np.ndarray:
        current = np.asarray(ctx.info["current_actions"], dtype=get_global_dtype())
        leg = LEG_ACTION_INDICES
        diff = (
            current[:, leg]
            - 2.0 * self._lingzu_prev_action[:, leg]
            + self._lingzu_prev_prev_action[:, leg]
        )
        reward = np.sum(np.square(diff), axis=1)
        reward *= self._lingzu_action_history_count >= 2
        reward *= self._getup_leg_regularization_multiplier[: ctx.num_envs]
        self._lingzu_prev_prev_action[:] = self._lingzu_prev_action
        self._lingzu_prev_action[:] = current
        self._lingzu_action_history_count += 1
        return self._clip_lingzu_reward("action_smooth_lingzu", reward)

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

    def _reward_getup_success_bonus(self, ctx: RewardContext) -> np.ndarray:
        """Emit a unit pulse only on the first successful step of an episode."""
        return np.asarray(
            self._getup_just_succeeded[: ctx.num_envs],
            dtype=get_global_dtype(),
        )

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
        required_hold_steps = max(
            int(round(self._reward_cfg.getup_success_hold_time_s / self._cfg.ctrl_dt)), 1
        )
        final_stable = self._getup_success_hold_steps[rows] >= required_hold_steps
        self._workspace_completed_episodes += int(rows.size)
        self._workspace_violation_episodes += int(
            np.count_nonzero(self._episode_workspace_violated[rows])
        )
        self._workspace_max_displacement_sum += float(
            np.sum(self._episode_max_abs_xy_displacement[rows])
        )

        for family in range(4):
            family_mask = categories == family
            self._family_episode_counts[family] += int(np.count_nonzero(family_mask))
            family_outcomes = final_stable if family == _RESET_BALANCE else outcomes
            self._family_success_counts[family] += int(
                np.count_nonzero(family_outcomes[family_mask])
            )
        standing = self._episode_standing_mask[rows]
        for command_family, command_mask in enumerate((standing, ~standing)):
            self._command_episode_counts[command_family] += int(np.count_nonzero(command_mask))
            self._command_success_counts[command_family] += int(
                np.count_nonzero(outcomes[command_mask])
            )
        if self._getup_curriculum_stage <= _STAGE_GETUP_WITH_HOME:
            self._success_truncation_count += int(np.count_nonzero(outcomes))
        else:
            self._post_success_evaluation_count += int(np.count_nonzero(outcomes))
            self._success_then_failure_count += int(np.count_nonzero(outcomes & ~final_stable))

        balance_mask = categories == _RESET_BALANCE
        for index, direction in enumerate((-1, 1)):
            direction_mask = balance_mask & (self._episode_balance_direction[rows] == direction)
            self._balance_episode_counts[index] += int(np.count_nonzero(direction_mask))
            self._balance_success_counts[index] += int(
                np.count_nonzero(final_stable[direction_mask])
            )

        c = self._cfg.getup_curriculum
        if c.forced_difficulty is not None or self._getup_curriculum_stage == _STAGE_ENHANCE:
            return
        if self._getup_curriculum_stage == _STAGE_MIXED:
            self._record_mixed_curriculum_outcomes(final_stable)
            return
        if self._getup_curriculum_stage == _STAGE_BALANCE:
            self._record_balance_curriculum_outcomes(rows, final_stable)
            return
        if self._getup_curriculum_stage == _STAGE_GETUP_WITH_HOME:
            self._record_consolidation_outcomes(rows, outcomes)
            return
        eligible = (
            self._episode_frontier[rows]
            if self._getup_curriculum_stage == _STAGE_HOME_TO_GETUP
            else categories == _RESET_EXACT_GETUP
        )
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
            if old_stage == _STAGE_HOME_TO_GETUP:
                if old_difficulty >= 1.0 and rate >= c.promote_success_rate:
                    self._getup_curriculum_stage = _STAGE_EXACT_GETUP
                    self._getup_curriculum_difficulty = 1.0
                elif rate >= c.promote_success_rate:
                    self._getup_curriculum_difficulty = min(1.0, old_difficulty + c.difficulty_step)
                    if self._getup_curriculum_difficulty > old_difficulty:
                        self._getup_curriculum_promotions += 1
                elif rate < c.demote_success_rate:
                    self._getup_curriculum_difficulty = max(0.0, old_difficulty - c.difficulty_step)
                    if self._getup_curriculum_difficulty < old_difficulty:
                        self._getup_curriculum_demotions += 1
            elif rate >= c.promote_success_rate:
                self._getup_curriculum_stage = _STAGE_GETUP_WITH_HOME
            self._getup_curriculum_window_completed = 0
            self._getup_curriculum_window_successes = 0
            # Every row in this reset batch was sampled against the old
            # boundary. Do not credit leftovers to a newly selected level.
            if (
                self._getup_curriculum_stage != old_stage
                or self._getup_curriculum_difficulty != old_difficulty
            ):
                break

    def _record_consolidation_outcomes(self, rows: np.ndarray, outcomes: np.ndarray) -> None:
        c = self._cfg.getup_curriculum
        target = c.window_episodes // 2
        for index, category in enumerate((_RESET_EXACT_GETUP, _RESET_HOME)):
            values = outcomes[self._episode_reset_category[rows] == category]
            capacity = target - int(self._consolidation_window_completed[index])
            take = min(capacity, int(values.size))
            if take > 0:
                self._consolidation_window_completed[index] += take
                self._consolidation_window_successes[index] += int(np.count_nonzero(values[:take]))
        if np.any(self._consolidation_window_completed < target):
            return
        rates = self._consolidation_window_successes / self._consolidation_window_completed
        self._consolidation_last_window_success_rates[:] = rates
        self._getup_curriculum_last_success_rate = float(np.min(rates))
        if np.all(rates >= c.promote_success_rate):
            self._getup_curriculum_stage = _STAGE_BALANCE
        self._consolidation_window_completed.fill(0)
        self._consolidation_window_successes.fill(0)

    def _record_balance_curriculum_outcomes(self, rows: np.ndarray, outcomes: np.ndarray) -> None:
        c = self._cfg.getup_curriculum
        per_direction_target = c.window_episodes // 2
        for index, direction in enumerate((-1, 1)):
            eligible = (self._episode_reset_category[rows] == _RESET_BALANCE) & (
                self._episode_balance_direction[rows] == direction
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
        if np.all(rates >= c.promote_success_rate):
            self._getup_curriculum_stage = _STAGE_MIXED
            self._getup_curriculum_mastered = False
        self._balance_window_completed.fill(0)
        self._balance_window_successes.fill(0)

    def _record_mixed_curriculum_outcomes(self, outcomes: np.ndarray) -> None:
        c = self._cfg.getup_curriculum
        values = np.asarray(outcomes, dtype=np.bool_).reshape(-1)
        offset = 0
        while offset < values.size:
            capacity = c.window_episodes - self._getup_curriculum_window_completed
            take = min(capacity, int(values.size - offset))
            selected = values[offset : offset + take]
            self._getup_curriculum_window_completed += take
            self._getup_curriculum_window_successes += int(np.count_nonzero(selected))
            offset += take
            if self._getup_curriculum_window_completed < c.window_episodes:
                continue
            rate = self._getup_curriculum_window_successes / c.window_episodes
            self._getup_curriculum_last_success_rate = float(rate)
            self._getup_curriculum_window_completed = 0
            self._getup_curriculum_window_successes = 0
            if rate >= c.promote_success_rate:
                self._getup_curriculum_stage = _STAGE_ENHANCE
                self._getup_curriculum_mastered = True
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
        family_names = ("home_to_getup", "exact_getup", "home", "balance")
        for family, name in enumerate(family_names):
            log[f"getup_curriculum/{name}_reset_fraction"] = float(np.mean(categories == family))
            episodes = int(self._family_episode_counts[family])
            successes = int(self._family_success_counts[family])
            log[f"getup_curriculum/{name}_episodes"] = float(episodes)
            log[f"getup_curriculum/{name}_success_rate"] = float(successes / max(episodes, 1))
        balance = categories == _RESET_BALANCE
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
        log["getup_curriculum/consolidation_window_completed"] = float(
            np.sum(self._consolidation_window_completed)
        )
        for index, name in enumerate(("getup", "home")):
            rate = self._consolidation_last_window_success_rates[index]
            log[f"getup_curriculum/consolidation_{name}_window_success_rate"] = (
                float(rate) if np.isfinite(rate) else 0.0
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
        height_target = self._getup_current_commands[:num_envs, 2]
        height_abs_error = np.abs(base_height - height_target)
        log["getup_base_height/actual_mean"] = float(np.mean(base_height))
        log["getup_base_height/actual_std"] = float(np.std(base_height))
        log["getup_base_height/actual_min"] = float(np.min(base_height))
        log["getup_base_height/actual_max"] = float(np.max(base_height))
        log["getup_base_height/target_mean"] = float(np.mean(height_target))
        log["getup_base_height/target_std"] = float(np.std(height_target))
        log["getup_base_height/target_min"] = float(np.min(height_target))
        log["getup_base_height/target_max"] = float(np.max(height_target))
        log["getup_base_height/abs_error_mean"] = float(np.mean(height_abs_error))
        log["getup_base_height/tracking_weight_mean"] = float(
            np.mean(self._getup_tracking_gate[:num_envs])
        )
        target_std = float(np.std(height_target))
        actual_std = float(np.std(base_height))
        log["getup_base_height/target_actual_correlation"] = (
            float(np.corrcoef(height_target, base_height)[0, 1])
            if target_std > 1.0e-6 and actual_std > 1.0e-6
            else 0.0
        )
        for command_family, name in enumerate(("standing", "moving")):
            episodes = int(self._command_episode_counts[command_family])
            successes = int(self._command_success_counts[command_family])
            log[f"getup_commands/{name}_episodes"] = float(episodes)
            log[f"getup_commands/{name}_success_rate"] = float(successes / max(episodes, 1))
        log["getup_commands/vx_error_normalized_mean"] = float(
            np.mean(self._getup_vx_error_normalized[:num_envs])
        )
        log["getup_commands/yaw_error_normalized_mean"] = float(
            np.mean(self._getup_yaw_error_normalized[:num_envs])
        )
        log["getup_reward_gate/height_mean"] = float(np.mean(self._getup_height_gate[:num_envs]))
        log["getup_reward_gate/contact_clear_fraction"] = float(
            np.mean(self._getup_contact_gate[:num_envs])
        )
        log["getup_reward_gate/tracking_mean"] = float(
            np.mean(self._getup_tracking_gate[:num_envs])
        )
        log["getup_reward_gate/lin_vel_z_multiplier_mean"] = float(
            np.mean(self._getup_lin_vel_z_multiplier[:num_envs])
        )
        log["getup_reward_gate/leg_regularization_multiplier_mean"] = float(
            np.mean(self._getup_leg_regularization_multiplier[:num_envs])
        )
        log["getup/success_truncations"] = float(self._success_truncation_count)
        log["getup/success_then_failures"] = float(self._success_then_failure_count)
        log["getup/success_then_failure_rate"] = float(
            self._success_then_failure_count / max(self._post_success_evaluation_count, 1)
        )

    def training_state_dict(self) -> dict[str, Any]:
        state = super().training_state_dict()
        state["getup_pose_curriculum"] = {
            "layout_version": 6,
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
            "command_episode_counts": self._command_episode_counts.tolist(),
            "command_success_counts": self._command_success_counts.tolist(),
            "success_truncation_count": int(self._success_truncation_count),
            "post_success_evaluation_count": int(self._post_success_evaluation_count),
            "success_then_failure_count": int(self._success_then_failure_count),
            "consolidation_window_completed": self._consolidation_window_completed.tolist(),
            "consolidation_window_successes": self._consolidation_window_successes.tolist(),
            "consolidation_last_window_success_rates": (
                self._consolidation_last_window_success_rates.tolist()
            ),
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
        if layout_version < 5:
            # V1-V3 used the old Front/Back stages; V4 also counted success
            # through vx/yaw tracking gates. Keep policy/optimizer weights but
            # restart the physical-success curriculum with a clean window.
            self._getup_curriculum_stage = _STAGE_HOME_TO_GETUP
            self._getup_curriculum_difficulty = 0.0
            self._getup_curriculum_window_completed = 0
            self._getup_curriculum_window_successes = 0
            self._getup_curriculum_last_success_rate = np.nan
            return
        stage = int(payload["stage"])
        if not _STAGE_HOME_TO_GETUP <= stage <= _STAGE_ENHANCE:
            raise ValueError(f"saved getup curriculum stage is invalid: {stage}")
        self._getup_curriculum_stage = stage
        self._getup_curriculum_window_completed = max(int(payload["window_completed"]), 0)
        self._getup_curriculum_window_successes = max(int(payload["window_successes"]), 0)
        self._getup_curriculum_last_success_rate = float(payload["last_success_rate"])
        self._load_counter(payload, "family_episode_counts", self._family_episode_counts)
        self._load_counter(payload, "family_success_counts", self._family_success_counts)
        self._load_counter(payload, "command_episode_counts", self._command_episode_counts)
        self._load_counter(payload, "command_success_counts", self._command_success_counts)
        self._load_counter(
            payload, "consolidation_window_completed", self._consolidation_window_completed
        )
        self._load_counter(
            payload, "consolidation_window_successes", self._consolidation_window_successes
        )
        self._load_counter(payload, "balance_window_completed", self._balance_window_completed)
        self._load_counter(payload, "balance_window_successes", self._balance_window_successes)
        self._load_counter(payload, "balance_episode_counts", self._balance_episode_counts)
        self._load_counter(payload, "balance_success_counts", self._balance_success_counts)
        for successes, completed in (
            (self._family_success_counts, self._family_episode_counts),
            (self._command_success_counts, self._command_episode_counts),
            (self._consolidation_window_successes, self._consolidation_window_completed),
            (self._balance_window_successes, self._balance_window_completed),
            (self._balance_success_counts, self._balance_episode_counts),
        ):
            if np.any(successes > completed):
                raise ValueError("saved Getup successes cannot exceed completed episodes")
        self._load_rate_pair(
            payload,
            "consolidation_last_window_success_rates",
            self._consolidation_last_window_success_rates,
        )
        self._load_rate_pair(
            payload, "balance_last_window_success_rates", self._balance_last_window_success_rates
        )
        self._success_truncation_count = max(int(payload.get("success_truncation_count", 0)), 0)
        self._post_success_evaluation_count = max(
            int(payload.get("post_success_evaluation_count", 0)), 0
        )
        self._success_then_failure_count = max(int(payload.get("success_then_failure_count", 0)), 0)
        if self._success_then_failure_count > self._post_success_evaluation_count:
            raise ValueError("saved post-success failure count exceeds evaluated successes")

    @staticmethod
    def _load_counter(payload: dict[str, Any], name: str, target: np.ndarray) -> None:
        values = np.asarray(payload.get(name, np.zeros_like(target)), dtype=np.int64)
        if values.shape != target.shape or np.any(values < 0):
            raise ValueError(f"saved {name} has an invalid counter layout")
        target[:] = values

    @staticmethod
    def _load_rate_pair(payload: dict[str, Any], name: str, target: np.ndarray) -> None:
        values = np.asarray(payload.get(name, [np.nan, np.nan]), dtype=np.float64)
        if values.shape != (2,) or np.any(np.isfinite(values) & ((values < 0.0) | (values > 1.0))):
            raise ValueError(f"saved {name} must contain two rates in [0, 1]")
        target[:] = values
