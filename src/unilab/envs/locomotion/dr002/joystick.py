from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.backend.base import BatchedMixedPdControl
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply, np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.dr002.base import (
    DEFAULT_DR002_ANGLES,
    JOINT_SENSOR_PREFIXES,
    JOINT_VEL_OBSERVATION_INDICES,
    LEG_ACTION_INDICES,
    NUM_DR002_ACTIONS,
    WHEEL_ACTION_INDICES,
    ControlConfig,
    DR002BaseCfg,
    DR002BaseEnv,
    NoiseConfig,
    compute_dr002_motor_ctrl,
    stack_joint_sensors,
)

_HISTORY_LENGTH = 5
_TERM_DIMS = (3, 3, 4, 6, 6, 3)
_WING_ANGLE_OBS_DIM = 2
_DR002_ASSET_ROOT = ASSETS_ROOT_PATH / "robots" / "dr002"
_JOINT_POS_SENSOR_NAMES = tuple(f"{prefix}_pos" for prefix in JOINT_SENSOR_PREFIXES)
_JOINT_VEL_SENSOR_NAMES = tuple(f"{prefix}_vel" for prefix in JOINT_SENSOR_PREFIXES)
_POSITION_CONTROL_MASK = np.ones((NUM_DR002_ACTIONS,), dtype=np.int32)
_POSITION_CONTROL_MASK[WHEEL_ACTION_INDICES] = 0
_PRIVILEGED_BODY_NAMES = tuple(f"{prefix}_joint" for prefix in JOINT_SENSOR_PREFIXES)
_ISAACLAB_CRITIC_TERM_DIMS = (
    3,  # base linear velocity
    3,  # base angular velocity
    3,  # projected gravity
    3,  # velocity command
    len(LEG_ACTION_INDICES),  # leg joint position error (wheel positions omitted)
    NUM_DR002_ACTIONS,  # joint velocity
    NUM_DR002_ACTIONS,  # actions
    NUM_DR002_ACTIONS,  # gym_last_action (same current action, matching IsaacLab)
    NUM_DR002_ACTIONS,  # gym_previous_action
    NUM_DR002_ACTIONS,  # gym_joint_acc
    77,  # gym_height_measurements
    NUM_DR002_ACTIONS,  # gym_joint_torque
    1,  # gym_base_mass_delta
    3,  # gym_base_com
    NUM_DR002_ACTIONS,  # gym_default_joint_pos_delta
    2,  # gym_material_properties: robot friction, restitution
    3,  # measured CSV force [Fx, Fy, Fz] (critic only)
)
_LEGACY_CRITIC_DIM = 52
_CRITIC_DIM = sum(_ISAACLAB_CRITIC_TERM_DIMS)
assert _CRITIC_DIM == 144
_CRITIC_WITH_MEASURED_MOMENT_DIM = _CRITIC_DIM + 3
assert _CRITIC_WITH_MEASURED_MOMENT_DIM == 147


@dataclass
class DR002Commands:
    lin_vel_x: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    ang_vel_z: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    height: list[float] = field(default_factory=lambda: [0.28, 0.28])
    resampling_time: float = 5.0
    startup_stand_seconds: float = 3.0
    rel_standing_envs: float = 0.0
    standing_envs_episode_persistent: bool = False
    curriculum: bool = True
    range_multiplier: list[float] = field(default_factory=lambda: [1.0, 2.0])
    ang_vel_z_range_multiplier: list[float] = field(default_factory=lambda: [0.3, 1.0])
    curriculum_threshold: float = 0.7
    curriculum_demote_threshold: float = 0.4
    curriculum_allow_demotion: bool = True
    curriculum_step: float = 0.1
    curriculum_min_episode_fraction: float = 0.8
    curriculum_moving_command_threshold: float = 0.05


@dataclass
class DR002DomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    init_xy_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    init_qvel_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])

    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    com_offset_y: list[float] = field(default_factory=lambda: [-0.01, 0.01])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.01, 0.01])

    randomize_torque_scale: bool = False
    torque_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    randomize_default_joint_pos: bool = False
    default_joint_pos_offset_range: list[float] = field(default_factory=lambda: [-0.02, 0.02])
    # When enabled, treat added base mass as preserving the base's nominal
    # mass distribution: scale all three principal inertias by
    # (nominal_mass + delta) / nominal_mass while leaving COM unchanged.
    couple_base_inertia_to_added_mass: bool = False
    # Preserve the existing UniLab behavior by default. Tasks that need exact
    # IsaacLab startup-event semantics can opt into independent left/right
    # non-base body draws and keep them fixed across episode resets.
    body_mass_share_bilateral_scale: bool = True
    body_mass_resample_on_reset: bool = True

    # Optional body-frame linear-velocity jump for interval pushes. The
    # provider rotates sampled vectors into the backend's world-frame contract.
    push_linear_velocity_delta_limit: list[float] | None = None
    # Optional one-control-step body-frame force that can be combined with the
    # velocity jump. The provider rotates it into world coordinates before
    # applying it. When both options are unset, max_force keeps the legacy
    # force-only behavior.
    push_force_limit: list[float] | None = None
    # Sample one trigger time independently inside every push_interval window.
    push_randomize_within_interval: bool = False

    csv_force_enabled: bool = False
    csv_force_path: str = "/home/esd_wch/lsaac_lab_ws/force_raw.csv"
    csv_force_curriculum_paths: list[str] = field(default_factory=list)
    csv_force_curriculum_hz: list[float] = field(default_factory=list)
    csv_force_period: float = 10.0
    csv_force_transition_seconds: float = 0.1
    csv_force_start_delay_range_s: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # Preserve legacy behavior by default: persistent standing-command
    # episodes do not receive the measured CSV replay. WE9 opts in so standing
    # changes only the command, not the independently sampled force level.
    csv_force_apply_to_standing: bool = False
    # Sample one scalar per environment at reset and hold it for the complete
    # episode. The same scalar multiplies Fx/Fy/Fz/Mx/My/Mz, changing only the
    # measured wrench amplitude while preserving its waveform and phase.
    csv_force_amplitude_scale_range: list[float] = field(default_factory=lambda: [1.0, 1.0])
    csv_force_zero_fy: bool = False
    csv_force_rotation: list[float] = field(
        default_factory=lambda: [
            0.7907964138, 0.0, -0.6120792693,
            0.0, 1.0, 0.0,
            0.6120792693, 0.0, 0.7907964138,
        ]
    )
    csv_force_body_name: str | None = None
    csv_force_push_point: list[float] = field(default_factory=lambda: [0.14543, 0.0, 0.1118])
    # Apply the measured free moment Mx/My/Mz about csv_force_push_point.
    # Disabled by default to preserve force-only legacy training contracts.
    csv_force_apply_measured_moment: bool = False
    csv_force_apply_point_torque: bool = True
    csv_force_observation_force_normalization: float = 50.0
    # Expose the rotated measured sensor moment Mx/My/Mz only to the privileged
    # critic. This is independent from whether the moment is applied to physics.
    csv_force_observation_include_measured_moment: bool = False
    csv_force_observation_moment_normalization: float = 15.0
    csv_force_curriculum: bool = True
    csv_force_curriculum_window_episodes: int = 4096
    csv_force_curriculum_promote_tracking_threshold: float = 0.65
    csv_force_curriculum_demote_tracking_threshold: float = 0.45
    csv_force_curriculum_promote_fail_rate_max: float = 0.20
    csv_force_curriculum_demote_fail_rate_min: float = 0.30
    csv_force_curriculum_promote_mean_episode_length_fraction: float = 0.0
    csv_force_curriculum_demote_mean_episode_length_fraction: float = 0.60
    csv_force_curriculum_step_levels: int = 1
    csv_force_curriculum_allow_demotion: bool = False


@dataclass
class WingAngleObservationConfig:
    enabled: bool = False
    curriculum_paths: list[str] = field(default_factory=list)
    normalization_deg: float = 180.0
    zero_offsets_deg: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # Sample one scalar per environment at reset and apply it only to the
    # positive-Hz CSV motor-position observation for the complete episode.
    # Left/right share the scalar; physical CSV wrench playback is unaffected.
    csv_amplitude_scale_range: list[float] = field(default_factory=lambda: [1.0, 1.0])
    standing_normalized_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # Per-frame uniform actor-observation noise, specified as a physical-angle
    # half range and normalized by ``normalization_deg`` before policy input.
    noise_half_range_deg: float = 0.0
    # Per-frame, per-motor multiplicative Gaussian noise. A value of 0.05 means
    # obs *= 1 + Normal(0, 0.05). This is dimensionless and independent from
    # the reset-owned CSV amplitude scale.
    gaussian_noise_relative_std: float = 0.0


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    # Optional forward-velocity width. None preserves the historical behavior
    # of sharing tracking_sigma with yaw tracking.
    track_lin_vel_x_std: float | None = None
    track_lin_vel_x_enhance_std: float = 0.8
    # Per-term weighted reward-rate limit for the two forward-velocity terms.
    # Keep the historical +/-1/s default; tasks using a larger scale can raise
    # this limit explicitly so that the configured scale is not clipped away.
    track_lin_vel_x_term_clip: float = 1.0
    base_height_std: float = 0.05
    base_height_clip: float = 4.0
    only_positive_rewards: bool = False
    undesired_contact_threshold: float = 0.1
    termination_contact_threshold: float = 0.1
    termination_contact_fail_steps: int = 1
    termination_gravity_z_threshold: float = 0.7
    termination_fail_time_s: float = 0.001


@dataclass
class JoystickSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"
    projected_gravity_x = "xvector"
    projected_gravity_y = "yvector"
    projected_gravity_z = "upvector"
    undesired_contacts: tuple[str, ...] = (
        "base_link_touch",
        "left_thigh_touch",
        "left_calf_touch",
        "right_thigh_touch",
        "right_calf_touch",
        "ancestor_dante_upper_left_touch",
        "ancestor_dante_upper_right_touch",
        "ancestor_dante_tail_rear_touch",
        "ancestor_dante_tail_mid_touch",
        "ancestor_dante_front_center_touch",
    )


@dataclass
class WE6JoystickSensor(JoystickSensor):
    """Contact sensors available in the streamlined WE6 model."""

    undesired_contacts: tuple[str, ...] = (
        "base_link_touch",
        "left_thigh_touch",
        "left_calf_touch",
        "right_thigh_touch",
        "right_calf_touch",
        "ancestor_dante_upper_left_touch",
        "ancestor_dante_upper_right_touch",
        "ancestor_dante_front_center_touch",
    )


@dataclass
class U9JoystickSensor(JoystickSensor):
    """Contact sensors exposed by the reviewed U9 model."""

    undesired_contacts: tuple[str, ...] = (
        "base_link_touch",
        "left_thigh_touch",
        "left_calf_touch",
        "right_thigh_touch",
        "right_calf_touch",
        "dandan_upper_left_touch",
        "dandan_upper_right_touch",
        "dandan_front_center_touch",
    )


@registry.envcfg("DR002JoystickFlat")
@dataclass
class DR002JoystickCfg(DR002BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(_DR002_ASSET_ROOT / "scene_flat_latest.xml")
        )
    )
    max_episode_seconds: float = 23.0
    commands: DR002Commands = field(default_factory=DR002Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)  # type: ignore[assignment]
    domain_rand: DR002DomainRandConfig = field(default_factory=DR002DomainRandConfig)
    wing_angle_obs: WingAngleObservationConfig = field(default_factory=WingAngleObservationConfig)
    critic_obs_mode: str = "legacy"


@dataclass
class WE6ControlConfig(ControlConfig):
    """WE6's 400 Hz training PD and pre-controller command FIFO."""

    motor_control_hz: float | None = 400.0
    action_delay_semantics: str = "pre_controller_command_fifo"
    torque_delay_steps: int = 0
    action_delay_min_steps: int = 6
    action_delay_max_steps: int = 14
    action_delay_steps_by_joint: list[int] | None = field(
        default_factory=lambda: [14, 6, 10, 14, 6, 10]
    )
    resample_action_delay: bool = False
    use_native_batched_pd: bool = True
    Kp: list[float] = field(  # noqa: N815
        default_factory=lambda: [3.75, 4.04, 0.0, 3.75, 4.04, 0.0]
    )
    Kd: list[float] = field(  # noqa: N815
        default_factory=lambda: [0.145, 0.2, 0.202, 0.145, 0.2, 0.202]
    )


@dataclass
class WE9ControlConfig(ControlConfig):
    """U9 PACE controller with a shared randomized command FIFO."""

    # Preserve the WE6 wheel action-to-torque gain with Kd reduced 4x to 0.05.
    wheel_action_scale: float = 10.0
    wheel_clip_actions: float = 2.5
    motor_control_hz: float | None = 200.0
    action_delay_semantics: str = "pre_controller_command_fifo"
    torque_delay_steps: int = 0
    action_delay_min_steps: int = 2
    action_delay_max_steps: int = 8
    action_delay_steps_by_joint: list[int] | None = None
    resample_action_delay: bool = True
    use_native_batched_pd: bool = False
    Kp: list[float] = field(  # noqa: N815
        default_factory=lambda: [4.11, 3.91, 0.0, 4.11, 3.91, 0.0]
    )
    Kd: list[float] = field(  # noqa: N815
        default_factory=lambda: [0.160, 0.193, 0.05, 0.160, 0.193, 0.05]
    )


@dataclass
class WE6NoiseConfig(NoiseConfig):
    """WE6 sensor noise with episode-fixed mounting error and slow IMU drift."""

    scale_gyro: float = 0.1
    scale_joint_angle: float = 0.001
    scale_joint_vel: float = 0.2
    scale_wheel_vel: float = 0.3
    scale_gravity: float = 0.0
    gravity_noise_mode: str = "tilt"
    gravity_installation_bias_max_deg: float = 5.0
    gravity_dynamic_noise_max_deg: float = 1.0
    gravity_dynamic_noise_time_constant_s: float = 3.0


@registry.envcfg("DR002JoystickFlatWE6")
@dataclass
class DR002JoystickFlatWE6Cfg(DR002JoystickCfg):
    """Walking Eagle 6 with Bode-PACE actuator dynamics."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(_DR002_ASSET_ROOT / "we6" / "scene_flat_we6.xml")
        )
    )
    sim_dt: float = 0.0025
    ctrl_dt: float = 0.02
    noise_config: NoiseConfig = field(default_factory=WE6NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=WE6ControlConfig)  # type: ignore[assignment]
    sensor: JoystickSensor = field(default_factory=WE6JoystickSensor)  # type: ignore[assignment]
    critic_obs_mode: str = "isaaclab"


@registry.envcfg("DR002JoystickFlatWE9")
@dataclass
class DR002JoystickFlatWE9Cfg(DR002JoystickFlatWE6Cfg):
    """Reviewed U9 morphology with its independently materialized PACE model."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(_DR002_ASSET_ROOT / "u9" / "scene_flat_u9_pace.xml")
        )
    )
    control_config: ControlConfig = field(  # type: ignore[assignment]
        default_factory=WE9ControlConfig
    )
    sensor: JoystickSensor = field(default_factory=U9JoystickSensor)  # type: ignore[assignment]


def _sample_dr002_commands(
    cfg: DR002Commands,
    num_samples: int,
    lin_vel_x_range: tuple[float, float] | None = None,
    ang_vel_z_range: tuple[float, float] | None = None,
    standing_mask: np.ndarray | None = None,
) -> np.ndarray:
    lin_vel_x = tuple(cfg.lin_vel_x) if lin_vel_x_range is None else lin_vel_x_range
    ang_vel_z = tuple(cfg.ang_vel_z) if ang_vel_z_range is None else ang_vel_z_range
    low = np.asarray([lin_vel_x[0], ang_vel_z[0], cfg.height[0]], dtype=get_global_dtype())
    high = np.asarray([lin_vel_x[1], ang_vel_z[1], cfg.height[1]], dtype=get_global_dtype())
    commands = np.random.uniform(low=low, high=high, size=(num_samples, 3)).astype(get_global_dtype())
    if standing_mask is not None:
        standing = np.asarray(standing_mask, dtype=np.bool_).reshape(-1)
        if standing.shape != (num_samples,):
            raise ValueError(
                f"standing_mask must have shape ({num_samples},), got {standing.shape}"
            )
        commands[standing, 0:2] = 0.0
    return commands


def _load_force_csv(path: str) -> np.ndarray:
    samples: list[list[float]] = []
    with open(path, "r", newline="", encoding="latin1") as file:
        for row in csv.reader(file):
            try:
                values = [float(value) for value in row[:7]]
            except (TypeError, ValueError):
                continue
            if len(values) >= 4:
                # Preserve compatibility with legacy force-only files while
                # making the six-axis measured-wrench contract explicit.
                samples.append(values[:7] + [0.0] * max(7 - len(values), 0))
    if not samples:
        raise ValueError(f"force CSV has no numeric rows: {path}")
    force = np.asarray(samples, dtype=np.float64)
    order = np.argsort(force[:, 0])
    force = force[order]
    if not np.all(np.isfinite(force)):
        raise ValueError(f"force CSV contains non-finite values: {path}")
    if force.shape[0] < 2 or np.any(np.diff(force[:, 0]) <= 0.0):
        raise ValueError(f"force CSV timestamps must be strictly increasing: {path}")
    return force


def _interp_force_csv(samples: np.ndarray, replay_t: float) -> np.ndarray | None:
    if samples.size == 0 or replay_t < float(samples[0, 0]) or replay_t > float(samples[-1, 0]):
        return None
    times = samples[:, 0]
    upper = int(np.searchsorted(times, replay_t, side="left"))
    if upper <= 0:
        return samples[0, 1:4].copy()
    if upper >= samples.shape[0]:
        return samples[-1, 1:4].copy()
    a = samples[upper - 1]
    b = samples[upper]
    denom = float(b[0] - a[0])
    alpha = (float(replay_t) - float(a[0])) / denom if denom > 0.0 else 0.0
    return np.asarray(a[1:4] + alpha * (b[1:4] - a[1:4]), dtype=np.float64)


def _interp_force_csv_batch(samples: np.ndarray, replay_t: np.ndarray) -> np.ndarray:
    replay = np.asarray(replay_t, dtype=np.float64).reshape(-1)
    force = np.zeros((replay.shape[0], 3), dtype=np.float64)
    if samples.size == 0:
        return force
    times = samples[:, 0]
    if times.size == 0:
        return force
    for axis in range(3):
        force[:, axis] = np.interp(replay, times, samples[:, axis + 1], left=0.0, right=0.0)
    return force


def _interp_wrench_csv_batch(samples: np.ndarray, replay_t: np.ndarray) -> np.ndarray:
    replay = np.asarray(replay_t, dtype=np.float64).reshape(-1)
    wrench = np.zeros((replay.shape[0], 6), dtype=np.float64)
    if samples.size == 0:
        return wrench
    times = samples[:, 0]
    if times.size == 0:
        return wrench
    for axis in range(6):
        wrench[:, axis] = np.interp(
            replay,
            times,
            samples[:, axis + 1],
            left=0.0,
            right=0.0,
        )
    return wrench


def _csv_replay_clock(
    domain_rand: DR002DomainRandConfig,
    episode_steps: np.ndarray,
    startup_steps: int,
    ctrl_dt: float,
    sample_offsets_s: np.ndarray | None = None,
    start_delay_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    elapsed_after_startup = np.asarray(episode_steps, dtype=np.int64) - int(startup_steps)
    if start_delay_steps is not None:
        delays = np.asarray(start_delay_steps, dtype=np.int64).reshape(-1)
        if delays.shape != elapsed_after_startup.shape:
            raise ValueError(
                "start_delay_steps must match the episode_steps shape, "
                f"got {delays.shape} and {elapsed_after_startup.shape}"
            )
        elapsed_after_startup = elapsed_after_startup - delays
    elapsed_s = np.maximum(elapsed_after_startup, 0).astype(np.float64) * float(ctrl_dt)
    if sample_offsets_s is None:
        replay_t = elapsed_s
    else:
        offsets = np.asarray(sample_offsets_s, dtype=np.float64).reshape(-1)
        replay_t = elapsed_s[:, None] + offsets[None, :]
    period = float(domain_rand.csv_force_period)
    if period <= 0.0:
        return elapsed_after_startup, replay_t

    period_steps = int(round(period / float(ctrl_dt)))
    aligned_period = abs(period_steps * float(ctrl_dt) - period)
    if sample_offsets_s is None and period_steps > 0 and aligned_period < float(ctrl_dt) * 0.25:
        replay_steps = np.mod(np.maximum(elapsed_after_startup, 0), period_steps)
        replay_t = replay_steps.astype(np.float64) * float(ctrl_dt)
    else:
        replay_t = np.fmod(replay_t, period)
        replay_t = np.where(replay_t < 0.0, replay_t + period, replay_t)
    return elapsed_after_startup, replay_t


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _csv_replay_activity_weight(
    samples: np.ndarray,
    replay_t: np.ndarray,
    elapsed_s: np.ndarray,
    *,
    period: float,
    transition_seconds: float,
) -> np.ndarray:
    replay = np.asarray(replay_t, dtype=np.float64)
    elapsed = np.broadcast_to(np.asarray(elapsed_s, dtype=np.float64), replay.shape)
    source_start = float(samples[0, 0])
    source_end = float(samples[-1, 0])
    active_end = min(source_end, period) if period > 0.0 else source_end
    active = (replay >= source_start) & (replay <= active_end)
    weight = active.astype(np.float64)
    transition = float(transition_seconds)
    if transition <= 0.0:
        return weight

    covers_period = period > 0.0 and source_end >= period
    if covers_period:
        initial_weight = _smoothstep01((elapsed - source_start) / transition)
        return weight * initial_weight

    fade_in = _smoothstep01((replay - source_start) / transition)
    fade_out = _smoothstep01((active_end - replay) / transition)
    return weight * np.minimum(fade_in, fade_out)


def _csv_replay_has_closed_seam(
    samples: np.ndarray,
    *,
    period: float,
    num_channels: int,
) -> bool:
    """Return whether the replay value at ``period`` exactly closes to its first row."""
    return _csv_replay_channels_have_closed_seam(
        samples,
        period=period,
        channel_columns=tuple(range(1, num_channels + 1)),
    )


def _csv_replay_channels_have_closed_seam(
    samples: np.ndarray,
    *,
    period: float,
    channel_columns: tuple[int, ...],
) -> bool:
    """Return whether selected CSV channels close across the replay period."""
    values = np.asarray(samples, dtype=np.float64)
    replay_period = float(period)
    if (
        replay_period <= 0.0
        or not channel_columns
        or values.ndim != 2
        or values.shape[0] == 0
        or min(channel_columns) < 1
        or max(channel_columns) >= values.shape[1]
        or not np.all(np.isfinite(values[:, (0, *channel_columns)]))
    ):
        return False

    times = values[:, 0]
    seam_atol = 1.0e-9
    if times[0] != 0.0 or times[-1] < replay_period:
        return False

    seam_value = np.asarray(
        [np.interp(replay_period, times, values[:, channel]) for channel in channel_columns],
        dtype=np.float64,
    )
    return bool(
        np.allclose(
            seam_value,
            values[0, channel_columns],
            rtol=0.0,
            atol=seam_atol,
        )
    )


def _csv_wrench_effective_seam_columns(
    domain_rand: DR002DomainRandConfig,
) -> tuple[int, ...]:
    force_columns = (1, 3) if domain_rand.csv_force_zero_fy else (1, 2, 3)
    moment_is_used = bool(
        domain_rand.csv_force_apply_measured_moment
        or domain_rand.csv_force_observation_include_measured_moment
    )
    moment_columns = (4, 5, 6) if moment_is_used else ()
    return force_columns + moment_columns


def _sample_force_csv_replay(
    samples: np.ndarray,
    replay_t: np.ndarray,
    elapsed_s: np.ndarray,
    *,
    period: float,
    transition_seconds: float,
    blend_period_tail: bool | None = None,
) -> np.ndarray:
    replay = np.asarray(replay_t, dtype=np.float64)
    flat_force = _interp_force_csv_batch(samples, replay.reshape(-1))
    force = flat_force.reshape(replay.shape + (3,))
    weight = _csv_replay_activity_weight(
        samples,
        replay,
        elapsed_s,
        period=period,
        transition_seconds=transition_seconds,
    )
    force *= weight[..., None]

    if blend_period_tail is None:
        blend_period_tail = not _csv_replay_has_closed_seam(
            samples,
            period=period,
            num_channels=3,
        )
    if not blend_period_tail:
        return force

    source_end = float(samples[-1, 0])
    transition = float(transition_seconds)
    if period <= 0.0 or source_end < period or transition <= 0.0:
        return force

    source_start = float(samples[0, 0])
    tail_start = max(source_start, period - transition)
    tail_width = period - tail_start
    if tail_width <= 0.0:
        return force
    tail = replay >= tail_start
    if not np.any(tail):
        return force
    alpha = _smoothstep01((replay[tail] - tail_start) / tail_width)
    first_force = np.asarray(samples[0, 1:4], dtype=np.float64)
    force[tail] = (1.0 - alpha[:, None]) * force[tail] + alpha[:, None] * first_force
    return force


def _apply_csv_force_channel_options(
    force_sensor: np.ndarray,
    *,
    zero_fy: bool,
) -> np.ndarray:
    force = np.asarray(force_sensor, dtype=np.float64)
    if force.shape[-1] != 3:
        raise ValueError(f"CSV force must have three channels, got shape {force.shape}")
    result = force.copy()
    if zero_fy:
        result[..., 1] = 0.0
    return result


def _transform_csv_wrench_to_base_com(
    wrench_sensor: np.ndarray,
    *,
    sensor_to_base_rotation: np.ndarray,
    moment_arm_base: np.ndarray,
    zero_fy: bool,
    apply_measured_moment: bool,
    apply_point_torque: bool,
) -> np.ndarray:
    """Express a sensor-origin wrench about the target body's COM in base axes."""
    wrench = np.asarray(wrench_sensor, dtype=np.float64)
    if wrench.shape[-1] != 6:
        raise ValueError(f"CSV wrench must have six channels, got shape {wrench.shape}")
    rotation = np.asarray(sensor_to_base_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"sensor-to-base rotation must have shape (3, 3), got {rotation.shape}")
    moment_arm = np.asarray(moment_arm_base, dtype=np.float64)
    if moment_arm.shape[-1] != 3:
        raise ValueError(f"base-frame moment arm must have three channels, got {moment_arm.shape}")

    force_sensor = _apply_csv_force_channel_options(wrench[..., :3], zero_fy=zero_fy)
    force_base = force_sensor @ rotation.T
    try:
        leading_shape = np.broadcast_shapes(force_base.shape[:-1], moment_arm.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "CSV wrench and base-frame moment arm leading dimensions are not broadcastable: "
            f"{wrench.shape[:-1]} and {moment_arm.shape[:-1]}"
        ) from exc
    force_base = np.broadcast_to(force_base, (*leading_shape, 3))
    moment_arm = np.broadcast_to(moment_arm, (*leading_shape, 3))

    moment_base = np.zeros((*leading_shape, 3), dtype=np.float64)
    if apply_measured_moment:
        # A proper rotation maps both polar force vectors and axial moment
        # vectors with the same matrix. Mx/My/Mz remain referenced at the
        # physical sensor origin, which coincides with csv_force_push_point.
        measured_moment_base = wrench[..., 3:6] @ rotation.T
        moment_base += np.broadcast_to(measured_moment_base, (*leading_shape, 3))
    if apply_point_torque:
        # MuJoCo xfrc_applied is a COM-based body wrench, so transport the
        # push-site force to the (possibly randomized) body COM exactly once.
        moment_base += np.cross(moment_arm, force_base)
    return np.concatenate([force_base, moment_base], axis=-1)


def _sample_wrench_csv_replay(
    samples: np.ndarray,
    replay_t: np.ndarray,
    elapsed_s: np.ndarray,
    *,
    period: float,
    transition_seconds: float,
    blend_period_tail: bool | None = None,
) -> np.ndarray:
    replay = np.asarray(replay_t, dtype=np.float64)
    flat_wrench = _interp_wrench_csv_batch(samples, replay.reshape(-1))
    wrench = flat_wrench.reshape(replay.shape + (6,))
    weight = _csv_replay_activity_weight(
        samples,
        replay,
        elapsed_s,
        period=period,
        transition_seconds=transition_seconds,
    )
    wrench *= weight[..., None]

    if blend_period_tail is None:
        blend_period_tail = not _csv_replay_has_closed_seam(
            samples,
            period=period,
            num_channels=6,
        )
    if not blend_period_tail:
        return wrench

    source_end = float(samples[-1, 0])
    transition = float(transition_seconds)
    if period <= 0.0 or source_end < period or transition <= 0.0:
        return wrench
    source_start = float(samples[0, 0])
    tail_start = max(source_start, period - transition)
    tail_width = period - tail_start
    if tail_width <= 0.0:
        return wrench
    tail = replay >= tail_start
    if not np.any(tail):
        return wrench
    alpha = _smoothstep01((replay[tail] - tail_start) / tail_width)
    first_wrench = np.asarray(samples[0, 1:7], dtype=np.float64)
    wrench[tail] = (1.0 - alpha[:, None]) * wrench[tail] + alpha[:, None] * first_wrench
    return wrench


def _load_wing_angle_csv(path: str | Path) -> np.ndarray:
    samples: list[list[float]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            try:
                values = [float(value) for value in row[:3]]
            except (TypeError, ValueError):
                continue
            if len(values) >= 3:
                samples.append(values[:3])
    if not samples:
        raise ValueError(f"wing angle CSV has no numeric rows: {path}")
    angles = np.asarray(samples, dtype=np.float64)
    order = np.argsort(angles[:, 0])
    angles = angles[order]
    if not np.all(np.isfinite(angles)):
        raise ValueError(f"wing angle CSV contains non-finite values: {path}")
    if angles.shape[0] < 2 or np.any(np.diff(angles[:, 0]) <= 0.0):
        raise ValueError(f"wing angle CSV timestamps must be strictly increasing: {path}")
    return angles


def _interp_wing_angle_csv_batch(samples: np.ndarray, phase_t: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_t, dtype=np.float64).reshape(-1)
    angles = np.zeros((phase.shape[0], _WING_ANGLE_OBS_DIM), dtype=np.float64)
    for motor_index in range(_WING_ANGLE_OBS_DIM):
        angles[:, motor_index] = np.interp(
            phase,
            samples[:, 0],
            samples[:, motor_index + 1],
        )
    return angles


def _wing_angle_curriculum_paths(cfg: WingAngleObservationConfig) -> list[Path]:
    paths: list[Path] = []
    for configured_path in cfg.curriculum_paths:
        path = Path(configured_path).expanduser()
        paths.append(path if path.is_absolute() else ASSETS_ROOT_PATH / path)
    return paths


def _wing_angle_path_for_level(
    domain_rand: DR002DomainRandConfig,
    cfg: WingAngleObservationConfig,
    level: int,
) -> Path | None:
    paths = _wing_angle_curriculum_paths(cfg)
    hz_values = _csv_force_curriculum_hz_values(domain_rand)
    if not hz_values:
        return paths[int(np.clip(level, 0, len(paths) - 1))]

    level = int(np.clip(level, 0, len(hz_values) - 1))
    if hz_values[level] <= 0.0:
        return None
    if len(paths) == len(hz_values):
        return paths[level]

    positive_path_index = sum(1 for hz in hz_values[: level + 1] if hz > 0.0) - 1
    if positive_path_index < 0:
        return None
    if positive_path_index >= len(paths):
        raise ValueError(
            "wing_angle_obs.curriculum_paths does not contain enough positive-Hz paths"
        )
    return paths[positive_path_index]


def _wing_angle_obs_csv_amplitude_scale_bounds(
    cfg: WingAngleObservationConfig,
) -> tuple[float, float]:
    scale_range = np.asarray(cfg.csv_amplitude_scale_range, dtype=np.float64)
    if scale_range.shape != (2,):
        raise ValueError("wing_angle_obs.csv_amplitude_scale_range must contain [min, max]")
    if not np.all(np.isfinite(scale_range)):
        raise ValueError("wing_angle_obs.csv_amplitude_scale_range must contain finite values")
    low, high = float(scale_range[0]), float(scale_range[1])
    if low < 0.0 or high < low:
        raise ValueError("wing_angle_obs.csv_amplitude_scale_range must satisfy 0 <= min <= max")
    return low, high


def _validate_wing_angle_observation_mapping(
    domain_rand: DR002DomainRandConfig,
    cfg: WingAngleObservationConfig,
) -> None:
    if not cfg.enabled:
        return
    if not domain_rand.csv_force_enabled:
        raise ValueError("wing_angle_obs.enabled requires domain_rand.csv_force_enabled")
    if float(cfg.normalization_deg) <= 0.0:
        raise ValueError("wing_angle_obs.normalization_deg must be positive")
    _wing_angle_obs_csv_amplitude_scale_bounds(cfg)
    noise_half_range_deg = float(cfg.noise_half_range_deg)
    if not np.isfinite(noise_half_range_deg) or noise_half_range_deg < 0.0:
        raise ValueError("wing_angle_obs.noise_half_range_deg must be finite and non-negative")
    gaussian_noise_relative_std = float(cfg.gaussian_noise_relative_std)
    if not np.isfinite(gaussian_noise_relative_std) or gaussian_noise_relative_std < 0.0:
        raise ValueError(
            "wing_angle_obs.gaussian_noise_relative_std must be finite and non-negative"
        )
    zero_offsets = np.asarray(cfg.zero_offsets_deg, dtype=np.float64)
    if zero_offsets.shape != (_WING_ANGLE_OBS_DIM,) or not np.all(np.isfinite(zero_offsets)):
        raise ValueError("wing_angle_obs.zero_offsets_deg must contain two finite values")
    standing_range = np.asarray(cfg.standing_normalized_range, dtype=np.float64)
    if standing_range.shape != (2,) or not np.all(np.isfinite(standing_range)):
        raise ValueError(
            "wing_angle_obs.standing_normalized_range must contain two finite values"
        )
    if standing_range[0] > standing_range[1]:
        raise ValueError(
            "wing_angle_obs.standing_normalized_range must be ordered [low, high]"
        )

    hz_values = _csv_force_curriculum_hz_values(domain_rand)
    if not hz_values:
        raise ValueError("wing angle observations require csv_force_curriculum_hz labels")
    paths = _wing_angle_curriculum_paths(cfg)
    positive_count = sum(1 for hz in hz_values if hz > 0.0)
    if len(paths) not in (len(hz_values), positive_count):
        raise ValueError(
            "wing_angle_obs.curriculum_paths must either match csv_force_curriculum_hz "
            "length or contain only the positive-Hz paths"
        )
    for level, source_hz in enumerate(hz_values):
        if source_hz <= 0.0:
            continue
        path = _wing_angle_path_for_level(domain_rand, cfg, level)
        if path is None:
            raise ValueError(f"wing angle path is missing for {source_hz:g}Hz")
        samples = _load_wing_angle_csv(path)
        expected_period = 1.0 / source_hz
        tolerance = max(1.0e-6, expected_period * 1.0e-3)
        if samples[0, 0] > tolerance or samples[-1, 0] < expected_period - tolerance:
            raise ValueError(
                f"wing angle CSV must span one full {source_hz:g}Hz period "
                f"[0, {expected_period:g}]s: {path}"
            )


def _csv_force_curriculum_paths(domain_rand: DR002DomainRandConfig) -> list[str]:
    configured_paths = (
        list(domain_rand.csv_force_curriculum_paths)
        if domain_rand.csv_force_curriculum_paths
        else [domain_rand.csv_force_path]
    )
    paths: list[str] = []
    for configured_path in configured_paths:
        path = Path(configured_path).expanduser()
        paths.append(str(path if path.is_absolute() else ASSETS_ROOT_PATH / path))
    return paths


def _csv_force_curriculum_hz_values(domain_rand: DR002DomainRandConfig) -> list[float]:
    return [float(hz) for hz in domain_rand.csv_force_curriculum_hz]


def _csv_force_curriculum_num_levels(domain_rand: DR002DomainRandConfig) -> int:
    hz_values = _csv_force_curriculum_hz_values(domain_rand)
    if hz_values:
        return len(hz_values)
    return len(_csv_force_curriculum_paths(domain_rand))


def _csv_force_curriculum_level_index(
    domain_rand: DR002DomainRandConfig,
    current_level: int | float,
) -> int:
    num_levels = _csv_force_curriculum_num_levels(domain_rand)
    if num_levels <= 1:
        return 0
    level = int(round(float(current_level)))
    return int(np.clip(level, 0, num_levels - 1))


def _csv_force_curriculum_source_hz(domain_rand: DR002DomainRandConfig, level: int) -> float:
    hz_values = _csv_force_curriculum_hz_values(domain_rand)
    if not hz_values:
        return float("nan")
    return float(hz_values[int(np.clip(level, 0, len(hz_values) - 1))])


def _csv_force_zero_hz_mask(
    domain_rand: DR002DomainRandConfig,
    levels: np.ndarray,
) -> np.ndarray:
    level_values = np.asarray(levels, dtype=np.int32)
    zero_hz = np.zeros(level_values.shape, dtype=np.bool_)
    for level in np.unique(level_values):
        source_hz = _csv_force_curriculum_source_hz(domain_rand, int(level))
        if np.isfinite(source_hz) and source_hz <= 0.0:
            zero_hz[level_values == int(level)] = True
    return zero_hz


def _csv_force_curriculum_path_for_level(
    domain_rand: DR002DomainRandConfig,
    level: int,
) -> str | None:
    paths = _csv_force_curriculum_paths(domain_rand)
    hz_values = _csv_force_curriculum_hz_values(domain_rand)
    if not hz_values:
        return paths[int(np.clip(level, 0, len(paths) - 1))]

    level = int(np.clip(level, 0, len(hz_values) - 1))
    if hz_values[level] <= 0.0:
        return None

    if len(paths) == len(hz_values):
        return paths[level]

    positive_path_index = sum(1 for hz in hz_values[: level + 1] if hz > 0.0) - 1
    if positive_path_index < 0:
        return None
    if positive_path_index >= len(paths):
        raise ValueError(
            "domain_rand.csv_force_curriculum_paths does not contain enough positive-Hz CSV paths"
        )
    return paths[positive_path_index]


def _validate_csv_force_curriculum_mapping(domain_rand: DR002DomainRandConfig) -> None:
    hz_values = _csv_force_curriculum_hz_values(domain_rand)
    if not hz_values:
        return
    paths = _csv_force_curriculum_paths(domain_rand)
    positive_count = sum(1 for hz in hz_values if hz > 0.0)
    if len(paths) not in (len(hz_values), positive_count):
        raise ValueError(
            "domain_rand.csv_force_curriculum_paths must either match "
            "csv_force_curriculum_hz length, or contain only the positive-Hz CSV paths"
        )


def _csv_force_curriculum_level_paths(
    domain_rand: DR002DomainRandConfig,
) -> list[str | None]:
    return [
        _csv_force_curriculum_path_for_level(domain_rand, level)
        for level in range(_csv_force_curriculum_num_levels(domain_rand))
    ]


def _checkpoint_stable_asset_path(path: str | None) -> str | None:
    if path is None:
        return None
    expanded = Path(path).expanduser()
    try:
        relative = expanded.relative_to(ASSETS_ROOT_PATH)
    except ValueError:
        return str(expanded)
    return f"asset://{relative.as_posix()}"


def _csv_force_curriculum_checkpoint_paths(
    domain_rand: DR002DomainRandConfig,
) -> tuple[list[str], list[str | None]]:
    paths = [
        str(_checkpoint_stable_asset_path(path))
        for path in _csv_force_curriculum_paths(domain_rand)
    ]
    level_paths = [
        _checkpoint_stable_asset_path(path)
        for path in _csv_force_curriculum_level_paths(domain_rand)
    ]
    return paths, level_paths


def _noise_curriculum_levels(noise_cfg: Any) -> np.ndarray:
    levels = np.asarray(
        getattr(noise_cfg, "curriculum_levels", []),
        dtype=np.float64,
    ).reshape(-1)
    if levels.size == 0:
        raise ValueError("noise_config.curriculum_levels must be non-empty")
    if not np.all(np.isfinite(levels)):
        raise ValueError("noise_config.curriculum_levels must be finite")
    if np.any(levels < 0.0):
        raise ValueError("noise_config.curriculum_levels must be non-negative")
    return levels


def _sample_dr002_body_mass_multipliers(
    env: Any,
    num_samples: int,
    body_mass_template: np.ndarray,
) -> np.ndarray:
    domain_rand = env.cfg.domain_rand
    template = np.asarray(body_mass_template, dtype=np.float64)
    if template.ndim != 1:
        raise ValueError(
            "body mass randomization requires body mass shape (nbody,), "
            f"got {template.shape}"
        )
    bounds = np.asarray(domain_rand.body_mass_multiplier_range, dtype=np.float64)
    if bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or bounds[1] < bounds[0]:
        raise ValueError(
            "body_mass_multiplier_range must contain finite [low, high] bounds"
        )
    multipliers = np.random.uniform(
        float(bounds[0]),
        float(bounds[1]),
        size=(num_samples, template.size),
    )
    base_body_id = int(env._backend.get_body_id(env.cfg.asset.base_name))
    if not 0 <= base_body_id < template.size:
        raise ValueError(f"base body id is outside the cached body mass table: {base_body_id}")
    multipliers[:, base_body_id] = 1.0

    if bool(getattr(domain_rand, "body_mass_share_bilateral_scale", True)):
        for suffix in ("thigh_joint", "calf_joint", "foot_joint"):
            try:
                left_id = int(env._backend.get_body_id(f"left_{suffix}"))
                right_id = int(env._backend.get_body_id(f"right_{suffix}"))
            except (KeyError, ValueError):
                continue
            if not 0 <= left_id < template.size or not 0 <= right_id < template.size:
                raise ValueError(
                    f"bilateral body ids for {suffix} are outside the cached body mass table"
                )
            pair_scale = np.random.uniform(
                float(bounds[0]),
                float(bounds[1]),
                size=(num_samples,),
            )
            multipliers[:, left_id] = pair_scale
            multipliers[:, right_id] = pair_scale
    return multipliers


def build_dr002_backend_reset_randomization(
    env: Any,
    num_reset: int,
    *,
    base_body_mass: np.ndarray | None = None,
    base_body_inertia: np.ndarray | None = None,
    base_geom_friction: np.ndarray | None = None,
    ground_geom_id: int | None = None,
    robot_geom_ids: np.ndarray | None = None,
    base_dof_armature: np.ndarray | None = None,
    body_mass_multipliers: np.ndarray | None = None,
) -> ResetRandomizationPayload | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()
    body_inertia = None
    body_inertia_template = None
    coupled_base_body_id: int | None = None
    coupled_base_inertia_scale: np.ndarray | None = None
    couple_base_inertia = bool(getattr(domain_rand, "couple_base_inertia_to_added_mass", False))
    if couple_base_inertia and not getattr(domain_rand, "randomize_base_mass", False):
        raise ValueError("couple_base_inertia_to_added_mass requires randomize_base_mass=True")
    if getattr(domain_rand, "randomize_base_mass", False):
        if couple_base_inertia:
            added_mass_range = np.asarray(domain_rand.added_mass_range, dtype=np.float64)
            if added_mass_range.shape != (2,):
                raise ValueError(
                    f"coupled added_mass_range must have shape (2,), got {added_mass_range.shape}"
                )
            low, high = (float(value) for value in added_mass_range)
            if np.any(~np.isfinite(added_mass_range)) or high < low:
                raise ValueError("coupled added_mass_range must contain finite [low, high] bounds")
            if base_body_mass is None:
                raise ValueError("base mass-inertia coupling requires cached body mass")
            if base_body_inertia is None:
                raise ValueError("base mass-inertia coupling requires cached body inertia")
            body_mass_template = np.asarray(base_body_mass, dtype=np.float64)
            body_inertia_template = np.asarray(base_body_inertia, dtype=np.float64)
            if body_mass_template.ndim != 1:
                raise ValueError(
                    "base mass-inertia coupling requires body mass shape (nbody,), "
                    f"got {body_mass_template.shape}"
                )
            if body_inertia_template.shape != (body_mass_template.size, 3):
                raise ValueError(
                    "base mass-inertia coupling requires principal inertia shape "
                    f"({body_mass_template.size}, 3), got {body_inertia_template.shape}"
                )
            coupled_base_body_id = int(env._backend.get_body_id(env.cfg.asset.base_name))
            if not 0 <= coupled_base_body_id < body_mass_template.size:
                raise ValueError(
                    f"base body id is outside the cached body mass table: {coupled_base_body_id}"
                )
            nominal_base_mass = float(body_mass_template[coupled_base_body_id])
            if not np.isfinite(nominal_base_mass) or nominal_base_mass <= 0.0:
                raise ValueError(
                    "base mass-inertia coupling requires a finite positive nominal base mass"
                )
            nominal_base_inertia = body_inertia_template[coupled_base_body_id]
            if np.any(~np.isfinite(nominal_base_inertia)) or np.any(nominal_base_inertia <= 0.0):
                raise ValueError(
                    "base mass-inertia coupling requires three finite positive "
                    "base principal inertias"
                )
            if nominal_base_mass + float(low) <= 0.0:
                raise ValueError(
                    "coupled added_mass_range must keep the base mass positive "
                    "for every possible draw"
                )
        else:
            low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))
        if couple_base_inertia:
            assert body_mass_template is not None
            assert coupled_base_body_id is not None
            nominal_base_mass = float(body_mass_template[coupled_base_body_id])
            effective_base_mass = nominal_base_mass + np.asarray(
                payload.base_mass_delta, dtype=np.float64
            )
            if np.any(~np.isfinite(effective_base_mass)) or np.any(effective_base_mass <= 0.0):
                raise ValueError(
                    "added_mass_range must keep the coupled effective base mass positive"
                )
            coupled_base_inertia_scale = effective_base_mass / nominal_base_mass
    if getattr(domain_rand, "randomize_body_mass", False):
        if base_body_mass is None:
            raise ValueError("body mass randomization requires cached body mass")
        if base_body_inertia is None:
            raise ValueError("body mass inertia recompute requires cached body inertia")
        body_mass_template = np.asarray(base_body_mass, dtype=np.float64)
        body_inertia_template = np.asarray(base_body_inertia, dtype=np.float64)
        if body_mass_multipliers is None:
            multipliers = _sample_dr002_body_mass_multipliers(
                env,
                num_reset,
                body_mass_template,
            )
        else:
            multipliers = np.asarray(body_mass_multipliers, dtype=np.float64)
            expected_shape = (num_reset, body_mass_template.size)
            if multipliers.shape != expected_shape:
                raise ValueError(
                    "body_mass_multipliers must have shape "
                    f"{expected_shape}, got {multipliers.shape}"
                )
            if np.any(~np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
                raise ValueError("body_mass_multipliers must contain finite positive values")
            base_body_id = int(env._backend.get_body_id(env.cfg.asset.base_name))
            if not np.allclose(multipliers[:, base_body_id], 1.0, rtol=0.0, atol=0.0):
                raise ValueError("body_mass_multipliers must leave the base body unchanged")
        body_mass = np.broadcast_to(body_mass_template, multipliers.shape).copy()
        randomized = body_mass_template > 0.0
        body_mass[:, randomized] *= multipliers[:, randomized]
        payload.body_mass = body_mass
        body_inertia = np.broadcast_to(
            body_inertia_template, (num_reset, *body_inertia_template.shape)
        ).copy()
        body_inertia[:, randomized, :] *= multipliers[:, randomized, None]
    if coupled_base_inertia_scale is not None:
        assert coupled_base_body_id is not None
        assert body_inertia_template is not None
        if body_inertia is None:
            body_inertia = np.broadcast_to(
                body_inertia_template, (num_reset, *body_inertia_template.shape)
            ).copy()
        body_inertia[:, coupled_base_body_id, :] *= coupled_base_inertia_scale[:, None]
    if getattr(domain_rand, "randomize_body_inertia", False):
        if body_inertia_template is None:
            if base_body_inertia is None:
                raise ValueError("body inertia randomization requires cached body inertia")
            body_inertia_template = np.asarray(base_body_inertia, dtype=np.float64)
        if body_inertia is None:
            body_inertia = np.broadcast_to(
                body_inertia_template, (num_reset, *body_inertia_template.shape)
            ).copy()
        low, high = domain_rand.body_inertia_multiplier_range
        base_body_id = env._backend.get_body_id(env.cfg.asset.base_name)
        scale = np.random.uniform(low, high, size=(num_reset, 1))
        body_inertia[:, int(base_body_id), :] *= scale
    if body_inertia is not None:
        payload.body_inertia = body_inertia
    if getattr(domain_rand, "random_com", False):
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        low, high = domain_rand.com_offset_x
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        low, high = domain_rand.com_offset_y
        base_com_offset[:, 1] = np.random.uniform(low, high, size=(num_reset,))
        low, high = domain_rand.com_offset_z
        base_com_offset[:, 2] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset
    if getattr(domain_rand, "randomize_gravity", False):
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        low = np.minimum(gravity_range[0], gravity_range[1])
        high = np.maximum(gravity_range[0], gravity_range[1])
        payload.gravity = np.random.uniform(low=low, high=high, size=(num_reset, 3))
    geom_friction = None
    geom_friction_template = None
    if getattr(domain_rand, "randomize_ground_friction", False) or getattr(
        domain_rand, "randomize_robot_geom_friction", False
    ):
        if base_geom_friction is None:
            raise ValueError("geom friction randomization requires cached geom friction")
        geom_friction_template = np.asarray(base_geom_friction, dtype=np.float64)
        geom_friction = np.broadcast_to(
            geom_friction_template, (num_reset, *geom_friction_template.shape)
        ).copy()
    if getattr(domain_rand, "randomize_ground_friction", False):
        if ground_geom_id is None:
            raise ValueError("ground friction randomization requires ground geom id")
        assert geom_friction is not None
        low, high = domain_rand.ground_friction_multiplier_range
        geom_friction[:, int(ground_geom_id), 0] *= np.random.uniform(low, high, size=(num_reset,))
    if getattr(domain_rand, "randomize_robot_geom_friction", False):
        if robot_geom_ids is None:
            raise ValueError("robot geom friction randomization requires robot geom ids")
        robot_geom_ids_np = np.asarray(robot_geom_ids, dtype=np.int32).reshape(-1)
        if robot_geom_ids_np.size == 0:
            raise ValueError("robot geom friction randomization requires at least one robot geom")
        assert geom_friction is not None
        low, high = domain_rand.robot_geom_friction_multiplier_range
        scale = np.random.uniform(low, high, size=(num_reset, 1, 1))
        geom_friction[:, robot_geom_ids_np, :] *= scale
    if geom_friction is not None:
        payload.geom_friction = geom_friction
    if getattr(domain_rand, "randomize_dof_armature", False):
        if base_dof_armature is None:
            raise ValueError("dof armature randomization requires cached dof armature")
        dof_armature_template = np.asarray(base_dof_armature, dtype=np.float64)
        low, high = domain_rand.dof_armature_multiplier_range
        dof_armature = np.broadcast_to(
            dof_armature_template, (num_reset, dof_armature_template.size)
        ).copy()
        randomized = dof_armature_template > 0.0
        dof_armature[:, randomized] *= np.random.uniform(
            low, high, size=(num_reset, int(np.count_nonzero(randomized)))
        )
        payload.dof_armature = dof_armature
    return None if payload.is_empty() else payload


class DR002JoystickDomainRandomizationProvider(LocomotionDRProvider):
    def __init__(
        self,
        *,
        base_body_mass: np.ndarray | None = None,
        base_body_inertia: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
        robot_geom_ids: np.ndarray | None = None,
        base_dof_armature: np.ndarray | None = None,
    ) -> None:
        self._base_body_mass = base_body_mass
        self._base_body_inertia = base_body_inertia
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id
        self._robot_geom_ids = robot_geom_ids
        self._base_dof_armature = base_dof_armature
        self._force_samples_by_path: dict[str, np.ndarray] = {}
        self._force_blend_period_tail_by_key: dict[tuple[str, float, tuple[int, ...]], bool] = {}
        self._push_next_elapsed_steps: np.ndarray | None = None
        self._startup_body_mass_multipliers: np.ndarray | None = None

    def _body_mass_multipliers_for_reset(
        self,
        env: Any,
        env_ids: np.ndarray,
    ) -> np.ndarray | None:
        domain_rand = env.cfg.domain_rand
        if not bool(getattr(domain_rand, "randomize_body_mass", False)):
            return None
        if self._base_body_mass is None:
            raise ValueError("body mass randomization requires cached body mass")

        rows = np.asarray(env_ids, dtype=np.int32).reshape(-1)
        if bool(getattr(domain_rand, "body_mass_resample_on_reset", True)):
            return _sample_dr002_body_mass_multipliers(
                env,
                rows.size,
                self._base_body_mass,
            )

        expected_shape = (int(env._num_envs), np.asarray(self._base_body_mass).size)
        if self._startup_body_mass_multipliers is None:
            self._startup_body_mass_multipliers = _sample_dr002_body_mass_multipliers(
                env,
                int(env._num_envs),
                self._base_body_mass,
            )
        elif self._startup_body_mass_multipliers.shape != expected_shape:
            raise ValueError(
                "cached startup body mass multipliers have shape "
                f"{self._startup_body_mass_multipliers.shape}, expected {expected_shape}"
            )
        if np.any(rows < 0) or np.any(rows >= int(env._num_envs)):
            raise ValueError("reset environment ids are outside the startup multiplier table")
        return self._startup_body_mass_multipliers[rows].copy()

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        payload = build_dr002_backend_reset_randomization(
            env,
            num_reset=1,
            base_body_mass=self._base_body_mass,
            base_body_inertia=self._base_body_inertia,
            base_geom_friction=self._base_geom_friction,
            ground_geom_id=self._ground_geom_id,
            robot_geom_ids=self._robot_geom_ids,
            base_dof_armature=self._base_dof_armature,
        )
        if payload is not None:
            unsupported = capabilities.get_unsupported_reset_terms(payload.requested_terms())
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support DR002 reset randomization terms: {names}"
                )
        push_velocity_limit = self._push_linear_velocity_delta_limit(env.cfg.domain_rand)
        push_force_limit = self._push_force_limit(
            env.cfg.domain_rand,
            fallback_to_max_force=push_velocity_limit is None,
        )
        if env.cfg.domain_rand.push_robots:
            if (
                push_velocity_limit is not None
                and not capabilities.supports_interval_body_velocity_delta
            ):
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support interval body velocity perturbation"
                )
            if push_force_limit is not None and not capabilities.supports_interval_body_force:
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support interval body force perturbation"
                )
        if env.cfg.domain_rand.csv_force_enabled and not capabilities.supports_interval_body_force:
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support interval CSV force perturbation"
            )
        if (
            env.cfg.domain_rand.csv_force_enabled
            and int(env.cfg.sim_substeps) > 1
            and not capabilities.supports_interval_body_force_trajectory
        ):
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support physics-substep CSV force trajectories"
            )
        if env.cfg.domain_rand.csv_force_enabled:
            rotation = np.asarray(env.cfg.domain_rand.csv_force_rotation, dtype=np.float64)
            if rotation.size != 9:
                raise ValueError("domain_rand.csv_force_rotation must contain 9 row-major values")
            rotation = rotation.reshape(3, 3)
            if not np.all(np.isfinite(rotation)):
                raise ValueError("domain_rand.csv_force_rotation must contain finite values")
            if not np.allclose(
                rotation @ rotation.T,
                np.eye(3, dtype=np.float64),
                rtol=1.0e-6,
                atol=1.0e-6,
            ) or not np.isclose(
                np.linalg.det(rotation),
                1.0,
                rtol=1.0e-6,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "domain_rand.csv_force_rotation must be an orthonormal "
                    "right-handed sensor-to-base rotation"
                )
            if float(env.cfg.domain_rand.csv_force_period) < 0.0:
                raise ValueError("domain_rand.csv_force_period must be >= 0")
            transition_seconds = float(env.cfg.domain_rand.csv_force_transition_seconds)
            if transition_seconds < 0.0:
                raise ValueError("domain_rand.csv_force_transition_seconds must be >= 0")
            if (
                float(env.cfg.domain_rand.csv_force_period) > 0.0
                and transition_seconds * 2.0 > float(env.cfg.domain_rand.csv_force_period)
            ):
                raise ValueError(
                    "domain_rand.csv_force_transition_seconds must not exceed half the replay period"
                )
            _validate_csv_force_curriculum_mapping(env.cfg.domain_rand)
            push_point = np.asarray(env.cfg.domain_rand.csv_force_push_point, dtype=np.float64)
            if push_point.shape != (3,):
                raise ValueError("domain_rand.csv_force_push_point must contain 3 values")
            csv_body_name = env.cfg.domain_rand.csv_force_body_name
            if (
                env.cfg.domain_rand.push_robots
                and csv_body_name is not None
                and csv_body_name != env.cfg.asset.base_name
            ):
                raise NotImplementedError(
                    "DR002 cannot combine push_robots with csv_force_body_name pointing to a "
                    "different body; interval plans currently share one body_ids array"
                )

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        domain_rand = env.cfg.domain_rand
        body_ids: np.ndarray | None = None
        push_force: np.ndarray | None = None
        push_velocity_delta: np.ndarray | None = None
        episode_steps = env.episode_steps()
        elapsed_after_startup = episode_steps - int(getattr(env, "_startup_stand_steps", 0))
        push_robots_enabled = bool(domain_rand.push_robots)
        if push_robots_enabled and domain_rand.push_interval > 0:
            push_due = self._interval_push_due(
                elapsed_after_startup,
                interval_steps=int(domain_rand.push_interval),
                randomize_within_interval=bool(domain_rand.push_randomize_within_interval),
            )
            if np.any(push_due):
                num_push = int(np.count_nonzero(push_due))
                body_id = env._backend.get_body_id(env.cfg.asset.base_name)
                body_ids = np.asarray([body_id], dtype=np.int32)
                velocity_limit = self._push_linear_velocity_delta_limit(domain_rand)
                force_limit = self._push_force_limit(
                    domain_rand,
                    fallback_to_max_force=velocity_limit is None,
                )
                push_quat: np.ndarray | None = None
                if velocity_limit is not None or domain_rand.push_force_limit is not None:
                    base_quat = np.asarray(env._backend.get_base_quat(), dtype=np.float64)
                    expected_quat_shape = (env._num_envs, 4)
                    if base_quat.shape != expected_quat_shape:
                        raise ValueError(
                            "base quaternion batch must have shape "
                            f"{expected_quat_shape}, got {base_quat.shape}"
                        )
                    push_quat = base_quat[push_due]
                if velocity_limit is not None:
                    assert push_quat is not None
                    push_velocity_delta = np.zeros((env._num_envs, 1, 3), dtype=np.float64)
                    velocity_delta_body = (
                        np.random.uniform(-1.0, 1.0, size=(num_push, 3)) * velocity_limit[None, :]
                    )
                    push_velocity_delta[push_due, 0, :] = np_quat_apply(
                        push_quat,
                        velocity_delta_body,
                    )
                if force_limit is not None:
                    push_force = np.zeros((env._num_envs, 1, 6), dtype=np.float64)
                    sampled_force = (
                        np.random.uniform(-1.0, 1.0, size=(num_push, 3)) * force_limit[None, :]
                    )
                    if domain_rand.push_force_limit is not None:
                        assert push_quat is not None
                        sampled_force = np_quat_apply(push_quat, sampled_force)
                    push_force[push_due, 0, :3] = sampled_force

        body_force_trajectory = self._build_csv_force_trajectory(env, step_counter)
        if push_force is not None and hasattr(env, "_external_disturbance_current_wrench"):
            env._external_disturbance_current_wrench += push_force[:, 0, :]
        if push_force is None and push_velocity_delta is None and body_force_trajectory is None:
            return None
        if body_ids is None:
            body_name = domain_rand.csv_force_body_name or env.cfg.asset.base_name
            body_ids = np.asarray([env._backend.get_body_id(body_name)], dtype=np.int32)
        return IntervalRandomizationPlan(
            body_ids=body_ids,
            body_linear_velocity_delta=push_velocity_delta,
            body_force=push_force,
            body_force_trajectory=body_force_trajectory,
        )

    @staticmethod
    def _push_linear_velocity_delta_limit(domain_rand: DR002DomainRandConfig) -> np.ndarray | None:
        configured = domain_rand.push_linear_velocity_delta_limit
        if configured is None:
            return None
        limit = np.asarray(configured, dtype=np.float64)
        if limit.shape != (3,):
            raise ValueError(
                "domain_rand.push_linear_velocity_delta_limit must have shape (3,), "
                f"got {limit.shape}"
            )
        if not np.all(np.isfinite(limit)) or np.any(limit < 0.0):
            raise ValueError(
                "domain_rand.push_linear_velocity_delta_limit must contain finite, "
                "non-negative values"
            )
        if not np.any(limit > 0.0):
            raise ValueError(
                "domain_rand.push_linear_velocity_delta_limit must enable at least one axis"
            )
        return limit

    def _interval_push_due(
        self,
        elapsed_after_startup: np.ndarray,
        *,
        interval_steps: int,
        randomize_within_interval: bool,
    ) -> np.ndarray:
        elapsed = np.asarray(elapsed_after_startup, dtype=np.int64)
        if not randomize_within_interval:
            return (elapsed > 0) & ((elapsed % interval_steps) == 0)

        if (
            self._push_next_elapsed_steps is None
            or self._push_next_elapsed_steps.shape != elapsed.shape
        ):
            self._push_next_elapsed_steps = np.zeros(elapsed.shape, dtype=np.int64)

        at_startup_boundary = elapsed == 0
        if np.any(at_startup_boundary):
            self._push_next_elapsed_steps[at_startup_boundary] = np.random.randint(
                1,
                interval_steps + 1,
                size=int(np.count_nonzero(at_startup_boundary)),
            )

        push_due = (elapsed > 0) & (elapsed >= self._push_next_elapsed_steps)
        push_due &= self._push_next_elapsed_steps > 0
        if np.any(push_due):
            due_elapsed = elapsed[push_due]
            next_window_start = ((due_elapsed - 1) // interval_steps + 1) * interval_steps
            self._push_next_elapsed_steps[push_due] = next_window_start + np.random.randint(
                1,
                interval_steps + 1,
                size=int(np.count_nonzero(push_due)),
            )
        return push_due

    @staticmethod
    def _push_force_limit(
        domain_rand: DR002DomainRandConfig,
        *,
        fallback_to_max_force: bool,
    ) -> np.ndarray | None:
        configured = domain_rand.push_force_limit
        field_name = "push_force_limit"
        if configured is None:
            if not fallback_to_max_force:
                return None
            configured = domain_rand.max_force
            field_name = "max_force"
        limit = np.asarray(configured, dtype=np.float64)
        if limit.shape != (3,):
            raise ValueError(f"domain_rand.{field_name} must have shape (3,), got {limit.shape}")
        if not np.all(np.isfinite(limit)) or np.any(limit < 0.0):
            raise ValueError(f"domain_rand.{field_name} must contain finite, non-negative values")
        if not np.any(limit > 0.0):
            raise ValueError(f"domain_rand.{field_name} must enable at least one axis")
        return limit

    def _build_csv_force_trajectory(self, env: Any, step_counter: int) -> np.ndarray | None:
        domain_rand = env.cfg.domain_rand
        if hasattr(env, "_external_disturbance_current_wrench"):
            env._external_disturbance_current_wrench.fill(0.0)
        if hasattr(env, "_measured_csv_force_base"):
            env._measured_csv_force_base.fill(0.0)
        if hasattr(env, "_measured_csv_moment_base"):
            env._measured_csv_moment_base.fill(0.0)
        if not domain_rand.csv_force_enabled:
            return None
        if domain_rand.csv_force_curriculum:
            levels = env.csv_force_active_levels()
        else:
            level = _csv_force_curriculum_level_index(
                domain_rand,
                int(getattr(env, "_csv_force_curriculum_level", 0)),
            )
            levels = np.full((env._num_envs,), level, dtype=np.int32)
        resume_suppressed_getter = getattr(env, "csv_force_resume_suppressed", None)
        if callable(resume_suppressed_getter):
            resume_suppressed = np.asarray(resume_suppressed_getter(), dtype=np.bool_)
            if resume_suppressed.shape != (env._num_envs,):
                raise ValueError(
                    "CSV force resume suppression must match the number of environments"
                )
        else:
            resume_suppressed = np.zeros((env._num_envs,), dtype=np.bool_)
        amplitude_scales = env.csv_force_amplitude_scales()
        if env._standing_envs_episode_persistent and not domain_rand.csv_force_apply_to_standing:
            levels = np.array(levels, copy=True)
            levels[env.episode_standing_mask()] = 0

        num_substeps = int(env.cfg.sim_substeps)
        substep_offsets = np.arange(num_substeps, dtype=np.float64) * float(env.cfg.sim_dt)
        elapsed_after_startup, replay_t = _csv_replay_clock(
            domain_rand,
            env.episode_steps(),
            int(getattr(env, "_startup_stand_steps", 0)),
            float(env.cfg.ctrl_dt),
            sample_offsets_s=substep_offsets,
            start_delay_steps=env.csv_force_start_delay_steps(),
        )
        elapsed_s = (
            np.maximum(elapsed_after_startup, 0).astype(np.float64)[:, None]
            * float(env.cfg.ctrl_dt)
            + substep_offsets[None, :]
        )

        rotation = np.asarray(domain_rand.csv_force_rotation, dtype=np.float64).reshape(3, 3)
        base_quat = env._backend.get_base_quat()
        force_world = np.zeros((env._num_envs, num_substeps, 3), dtype=np.float64)
        torque_world = np.zeros_like(force_world)
        body_name = domain_rand.csv_force_body_name or env.cfg.asset.base_name
        body_id = int(env._backend.get_body_id(body_name))
        nominal_body_com = np.asarray(env._backend.get_body_ipos()[body_id], dtype=np.float64)
        body_com = np.broadcast_to(nominal_body_com, (env._num_envs, 3)).copy()
        if body_name == env.cfg.asset.base_name:
            body_com += np.asarray(env._privileged_base_com_offset, dtype=np.float64)
        push_point_base = np.asarray(domain_rand.csv_force_push_point, dtype=np.float64)
        moment_arm_base = push_point_base[None, :] - body_com
        for level in np.unique(levels):
            level_int = int(level)
            rows = (levels == level_int) & ~resume_suppressed
            source_hz = _csv_force_curriculum_source_hz(domain_rand, level_int)
            force_path = _csv_force_curriculum_path_for_level(domain_rand, level_int)
            if source_hz <= 0.0 or force_path is None or not np.any(rows):
                continue
            if force_path not in self._force_samples_by_path:
                wrench_samples = _load_force_csv(force_path)
                self._force_samples_by_path[force_path] = wrench_samples
            wrench_samples = self._force_samples_by_path[force_path]
            seam_columns = _csv_wrench_effective_seam_columns(domain_rand)
            seam_key = (force_path, float(domain_rand.csv_force_period), seam_columns)
            if seam_key not in self._force_blend_period_tail_by_key:
                self._force_blend_period_tail_by_key[seam_key] = not (
                    _csv_replay_channels_have_closed_seam(
                        wrench_samples,
                        period=float(domain_rand.csv_force_period),
                        channel_columns=seam_columns,
                    )
                )
            wrench_sensor_batch = _sample_wrench_csv_replay(
                wrench_samples,
                replay_t[rows],
                elapsed_s[rows],
                period=float(domain_rand.csv_force_period),
                transition_seconds=float(domain_rand.csv_force_transition_seconds),
                blend_period_tail=self._force_blend_period_tail_by_key[seam_key],
            )
            wrench_sensor_batch[elapsed_after_startup[rows] < 0] = 0.0
            wrench_sensor_batch *= amplitude_scales[rows, None, None]
            wrench_base_batch = _transform_csv_wrench_to_base_com(
                wrench_sensor_batch,
                sensor_to_base_rotation=rotation,
                moment_arm_base=moment_arm_base[rows, None, :],
                zero_fy=bool(domain_rand.csv_force_zero_fy),
                apply_measured_moment=bool(domain_rand.csv_force_apply_measured_moment),
                apply_point_torque=bool(domain_rand.csv_force_apply_point_torque),
            )
            force_base_batch = wrench_base_batch[..., :3]
            torque_base_batch = wrench_base_batch[..., 3:6]
            if hasattr(env, "_measured_csv_force_base"):
                env._measured_csv_force_base[rows] = force_base_batch[:, -1, :]
            if hasattr(env, "_measured_csv_moment_base"):
                measured_moment_base_batch = wrench_sensor_batch[..., 3:6] @ rotation.T
                env._measured_csv_moment_base[rows] = measured_moment_base_batch[:, -1, :]
            row_count = int(np.count_nonzero(rows))
            repeated_quat = np.repeat(base_quat[rows], num_substeps, axis=0)
            force_world[rows] = np.asarray(
                np_quat_apply(repeated_quat, force_base_batch.reshape(-1, 3)),
                dtype=np.float64,
            ).reshape(row_count, num_substeps, 3)
            torque_world[rows] = np.asarray(
                np_quat_apply(repeated_quat, torque_base_batch.reshape(-1, 3)),
                dtype=np.float64,
            ).reshape(row_count, num_substeps, 3)
        wrench_world = np.concatenate([force_world, torque_world], axis=2)
        if hasattr(env, "_external_disturbance_current_wrench"):
            env._external_disturbance_current_wrench[: wrench_world.shape[0]] = wrench_world[:, -1, :]
        return wrench_world[:, :, None, :]

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        low_xy, high_xy = env.cfg.domain_rand.init_xy_range
        qpos[:, 0:2] += np.random.uniform(low_xy, high_xy, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        if env.cfg.domain_rand.randomize_init_yaw:
            yaw_low, yaw_high = env.cfg.domain_rand.init_yaw_range
            yaw = np.random.uniform(yaw_low, yaw_high, (num_reset,))
            qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        if env._startup_stand_steps > 0:
            qvel[:, 0:6] = 0.0
        else:
            qvel_low, qvel_high = env.cfg.domain_rand.init_qvel_range
            qvel[:, 0:6] = np.asarray(
                np.random.uniform(qvel_low, qvel_high, size=(num_reset, 6)),
                dtype=get_global_dtype(),
            )

        motor_kp, motor_kd = env.sample_reset_motor_gains(num_reset)
        env.set_motor_gains(env_ids, motor_kp, motor_kd)
        torque_scale, default_joint_pos_offset = env.sample_reset_motor_runtime_randomization(num_reset)
        env.set_motor_runtime_randomization(env_ids, torque_scale, default_joint_pos_offset)
        standing_mask = env.sample_reset_standing_mask(env_ids)
        env.set_episode_standing_mask(env_ids, standing_mask)
        csv_force_levels = env.sample_reset_csv_force_levels(num_reset)
        csv_force_start_delay_steps = env.sample_reset_csv_force_start_delay_steps(num_reset)
        csv_force_amplitude_scales = env.sample_reset_csv_force_amplitude_scales(num_reset)
        wing_angle_obs_amplitude_scales = env.sample_reset_wing_angle_obs_amplitude_scales(
            num_reset
        )
        if (
            env._standing_envs_episode_persistent
            and not env.cfg.domain_rand.csv_force_apply_to_standing
        ):
            csv_force_levels[standing_mask] = 0
            csv_force_start_delay_steps[standing_mask] = 0
        zero_hz_wing_angle_obs = env.sample_reset_zero_hz_wing_angle_obs(
            _csv_force_zero_hz_mask(env.cfg.domain_rand, csv_force_levels)
        )
        env.set_zero_hz_wing_angle_obs(env_ids, zero_hz_wing_angle_obs)
        env.set_csv_force_active_levels(env_ids, csv_force_levels)
        env.set_csv_force_start_delay_steps(env_ids, csv_force_start_delay_steps)
        env.set_csv_force_amplitude_scales(env_ids, csv_force_amplitude_scales)
        env.set_wing_angle_obs_amplitude_scales(env_ids, wing_angle_obs_amplitude_scales)
        body_mass_multipliers = self._body_mass_multipliers_for_reset(env, env_ids)
        reset_randomization = build_dr002_backend_reset_randomization(
            env,
            num_reset,
            base_body_mass=self._base_body_mass,
            base_body_inertia=self._base_body_inertia,
            base_geom_friction=self._base_geom_friction,
            ground_geom_id=self._ground_geom_id,
            robot_geom_ids=self._robot_geom_ids,
            base_dof_armature=self._base_dof_armature,
            body_mass_multipliers=body_mass_multipliers,
        )
        env.set_privileged_reset_randomization(env_ids, reset_randomization)
        info_updates: dict[str, Any] = {
            "commands": (
                env.startup_commands(num_reset)
                if env._startup_stand_steps > 0
                else env.sample_commands(num_reset, env_ids=env_ids)
            ),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "motor_kp": motor_kp.astype(get_global_dtype()),
            "motor_kd": motor_kd.astype(get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=reset_randomization,
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        projected_gravity = env.get_projected_gravity()[env_ids]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
                linvel,
                gyro,
                gravity,
                projected_gravity,
                dof_pos,
                dof_vel,
                env_ids=env_ids,
                reset_history=True,
            ),
        )


@registry.env("DR002JoystickFlat", sim_backend="mujoco")
class DR002JoystickEnv(DR002BaseEnv):
    _cfg: DR002JoystickCfg
    _TRAINING_STATE_KIND = "unilab.dr002_joystick.curriculum"
    _TRAINING_STATE_VERSION = 1

    def __init__(self, cfg: DR002JoystickCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._wing_angle_obs_enabled = bool(cfg.wing_angle_obs.enabled)
        self._history_term_dims = (
            _TERM_DIMS[:-1] + (_WING_ANGLE_OBS_DIM,) + _TERM_DIMS[-1:]
            if self._wing_angle_obs_enabled
            else _TERM_DIMS
        )
        self._actor_dim = _HISTORY_LENGTH * sum(self._history_term_dims)
        critic_obs_mode = str(cfg.critic_obs_mode)
        if critic_obs_mode not in {"legacy", "isaaclab"}:
            raise ValueError(
                "critic_obs_mode must be either 'legacy' or 'isaaclab', "
                f"got {critic_obs_mode!r}"
            )
        self._use_isaaclab_critic = critic_obs_mode == "isaaclab"
        self._critic_includes_measured_moment = self._use_isaaclab_critic and bool(
            cfg.domain_rand.csv_force_observation_include_measured_moment
        )
        # Force-only IsaacLab critics retain 144D. WE9 can opt into the rotated
        # measured Mx/My/Mz for 147D, while legacy DR002 stays checkpoint-safe.
        if not self._use_isaaclab_critic:
            self._critic_dim = _LEGACY_CRITIC_DIM
        elif self._critic_includes_measured_moment:
            self._critic_dim = _CRITIC_WITH_MEASURED_MOMENT_DIM
        else:
            self._critic_dim = _CRITIC_DIM
        self._wing_angle_samples_by_path: dict[Path, np.ndarray] = {}
        self._wing_angle_force_samples_by_path: dict[str, np.ndarray] = {}
        self._enable_reward_log = True
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._validate_motor_control_contract(ctrl_range, num_envs)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        self._base_motor_kp = np.asarray(cfg.control_config.Kp, dtype=np.float64)
        self._base_motor_kd = np.asarray(cfg.control_config.Kd, dtype=np.float64)
        self._motor_kp = np.broadcast_to(self._base_motor_kp, (num_envs, NUM_DR002_ACTIONS)).copy()
        self._motor_kd = np.broadcast_to(self._base_motor_kd, (num_envs, NUM_DR002_ACTIONS)).copy()
        self._motor_torque_scale = np.ones((num_envs, NUM_DR002_ACTIONS), dtype=np.float64)
        self._default_joint_pos_offset = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=self._np_dtype)
        self._privileged_body_ids = np.asarray(
            [self._backend.get_body_id(name) for name in _PRIVILEGED_BODY_NAMES],
            dtype=np.int32,
        )
        self._privileged_joint_dof_ids = self._backend.get_joint_dof_indices(_PRIVILEGED_BODY_NAMES)
        self._privileged_base_mass_delta = np.zeros((num_envs, 1), dtype=np.float64)
        self._privileged_body_mass_scale = np.ones((num_envs, NUM_DR002_ACTIONS), dtype=np.float64)
        self._privileged_base_com_offset = np.zeros((num_envs, 3), dtype=np.float64)
        self._privileged_base_inertia_scale = np.ones((num_envs, 1), dtype=np.float64)
        self._privileged_ground_friction_scale = np.ones((num_envs, 1), dtype=np.float64)
        self._privileged_robot_friction_scale = np.ones((num_envs, 1), dtype=np.float64)
        self._privileged_dof_armature_scale = np.ones((num_envs, NUM_DR002_ACTIONS), dtype=np.float64)
        self._last_motor_ctrl = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=self._np_dtype)
        self._last_dof_vel_for_acc = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=get_global_dtype())
        delay_semantics = str(cfg.control_config.action_delay_semantics)
        torque_delay_steps = int(cfg.control_config.torque_delay_steps)
        if delay_semantics != "pre_controller_command_fifo" or torque_delay_steps != 0:
            raise ValueError(
                "DR002 supports only pre-controller command delay with torque_delay_steps=0, "
                f"got semantics={delay_semantics!r}, torque_delay_steps={torque_delay_steps}"
            )
        motor_control_hz = cfg.control_config.motor_control_hz
        if motor_control_hz is None:
            self._motor_control_hz = 1.0 / float(cfg.sim_dt)
            self._motor_control_decimation = 1
        else:
            self._motor_control_hz = float(motor_control_hz)
            if not np.isfinite(self._motor_control_hz) or self._motor_control_hz <= 0.0:
                raise ValueError("motor_control_hz must be finite and positive")
            decimation = (1.0 / float(cfg.sim_dt)) / self._motor_control_hz
            rounded_decimation = int(round(decimation))
            if rounded_decimation < 1 or not np.isclose(
                decimation, rounded_decimation, rtol=0.0, atol=1.0e-9
            ):
                raise ValueError(
                    "physics_hz must be an integer multiple of motor_control_hz, "
                    f"got physics_hz={1.0 / float(cfg.sim_dt):g}, "
                    f"motor_control_hz={self._motor_control_hz:g}"
                )
            self._motor_control_decimation = rounded_decimation
        if int(cfg.sim_substeps) % self._motor_control_decimation != 0:
            raise ValueError(
                "policy interval must contain an integer number of motor-control updates, "
                f"got sim_substeps={cfg.sim_substeps}, decimation={self._motor_control_decimation}"
            )
        if self._motor_control_decimation > 1 and cfg.control_config.use_native_batched_pd:
            raise ValueError(
                "native batched PD recomputes torque every physics substep and cannot be used "
                "with a lower-rate zero-order-held motor controller"
            )
        self._motor_control_substep_index = 0
        self._action_delay_enabled = bool(cfg.control_config.simulate_action_latency)
        fixed_delay_steps = cfg.control_config.action_delay_steps_by_joint
        self._fixed_action_delay_steps: np.ndarray | None = None
        if fixed_delay_steps is not None:
            raw_delay_steps = np.asarray(fixed_delay_steps, dtype=np.float64)
            if raw_delay_steps.shape != (NUM_DR002_ACTIONS,):
                raise ValueError(
                    "action_delay_steps_by_joint must follow the six-joint DR002 order, "
                    f"got shape {raw_delay_steps.shape}"
                )
            if np.any(~np.isfinite(raw_delay_steps)) or np.any(raw_delay_steps < 0.0) or np.any(
                raw_delay_steps != np.rint(raw_delay_steps)
            ):
                raise ValueError("action_delay_steps_by_joint must contain non-negative integers")
            if not self._action_delay_enabled:
                raise ValueError("action_delay_steps_by_joint requires simulate_action_latency=True")
            self._fixed_action_delay_steps = raw_delay_steps.astype(np.int32)
            self._action_delay_min_steps = int(np.min(self._fixed_action_delay_steps))
            self._action_delay_max_steps = int(np.max(self._fixed_action_delay_steps))
        else:
            self._action_delay_min_steps = int(cfg.control_config.action_delay_min_steps)
            self._action_delay_max_steps = int(cfg.control_config.action_delay_max_steps)
        if self._action_delay_min_steps < 0 or self._action_delay_max_steps < self._action_delay_min_steps:
            raise ValueError(
                "DR002 action delay requires 0 <= action_delay_min_steps <= action_delay_max_steps, "
                f"got [{self._action_delay_min_steps}, {self._action_delay_max_steps}]"
            )
        startup_seconds = float(cfg.commands.startup_stand_seconds)
        if startup_seconds < 0.0:
            raise ValueError("commands.startup_stand_seconds must be non-negative")
        self._startup_stand_steps = int(round(startup_seconds / float(cfg.ctrl_dt)))
        self._leg_clip_actions = float(cfg.control_config.clip_actions)
        self._wheel_clip_actions = float(cfg.control_config.wheel_clip_actions)
        if self._leg_clip_actions <= 0.0 or self._wheel_clip_actions <= 0.0:
            raise ValueError(
                "DR002 action clipping limits must be positive, "
                f"got leg={self._leg_clip_actions}, wheel={self._wheel_clip_actions}"
            )
        self._action_delay_buffer = np.zeros(
            (num_envs, self._action_delay_max_steps + 1, NUM_DR002_ACTIONS), dtype=self._np_dtype
        )
        if self._fixed_action_delay_steps is None:
            self._action_delay_indices = np.zeros((num_envs,), dtype=np.int32)
        else:
            self._action_delay_indices = np.broadcast_to(
                self._fixed_action_delay_steps, (num_envs, NUM_DR002_ACTIONS)
            ).copy()
        self._reset_action_delay(np.arange(num_envs, dtype=np.int32), resample=True)
        self._joint_range = self._backend.get_joint_range()
        self._lingzu_prev_action = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=get_global_dtype())
        self._lingzu_prev_prev_action = np.zeros((num_envs, NUM_DR002_ACTIONS), dtype=get_global_dtype())
        self._lingzu_action_history_count = np.zeros((num_envs,), dtype=np.int32)
        self._lingzu_fail_steps = np.zeros((num_envs,), dtype=np.int32)
        self._lingzu_contact_fail_accum_steps = np.zeros((num_envs,), dtype=np.int32)
        self._last_termination_contact_now_fraction = 0.0
        self._last_termination_contact_accum_mean = 0.0
        self._last_termination_contact_accum_max = 0.0
        self._last_termination_contact_done_fraction = 0.0
        self._last_termination_gravity_done_fraction = 0.0
        self._base_command_lin_vel_x = np.asarray(cfg.commands.lin_vel_x, dtype=np.float64)
        self._base_command_ang_vel_z = np.asarray(cfg.commands.ang_vel_z, dtype=np.float64)
        self._standing_probability = float(cfg.commands.rel_standing_envs)
        self._standing_envs_episode_persistent = bool(
            cfg.commands.standing_envs_episode_persistent
        )
        if not 0.0 <= self._standing_probability <= 1.0:
            raise ValueError(
                "commands.rel_standing_envs must be between 0 and 1, "
                f"got {self._standing_probability}"
            )
        self._episode_standing_mask = np.zeros((num_envs,), dtype=np.bool_)
        self._zero_hz_wing_angle_obs = np.zeros(
            (num_envs, _WING_ANGLE_OBS_DIM), dtype=self._np_dtype
        )
        self._wing_angle_obs_amplitude_scales = np.ones((num_envs,), dtype=self._np_dtype)
        self._standing_window_episodes = 0
        self._standing_window_fail_count = 0
        self._standing_window_episode_length_fraction_sum = 0.0
        self._last_standing_window_fail_rate = np.nan
        self._last_standing_window_mean_episode_length_fraction = np.nan
        self._last_standing_fraction = 0.0
        self._last_standing_mean_abs_vx = np.nan
        self._last_standing_mean_abs_vy = np.nan
        self._last_standing_mean_abs_yaw_rate = np.nan
        multiplier = np.asarray(cfg.commands.range_multiplier, dtype=np.float64)
        if multiplier.shape != (2,):
            raise ValueError("commands.range_multiplier must contain [initial, final]")
        yaw_multiplier = np.asarray(cfg.commands.ang_vel_z_range_multiplier, dtype=np.float64)
        if yaw_multiplier.shape != (2,):
            raise ValueError("commands.ang_vel_z_range_multiplier must contain [initial, final]")
        self._command_curriculum_scale = float(multiplier[0])
        self._command_curriculum_initial_scale = float(multiplier[0])
        self._command_curriculum_final_scale = float(multiplier[1])
        self._command_curriculum_yaw_scale = float(yaw_multiplier[0])
        self._command_curriculum_yaw_initial_scale = float(yaw_multiplier[0])
        self._command_curriculum_yaw_final_scale = float(yaw_multiplier[1])
        self._last_command_curriculum_mean_tracking = np.nan
        self._last_command_curriculum_mature_mean_tracking = np.nan
        self._last_command_curriculum_mean_episode_steps = np.nan
        self._last_command_curriculum_mean_moving_steps = np.nan
        self._last_command_curriculum_mature_fraction = np.nan
        self._csv_force_curriculum_level = 0
        self._csv_force_active_levels = np.zeros((num_envs,), dtype=np.int32)
        self._csv_force_start_delay_steps = np.zeros((num_envs,), dtype=np.int32)
        self._csv_force_resume_pending_reset = np.zeros((num_envs,), dtype=np.bool_)
        self._csv_force_amplitude_scales = np.ones((num_envs,), dtype=get_global_dtype())
        self._csv_force_curriculum_num_promoted = 0
        self._csv_force_curriculum_num_demoted = 0
        self._csv_force_curriculum_last_direction = 0
        self._csv_force_window_episodes = 0
        self._csv_force_window_tracking_episodes = 0
        self._csv_force_window_tracking_sum = 0.0
        self._csv_force_window_episode_length_fraction_sum = 0.0
        self._csv_force_window_mature_count = 0
        self._csv_force_window_fail_count = 0
        self._last_csv_force_window_tracking = np.nan
        self._last_csv_force_window_fail_rate = np.nan
        self._last_csv_force_window_mean_episode_length_fraction = np.nan
        self._last_csv_force_window_mature_fraction = np.nan
        self._noise_curriculum_level = 0
        self._noise_curriculum_num_promoted = 0
        self._noise_curriculum_num_demoted = 0
        self._noise_curriculum_last_direction = 0
        self._noise_window_episodes = 0
        self._noise_window_tracking_episodes = 0
        self._noise_window_tracking_sum = 0.0
        self._noise_window_episode_length_fraction_sum = 0.0
        self._noise_window_mature_count = 0
        self._noise_window_fail_count = 0
        self._last_noise_window_tracking = np.nan
        self._last_noise_window_fail_rate = np.nan
        self._last_noise_window_mean_episode_length_fraction = np.nan
        self._last_noise_window_mature_fraction = np.nan
        self._gravity_installation_bias_rp = np.zeros(
            (num_envs, 2), dtype=self._np_dtype
        )
        self._gravity_dynamic_noise_rp = np.zeros(
            (num_envs, 2), dtype=self._np_dtype
        )
        self._external_disturbance_current_wrench = np.zeros(
            (num_envs, 6), dtype=get_global_dtype()
        )
        self._measured_csv_force_base = np.zeros(
            (num_envs, 3), dtype=get_global_dtype()
        )
        self._measured_csv_moment_base = np.zeros((num_envs, 3), dtype=get_global_dtype())
        self._validate_csv_force_curriculum_cfg()
        _validate_wing_angle_observation_mapping(cfg.domain_rand, cfg.wing_angle_obs)
        self._validate_noise_curriculum_cfg()
        self._episode_track_lin_vel_x_sum = np.zeros((num_envs,), dtype=np.float64)
        self._episode_track_ang_vel_z_sum = np.zeros((num_envs,), dtype=np.float64)
        self._episode_force_track_lin_vel_x_sum = np.zeros((num_envs,), dtype=np.float64)
        self._episode_track_lin_vel_x_steps = np.zeros((num_envs,), dtype=np.int32)
        self._episode_track_total_steps = np.zeros((num_envs,), dtype=np.int32)
        self._episode_alive_sum = np.zeros((num_envs,), dtype=np.float64)
        self._episode_alive_steps = np.zeros((num_envs,), dtype=np.int32)
        self._last_episode_alive_return = np.nan
        self._last_episode_alive_fraction = np.nan
        self._last_episode_alive_steps = np.nan
        self._last_episode_alive_episode_steps = np.nan
        self._history_terms = [
            np.zeros((num_envs, _HISTORY_LENGTH, dim), dtype=get_global_dtype())
            for dim in self._history_term_dims
        ]
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        if cfg.control_config.use_native_batched_pd:
            self._backend.set_batched_mixed_pd_control(self._batched_motor_control)
        self._init_reward_functions()
        couple_base_inertia = cfg.domain_rand.couple_base_inertia_to_added_mass
        self._dr_base_body_mass = (
            self._backend.get_body_mass()
            if cfg.domain_rand.randomize_body_mass or couple_base_inertia
            else None
        )
        self._dr_base_body_inertia = (
            self._backend.get_body_inertia()
            if (
                cfg.domain_rand.randomize_body_mass
                or cfg.domain_rand.randomize_body_inertia
                or couple_base_inertia
            )
            else None
        )
        self._dr_base_geom_friction = None
        self._dr_ground_geom_id = None
        self._dr_robot_geom_ids = None
        if cfg.domain_rand.randomize_ground_friction or cfg.domain_rand.randomize_robot_geom_friction:
            self._dr_base_geom_friction = self._backend.get_geom_friction()
        if cfg.domain_rand.randomize_ground_friction:
            self._dr_ground_geom_id = self._backend.get_geom_id(cfg.asset.ground)
        if cfg.domain_rand.randomize_robot_geom_friction:
            base_body_id = self._backend.get_body_id(cfg.asset.base_name)
            robot_body_ids = self._backend.get_body_subtree_ids(base_body_id)
            geom_body_ids = self._backend.get_geom_body_ids()
            robot_geom_mask = np.isin(geom_body_ids, robot_body_ids)
            contype, conaffinity = self._backend.get_geom_contact_masks()
            contact_mask = (contype != 0) | (conaffinity != 0)
            self._dr_robot_geom_ids = np.flatnonzero(robot_geom_mask & contact_mask).astype(np.int32)
            if self._dr_robot_geom_ids.size == 0:
                raise ValueError("DR002 robot geom friction randomization found no contact-enabled robot geoms")
        self._dr_base_dof_armature = (
            self._backend.get_dof_armature() if cfg.domain_rand.randomize_dof_armature else None
        )
        self._init_domain_randomization(
            DR002JoystickDomainRandomizationProvider(
                base_body_mass=self._dr_base_body_mass,
                base_body_inertia=self._dr_base_body_inertia,
                base_geom_friction=self._dr_base_geom_friction,
                ground_geom_id=self._dr_ground_geom_id,
                robot_geom_ids=self._dr_robot_geom_ids,
                base_dof_armature=self._dr_base_dof_armature,
            )
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._actor_dim, "critic": self._critic_dim, "privileged_target": 3}

    def training_state_dict(self) -> dict[str, Any]:
        """Serialize aggregate curriculum progress without per-environment runtime state."""
        domain_rand = self._cfg.domain_rand
        noise_cfg = self._cfg.noise_config
        noise_curriculum_enabled = bool(getattr(noise_cfg, "curriculum", False))
        noise_levels = (
            _noise_curriculum_levels(noise_cfg).tolist() if noise_curriculum_enabled else []
        )
        checkpoint_paths, checkpoint_level_paths = _csv_force_curriculum_checkpoint_paths(
            domain_rand
        )
        return {
            "kind": self._TRAINING_STATE_KIND,
            "version": self._TRAINING_STATE_VERSION,
            "command_curriculum": {
                "scale": float(self._command_curriculum_scale),
                "yaw_scale": float(self._command_curriculum_yaw_scale),
            },
            "csv_force_curriculum": {
                "level": int(self._csv_force_curriculum_level),
                "num_promoted": int(self._csv_force_curriculum_num_promoted),
                "num_demoted": int(self._csv_force_curriculum_num_demoted),
                "last_direction": int(self._csv_force_curriculum_last_direction),
                "config": {
                    "hz_values": _csv_force_curriculum_hz_values(domain_rand),
                    "paths": checkpoint_paths,
                    "level_paths": checkpoint_level_paths,
                },
            },
            "noise_curriculum": {
                "level": int(self._noise_curriculum_level),
                "num_promoted": int(self._noise_curriculum_num_promoted),
                "num_demoted": int(self._noise_curriculum_num_demoted),
                "last_direction": int(self._noise_curriculum_last_direction),
                "config": {
                    "enabled": noise_curriculum_enabled,
                    "follows_csv_force": bool(self._noise_curriculum_follows_csv_force()),
                    "levels": noise_levels,
                },
            },
        }

    def load_training_state_dict(self, state: dict[str, Any]) -> None:
        """Restore global progress, restart partial windows, and safely phase in replay."""
        if not isinstance(state, dict):
            raise TypeError("DR002 training state must be a dictionary")
        if state.get("kind") != self._TRAINING_STATE_KIND:
            raise ValueError(f"unsupported DR002 training state kind: {state.get('kind')!r}")
        if state.get("version") != self._TRAINING_STATE_VERSION:
            raise ValueError(f"unsupported DR002 training state version: {state.get('version')!r}")

        command_state = state.get("command_curriculum")
        force_state = state.get("csv_force_curriculum")
        noise_state = state.get("noise_curriculum")
        if (
            not isinstance(command_state, dict)
            or not isinstance(force_state, dict)
            or not isinstance(noise_state, dict)
        ):
            raise ValueError(
                "DR002 training state must contain command, CSV-force, and noise curricula"
            )

        def finite_float(payload: dict[str, Any], key: str) -> float:
            value = float(payload[key])
            if not np.isfinite(value):
                raise ValueError(f"DR002 training state field {key!r} must be finite")
            return value

        def nonnegative_int(payload: dict[str, Any], key: str) -> int:
            raw = payload[key]
            value = int(raw)
            if isinstance(raw, (float, np.floating)) and float(raw) != float(value):
                raise ValueError(f"DR002 training state field {key!r} must be an integer")
            if value < 0:
                raise ValueError(f"DR002 training state field {key!r} must be non-negative")
            return value

        def direction(payload: dict[str, Any]) -> int:
            value = int(payload["last_direction"])
            if value not in (-1, 0, 1):
                raise ValueError("DR002 curriculum last_direction must be -1, 0, or 1")
            return value

        def optional_path_list(value: Any) -> list[str | None]:
            if not isinstance(value, (list, tuple)):
                return []
            return [None if path is None else str(path) for path in value]

        def float_list(value: Any) -> list[float]:
            if not isinstance(value, (list, tuple)):
                return []
            try:
                values = [float(item) for item in value]
            except (TypeError, ValueError):
                return []
            return values if np.all(np.isfinite(values)) else []

        def restore_level(
            *,
            saved_level: int,
            saved_values: list[float],
            current_values: list[float],
            saved_paths: list[str | None] | None = None,
            current_paths: list[str | None] | None = None,
            mapping_matches: bool,
        ) -> int:
            current_size = len(current_values) or len(current_paths or [])
            max_level = max(current_size - 1, 0)
            if mapping_matches:
                return int(np.clip(saved_level, 0, max_level))
            if not current_values:
                if saved_paths and current_paths:
                    saved_index = int(np.clip(saved_level, 0, len(saved_paths) - 1))
                    saved_path = saved_paths[saved_index]
                    matching = [
                        index for index, path in enumerate(current_paths) if path == saved_path
                    ]
                    return matching[0] if matching else 0
                return 0

            saved_index = int(np.clip(saved_level, 0, max(len(saved_values) - 1, 0)))
            saved_value = saved_values[saved_index] if saved_values else float("nan")
            saved_path = (
                saved_paths[saved_index]
                if saved_paths is not None and saved_index < len(saved_paths)
                else None
            )
            identity_matches = [
                index
                for index, current_value in enumerate(current_values)
                if np.isclose(current_value, saved_value, rtol=0.0, atol=1.0e-9)
                and (
                    saved_paths is None
                    or current_paths is None
                    or (index < len(current_paths) and current_paths[index] == saved_path)
                )
            ]
            if identity_matches:
                saved_identity_rank = 0
                for index in range(saved_index):
                    same_value = index < len(saved_values) and np.isclose(
                        saved_values[index],
                        saved_value,
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                    same_path = (
                        saved_paths is None
                        or index >= len(saved_paths)
                        or saved_paths[index] == saved_path
                    )
                    if same_value and same_path:
                        saved_identity_rank += 1
                return int(identity_matches[min(saved_identity_rank, len(identity_matches) - 1)])
            same_value_matches = [
                index
                for index, current_value in enumerate(current_values)
                if np.isclose(current_value, saved_value, rtol=0.0, atol=1.0e-9)
            ]
            if same_value_matches:
                saved_value_rank = sum(
                    1
                    for value in saved_values[:saved_index]
                    if np.isclose(value, saved_value, rtol=0.0, atol=1.0e-9)
                )
                return int(same_value_matches[min(saved_value_rank, len(same_value_matches) - 1)])
            if np.isfinite(saved_value):
                eligible = [
                    (current_value, index)
                    for index, current_value in enumerate(current_values)
                    if current_value <= saved_value + 1.0e-9
                ]
                if eligible:
                    easiest_at_best_value = min(
                        index
                        for value, index in eligible
                        if value == max(item[0] for item in eligible)
                    )
                    return int(easiest_at_best_value)
            return 0

        self._command_curriculum_scale = float(
            np.clip(
                finite_float(command_state, "scale"),
                min(
                    self._command_curriculum_initial_scale,
                    self._command_curriculum_final_scale,
                ),
                max(
                    self._command_curriculum_initial_scale,
                    self._command_curriculum_final_scale,
                ),
            )
        )
        self._command_curriculum_yaw_scale = float(
            np.clip(
                finite_float(command_state, "yaw_scale"),
                min(
                    self._command_curriculum_yaw_initial_scale,
                    self._command_curriculum_yaw_final_scale,
                ),
                max(
                    self._command_curriculum_yaw_initial_scale,
                    self._command_curriculum_yaw_final_scale,
                ),
            )
        )

        domain_rand = self._cfg.domain_rand
        current_hz_values = _csv_force_curriculum_hz_values(domain_rand)
        current_paths, current_level_paths = _csv_force_curriculum_checkpoint_paths(domain_rand)
        saved_config = force_state.get("config")
        if not isinstance(saved_config, dict):
            saved_config = {}
        saved_hz_values = float_list(saved_config.get("hz_values"))
        saved_paths = optional_path_list(saved_config.get("paths"))
        saved_level_paths = optional_path_list(saved_config.get("level_paths"))
        force_mapping_matches = (
            saved_hz_values == current_hz_values
            and saved_paths == current_paths
            and saved_level_paths == current_level_paths
        )
        saved_force_level = nonnegative_int(force_state, "level")
        restored_force_level = restore_level(
            saved_level=saved_force_level,
            saved_values=saved_hz_values,
            current_values=current_hz_values,
            saved_paths=saved_level_paths,
            current_paths=current_level_paths,
            mapping_matches=force_mapping_matches,
        )
        self._csv_force_curriculum_num_promoted = nonnegative_int(force_state, "num_promoted")
        self._csv_force_curriculum_num_demoted = nonnegative_int(force_state, "num_demoted")
        self._csv_force_curriculum_last_direction = direction(force_state)
        command_is_full = self._command_curriculum_is_full()
        self._csv_force_curriculum_level = int(restored_force_level) if command_is_full else 0

        self._reset_csv_force_curriculum_window()

        noise_cfg = self._cfg.noise_config
        current_noise_enabled = bool(getattr(noise_cfg, "curriculum", False))
        current_noise_follows = self._noise_curriculum_follows_csv_force()
        current_noise_levels = (
            _noise_curriculum_levels(noise_cfg).tolist() if current_noise_enabled else []
        )
        saved_noise_config = noise_state.get("config")
        if not isinstance(saved_noise_config, dict):
            saved_noise_config = {}
        saved_noise_levels = float_list(saved_noise_config.get("levels"))
        noise_mapping_matches = saved_noise_levels == current_noise_levels
        saved_noise_level = nonnegative_int(noise_state, "level")
        restored_noise_level = restore_level(
            saved_level=saved_noise_level,
            saved_values=saved_noise_levels,
            current_values=current_noise_levels,
            mapping_matches=noise_mapping_matches,
        )
        self._noise_curriculum_num_promoted = nonnegative_int(noise_state, "num_promoted")
        self._noise_curriculum_num_demoted = nonnegative_int(noise_state, "num_demoted")
        self._noise_curriculum_last_direction = direction(noise_state)
        if not command_is_full or not current_noise_enabled:
            self._noise_curriculum_level = 0
        elif current_noise_follows:
            self._noise_curriculum_level = int(self._csv_force_curriculum_level)
        else:
            self._noise_curriculum_level = int(restored_noise_level)

        self._reset_noise_curriculum_window()

        # The concrete environment clock is deliberately staggered at startup.
        # Suppress replay/paired observations and ignore each transition episode
        # until its first natural reset installs a fresh level and start delay.
        self._csv_force_active_levels.fill(0)
        self._csv_force_start_delay_steps.fill(0)
        self._csv_force_resume_pending_reset.fill(True)
        self._external_disturbance_current_wrench.fill(0.0)
        self._measured_csv_force_base.fill(0.0)
        self._measured_csv_moment_base.fill(0.0)
        current_state = getattr(self, "_state", None)
        current_obs = getattr(current_state, "obs", None)
        if bool(getattr(self, "_wing_angle_obs_enabled", False)):
            self._history_terms[-2].fill(0.0)
            if isinstance(current_obs, dict) and "obs" in current_obs:
                current_obs["obs"][:] = np.concatenate(
                    [hist.reshape(self._num_envs, -1) for hist in self._history_terms],
                    axis=1,
                )
        if bool(getattr(self, "_use_isaaclab_critic", False)) and isinstance(current_obs, dict):
            critic = current_obs.get("critic")
            if isinstance(critic, np.ndarray):
                measured_wrench_dim = (
                    6 if bool(getattr(self, "_critic_includes_measured_moment", False)) else 3
                )
                critic[:, -measured_wrench_dim:] = 0.0
        self._last_csv_force_window_tracking = np.nan
        self._last_csv_force_window_fail_rate = np.nan
        self._last_csv_force_window_mean_episode_length_fraction = np.nan
        self._last_csv_force_window_mature_fraction = np.nan
        self._last_noise_window_tracking = np.nan
        self._last_noise_window_fail_rate = np.nan
        self._last_noise_window_mean_episode_length_fraction = np.nan
        self._last_noise_window_mature_fraction = np.nan

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        if hasattr(self, "_episode_standing_mask"):
            self._update_standing_episode_stats(env_ids)
        if hasattr(self, "_episode_alive_sum"):
            self._update_episode_alive_stats(env_ids)
        if hasattr(self, "_episode_track_lin_vel_x_sum"):
            self._update_command_curriculum(env_ids)
        self._reset_gravity_tilt_noise(env_ids)
        self._external_disturbance_current_wrench[env_ids] = 0.0
        self._measured_csv_force_base[env_ids] = 0.0
        self._measured_csv_moment_base[env_ids] = 0.0
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        self._lingzu_prev_action[env_ids] = 0.0
        self._lingzu_prev_prev_action[env_ids] = 0.0
        self._lingzu_action_history_count[env_ids] = 0
        self._lingzu_fail_steps[env_ids] = 0
        self._lingzu_contact_fail_accum_steps[env_ids] = 0
        self._last_motor_ctrl[env_ids] = 0.0
        self._reset_action_delay(env_ids, resample=self._cfg.control_config.resample_action_delay)
        self._episode_track_lin_vel_x_sum[env_ids] = 0.0
        self._episode_track_ang_vel_z_sum[env_ids] = 0.0
        self._episode_force_track_lin_vel_x_sum[env_ids] = 0.0
        self._episode_track_lin_vel_x_steps[env_ids] = 0
        self._episode_track_total_steps[env_ids] = 0
        self._episode_alive_sum[env_ids] = 0.0
        self._episode_alive_steps[env_ids] = 0
        return obs, info

    def _neutral_policy_ctrl(self, env_ids: np.ndarray) -> np.ndarray:
        env_ids = np.asarray(env_ids, dtype=np.int32)
        neutral = np.zeros((len(env_ids), NUM_DR002_ACTIONS), dtype=self._np_dtype)
        neutral[:, LEG_ACTION_INDICES] = (
            self.default_angles[LEG_ACTION_INDICES]
            + self._default_joint_pos_offset[env_ids][:, LEG_ACTION_INDICES]
        )
        return neutral

    def _reset_action_delay(self, env_ids: np.ndarray, *, resample: bool) -> None:
        if env_ids.size == 0:
            return
        neutral = self._neutral_policy_ctrl(env_ids)
        self._action_delay_buffer[env_ids] = neutral[:, None, :]
        if self._fixed_action_delay_steps is not None:
            self._action_delay_indices[env_ids] = self._fixed_action_delay_steps
        elif self._action_delay_enabled and resample:
            self._action_delay_indices[env_ids] = np.random.randint(
                self._action_delay_min_steps,
                self._action_delay_max_steps + 1,
                size=(len(env_ids),),
                dtype=np.int32,
            )
        elif not self._action_delay_enabled:
            self._action_delay_indices[env_ids] = 0

    def _delayed_policy_ctrl(self, policy_ctrl: np.ndarray) -> np.ndarray:
        if not self._action_delay_enabled:
            return policy_ctrl
        self._action_delay_buffer[:, 1:] = self._action_delay_buffer[:, :-1].copy()
        self._action_delay_buffer[:, 0] = policy_ctrl
        env_ids = np.arange(policy_ctrl.shape[0], dtype=np.int32)
        if self._action_delay_indices.ndim == 1:
            return self._action_delay_buffer[env_ids, self._action_delay_indices[env_ids]]
        joint_ids = np.arange(NUM_DR002_ACTIONS, dtype=np.int32)
        return self._action_delay_buffer[
            env_ids[:, None], self._action_delay_indices[env_ids], joint_ids[None, :]
        ]

    def _validate_motor_control_contract(self, ctrl_range: np.ndarray, num_envs: int) -> None:
        if self._backend.num_actuators != NUM_DR002_ACTIONS:
            raise ValueError(
                f"DR002 requires {NUM_DR002_ACTIONS} motor actuators, got {self._backend.num_actuators}"
            )
        if ctrl_range.shape != (NUM_DR002_ACTIONS, 2):
            raise ValueError(
                f"DR002 actuator ctrl_range must have shape ({NUM_DR002_ACTIONS}, 2), got {ctrl_range.shape}"
            )
        expected_shape = (num_envs, NUM_DR002_ACTIONS)
        pos = stack_joint_sensors(self._backend, "pos", dtype=self.default_angles.dtype)
        vel = stack_joint_sensors(self._backend, "vel", dtype=self.default_angles.dtype)
        if pos.shape != expected_shape:
            raise ValueError(f"DR002 joint position sensor stack must have shape {expected_shape}")
        if vel.shape != expected_shape:
            raise ValueError(f"DR002 joint velocity sensor stack must have shape {expected_shape}")

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "track_lin_vel_x": self._reward_track_lin_vel_x,
            "track_ang_vel_z": self._reward_track_ang_vel_z,
            "track_lin_vel_x_enhance": self._reward_track_lin_vel_x_enhance,
            "lin_vel_z": self._reward_lin_vel_z_lingzu,
            "ang_vel_xy": self._reward_ang_vel_xy_lingzu,
            "orientation": self._reward_orientation_lingzu,
            "base_height": self._reward_base_height_cmd,
            "joint_torques_l2": self._reward_joint_torques_l2,
            "joint_torques_wheel_l2": self._reward_joint_torques_wheel_l2,
            "joint_vel_l2": self._reward_joint_vel_l2,
            "joint_acc_l2": self._reward_joint_acc_l2,
            "joint_acc_wheel_l2": self._reward_joint_acc_wheel_l2,
            "joint_pos_limits": self._reward_joint_pos_limits,
            "nominal_state_lingzu": self._reward_nominal_state_lingzu,
            "action_rate_l2": self._reward_action_rate_lingzu,
            "action_smooth_lingzu": self._reward_action_smooth_lingzu,
            "undesired_contacts": self._reward_undesired_contacts,
            "alive": self._reward_alive,
        }

    def sample_reset_motor_gains(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        kp = np.broadcast_to(self._base_motor_kp, (num_reset, NUM_DR002_ACTIONS)).copy()
        kd = np.broadcast_to(self._base_motor_kd, (num_reset, NUM_DR002_ACTIONS)).copy()
        domain_rand = self._cfg.domain_rand
        if domain_rand.randomize_kp:
            low, high = domain_rand.kp_multiplier_range
            kp *= np.random.uniform(low, high, size=(num_reset, 1))
        if domain_rand.randomize_kd:
            low, high = domain_rand.kd_multiplier_range
            kd *= np.random.uniform(low, high, size=(num_reset, 1))
        kp[:, WHEEL_ACTION_INDICES] = 0.0
        return kp, kd

    def sample_reset_motor_runtime_randomization(self, num_reset: int) -> tuple[np.ndarray, np.ndarray]:
        torque_scale = np.ones((num_reset, NUM_DR002_ACTIONS), dtype=np.float64)
        default_joint_pos_offset = np.zeros((num_reset, NUM_DR002_ACTIONS), dtype=self._np_dtype)
        domain_rand = self._cfg.domain_rand
        if domain_rand.randomize_torque_scale:
            low, high = domain_rand.torque_scale_range
            torque_scale *= np.random.uniform(low, high, size=(num_reset, NUM_DR002_ACTIONS))
        if domain_rand.randomize_default_joint_pos:
            low, high = domain_rand.default_joint_pos_offset_range
            default_joint_pos_offset[:, LEG_ACTION_INDICES] = np.random.uniform(
                low, high, size=(num_reset, len(LEG_ACTION_INDICES))
            )
        return torque_scale, default_joint_pos_offset

    def set_motor_gains(self, env_ids: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        self._motor_kp[env_ids] = np.asarray(kp, dtype=np.float64)
        self._motor_kd[env_ids] = np.asarray(kd, dtype=np.float64)

    def set_motor_runtime_randomization(
        self,
        env_ids: np.ndarray,
        torque_scale: np.ndarray,
        default_joint_pos_offset: np.ndarray,
    ) -> None:
        self._motor_torque_scale[env_ids] = np.asarray(torque_scale, dtype=np.float64)
        self._default_joint_pos_offset[env_ids] = np.asarray(
            default_joint_pos_offset, dtype=self._np_dtype
        )

    def sample_reset_csv_force_levels(self, num_reset: int) -> np.ndarray:
        domain_rand = self._cfg.domain_rand
        levels = np.zeros((num_reset,), dtype=np.int32)
        if (
            not domain_rand.csv_force_enabled
            or not domain_rand.csv_force_curriculum
            or num_reset <= 0
        ):
            return levels
        max_level = _csv_force_curriculum_level_index(
            domain_rand,
            int(self._csv_force_curriculum_level),
        )
        if max_level <= 0:
            return levels
        return np.random.randint(0, max_level + 1, size=(num_reset,)).astype(np.int32)

    def _csv_force_start_delay_step_bounds(self) -> tuple[int, int]:
        delay_range = np.asarray(
            self._cfg.domain_rand.csv_force_start_delay_range_s,
            dtype=np.float64,
        )
        if delay_range.shape != (2,):
            raise ValueError("domain_rand.csv_force_start_delay_range_s must contain [min, max]")
        if not np.all(np.isfinite(delay_range)):
            raise ValueError("domain_rand.csv_force_start_delay_range_s must be finite")
        low_s, high_s = float(delay_range[0]), float(delay_range[1])
        if low_s < 0.0 or high_s < low_s:
            raise ValueError(
                "domain_rand.csv_force_start_delay_range_s must satisfy 0 <= min <= max"
            )
        ctrl_dt = float(self._cfg.ctrl_dt)
        low_steps = int(np.ceil(low_s / ctrl_dt - 1.0e-9))
        high_steps = int(np.floor(high_s / ctrl_dt + 1.0e-9))
        if high_steps < low_steps:
            raise ValueError(
                "domain_rand.csv_force_start_delay_range_s contains no value aligned "
                f"to ctrl_dt={ctrl_dt}"
            )
        return low_steps, high_steps

    def sample_reset_csv_force_start_delay_steps(self, num_reset: int) -> np.ndarray:
        low_steps, high_steps = self._csv_force_start_delay_step_bounds()
        if num_reset <= 0 or high_steps <= low_steps:
            return np.full((max(num_reset, 0),), low_steps, dtype=np.int32)
        return np.random.randint(
            low_steps,
            high_steps + 1,
            size=(num_reset,),
            dtype=np.int32,
        )

    def _csv_force_amplitude_scale_bounds(self) -> tuple[float, float]:
        scale_range = np.asarray(
            self._cfg.domain_rand.csv_force_amplitude_scale_range,
            dtype=np.float64,
        )
        if scale_range.shape != (2,):
            raise ValueError("domain_rand.csv_force_amplitude_scale_range must contain [min, max]")
        if not np.all(np.isfinite(scale_range)):
            raise ValueError(
                "domain_rand.csv_force_amplitude_scale_range must contain finite values"
            )
        low, high = float(scale_range[0]), float(scale_range[1])
        if low < 0.0 or high < low:
            raise ValueError(
                "domain_rand.csv_force_amplitude_scale_range must satisfy 0 <= min <= max"
            )
        return low, high

    def sample_reset_csv_force_amplitude_scales(self, num_reset: int) -> np.ndarray:
        low, high = self._csv_force_amplitude_scale_bounds()
        if num_reset <= 0:
            return np.ones((0,), dtype=get_global_dtype())
        if high <= low:
            return np.full((num_reset,), low, dtype=get_global_dtype())
        return np.asarray(
            np.random.uniform(low, high, size=(num_reset,)),
            dtype=get_global_dtype(),
        )

    def set_csv_force_amplitude_scales(
        self,
        env_ids: np.ndarray,
        amplitude_scales: np.ndarray,
    ) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        values = np.asarray(amplitude_scales, dtype=get_global_dtype()).reshape(-1)
        if values.shape != (rows.size,):
            raise ValueError(
                f"CSV force amplitude scales must have shape ({rows.size},), got {values.shape}"
            )
        low, high = self._csv_force_amplitude_scale_bounds()
        tolerance = max(1.0, abs(low), abs(high)) * 1.0e-6
        if (
            not np.all(np.isfinite(values))
            or np.any(values < low - tolerance)
            or np.any(values > high + tolerance)
        ):
            raise ValueError(
                "CSV force amplitude scales must be finite and inside the configured "
                f"[{low}, {high}] range"
            )
        self._csv_force_amplitude_scales[rows] = values

    def csv_force_amplitude_scales(self) -> np.ndarray:
        return np.asarray(
            self._csv_force_amplitude_scales[: self._num_envs],
            dtype=np.float64,
        )

    def set_csv_force_active_levels(self, env_ids: np.ndarray, levels: np.ndarray) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        if rows.size == 0:
            return
        self._csv_force_active_levels[rows] = np.asarray(levels, dtype=np.int32).reshape(rows.size)

    def csv_force_active_levels(self) -> np.ndarray:
        domain_rand = self._cfg.domain_rand
        max_level = max(_csv_force_curriculum_num_levels(domain_rand) - 1, 0)
        return np.asarray(
            np.clip(self._csv_force_active_levels[: self._num_envs], 0, max_level),
            dtype=np.int32,
        )

    def csv_force_resume_suppressed(self) -> np.ndarray:
        resume_pending = getattr(self, "_csv_force_resume_pending_reset", None)
        if resume_pending is None:
            return np.zeros((self._num_envs,), dtype=np.bool_)
        return np.asarray(
            resume_pending[: self._num_envs],
            dtype=np.bool_,
        )

    def set_csv_force_start_delay_steps(
        self,
        env_ids: np.ndarray,
        delay_steps: np.ndarray,
    ) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        values = np.asarray(delay_steps, dtype=np.int32).reshape(-1)
        if values.shape != (rows.size,):
            raise ValueError(
                f"CSV force start delays must have shape ({rows.size},), got {values.shape}"
            )
        self._csv_force_start_delay_steps[rows] = values

    def csv_force_start_delay_steps(self) -> np.ndarray:
        return np.asarray(
            self._csv_force_start_delay_steps[: self._num_envs],
            dtype=np.int32,
        )

    def csv_force_active_hz(self) -> np.ndarray:
        domain_rand = self._cfg.domain_rand
        levels = self.csv_force_active_levels()
        hz = np.full((levels.shape[0],), np.nan, dtype=np.float64)
        for level in np.unique(levels):
            hz[levels == int(level)] = _csv_force_curriculum_source_hz(domain_rand, int(level))
        return hz

    @staticmethod
    def _safe_scale_ratio(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        baseline_arr = np.asarray(baseline, dtype=np.float64)
        values_arr = np.asarray(values, dtype=np.float64)
        ratio = np.ones_like(values_arr, dtype=np.float64)
        valid = np.abs(baseline_arr) > 1.0e-12
        ratio[..., valid] = values_arr[..., valid] / baseline_arr[valid]
        return ratio

    def _select_env_rows(
        self,
        values: np.ndarray,
        num_obs: int,
        env_ids: np.ndarray | None,
    ) -> np.ndarray:
        arr = np.asarray(values)
        if env_ids is None:
            return arr[:num_obs]
        return arr[np.asarray(env_ids, dtype=np.int32)]

    def set_privileged_reset_randomization(
        self,
        env_ids: np.ndarray,
        randomization: ResetRandomizationPayload | None,
    ) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        num_reset = rows.shape[0]
        self._privileged_base_mass_delta[rows] = 0.0
        self._privileged_body_mass_scale[rows] = 1.0
        self._privileged_base_com_offset[rows] = 0.0
        self._privileged_base_inertia_scale[rows] = 1.0
        self._privileged_ground_friction_scale[rows] = 1.0
        self._privileged_robot_friction_scale[rows] = 1.0
        self._privileged_dof_armature_scale[rows] = 1.0
        if randomization is None:
            return

        if randomization.base_mass_delta is not None:
            self._privileged_base_mass_delta[rows, 0] = np.asarray(
                randomization.base_mass_delta, dtype=np.float64
            ).reshape(num_reset)

        if randomization.body_mass is not None and self._dr_base_body_mass is not None:
            body_mass = np.asarray(randomization.body_mass, dtype=np.float64).reshape(num_reset, -1)
            baseline = np.asarray(self._dr_base_body_mass, dtype=np.float64)[
                self._privileged_body_ids
            ]
            self._privileged_body_mass_scale[rows] = self._safe_scale_ratio(
                body_mass[:, self._privileged_body_ids],
                baseline,
            )

        if randomization.base_com_offset is not None:
            self._privileged_base_com_offset[rows] = np.asarray(
                randomization.base_com_offset, dtype=np.float64
            ).reshape(num_reset, 3)

        if randomization.body_inertia is not None and self._dr_base_body_inertia is not None:
            body_inertia = np.asarray(randomization.body_inertia, dtype=np.float64).reshape(
                num_reset, -1, 3
            )
            base_body_id = self._backend.get_body_id(self._cfg.asset.base_name)
            baseline = np.asarray(self._dr_base_body_inertia, dtype=np.float64)[int(base_body_id)]
            ratio = self._safe_scale_ratio(body_inertia[:, int(base_body_id), :], baseline)
            self._privileged_base_inertia_scale[rows, 0] = np.mean(ratio, axis=1)

        if randomization.geom_friction is not None and self._dr_base_geom_friction is not None:
            geom_friction = np.asarray(randomization.geom_friction, dtype=np.float64).reshape(
                num_reset, -1, 3
            )
            base_geom_friction = np.asarray(self._dr_base_geom_friction, dtype=np.float64)
            if self._dr_ground_geom_id is not None:
                ground_id = int(self._dr_ground_geom_id)
                baseline = base_geom_friction[ground_id, 0]
                if abs(float(baseline)) > 1.0e-12:
                    self._privileged_ground_friction_scale[rows, 0] = (
                        geom_friction[:, ground_id, 0] / baseline
                    )
            if self._dr_robot_geom_ids is not None and self._dr_robot_geom_ids.size > 0:
                ids = np.asarray(self._dr_robot_geom_ids, dtype=np.int32)
                baseline = base_geom_friction[ids, 0]
                valid = np.abs(baseline) > 1.0e-12
                if np.any(valid):
                    ratios = geom_friction[:, ids[valid], 0] / baseline[valid]
                    self._privileged_robot_friction_scale[rows, 0] = np.mean(ratios, axis=1)

        if randomization.dof_armature is not None and self._dr_base_dof_armature is not None:
            dof_armature = np.asarray(randomization.dof_armature, dtype=np.float64).reshape(
                num_reset, -1
            )
            baseline = np.asarray(self._dr_base_dof_armature, dtype=np.float64)[
                self._privileged_joint_dof_ids
            ]
            self._privileged_dof_armature_scale[rows] = self._safe_scale_ratio(
                dof_armature[:, self._privileged_joint_dof_ids],
                baseline,
            )

    def _privileged_randomization_obs(
        self,
        num_obs: int,
        env_ids: np.ndarray | None,
    ) -> np.ndarray:
        terms = [
            self._select_env_rows(self._motor_torque_scale, num_obs, env_ids),
            self._select_env_rows(self._default_joint_pos_offset, num_obs, env_ids),
            self._select_env_rows(self._privileged_base_mass_delta, num_obs, env_ids),
            self._select_env_rows(self._privileged_body_mass_scale, num_obs, env_ids),
            self._select_env_rows(self._privileged_base_com_offset, num_obs, env_ids),
            self._select_env_rows(self._privileged_base_inertia_scale, num_obs, env_ids),
            self._select_env_rows(self._privileged_ground_friction_scale, num_obs, env_ids),
            self._select_env_rows(self._privileged_robot_friction_scale, num_obs, env_ids),
            self._select_env_rows(self._privileged_dof_armature_scale, num_obs, env_ids),
        ]
        return np.concatenate(terms, axis=1, dtype=get_global_dtype())

    def _current_lin_vel_x_range(self) -> tuple[float, float]:
        scale = self._command_curriculum_scale if self._cfg.commands.curriculum else 1.0
        low, high = self._base_command_lin_vel_x * scale
        return float(low), float(high)

    def _current_ang_vel_z_range(self) -> tuple[float, float]:
        scale = self._command_curriculum_yaw_scale if self._cfg.commands.curriculum else 1.0
        low, high = self._base_command_ang_vel_z * scale
        return float(low), float(high)

    def sample_reset_standing_mask(self, env_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_ids, dtype=np.int32)
        num_reset = int(rows.size)
        if num_reset <= 0 or self._standing_probability <= 0.0:
            return np.zeros((max(num_reset, 0),), dtype=np.bool_)
        if self._standing_probability >= 1.0:
            return np.ones((num_reset,), dtype=np.bool_)
        return np.asarray(
            np.random.uniform(size=(num_reset,)) < self._standing_probability,
            dtype=np.bool_,
        )

    def set_episode_standing_mask(self, env_ids: np.ndarray, standing_mask: np.ndarray) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        mask = np.asarray(standing_mask, dtype=np.bool_).reshape(-1)
        if mask.shape != (rows.shape[0],):
            raise ValueError(f"standing_mask must have shape ({rows.shape[0]},), got {mask.shape}")
        self._episode_standing_mask[rows] = mask

    def episode_standing_mask(self) -> np.ndarray:
        return np.asarray(
            self._episode_standing_mask[: self._num_envs],
            dtype=np.bool_,
        )

    def sample_reset_zero_hz_wing_angle_obs(
        self,
        zero_hz_mask: np.ndarray,
    ) -> np.ndarray:
        mask = np.asarray(zero_hz_mask, dtype=np.bool_).reshape(-1)
        result = np.zeros((mask.size, _WING_ANGLE_OBS_DIM), dtype=self._np_dtype)
        if not self._wing_angle_obs_enabled or not np.any(mask):
            return result
        low, high = np.asarray(
            self._cfg.wing_angle_obs.standing_normalized_range,
            dtype=np.float64,
        )
        result[mask] = np.random.uniform(
            low,
            high,
            size=(int(np.count_nonzero(mask)), _WING_ANGLE_OBS_DIM),
        ).astype(self._np_dtype)
        return result

    def set_zero_hz_wing_angle_obs(
        self,
        env_ids: np.ndarray,
        wing_angle_obs: np.ndarray,
    ) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        values = np.asarray(wing_angle_obs, dtype=self._np_dtype)
        expected_shape = (rows.size, _WING_ANGLE_OBS_DIM)
        if values.shape != expected_shape:
            raise ValueError(
                f"zero-Hz wing angle obs must have shape {expected_shape}, got {values.shape}"
            )
        self._zero_hz_wing_angle_obs[rows] = values

    def sample_reset_wing_angle_obs_amplitude_scales(self, num_reset: int) -> np.ndarray:
        if num_reset <= 0:
            return np.ones((0,), dtype=self._np_dtype)
        if not self._wing_angle_obs_enabled:
            return np.ones((num_reset,), dtype=self._np_dtype)
        low, high = _wing_angle_obs_csv_amplitude_scale_bounds(self._cfg.wing_angle_obs)
        if high <= low:
            return np.full((num_reset,), low, dtype=self._np_dtype)
        return np.asarray(
            np.random.uniform(low, high, size=(num_reset,)),
            dtype=self._np_dtype,
        )

    def set_wing_angle_obs_amplitude_scales(
        self,
        env_ids: np.ndarray,
        amplitude_scales: np.ndarray,
    ) -> None:
        rows = np.asarray(env_ids, dtype=np.int32)
        values = np.asarray(amplitude_scales, dtype=self._np_dtype).reshape(-1)
        if values.shape != (rows.size,):
            raise ValueError(
                "wing-angle observation amplitude scales must have shape "
                f"({rows.size},), got {values.shape}"
            )
        low, high = _wing_angle_obs_csv_amplitude_scale_bounds(self._cfg.wing_angle_obs)
        tolerance = max(1.0, abs(low), abs(high)) * 1.0e-6
        if (
            not np.all(np.isfinite(values))
            or np.any(values < low - tolerance)
            or np.any(values > high + tolerance)
        ):
            raise ValueError(
                "wing-angle observation amplitude scales must be finite and inside "
                f"the configured [{low}, {high}] range"
            )
        self._wing_angle_obs_amplitude_scales[rows] = values

    def wing_angle_obs_amplitude_scales(self) -> np.ndarray:
        return np.asarray(
            self._wing_angle_obs_amplitude_scales[: self._num_envs],
            dtype=np.float64,
        )

    def sample_commands(
        self,
        num_samples: int,
        *,
        env_ids: np.ndarray | None = None,
        resample_standing: bool = False,
    ) -> np.ndarray:
        standing_mask = None
        if env_ids is not None:
            rows = np.asarray(env_ids, dtype=np.int32)
            if rows.shape != (num_samples,):
                raise ValueError(f"env_ids must have shape ({num_samples},), got {rows.shape}")
            if resample_standing and not self._standing_envs_episode_persistent:
                resampled_mask = self.sample_reset_standing_mask(rows)
                self.set_episode_standing_mask(rows, resampled_mask)
            standing_mask = self._episode_standing_mask[rows]
        return _sample_dr002_commands(
            self._cfg.commands,
            num_samples,
            lin_vel_x_range=self._current_lin_vel_x_range(),
            ang_vel_z_range=self._current_ang_vel_z_range(),
            standing_mask=standing_mask,
        )

    def startup_commands(self, num_samples: int) -> np.ndarray:
        commands = np.zeros((num_samples, 3), dtype=get_global_dtype())
        height = np.asarray(self._cfg.commands.height, dtype=np.float64)
        commands[:, 2] = float(np.mean(height))
        return commands

    def episode_steps(self) -> np.ndarray:
        steps = np.asarray(self._state.info.get("steps"), dtype=np.int64)
        if steps.shape != (self._num_envs,):
            return np.zeros((self._num_envs,), dtype=np.int64)
        return steps

    def _validate_csv_force_curriculum_cfg(self) -> None:
        domain_rand = self._cfg.domain_rand
        self._csv_force_start_delay_step_bounds()
        self._csv_force_amplitude_scale_bounds()
        force_normalization = float(domain_rand.csv_force_observation_force_normalization)
        if not np.isfinite(force_normalization) or force_normalization <= 0.0:
            raise ValueError(
                "domain_rand.csv_force_observation_force_normalization must be positive"
            )
        moment_normalization = float(domain_rand.csv_force_observation_moment_normalization)
        if not np.isfinite(moment_normalization) or moment_normalization <= 0.0:
            raise ValueError(
                "domain_rand.csv_force_observation_moment_normalization must be positive"
            )
        if int(domain_rand.csv_force_curriculum_window_episodes) <= 0:
            raise ValueError("domain_rand.csv_force_curriculum_window_episodes must be positive")
        if int(domain_rand.csv_force_curriculum_step_levels) <= 0:
            raise ValueError("domain_rand.csv_force_curriculum_step_levels must be positive")
        rate_fields = {
            "csv_force_curriculum_promote_fail_rate_max": (
                domain_rand.csv_force_curriculum_promote_fail_rate_max
            ),
            "csv_force_curriculum_demote_fail_rate_min": (
                domain_rand.csv_force_curriculum_demote_fail_rate_min
            ),
            "csv_force_curriculum_promote_mean_episode_length_fraction": (
                domain_rand.csv_force_curriculum_promote_mean_episode_length_fraction
            ),
            "csv_force_curriculum_demote_mean_episode_length_fraction": (
                domain_rand.csv_force_curriculum_demote_mean_episode_length_fraction
            ),
        }
        for name, value in rate_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"domain_rand.{name} must be in [0, 1], got {value}")

    def _validate_noise_curriculum_cfg(self) -> None:
        noise_cfg = self._cfg.noise_config
        gravity_noise_mode = (
            str(getattr(noise_cfg, "gravity_noise_mode", "additive")).strip().lower()
        )
        if gravity_noise_mode not in {"additive", "tilt"}:
            raise ValueError(
                "noise_config.gravity_noise_mode must be 'additive' or 'tilt', "
                f"got {gravity_noise_mode!r}"
            )
        gravity_angle_fields = (
            "gravity_installation_bias_max_deg",
            "gravity_dynamic_noise_max_deg",
        )
        for name in gravity_angle_fields:
            value = float(getattr(noise_cfg, name, 0.0))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"noise_config.{name} must be finite and non-negative")
        dynamic_max_deg = float(getattr(noise_cfg, "gravity_dynamic_noise_max_deg", 0.0))
        dynamic_time_constant_s = float(
            getattr(noise_cfg, "gravity_dynamic_noise_time_constant_s", 3.0)
        )
        if (
            gravity_noise_mode == "tilt"
            and dynamic_max_deg > 0.0
            and (not np.isfinite(dynamic_time_constant_s) or dynamic_time_constant_s <= 0.0)
        ):
            raise ValueError(
                "noise_config.gravity_dynamic_noise_time_constant_s must be finite and "
                "positive when tilt-mode dynamic gravity noise is enabled"
            )

        if not bool(getattr(noise_cfg, "curriculum", False)):
            return
        levels = _noise_curriculum_levels(noise_cfg)
        domain_rand = self._cfg.domain_rand
        if domain_rand.csv_force_enabled and domain_rand.csv_force_curriculum:
            force_num_levels = _csv_force_curriculum_num_levels(domain_rand)
            if levels.size != force_num_levels:
                raise ValueError(
                    "noise curriculum must have exactly one level per CSV force curriculum "
                    f"level, got {levels.size} noise levels and {force_num_levels} force levels"
                )

    def _noise_curriculum_follows_csv_force(self) -> bool:
        domain_rand = self._cfg.domain_rand
        return bool(
            getattr(self._cfg.noise_config, "curriculum", False)
            and domain_rand.csv_force_enabled
            and domain_rand.csv_force_curriculum
        )

    def _current_noise_level(self) -> float:
        noise_cfg = self._cfg.noise_config
        if not bool(getattr(noise_cfg, "curriculum", False)):
            return float(getattr(noise_cfg, "level", 0.0))
        levels = _noise_curriculum_levels(noise_cfg)
        curriculum_level = (
            self._csv_force_curriculum_level
            if self._noise_curriculum_follows_csv_force()
            else self._noise_curriculum_level
        )
        level_index = int(np.clip(curriculum_level, 0, levels.size - 1))
        return float(levels[level_index])

    def _compute_wing_angle_obs(
        self,
        num_obs: int,
        env_ids: np.ndarray | None,
        *,
        preview_next_step: bool,
        episode_steps_override: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.zeros((num_obs, _WING_ANGLE_OBS_DIM), dtype=get_global_dtype())
        if not self._wing_angle_obs_enabled:
            return result

        domain_rand = self._cfg.domain_rand
        angle_cfg = self._cfg.wing_angle_obs
        levels = np.asarray(
            self._select_env_rows(self.csv_force_active_levels(), num_obs, env_ids),
            dtype=np.int32,
        )
        resume_suppressed = np.asarray(
            self._select_env_rows(
                self.csv_force_resume_suppressed(),
                num_obs,
                env_ids,
            ),
            dtype=np.bool_,
        )
        amplitude_scales = np.asarray(
            self._select_env_rows(
                self.wing_angle_obs_amplitude_scales(),
                num_obs,
                env_ids,
            ),
            dtype=np.float64,
        )
        if episode_steps_override is None:
            state = getattr(self, "_state", None)
            if state is None:
                episode_steps = np.zeros((num_obs,), dtype=np.int64)
            else:
                episode_steps = np.asarray(
                    self._select_env_rows(self.episode_steps(), num_obs, env_ids),
                    dtype=np.int64,
                )
        else:
            episode_steps = np.asarray(episode_steps_override, dtype=np.int64).reshape(-1)
            if episode_steps.shape != (num_obs,):
                raise ValueError(
                    "episode_steps_override must match observation rows, "
                    f"got {episode_steps.shape} and ({num_obs},)"
                )
        if preview_next_step:
            episode_steps = episode_steps + 1
        start_delay_steps = np.asarray(
            self._select_env_rows(self.csv_force_start_delay_steps(), num_obs, env_ids),
            dtype=np.int32,
        )
        elapsed_after_startup, replay_t = _csv_replay_clock(
            domain_rand,
            episode_steps,
            self._startup_stand_steps,
            float(self._cfg.ctrl_dt),
            start_delay_steps=start_delay_steps,
        )
        elapsed_s = np.maximum(elapsed_after_startup, 0).astype(np.float64) * float(
            self._cfg.ctrl_dt
        )

        zero_offsets = np.asarray(angle_cfg.zero_offsets_deg, dtype=np.float64)
        normalization_deg = float(angle_cfg.normalization_deg)
        for level in np.unique(levels):
            level_int = int(level)
            source_hz = _csv_force_curriculum_source_hz(domain_rand, level_int)
            force_path = _csv_force_curriculum_path_for_level(domain_rand, level_int)
            angle_path = _wing_angle_path_for_level(domain_rand, angle_cfg, level_int)
            if source_hz <= 0.0 or force_path is None or angle_path is None:
                continue

            if force_path not in self._wing_angle_force_samples_by_path:
                self._wing_angle_force_samples_by_path[force_path] = _load_force_csv(force_path)
            if angle_path not in self._wing_angle_samples_by_path:
                self._wing_angle_samples_by_path[angle_path] = _load_wing_angle_csv(angle_path)
            force_samples = self._wing_angle_force_samples_by_path[force_path]
            angle_samples = self._wing_angle_samples_by_path[angle_path]

            level_rows = levels == level_int
            level_indices = np.flatnonzero(level_rows)
            activity_weight = _csv_replay_activity_weight(
                force_samples,
                replay_t[level_rows],
                elapsed_s[level_rows],
                period=float(domain_rand.csv_force_period),
                transition_seconds=float(domain_rand.csv_force_transition_seconds),
            )
            activity_weight[elapsed_after_startup[level_rows] < 0] = 0.0
            active_local = activity_weight > 0.0
            if not np.any(active_local):
                continue
            active_indices = level_indices[active_local]
            phase_t = np.mod(replay_t[active_indices], 1.0 / source_hz)
            angles_deg = _interp_wing_angle_csv_batch(angle_samples, phase_t)
            relative_deg = angles_deg - zero_offsets[None, :]
            wrapped_deg = np.mod(relative_deg + 180.0, 360.0) - 180.0
            result[active_indices] = np.asarray(
                (wrapped_deg / normalization_deg)
                * activity_weight[active_local, None]
                * amplitude_scales[active_indices, None],
                dtype=get_global_dtype(),
            )
        zero_hz_mask = _csv_force_zero_hz_mask(domain_rand, levels)
        if np.any(zero_hz_mask):
            zero_hz_obs = self._select_env_rows(
                self._zero_hz_wing_angle_obs,
                num_obs,
                env_ids,
            )
            result[zero_hz_mask] = zero_hz_obs[zero_hz_mask]
        result[resume_suppressed] = 0.0
        return result

    @staticmethod
    def _obs_noise_at_level(data: np.ndarray, scale: float, level: float) -> np.ndarray:
        if float(level) <= 0.0 or float(scale) <= 0.0:
            return data
        noise = (
            np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype)
            * float(level)
            * float(scale)
        )
        return data + noise

    @staticmethod
    def _obs_multiplicative_gaussian_noise_at_level(
        data: np.ndarray,
        relative_std: float,
        level: float,
    ) -> np.ndarray:
        if float(level) <= 0.0 or float(relative_std) <= 0.0:
            return data
        relative_noise = (
            np.random.normal(0.0, 1.0, data.shape).astype(data.dtype)
            * float(level)
            * float(relative_std)
        )
        return data * (1.0 + relative_noise)

    def _gravity_noise_mode(self) -> str:
        return (
            str(getattr(self._cfg.noise_config, "gravity_noise_mode", "additive")).strip().lower()
        )

    def _reset_gravity_tilt_noise(self, env_ids: np.ndarray) -> None:
        rows = np.asarray(env_ids, dtype=np.intp)
        if rows.size == 0:
            return
        self._gravity_dynamic_noise_rp[rows] = 0.0
        if self._gravity_noise_mode() != "tilt":
            self._gravity_installation_bias_rp[rows] = 0.0
            return

        noise_level = max(self._current_noise_level(), 0.0)
        max_bias_rad = np.deg2rad(
            float(self._cfg.noise_config.gravity_installation_bias_max_deg) * noise_level
        )
        if max_bias_rad <= 0.0:
            self._gravity_installation_bias_rp[rows] = 0.0
            return
        self._gravity_installation_bias_rp[rows] = np.random.uniform(
            -max_bias_rad,
            max_bias_rad,
            size=(rows.size, 2),
        ).astype(self._np_dtype)

    def _advance_gravity_dynamic_noise(
        self,
        env_ids: np.ndarray,
        noise_level: float,
    ) -> None:
        rows = np.asarray(env_ids, dtype=np.intp)
        if rows.size == 0:
            return
        max_noise_rad = np.deg2rad(
            float(self._cfg.noise_config.gravity_dynamic_noise_max_deg)
            * max(float(noise_level), 0.0)
        )
        if max_noise_rad <= 0.0:
            self._gravity_dynamic_noise_rp[rows] = 0.0
            return

        time_constant_s = float(self._cfg.noise_config.gravity_dynamic_noise_time_constant_s)
        rho = float(np.exp(-float(self._cfg.ctrl_dt) / time_constant_s))
        # Treat the configured bound as three stationary standard deviations,
        # then clip exactly so every roll/pitch sample remains inside it.
        stationary_std = max_noise_rad / 3.0
        innovation_std = stationary_std * np.sqrt(max(1.0 - rho * rho, 0.0))
        standard_normal = np.random.normal(
            0.0,
            1.0,
            size=(rows.size, 2),
        )
        innovation = (np.clip(standard_normal, -3.0, 3.0) * innovation_std).astype(self._np_dtype)
        next_noise = rho * self._gravity_dynamic_noise_rp[rows] + innovation
        self._gravity_dynamic_noise_rp[rows] = np.clip(
            next_noise,
            -max_noise_rad,
            max_noise_rad,
        )

    @staticmethod
    def _rotate_and_normalize_projected_gravity(
        projected_gravity: np.ndarray,
        roll_pitch_rad: np.ndarray,
    ) -> np.ndarray:
        gravity = np.asarray(projected_gravity)
        roll_pitch = np.asarray(roll_pitch_rad)
        if gravity.ndim != 2 or gravity.shape[1] != 3:
            raise ValueError(f"projected_gravity must have shape (N, 3), got {gravity.shape}")
        if roll_pitch.shape != (gravity.shape[0], 2):
            raise ValueError(
                "roll_pitch_rad must have shape (N, 2), "
                f"got {roll_pitch.shape} for {gravity.shape[0]} gravity rows"
            )

        work = np.asarray(gravity, dtype=np.float64)
        norm = np.linalg.norm(work, axis=1, keepdims=True)
        unit = np.divide(work, norm, out=np.zeros_like(work), where=norm > 1.0e-12)
        roll = np.asarray(roll_pitch[:, 0], dtype=np.float64)
        pitch = np.asarray(roll_pitch[:, 1], dtype=np.float64)
        sin_roll, cos_roll = np.sin(roll), np.cos(roll)
        sin_pitch, cos_pitch = np.sin(pitch), np.cos(pitch)

        # Sensor-frame tilt: first rotate about X (roll), then about Y (pitch).
        x_roll = unit[:, 0]
        y_roll = cos_roll * unit[:, 1] - sin_roll * unit[:, 2]
        z_roll = sin_roll * unit[:, 1] + cos_roll * unit[:, 2]
        rotated = np.stack(
            (
                cos_pitch * x_roll + sin_pitch * z_roll,
                y_roll,
                -sin_pitch * x_roll + cos_pitch * z_roll,
            ),
            axis=1,
        )
        rotated_norm = np.linalg.norm(rotated, axis=1, keepdims=True)
        normalized = np.divide(
            rotated,
            rotated_norm,
            out=np.zeros_like(rotated),
            where=rotated_norm > 1.0e-12,
        )
        return np.asarray(normalized, dtype=gravity.dtype)

    def _actor_projected_gravity(
        self,
        projected_gravity: np.ndarray,
        *,
        num_obs: int,
        env_ids: np.ndarray | None,
        noise_level: float,
        advance_dynamic: bool,
    ) -> np.ndarray:
        if self._gravity_noise_mode() != "tilt":
            return self._obs_noise_at_level(
                projected_gravity,
                self._cfg.noise_config.scale_gravity,
                noise_level,
            )

        rows = (
            np.arange(num_obs, dtype=np.intp)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.intp)
        )
        if advance_dynamic:
            self._advance_gravity_dynamic_noise(rows, noise_level)
        roll_pitch = self._gravity_installation_bias_rp[rows] + self._gravity_dynamic_noise_rp[rows]
        return self._rotate_and_normalize_projected_gravity(
            projected_gravity,
            roll_pitch,
        )

    def _clip_policy_actions(self, actions: np.ndarray) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(actions, -self._leg_clip_actions, self._leg_clip_actions),
            dtype=self._np_dtype,
        )
        clipped_actions[:, WHEEL_ACTION_INDICES] = np.clip(
            actions[:, WHEEL_ACTION_INDICES],
            -self._wheel_clip_actions,
            self._wheel_clip_actions,
        )
        return clipped_actions

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = self._clip_policy_actions(actions)
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions

        ctrl = np.zeros((clipped_actions.shape[0], NUM_DR002_ACTIONS), dtype=self._np_dtype)
        num_actions = clipped_actions.shape[0]
        ctrl[:, LEG_ACTION_INDICES] = (
            clipped_actions[:, LEG_ACTION_INDICES] * self._cfg.control_config.action_scale
            + self.default_angles[LEG_ACTION_INDICES]
            + self._default_joint_pos_offset[:num_actions, LEG_ACTION_INDICES]
        )
        ctrl[:, WHEEL_ACTION_INDICES] = (
            clipped_actions[:, WHEEL_ACTION_INDICES] * self._cfg.control_config.wheel_action_scale
        )
        return ctrl

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        if self._motor_control_substep_index == 0:
            delayed_policy_ctrl = self._delayed_policy_ctrl(policy_ctrl)
            joint_pos = stack_joint_sensors(backend, "pos", dtype=self.default_angles.dtype)
            joint_vel = stack_joint_sensors(backend, "vel", dtype=self.default_angles.dtype)
            compute_dr002_motor_ctrl(
                delayed_policy_ctrl,
                joint_pos,
                joint_vel,
                self._motor_kp,
                self._motor_kd,
                self._ctrl_lower,
                self._ctrl_upper,
                self._last_motor_ctrl,
                self._motor_torque_scale,
            )
        self._motor_control_substep_index = (
            self._motor_control_substep_index + 1
        ) % self._motor_control_decimation
        return self._last_motor_ctrl

    def _batched_motor_control(
        self, backend: Any, policy_ctrl: np.ndarray, nsteps: int
    ) -> BatchedMixedPdControl:
        if nsteps < 1:
            raise ValueError("DR002 batched motor control requires at least one physics substep")
        joint_pos = stack_joint_sensors(backend, "pos", dtype=self.default_angles.dtype)
        joint_vel = stack_joint_sensors(backend, "vel", dtype=self.default_angles.dtype)
        trajectory_shape = (policy_ctrl.shape[0], int(nsteps), NUM_DR002_ACTIONS)
        target_trajectory = getattr(self, "_batched_policy_ctrl_trajectory", None)
        if target_trajectory is None or target_trajectory.shape != trajectory_shape:
            target_trajectory = np.empty(trajectory_shape, dtype=self._np_dtype)
            self._batched_policy_ctrl_trajectory = target_trajectory
        for substep in range(int(nsteps)):
            target_trajectory[:, substep] = self._delayed_policy_ctrl(policy_ctrl)
        return BatchedMixedPdControl(
            target_trajectory=target_trajectory,
            kp=self._motor_kp,
            kd=self._motor_kd,
            torque_scale=self._motor_torque_scale,
            initial_joint_pos=joint_pos,
            initial_joint_vel=joint_vel,
            position_control_mask=_POSITION_CONTROL_MASK,
            position_sensor_names=_JOINT_POS_SENSOR_NAMES,
            velocity_sensor_names=_JOINT_VEL_SENSOR_NAMES,
            ctrl_lower=self._ctrl_lower,
            ctrl_upper=self._ctrl_upper,
            final_ctrl_out=self._last_motor_ctrl,
        )

    def get_projected_gravity(self) -> np.ndarray:
        """Return standard body-frame gravity: R^T * [0, 0, -1]."""
        x_axis = self._backend.get_sensor_data(self._cfg.sensor.projected_gravity_x)
        y_axis = self._backend.get_sensor_data(self._cfg.sensor.projected_gravity_y)
        z_axis = self._backend.get_sensor_data(self._cfg.sensor.projected_gravity_z)
        projected = np.stack(
            [
                -x_axis[:, 2],
                -y_axis[:, 2],
                -z_axis[:, 2],
            ],
            axis=1,
        )
        return projected.astype(get_global_dtype(), copy=False)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        projected_gravity = self.get_projected_gravity()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._last_motor_ctrl.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._update_standing_step_stats(linvel, gyro)
        self._accumulate_command_curriculum(state.info, linvel)
        self._log_command_curriculum(state.info)
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, projected_gravity, dof_pos, dof_vel
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        gravity_failed_now = gravity[:, 2] <= self._reward_cfg.termination_gravity_z_threshold
        contact_failed_now = self._has_undesired_contact(
            gravity.shape[0], threshold=self._reward_cfg.termination_contact_threshold
        )
        self._lingzu_fail_steps = np.where(gravity_failed_now, self._lingzu_fail_steps + 1, 0)
        # Contact failure, like gravity failure, requires an uninterrupted streak.
        # A contact-free control step clears only the contact counter.
        self._lingzu_contact_fail_accum_steps = np.where(
            contact_failed_now,
            self._lingzu_contact_fail_accum_steps + 1,
            0,
        )
        fail_steps = max(
            int(round(self._reward_cfg.termination_fail_time_s / self._cfg.ctrl_dt)), 1
        )
        contact_fail_steps = max(int(self._reward_cfg.termination_contact_fail_steps), 1)
        gravity_done = self._lingzu_fail_steps >= fail_steps
        contact_done = self._lingzu_contact_fail_accum_steps >= contact_fail_steps
        self._last_termination_contact_now_fraction = float(np.mean(contact_failed_now))
        self._last_termination_contact_accum_mean = float(
            np.mean(self._lingzu_contact_fail_accum_steps)
        )
        self._last_termination_contact_accum_max = float(
            np.max(self._lingzu_contact_fail_accum_steps)
        )
        self._last_termination_contact_done_fraction = float(np.mean(contact_done))
        self._last_termination_gravity_done_fraction = float(np.mean(gravity_done))
        return gravity_done | contact_done

    def _undesired_contact_values(self, num_envs: int) -> np.ndarray:
        contacts = []
        for name in self._cfg.sensor.undesired_contacts:
            sensor = np.asarray(self._backend.get_sensor_data(name)).reshape(num_envs, -1)[:, 0]
            contacts.append(sensor)
        if not contacts:
            return np.zeros((num_envs, 0), dtype=get_global_dtype())
        return np.asarray(np.stack(contacts, axis=1), dtype=get_global_dtype())

    def _has_undesired_contact(self, num_envs: int, *, threshold: float) -> np.ndarray:
        contacts = self._undesired_contact_values(num_envs)
        if contacts.shape[1] == 0:
            return np.zeros((num_envs,), dtype=np.bool_)
        return np.any(contacts > float(threshold), axis=1)

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        interval = max(int(round(float(self._cfg.commands.resampling_time) / self._cfg.ctrl_dt)), 1)
        steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
        startup_steps = int(self._startup_stand_steps)
        if startup_steps > 0:
            startup_mask = steps < startup_steps
            if np.any(startup_mask):
                commands_arr[startup_mask] = self.startup_commands(
                    int(np.count_nonzero(startup_mask))
                )
            elapsed_after_startup = steps.astype(np.int64) - startup_steps
            resample_mask = (steps == startup_steps) | (
                (steps > startup_steps) & ((elapsed_after_startup % interval) == 0)
            )
        else:
            resample_mask = (steps > 0) & ((steps % interval) == 0)
        if np.any(resample_mask):
            env_ids = np.flatnonzero(resample_mask).astype(np.int32)
            commands_arr[resample_mask] = self.sample_commands(
                len(env_ids),
                env_ids=env_ids,
                resample_standing=True,
            )
        info["commands"] = commands_arr

    def _accumulate_command_curriculum(self, info: dict, linvel: np.ndarray) -> None:
        if not self._cfg.commands.curriculum:
            return
        commands = np.asarray(info.get("commands"), dtype=get_global_dtype())
        if commands.shape[0] != linvel.shape[0]:
            return
        lin_error = np.square(commands[:, 0] - linvel[:, 0])
        lin_tracking = np.exp(-lin_error / (self._reward_cfg.tracking_sigma**2))
        configured_std = self._reward_cfg.track_lin_vel_x_std
        force_tracking_std = (
            self._reward_cfg.tracking_sigma
            if configured_std is None
            else max(float(configured_std), 1.0e-6)
        )
        force_lin_tracking = np.exp(-lin_error / (force_tracking_std**2))
        try:
            gyro = self.get_gyro()
        except Exception:
            gyro = None
        if isinstance(gyro, np.ndarray) and gyro.shape[0] == commands.shape[0]:
            yaw_error = np.square(commands[:, 1] - gyro[:, 2])
            yaw_tracking = np.exp(-yaw_error / (self._reward_cfg.tracking_sigma**2))
        else:
            yaw_tracking = lin_tracking
        num_envs = lin_tracking.shape[0]
        self._episode_track_total_steps[:num_envs] += 1
        threshold = max(float(self._cfg.commands.curriculum_moving_command_threshold), 0.0)
        moving = ~self._episode_standing_mask[:num_envs] & (
            (np.abs(commands[:, 0]) > threshold) | (np.abs(commands[:, 1]) > threshold)
        )
        if not np.any(moving):
            return
        env_rows = np.nonzero(moving)[0]
        self._episode_track_lin_vel_x_sum[env_rows] += lin_tracking[env_rows]
        self._episode_track_ang_vel_z_sum[env_rows] += yaw_tracking[env_rows]
        self._episode_force_track_lin_vel_x_sum[env_rows] += force_lin_tracking[env_rows]
        self._episode_track_lin_vel_x_steps[env_rows] += 1

    def _update_standing_step_stats(self, linvel: np.ndarray, gyro: np.ndarray) -> None:
        num_envs = linvel.shape[0]
        standing = self._episode_standing_mask[:num_envs]
        self._last_standing_fraction = float(np.mean(standing)) if num_envs > 0 else 0.0
        if not np.any(standing):
            self._last_standing_mean_abs_vx = np.nan
            self._last_standing_mean_abs_vy = np.nan
            self._last_standing_mean_abs_yaw_rate = np.nan
            return
        self._last_standing_mean_abs_vx = float(np.mean(np.abs(linvel[standing, 0])))
        self._last_standing_mean_abs_vy = float(np.mean(np.abs(linvel[standing, 1])))
        self._last_standing_mean_abs_yaw_rate = float(np.mean(np.abs(gyro[standing, 2])))

    def _standing_window_target_episodes(self) -> int:
        base_window = int(self._cfg.domain_rand.csv_force_curriculum_window_episodes)
        return max(int(round(base_window * self._standing_probability)), 1)

    def _reset_standing_episode_window(self) -> None:
        self._standing_window_episodes = 0
        self._standing_window_fail_count = 0
        self._standing_window_episode_length_fraction_sum = 0.0

    def _update_standing_episode_stats(self, env_ids: np.ndarray) -> None:
        if self._state is None or env_ids.size == 0 or self._standing_probability <= 0.0:
            return
        standing = self._episode_standing_mask[env_ids]
        if not np.any(standing):
            return
        total_steps = self._episode_alive_steps[env_ids]
        done = np.asarray(
            self._state.terminated[env_ids] | self._state.truncated[env_ids],
            dtype=np.bool_,
        )
        valid = standing & (total_steps > 0) & done
        if not np.any(valid):
            return

        valid_steps = total_steps[valid].astype(np.float64)
        failed = np.asarray(self._state.terminated[env_ids], dtype=np.bool_)[valid]
        max_steps = int(self._cfg.max_episode_steps or 0)
        length_fraction = (
            np.clip(valid_steps / float(max_steps), 0.0, 1.0)
            if max_steps > 0
            else np.ones_like(valid_steps)
        )
        self._standing_window_episodes += int(valid_steps.size)
        self._standing_window_fail_count += int(np.count_nonzero(failed))
        self._standing_window_episode_length_fraction_sum += float(np.sum(length_fraction))

        if self._standing_window_episodes < self._standing_window_target_episodes():
            return
        episodes = max(self._standing_window_episodes, 1)
        self._last_standing_window_fail_rate = self._standing_window_fail_count / float(episodes)
        self._last_standing_window_mean_episode_length_fraction = (
            self._standing_window_episode_length_fraction_sum / float(episodes)
        )
        self._reset_standing_episode_window()

    def _alive_values(self, num_envs: int) -> np.ndarray:
        fail_steps = max(
            int(round(self._reward_cfg.termination_fail_time_s / self._cfg.ctrl_dt)), 1
        )
        contact_fail_steps = max(int(self._reward_cfg.termination_contact_fail_steps), 1)
        alive = (self._lingzu_fail_steps[:num_envs] < fail_steps) & (
            self._lingzu_contact_fail_accum_steps[:num_envs] < contact_fail_steps
        )
        return np.asarray(alive, dtype=np.bool_)

    def _accumulate_alive_episode(self, num_envs: int) -> None:
        if not hasattr(self, "_episode_alive_sum"):
            return
        alive = self._alive_values(num_envs).astype(np.float64)
        self._episode_alive_sum[:num_envs] += alive
        self._episode_alive_steps[:num_envs] += 1

    def _update_episode_alive_stats(self, env_ids: np.ndarray) -> None:
        if env_ids.size == 0:
            return
        steps = self._episode_alive_steps[env_ids]
        done = np.ones_like(steps, dtype=np.bool_)
        if self._state is not None:
            done = np.asarray(
                self._state.terminated[env_ids] | self._state.truncated[env_ids], dtype=np.bool_
            )
        valid = (steps > 0) & done
        if not np.any(valid):
            return
        alive_steps = self._episode_alive_sum[env_ids][valid]
        episode_steps = steps[valid].astype(np.float64)
        scale = float(self._reward_cfg.scales.get("alive", 0.0))
        max_steps = int(self._cfg.max_episode_steps or 0)
        fraction_denominator = (
            np.full_like(episode_steps, float(max_steps))
            if max_steps > 0
            else np.maximum(episode_steps, 1.0)
        )
        self._last_episode_alive_return = float(np.mean(alive_steps * scale * self._cfg.ctrl_dt))
        self._last_episode_alive_fraction = float(np.mean(alive_steps / fraction_denominator))
        self._last_episode_alive_steps = float(np.mean(alive_steps))
        self._last_episode_alive_episode_steps = float(np.mean(episode_steps))

    def _log_alive_reward(self, info: dict, num_envs: int) -> None:
        if not self._enable_reward_log:
            return
        step_count = info.get("steps", np.zeros((num_envs,), dtype=np.uint32))
        if int(step_count[0]) % 4 != 0:
            return
        log = info.setdefault("log", {})
        if np.isfinite(self._last_episode_alive_return):
            log["reward/alive"] = float(self._last_episode_alive_return)
            log["reward/alive_fraction"] = float(self._last_episode_alive_fraction)
            log["reward/alive_steps"] = float(self._last_episode_alive_steps)
            log["reward/alive_episode_steps"] = float(self._last_episode_alive_episode_steps)
            return

        scale = float(self._reward_cfg.scales.get("alive", 0.0))
        alive_steps = self._episode_alive_sum[:num_envs]
        episode_steps = self._episode_alive_steps[:num_envs].astype(np.float64)
        max_steps = int(self._cfg.max_episode_steps or 0)
        fraction_denominator = (
            np.full_like(episode_steps, float(max_steps))
            if max_steps > 0
            else np.maximum(episode_steps, 1.0)
        )
        log["reward/alive"] = float(np.mean(alive_steps * scale * self._cfg.ctrl_dt))
        log["reward/alive_fraction"] = float(np.mean(alive_steps / fraction_denominator))
        log["reward/alive_steps"] = float(np.mean(alive_steps))
        log["reward/alive_episode_steps"] = float(np.mean(episode_steps))

    def _update_command_curriculum(self, env_ids: np.ndarray) -> None:
        if env_ids.size == 0:
            return
        resume_pending = getattr(self, "_csv_force_resume_pending_reset", None)
        if isinstance(resume_pending, np.ndarray):
            skip_transition_episode = resume_pending[env_ids]
            if np.any(skip_transition_episode):
                resume_pending[env_ids[skip_transition_episode]] = False
                env_ids = env_ids[~skip_transition_episode]
                if env_ids.size == 0:
                    return
        moving_steps = self._episode_track_lin_vel_x_steps[env_ids]
        total_steps = self._episode_track_total_steps[env_ids]
        done = np.ones_like(total_steps, dtype=np.bool_)
        failed = np.zeros_like(total_steps, dtype=np.bool_)
        if self._state is not None:
            done = np.asarray(
                self._state.terminated[env_ids] | self._state.truncated[env_ids], dtype=np.bool_
            )
            failed = np.asarray(self._state.terminated[env_ids], dtype=np.bool_)
        episode_valid = (total_steps > 0) & done
        if not np.any(episode_valid):
            return
        valid_total_steps = total_steps[episode_valid]
        valid_moving_steps = moving_steps[episode_valid]
        failed = failed[episode_valid]
        tracking_mask = valid_moving_steps > 0
        tracking = np.zeros((valid_total_steps.size,), dtype=np.float64)
        force_tracking = np.zeros((valid_total_steps.size,), dtype=np.float64)
        if np.any(tracking_mask):
            lin_tracking = self._episode_track_lin_vel_x_sum[env_ids][episode_valid][tracking_mask]
            yaw_tracking = self._episode_track_ang_vel_z_sum[env_ids][episode_valid][tracking_mask]
            force_lin_tracking = self._episode_force_track_lin_vel_x_sum[env_ids][episode_valid][
                tracking_mask
            ]
            moving_denominator = np.maximum(valid_moving_steps[tracking_mask], 1)
            tracking[tracking_mask] = 0.5 * (
                lin_tracking / moving_denominator + yaw_tracking / moving_denominator
            )
            force_tracking[tracking_mask] = force_lin_tracking / moving_denominator

        min_fraction = float(self._cfg.commands.curriculum_min_episode_fraction)
        if min_fraction < 0.0:
            raise ValueError("commands.curriculum_min_episode_fraction must be non-negative")
        max_steps = self._cfg.max_episode_steps or 0
        min_episode_steps = int(np.ceil(float(max_steps) * min_fraction)) if max_steps else 0
        mature = valid_total_steps >= min_episode_steps
        mature_tracking_mask = mature & tracking_mask
        mean_tracking = np.nan
        mature_mean_tracking = np.nan
        if np.any(tracking_mask):
            mean_tracking = float(np.mean(tracking[tracking_mask]))
            self._last_command_curriculum_mean_tracking = mean_tracking
            self._last_command_curriculum_mean_episode_steps = float(
                np.mean(valid_total_steps[tracking_mask])
            )
            self._last_command_curriculum_mean_moving_steps = float(
                np.mean(valid_moving_steps[tracking_mask])
            )
            self._last_command_curriculum_mature_fraction = float(np.mean(mature[tracking_mask]))
            if np.any(mature_tracking_mask):
                mature_mean_tracking = float(np.mean(tracking[mature_tracking_mask]))
            self._last_command_curriculum_mature_mean_tracking = mature_mean_tracking

        if not self._command_curriculum_is_full():
            self._csv_force_curriculum_level = 0
            self._csv_force_active_levels.fill(0)
            self._reset_csv_force_curriculum_window()
            self._noise_curriculum_level = 0
            self._reset_noise_curriculum_window()
            if not np.isfinite(mean_tracking):
                return
            if bool(self._cfg.commands.curriculum_allow_demotion) and mean_tracking < float(
                self._cfg.commands.curriculum_demote_threshold
            ):
                self._update_command_velocity_curriculum(promote=False)
            elif np.isfinite(mature_mean_tracking) and mature_mean_tracking > float(
                self._cfg.commands.curriculum_threshold
            ):
                self._update_command_velocity_curriculum(promote=True)
            return

        self._update_csv_force_curriculum_from_window(
            tracking=force_tracking,
            tracking_mask=tracking_mask,
            valid_steps=valid_total_steps,
            mature=mature,
            failed=failed,
        )
        if not self._noise_curriculum_follows_csv_force():
            self._update_noise_curriculum_from_window(
                tracking=tracking,
                tracking_mask=tracking_mask,
                valid_steps=valid_total_steps,
                mature=mature,
                failed=failed,
            )

    def _reset_csv_force_curriculum_window(self) -> None:
        self._csv_force_window_episodes = 0
        self._csv_force_window_tracking_episodes = 0
        self._csv_force_window_tracking_sum = 0.0
        self._csv_force_window_episode_length_fraction_sum = 0.0
        self._csv_force_window_mature_count = 0
        self._csv_force_window_fail_count = 0

    def _update_csv_force_curriculum_from_window(
        self,
        *,
        tracking: np.ndarray,
        tracking_mask: np.ndarray,
        valid_steps: np.ndarray,
        mature: np.ndarray,
        failed: np.ndarray,
    ) -> None:
        domain_rand = self._cfg.domain_rand
        if not domain_rand.csv_force_enabled or not domain_rand.csv_force_curriculum:
            return
        num_episodes = int(valid_steps.shape[0])
        if num_episodes <= 0:
            return
        max_steps = int(self._cfg.max_episode_steps or 0)
        episode_length_fraction = (
            valid_steps.astype(np.float64) / float(max_steps)
            if max_steps > 0
            else np.ones_like(valid_steps, dtype=np.float64)
        )
        episode_length_fraction = np.clip(episode_length_fraction, 0.0, 1.0)
        self._csv_force_window_episodes += num_episodes
        self._csv_force_window_tracking_episodes += int(np.count_nonzero(tracking_mask))
        self._csv_force_window_tracking_sum += float(np.sum(tracking[tracking_mask]))
        self._csv_force_window_episode_length_fraction_sum += float(np.sum(episode_length_fraction))
        self._csv_force_window_mature_count += int(np.count_nonzero(mature))
        self._csv_force_window_fail_count += int(np.count_nonzero(failed))

        window_target = int(domain_rand.csv_force_curriculum_window_episodes)
        if self._csv_force_window_episodes < window_target:
            return

        window_episodes = max(self._csv_force_window_episodes, 1)
        window_tracking = (
            self._csv_force_window_tracking_sum / float(self._csv_force_window_tracking_episodes)
            if self._csv_force_window_tracking_episodes > 0
            else np.nan
        )
        window_fail_rate = self._csv_force_window_fail_count / float(window_episodes)
        window_mean_length_fraction = self._csv_force_window_episode_length_fraction_sum / float(
            window_episodes
        )
        window_mature_fraction = self._csv_force_window_mature_count / float(window_episodes)
        self._last_csv_force_window_tracking = float(window_tracking)
        self._last_csv_force_window_fail_rate = float(window_fail_rate)
        self._last_csv_force_window_mean_episode_length_fraction = float(
            window_mean_length_fraction
        )
        self._last_csv_force_window_mature_fraction = float(window_mature_fraction)

        should_promote = (
            np.isfinite(window_tracking)
            and window_tracking > float(domain_rand.csv_force_curriculum_promote_tracking_threshold)
            and window_fail_rate < float(domain_rand.csv_force_curriculum_promote_fail_rate_max)
            and window_mean_length_fraction
            >= float(domain_rand.csv_force_curriculum_promote_mean_episode_length_fraction)
        )
        should_demote = bool(domain_rand.csv_force_curriculum_allow_demotion) and (
            (
                np.isfinite(window_tracking)
                and window_tracking
                <= float(domain_rand.csv_force_curriculum_demote_tracking_threshold)
            )
            or window_fail_rate >= float(domain_rand.csv_force_curriculum_demote_fail_rate_min)
            or window_mean_length_fraction
            <= float(domain_rand.csv_force_curriculum_demote_mean_episode_length_fraction)
        )
        if should_promote:
            self._update_csv_force_curriculum(promote=True)
        elif should_demote:
            self._update_csv_force_curriculum(promote=False)
        else:
            self._csv_force_curriculum_last_direction = 0
        self._reset_csv_force_curriculum_window()

    def _reset_noise_curriculum_window(self) -> None:
        self._noise_window_episodes = 0
        self._noise_window_tracking_episodes = 0
        self._noise_window_tracking_sum = 0.0
        self._noise_window_episode_length_fraction_sum = 0.0
        self._noise_window_mature_count = 0
        self._noise_window_fail_count = 0

    def _update_noise_curriculum_from_window(
        self,
        *,
        tracking: np.ndarray,
        tracking_mask: np.ndarray,
        valid_steps: np.ndarray,
        mature: np.ndarray,
        failed: np.ndarray,
    ) -> None:
        noise_cfg = self._cfg.noise_config
        if not bool(getattr(noise_cfg, "curriculum", False)):
            return
        num_episodes = int(valid_steps.shape[0])
        if num_episodes <= 0:
            return
        max_steps = int(self._cfg.max_episode_steps or 0)
        episode_length_fraction = (
            valid_steps.astype(np.float64) / float(max_steps)
            if max_steps > 0
            else np.ones_like(valid_steps, dtype=np.float64)
        )
        episode_length_fraction = np.clip(episode_length_fraction, 0.0, 1.0)
        self._noise_window_episodes += num_episodes
        self._noise_window_tracking_episodes += int(np.count_nonzero(tracking_mask))
        self._noise_window_tracking_sum += float(np.sum(tracking[tracking_mask]))
        self._noise_window_episode_length_fraction_sum += float(np.sum(episode_length_fraction))
        self._noise_window_mature_count += int(np.count_nonzero(mature))
        self._noise_window_fail_count += int(np.count_nonzero(failed))

        domain_rand = self._cfg.domain_rand
        window_target = int(domain_rand.csv_force_curriculum_window_episodes)
        if self._noise_window_episodes < window_target:
            return

        window_episodes = max(self._noise_window_episodes, 1)
        window_tracking = (
            self._noise_window_tracking_sum / float(self._noise_window_tracking_episodes)
            if self._noise_window_tracking_episodes > 0
            else np.nan
        )
        window_fail_rate = self._noise_window_fail_count / float(window_episodes)
        window_mean_length_fraction = self._noise_window_episode_length_fraction_sum / float(
            window_episodes
        )
        window_mature_fraction = self._noise_window_mature_count / float(window_episodes)
        self._last_noise_window_tracking = float(window_tracking)
        self._last_noise_window_fail_rate = float(window_fail_rate)
        self._last_noise_window_mean_episode_length_fraction = float(window_mean_length_fraction)
        self._last_noise_window_mature_fraction = float(window_mature_fraction)

        should_promote = (
            np.isfinite(window_tracking)
            and window_tracking > float(domain_rand.csv_force_curriculum_promote_tracking_threshold)
            and window_fail_rate < float(domain_rand.csv_force_curriculum_promote_fail_rate_max)
            and window_mean_length_fraction
            >= float(domain_rand.csv_force_curriculum_promote_mean_episode_length_fraction)
        )
        should_demote = bool(domain_rand.csv_force_curriculum_allow_demotion) and (
            (
                np.isfinite(window_tracking)
                and window_tracking
                <= float(domain_rand.csv_force_curriculum_demote_tracking_threshold)
            )
            or window_fail_rate >= float(domain_rand.csv_force_curriculum_demote_fail_rate_min)
            or window_mean_length_fraction
            <= float(domain_rand.csv_force_curriculum_demote_mean_episode_length_fraction)
        )
        if should_promote:
            self._update_noise_curriculum(promote=True)
        elif should_demote:
            self._update_noise_curriculum(promote=False)
        else:
            self._noise_curriculum_last_direction = 0
        self._reset_noise_curriculum_window()

    def _command_curriculum_is_full(self) -> bool:
        if not self._cfg.commands.curriculum:
            return True
        eps = 1.0e-9
        lin_full = self._curriculum_value_reached(
            self._command_curriculum_scale,
            self._command_curriculum_initial_scale,
            self._command_curriculum_final_scale,
            eps=eps,
        )
        yaw_full = self._curriculum_value_reached(
            self._command_curriculum_yaw_scale,
            self._command_curriculum_yaw_initial_scale,
            self._command_curriculum_yaw_final_scale,
            eps=eps,
        )
        return lin_full and yaw_full

    @staticmethod
    def _curriculum_value_reached(
        current: float, initial: float, final: float, *, eps: float
    ) -> bool:
        if final >= initial:
            return current >= final - eps
        return current <= final + eps

    def _csv_force_curriculum_is_above_initial(self) -> bool:
        return int(self._csv_force_curriculum_level) > 0

    def _update_command_velocity_curriculum(self, *, promote: bool) -> None:
        step = float(self._cfg.commands.curriculum_step)
        if promote:
            self._command_curriculum_scale = min(
                self._command_curriculum_final_scale,
                self._command_curriculum_scale + step,
            )
            self._command_curriculum_yaw_scale = min(
                self._command_curriculum_yaw_final_scale,
                self._command_curriculum_yaw_scale + step,
            )
        else:
            self._command_curriculum_scale = max(
                self._command_curriculum_initial_scale,
                self._command_curriculum_scale - step,
            )
            self._command_curriculum_yaw_scale = max(
                self._command_curriculum_yaw_initial_scale,
                self._command_curriculum_yaw_scale - step,
            )

    def _update_csv_force_curriculum(self, *, promote: bool) -> None:
        domain_rand = self._cfg.domain_rand
        if not domain_rand.csv_force_enabled or not domain_rand.csv_force_curriculum:
            return
        max_level = max(_csv_force_curriculum_num_levels(domain_rand) - 1, 0)
        old_level = int(np.clip(self._csv_force_curriculum_level, 0, max_level))
        step = int(domain_rand.csv_force_curriculum_step_levels)
        if promote:
            new_level = min(max_level, old_level + step)
        else:
            new_level = max(0, old_level - step)
        self._csv_force_curriculum_level = int(new_level)
        if self._noise_curriculum_follows_csv_force():
            old_noise_level = int(self._noise_curriculum_level)
            self._noise_curriculum_level = int(new_level)
            if new_level > old_noise_level:
                self._noise_curriculum_num_promoted += 1
                self._noise_curriculum_last_direction = 1
            elif new_level < old_noise_level:
                self._noise_curriculum_num_demoted += 1
                self._noise_curriculum_last_direction = -1
            else:
                self._noise_curriculum_last_direction = 0
            self._reset_noise_curriculum_window()
        if new_level < old_level:
            self._csv_force_active_levels = np.minimum(
                self._csv_force_active_levels,
                int(new_level),
            ).astype(np.int32)
        if new_level > old_level:
            self._csv_force_curriculum_num_promoted += 1
            self._csv_force_curriculum_last_direction = 1
        elif new_level < old_level:
            self._csv_force_curriculum_num_demoted += 1
            self._csv_force_curriculum_last_direction = -1
        else:
            self._csv_force_curriculum_last_direction = 0

    def _update_noise_curriculum(self, *, promote: bool) -> None:
        noise_cfg = self._cfg.noise_config
        if not bool(getattr(noise_cfg, "curriculum", False)):
            return
        levels = _noise_curriculum_levels(noise_cfg)
        max_level = int(levels.size - 1)
        old_level = int(np.clip(self._noise_curriculum_level, 0, max_level))
        step = int(self._cfg.domain_rand.csv_force_curriculum_step_levels)
        if promote:
            new_level = min(max_level, old_level + step)
        else:
            new_level = max(0, old_level - step)
        self._noise_curriculum_level = int(new_level)
        if new_level > old_level:
            self._noise_curriculum_num_promoted += 1
            self._noise_curriculum_last_direction = 1
        elif new_level < old_level:
            self._noise_curriculum_num_demoted += 1
            self._noise_curriculum_last_direction = -1
        else:
            self._noise_curriculum_last_direction = 0

    def _log_noise_curriculum(self, log: dict) -> None:
        noise_cfg = self._cfg.noise_config
        is_curriculum = bool(getattr(noise_cfg, "curriculum", False))
        level = self._current_noise_level()
        log["noise_curriculum/enabled"] = float(is_curriculum)
        log["noise_curriculum/level"] = float(level)
        if not is_curriculum:
            return
        levels = _noise_curriculum_levels(noise_cfg)
        curriculum_level = (
            self._csv_force_curriculum_level
            if self._noise_curriculum_follows_csv_force()
            else self._noise_curriculum_level
        )
        level_index = int(np.clip(curriculum_level, 0, levels.size - 1))
        final_level_index = int(levels.size - 1)
        progress = float(level_index / final_level_index) if final_level_index > 0 else 1.0
        domain_rand = self._cfg.domain_rand
        log["noise_curriculum/level_index"] = float(level_index)
        log["noise_curriculum/final_level_index"] = float(final_level_index)
        log["noise_curriculum/progress"] = progress
        log["noise_curriculum/num_levels"] = float(levels.size)
        log["noise_curriculum/command_ready"] = float(self._command_curriculum_is_full())
        log["noise_curriculum/window_episodes"] = float(self._noise_window_episodes)
        log["noise_curriculum/window_tracking_episodes"] = float(
            self._noise_window_tracking_episodes
        )
        log["noise_curriculum/window_target_episodes"] = float(
            domain_rand.csv_force_curriculum_window_episodes
        )
        log["noise_curriculum/num_promoted"] = float(self._noise_curriculum_num_promoted)
        log["noise_curriculum/num_demoted"] = float(self._noise_curriculum_num_demoted)
        log["noise_curriculum/last_direction"] = float(self._noise_curriculum_last_direction)
        if np.isfinite(self._last_noise_window_tracking):
            log["noise_curriculum/window_tracking"] = float(self._last_noise_window_tracking)
        if np.isfinite(self._last_noise_window_fail_rate):
            log["noise_curriculum/window_fail_rate"] = float(self._last_noise_window_fail_rate)
        if np.isfinite(self._last_noise_window_mean_episode_length_fraction):
            log["noise_curriculum/window_mean_episode_length_frac"] = float(
                self._last_noise_window_mean_episode_length_fraction
            )
        if np.isfinite(self._last_noise_window_mature_fraction):
            log["noise_curriculum/window_mature_fraction"] = float(
                self._last_noise_window_mature_fraction
            )
        for index, source_level in enumerate(levels):
            log[f"noise_curriculum/level_value_{index}"] = float(source_level)

    def _log_command_curriculum(self, info: dict) -> None:
        log = info.setdefault("log", {})
        self._log_noise_curriculum(log)
        log["standing/target_fraction"] = float(self._standing_probability)
        log["standing/fraction"] = float(self._last_standing_fraction)
        log["standing/window_episodes"] = float(self._standing_window_episodes)
        log["standing/window_target_episodes"] = float(self._standing_window_target_episodes())
        if np.isfinite(self._last_standing_mean_abs_vx):
            log["standing/mean_abs_vx"] = float(self._last_standing_mean_abs_vx)
        if np.isfinite(self._last_standing_mean_abs_vy):
            log["standing/mean_abs_vy"] = float(self._last_standing_mean_abs_vy)
        if np.isfinite(self._last_standing_mean_abs_yaw_rate):
            log["standing/mean_abs_yaw_rate"] = float(self._last_standing_mean_abs_yaw_rate)
        if np.isfinite(self._last_standing_window_fail_rate):
            log["standing/fail_rate"] = float(self._last_standing_window_fail_rate)
        if np.isfinite(self._last_standing_window_mean_episode_length_fraction):
            log["standing/mean_episode_length_frac"] = float(
                self._last_standing_window_mean_episode_length_fraction
            )
        if self._cfg.commands.curriculum:
            low, high = self._current_lin_vel_x_range()
            yaw_low, yaw_high = self._current_ang_vel_z_range()
            log["command_curriculum/scale"] = float(self._command_curriculum_scale)
            log["command_curriculum/lin_vel_x_min"] = low
            log["command_curriculum/lin_vel_x_max"] = high
            log["command_curriculum/yaw_scale"] = float(self._command_curriculum_yaw_scale)
            log["command_curriculum/ang_vel_z_min"] = yaw_low
            log["command_curriculum/ang_vel_z_max"] = yaw_high
            log["command_curriculum/moving_command_threshold"] = float(
                self._cfg.commands.curriculum_moving_command_threshold
            )
            log["command_curriculum/is_full"] = float(self._command_curriculum_is_full())
            if np.isfinite(self._last_command_curriculum_mean_tracking):
                log["command_curriculum/last_mean_tracking"] = float(
                    self._last_command_curriculum_mean_tracking
                )
            if np.isfinite(self._last_command_curriculum_mature_mean_tracking):
                log["command_curriculum/last_mature_mean_tracking"] = float(
                    self._last_command_curriculum_mature_mean_tracking
                )
            if np.isfinite(self._last_command_curriculum_mean_episode_steps):
                log["command_curriculum/mean_episode_steps"] = float(
                    self._last_command_curriculum_mean_episode_steps
                )
            if np.isfinite(self._last_command_curriculum_mean_moving_steps):
                log["command_curriculum/mean_moving_steps"] = float(
                    self._last_command_curriculum_mean_moving_steps
                )
            if np.isfinite(self._last_command_curriculum_mature_fraction):
                log["command_curriculum/mature_fraction"] = float(
                    self._last_command_curriculum_mature_fraction
                )
        if self._cfg.domain_rand.csv_force_enabled:
            domain_rand = self._cfg.domain_rand
            level = _csv_force_curriculum_level_index(
                domain_rand,
                self._csv_force_curriculum_level,
            )
            num_levels = _csv_force_curriculum_num_levels(domain_rand)
            final_level = max(num_levels - 1, 0)
            progress = float(level / final_level) if final_level > 0 else 1.0
            log["force_curriculum/progress"] = progress
            log["force_curriculum/level"] = level
            log["force_curriculum/final_progress"] = 1.0
            log["force_curriculum/final_level"] = final_level
            log["force_curriculum/num_levels"] = num_levels
            log["force_curriculum/applied_scale"] = 1.0
            command_ready = self._command_curriculum_is_full()
            log["force_curriculum/command_ready"] = float(command_ready)
            log["force_curriculum/locked_by_command"] = float(not command_ready)
            log["force_curriculum/window_episodes"] = float(self._csv_force_window_episodes)
            log["force_curriculum/window_tracking_episodes"] = float(
                self._csv_force_window_tracking_episodes
            )
            log["force_curriculum/window_target_episodes"] = float(
                domain_rand.csv_force_curriculum_window_episodes
            )
            log["force_curriculum/num_promoted"] = float(self._csv_force_curriculum_num_promoted)
            log["force_curriculum/num_demoted"] = float(self._csv_force_curriculum_num_demoted)
            log["force_curriculum/last_direction"] = float(
                self._csv_force_curriculum_last_direction
            )
            log["force_curriculum/allow_demotion"] = float(
                bool(domain_rand.csv_force_curriculum_allow_demotion)
            )
            if np.isfinite(self._last_csv_force_window_tracking):
                log["force_curriculum/window_tracking"] = float(
                    self._last_csv_force_window_tracking
                )
            if np.isfinite(self._last_csv_force_window_fail_rate):
                log["force_curriculum/window_fail_rate"] = float(
                    self._last_csv_force_window_fail_rate
                )
            if np.isfinite(self._last_csv_force_window_mean_episode_length_fraction):
                log["force_curriculum/window_mean_episode_length_frac"] = float(
                    self._last_csv_force_window_mean_episode_length_fraction
                )
            if np.isfinite(self._last_csv_force_window_mature_fraction):
                log["force_curriculum/window_mature_fraction"] = float(
                    self._last_csv_force_window_mature_fraction
                )
            hz = _csv_force_curriculum_source_hz(domain_rand, level)
            if np.isfinite(hz):
                log["force_curriculum/source_hz"] = hz
            active_levels = self.csv_force_active_levels()
            active_hz = self.csv_force_active_hz()
            log["force_curriculum/active_level_mean"] = float(np.mean(active_levels))
            finite_active_hz = active_hz[np.isfinite(active_hz)]
            if finite_active_hz.size > 0:
                log["force_curriculum/active_source_hz_mean"] = float(np.mean(finite_active_hz))
                log["force_curriculum/push_enabled"] = float(bool(domain_rand.push_robots))
            for active_level in range(num_levels):
                active_level_hz = _csv_force_curriculum_source_hz(domain_rand, active_level)
                if not np.isfinite(active_level_hz):
                    continue
                hz_label = (
                    int(active_level_hz) if float(active_level_hz).is_integer() else active_level_hz
                )
                log[f"force_curriculum/active_fraction_{hz_label}hz"] = float(
                    np.mean(active_levels == active_level)
                )

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        *,
        env_ids: np.ndarray | None = None,
        reset_history: bool = False,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        num_obs = gyro.shape[0]
        current_actions = np.asarray(
            info.get("current_actions", np.zeros((num_obs, self._num_action))),
            dtype=get_global_dtype(),
        )
        dof_vel_obs = dof_vel[:, JOINT_VEL_OBSERVATION_INDICES]
        default_joint_pos_offset = np.asarray(
            self._select_env_rows(self._default_joint_pos_offset, num_obs, env_ids),
            dtype=get_global_dtype(),
        )
        ordered_default_joint_pos_offset = default_joint_pos_offset[
            :, JOINT_VEL_OBSERVATION_INDICES
        ]
        leg_joint_pos_rel = (
            dof_pos[:, LEG_ACTION_INDICES]
            - self.default_angles[LEG_ACTION_INDICES]
            - default_joint_pos_offset[:, LEG_ACTION_INDICES]
        )
        noise_level = self._current_noise_level()
        num_leg_vel_obs = len(LEG_ACTION_INDICES)
        noisy_dof_vel_obs = np.empty_like(dof_vel_obs)
        noisy_dof_vel_obs[:, :num_leg_vel_obs] = self._obs_noise_at_level(
            dof_vel_obs[:, :num_leg_vel_obs],
            noise_cfg.scale_joint_vel,
            noise_level,
        )
        noisy_dof_vel_obs[:, num_leg_vel_obs:] = self._obs_noise_at_level(
            dof_vel_obs[:, num_leg_vel_obs:],
            noise_cfg.scale_wheel_vel,
            noise_level,
        )
        commands = np.asarray(info["commands"], dtype=get_global_dtype())
        noisy_projected_gravity = self._actor_projected_gravity(
            projected_gravity,
            num_obs=num_obs,
            env_ids=env_ids,
            noise_level=noise_level,
            advance_dynamic=not reset_history,
        )
        frame_terms = [
            self._obs_noise_at_level(gyro, noise_cfg.scale_gyro, noise_level),
            noisy_projected_gravity,
            self._obs_noise_at_level(
                leg_joint_pos_rel,
                noise_cfg.scale_joint_angle,
                noise_level,
            ),
            noisy_dof_vel_obs * 0.1,
            current_actions,
        ]
        wing_angle_obs = None
        if self._wing_angle_obs_enabled:
            wing_angle_obs = self._compute_wing_angle_obs(
                num_obs,
                env_ids,
                preview_next_step=not reset_history,
                episode_steps_override=(
                    np.zeros((num_obs,), dtype=np.int64) if reset_history else None
                ),
            )
            wing_angle_obs = self._obs_multiplicative_gaussian_noise_at_level(
                wing_angle_obs,
                float(self._cfg.wing_angle_obs.gaussian_noise_relative_std),
                noise_level,
            )
            wing_angle_obs = self._obs_noise_at_level(
                wing_angle_obs,
                float(self._cfg.wing_angle_obs.noise_half_range_deg)
                / float(self._cfg.wing_angle_obs.normalization_deg),
                noise_level,
            )
            frame_terms.append(wing_angle_obs)
        frame_terms.append(commands)
        actor = self._update_history(frame_terms, env_ids=env_ids, reset_history=reset_history)
        previous_actions = np.asarray(
            info.get("last_actions", np.zeros((num_obs, self._num_action))),
            dtype=get_global_dtype(),
        )
        joint_acc = np.asarray(
            info.get("qacc", np.zeros((num_obs, self._num_action))),
            dtype=get_global_dtype(),
        )
        motor_ctrl = np.asarray(
            info.get("torques", np.zeros((num_obs, self._num_action), dtype=dof_pos.dtype)),
            dtype=get_global_dtype(),
        )
        if not self._use_isaaclab_critic:
            critic = np.concatenate(
                [
                    linvel,
                    gyro,
                    projected_gravity,
                    dof_pos[:, LEG_ACTION_INDICES] - self.default_angles[LEG_ACTION_INDICES],
                    dof_vel_obs * 0.1,
                    current_actions,
                    commands,
                    motor_ctrl,
                    self._select_env_rows(self._motor_kp, num_obs, env_ids),
                    self._select_env_rows(self._motor_kd, num_obs, env_ids),
                    self._select_env_rows(
                        self._external_disturbance_current_wrench,
                        num_obs,
                        env_ids,
                    ),
                ],
                axis=1,
                dtype=get_global_dtype(),
            )
            if critic.shape[1] != self._critic_dim:
                raise RuntimeError(
                    "DR002 legacy critic contract mismatch: "
                    f"expected {self._critic_dim}, got {critic.shape[1]}"
                )
            return {
                "obs": actor,
                "critic": critic,
                "privileged_target": linvel.astype(get_global_dtype()),
            }

        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        base_pos = self._select_env_rows(base_pos, num_obs, env_ids)
        height_scalar = np.clip(base_pos[:, 2:3] - 0.5, -1.0, 1.0)
        height_measurements = np.repeat(height_scalar, 77, axis=1)
        material_properties = np.concatenate(
            [
                self._select_env_rows(self._privileged_robot_friction_scale, num_obs, env_ids),
                # UniLab currently does not randomize MuJoCo restitution. Keep the
                # IsaacLab slot semantic explicit rather than filling it with an
                # unrelated ground-friction value.
                np.zeros((num_obs, 1), dtype=get_global_dtype()),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        measured_csv_force = np.asarray(
            self._select_env_rows(self._measured_csv_force_base, num_obs, env_ids),
            dtype=get_global_dtype(),
        ).copy()
        measured_csv_force /= float(self._cfg.domain_rand.csv_force_observation_force_normalization)
        critic_terms = [
            linvel,
            gyro,
            projected_gravity,
            commands,
            leg_joint_pos_rel,
            dof_vel_obs * 0.1,
            current_actions,
            current_actions,
            previous_actions,
            joint_acc[:, JOINT_VEL_OBSERVATION_INDICES] * 0.0025,
            height_measurements,
            motor_ctrl[:, JOINT_VEL_OBSERVATION_INDICES] * 0.05,
            self._select_env_rows(self._privileged_base_mass_delta, num_obs, env_ids),
            self._select_env_rows(self._privileged_base_com_offset, num_obs, env_ids),
            ordered_default_joint_pos_offset,
            material_properties,
            measured_csv_force,
        ]
        if self._critic_includes_measured_moment:
            measured_csv_moment = np.asarray(
                self._select_env_rows(self._measured_csv_moment_base, num_obs, env_ids),
                dtype=get_global_dtype(),
            ).copy()
            measured_csv_moment /= float(
                self._cfg.domain_rand.csv_force_observation_moment_normalization
            )
            critic_terms.append(measured_csv_moment)
        critic = np.concatenate(
            critic_terms,
            axis=1,
            dtype=get_global_dtype(),
        )
        if critic.shape[1] != self._critic_dim:
            raise RuntimeError(
                "DR002 critic contract mismatch: "
                f"expected {self._critic_dim}, got {critic.shape[1]}"
            )
        return {
            "obs": actor,
            "critic": critic,
            "privileged_target": linvel.astype(get_global_dtype()),
        }

    def _update_history(
        self,
        frame_terms: list[np.ndarray],
        *,
        env_ids: np.ndarray | None,
        reset_history: bool,
    ) -> np.ndarray:
        if env_ids is None:
            if reset_history:
                for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                    hist[:] = frame[:, None, :]
            else:
                for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                    hist[:, :-1] = hist[:, 1:]
                    hist[:, -1] = frame
            return np.concatenate(
                [hist.reshape(self._num_envs, -1) for hist in self._history_terms], axis=1
            )

        env_ids = np.asarray(env_ids, dtype=np.intp)
        if reset_history:
            for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                hist[env_ids] = frame[:, None, :]
        else:
            for hist, frame in zip(self._history_terms, frame_terms, strict=True):
                hist[env_ids, :-1] = hist[env_ids, 1:]
                hist[env_ids, -1] = frame
        return np.concatenate(
            [hist[env_ids].reshape(len(env_ids), -1) for hist in self._history_terms], axis=1
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        base_height = self._reward_base_height_values(linvel.shape[0])
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=linvel.shape[0],
            default_angles=DEFAULT_DR002_ANGLES.astype(get_global_dtype()),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height=base_height,
            gravity=gravity,
        )
        reward = rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )
        self._accumulate_alive_episode(ctx.num_envs)
        self._log_alive_reward(info, ctx.num_envs)
        step_count = info.get("steps", np.zeros((ctx.num_envs,), dtype=np.uint32))
        if (
            self._enable_reward_log
            and int(step_count[0]) % 4 == 0
            and base_height.shape[0] == ctx.num_envs
        ):
            target = np.asarray(info["commands"], dtype=get_global_dtype())[:, 2]
            height_error = base_height - target
            log = info.setdefault("log", {})
            log["base_height/mean"] = float(np.mean(base_height))
            log["base_height/target_mean"] = float(np.mean(target))
            log["base_height/error_mean"] = float(np.mean(height_error))
            log["base_height/abs_error_mean"] = float(np.mean(np.abs(height_error)))
            log["termination/contact_now_frac"] = self._last_termination_contact_now_fraction
            log["termination/contact_accum_steps_mean"] = self._last_termination_contact_accum_mean
            log["termination/contact_accum_steps_max"] = self._last_termination_contact_accum_max
            log["termination/contact_done_frac"] = self._last_termination_contact_done_fraction
            log["termination/gravity_done_frac"] = self._last_termination_gravity_done_fraction
        return reward

    def _clip_lingzu_reward(
        self, name: str, reward: np.ndarray, clip_single_reward: float = 1.0
    ) -> np.ndarray:
        scale = float(self._reward_cfg.scales.get(name, 0.0))
        if scale == 0.0:
            return np.asarray(reward, dtype=get_global_dtype())
        weighted_dt = np.asarray(reward, dtype=get_global_dtype()) * scale * self._cfg.ctrl_dt
        clipped = np.clip(
            weighted_dt,
            -float(clip_single_reward) * self._cfg.ctrl_dt,
            float(clip_single_reward) * self._cfg.ctrl_dt,
        )
        return np.asarray(clipped / (scale * self._cfg.ctrl_dt), dtype=get_global_dtype())

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())

    def _reward_track_lin_vel_x(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 0] - ctx.linvel[:, 0])
        configured_std = self._reward_cfg.track_lin_vel_x_std
        std = ctx.tracking_sigma if configured_std is None else max(float(configured_std), 1.0e-6)
        reward = np.asarray(np.exp(-error / (std * std)), dtype=get_global_dtype())
        return self._clip_lingzu_reward(
            "track_lin_vel_x",
            reward,
            clip_single_reward=float(self._reward_cfg.track_lin_vel_x_term_clip),
        )

    def _reward_track_lin_vel_x_enhance(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 0] - ctx.linvel[:, 0])
        std = max(float(self._reward_cfg.track_lin_vel_x_enhance_std), 1.0e-6)
        reward = np.asarray(np.exp(-error / (std * std)) - 1.0, dtype=get_global_dtype())
        return self._clip_lingzu_reward(
            "track_lin_vel_x_enhance",
            reward,
            clip_single_reward=float(self._reward_cfg.track_lin_vel_x_term_clip),
        )

    def _reward_track_ang_vel_z(self, ctx: RewardContext) -> np.ndarray:
        error = np.square(ctx.info["commands"][:, 1] - ctx.gyro[:, 2])
        reward = np.asarray(np.exp(-error / (ctx.tracking_sigma**2)), dtype=get_global_dtype())
        return self._clip_lingzu_reward("track_ang_vel_z", reward)

    def _reward_lin_vel_z_lingzu(self, ctx: RewardContext) -> np.ndarray:
        reward = np.asarray(np.square(ctx.linvel[:, 2]), dtype=get_global_dtype())
        return self._clip_lingzu_reward("lin_vel_z", reward)

    def _reward_ang_vel_xy_lingzu(self, ctx: RewardContext) -> np.ndarray:
        reward = np.asarray(np.sum(np.square(ctx.gyro[:, :2]), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("ang_vel_xy", reward)

    def _reward_orientation_lingzu(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.gravity is not None
        reward = np.asarray(
            np.square(ctx.gravity[:, 0]) + np.square(ctx.gravity[:, 1]), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward("orientation", reward)

    def _reward_base_height_cmd(self, ctx: RewardContext) -> np.ndarray:
        target = ctx.info["commands"][:, 2]
        std = max(float(self._reward_cfg.base_height_std), 1.0e-6)
        reward = np.asarray(
            np.square(ctx.base_height - target) / (std * std), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward(
            "base_height",
            reward,
            clip_single_reward=float(self._reward_cfg.base_height_clip),
        )

    def _reward_joint_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        reward = np.asarray(
            np.sum(np.square(torques[:, LEG_ACTION_INDICES]), axis=1), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward("joint_torques_l2", reward)

    def _reward_joint_torques_wheel_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        reward = np.asarray(
            np.sum(np.square(torques[:, WHEEL_ACTION_INDICES]), axis=1), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward("joint_torques_wheel_l2", reward)

    def _reward_joint_vel_l2(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        reward = np.asarray(
            np.sum(np.square(ctx.dof_vel[:, LEG_ACTION_INDICES]), axis=1), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward("joint_vel_l2", reward)

    def _reward_joint_acc_l2(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(
            ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_DR002_ACTIONS))),
            dtype=get_global_dtype(),
        )
        reward = np.asarray(
            np.sum(np.square(qacc[:, LEG_ACTION_INDICES]), axis=1), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward("joint_acc_l2", reward)

    def _reward_joint_acc_wheel_l2(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(
            ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_DR002_ACTIONS))),
            dtype=get_global_dtype(),
        )
        reward = np.asarray(
            np.sum(np.square(qacc[:, WHEEL_ACTION_INDICES]), axis=1), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward("joint_acc_wheel_l2", reward)

    def _reward_joint_pos_limits(self, ctx: RewardContext) -> np.ndarray:
        if self._joint_range is None:
            return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        leg = LEG_ACTION_INDICES
        joint_range = np.asarray(self._joint_range, dtype=get_global_dtype())
        lower = joint_range[leg, 0]
        upper = joint_range[leg, 1]
        low_error = np.clip(lower - ctx.dof_pos[:, leg], 0.0, None)
        high_error = np.clip(ctx.dof_pos[:, leg] - upper, 0.0, None)
        reward = np.asarray(np.sum(low_error + high_error, axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("joint_pos_limits", reward)

    def _reward_nominal_state_lingzu(self, ctx: RewardContext) -> np.ndarray:
        thigh_error = ctx.dof_pos[:, 0] - ctx.dof_pos[:, 3]
        calf_error = ctx.dof_pos[:, 1] - ctx.dof_pos[:, 4]
        reward = np.asarray(
            np.square(thigh_error) + np.square(calf_error), dtype=get_global_dtype()
        )
        return self._clip_lingzu_reward("nominal_state_lingzu", reward)

    def _reward_action_rate_lingzu(self, ctx: RewardContext) -> np.ndarray:
        current = np.asarray(ctx.info["current_actions"], dtype=get_global_dtype())
        last = np.asarray(ctx.info["last_actions"], dtype=get_global_dtype())
        reward = np.asarray(np.sum(np.square(current - last), axis=1), dtype=get_global_dtype())
        return self._clip_lingzu_reward("action_rate_l2", reward)

    def _reward_action_smooth_lingzu(self, ctx: RewardContext) -> np.ndarray:
        current = np.asarray(ctx.info["current_actions"], dtype=get_global_dtype())
        leg = LEG_ACTION_INDICES
        diff = (
            current[:, leg]
            - 2.0 * self._lingzu_prev_action[:, leg]
            + self._lingzu_prev_prev_action[:, leg]
        )
        reward = np.asarray(np.sum(np.square(diff), axis=1), dtype=get_global_dtype())
        reward *= self._lingzu_action_history_count >= 2
        self._lingzu_prev_prev_action[:] = self._lingzu_prev_action
        self._lingzu_prev_action[:] = current
        self._lingzu_action_history_count += 1
        return self._clip_lingzu_reward("action_smooth_lingzu", reward)

    def _reward_undesired_contacts(self, ctx: RewardContext) -> np.ndarray:
        contacts = self._undesired_contact_values(ctx.num_envs)
        reward = np.asarray(
            np.sum(contacts > self._reward_cfg.undesired_contact_threshold, axis=1),
            dtype=get_global_dtype(),
        )
        return self._clip_lingzu_reward("undesired_contacts", reward)

    def _reward_alive(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(self._alive_values(ctx.num_envs), dtype=get_global_dtype())


registry.register_env("DR002JoystickFlat", DR002JoystickEnv, sim_backend="motrix")
registry.register_env("DR002JoystickFlatWE6", DR002JoystickEnv, sim_backend="mujoco")
registry.register_env("DR002JoystickFlatWE9", DR002JoystickEnv, sim_backend="mujoco")
