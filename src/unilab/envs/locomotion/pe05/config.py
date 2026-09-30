"""Independent Hydra configuration owner for PE05."""

import math
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[5]


def load_config(overrides: Sequence[str] = (), *, name: str = "config") -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/pe05")):
        config = compose(config_name=name, overrides=list(overrides))
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    if config.robot != "pe05" or config.task_id != "pe05_flat":
        raise ValueError("PE05 configuration requires robot=pe05 and task=pe05_flat")
    expected = {
        "policy": "pe05_encoder_mlp",
        "algorithm": "pe05_custom_ppo",
        "simulator": "mujoco",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"PE05 configuration requires {key}={value}")
    if config.observation != "pe05_v1":
        raise ValueError("unsupported PE05 observation version")
    if config.env.history_length != 10:
        raise ValueError("pe05_v1 requires ten history frames")
    count = len(config.env.joint_order)
    if count < 1 or len(set(config.env.joint_order)) != count:
        raise ValueError("PE05 joint_order must contain unique joints")
    if config.env.frame_size < 3 * count + 6 or config.env.history_length < 1:
        raise ValueError("PE05 frame/history dimensions cannot hold the observation layout")
    keyframe = config.env.get("reset_keyframe")
    if keyframe is not None and (not isinstance(keyframe, str) or not keyframe.strip()):
        raise ValueError("PE05 reset_keyframe must be a nonempty name or null")
    height = config.env.initial_height
    if height is not None and (not math.isfinite(float(height)) or float(height) <= 0):
        raise ValueError("PE05 initial_height must be positive and finite or null")
    physics, policy = int(config.control.physics_hz), int(config.control.policy_hz)
    if physics <= 0 or policy <= 0 or physics % policy:
        raise ValueError("physics_hz must be a positive integer multiple of policy_hz")
    if config.control.action_clip <= 0 or config.control.action_scale <= 0:
        raise ValueError("action_clip and action_scale must be positive")
    if config.network.command_size != 3:
        raise ValueError("the PE05 joystick command contract requires three values")
    if config.network.latent_dim < 1 or config.network.initial_std <= 0:
        raise ValueError("PE05 latent_dim and initial_std must be positive")
    if any(
        width < 1
        for widths in (
            config.network.encoder_hidden_dims,
            config.network.actor_hidden_dims,
            config.network.critic_hidden_dims,
        )
        for width in widths
    ):
        raise ValueError("PE05 hidden layer widths must be positive")
    if count != 6 or config.env.frame_size != 30 or config.network.latent_dim != 3:
        raise ValueError("pe05_v1 requires 6 joints, a 30D frame and 3D velocity encoder")
    if keyframe is None:
        raise ValueError("pe05_v1 requires a named reset keyframe for its PD reference")
    for field in ("cpu_threads", "mujoco_threads", "evaluation_episodes"):
        if int(config.training[field]) < 1:
            raise ValueError(f"training.{field} must be positive")
    for field in ("evaluation_joint_noise", "evaluation_velocity_noise"):
        noise_value = float(config.training.get(field, 0.0))
        if not math.isfinite(noise_value) or noise_value < 0:
            raise ValueError(f"training.{field} must be nonnegative and finite")
    if len(config.play.command) != 3 or len(config.play.gait) != 4:
        raise ValueError("pe05_v1 play requires three command values and four gait values")
    if not 0 < config.play.gait[2] < 1:
        raise ValueError("play gait duration must be strictly between zero and one")
    if config.control.type != "P" or config.control.motor_hz != physics:
        raise ValueError("pe05_v1 uses position PD at each physics step")
    for field in ("kp", "kd", "torque_limits"):
        values = list(config.control[field])
        if len(values) != count or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"control.{field} requires one positive finite value per joint")
    for field in (
        "num_envs",
        "num_steps_per_env",
        "max_iterations",
        "save_interval",
        "num_learning_epochs",
        "num_mini_batches",
    ):
        if int(config.algo[field]) < 1:
            raise ValueError(f"algo.{field} must be positive")
    if config.algo.num_envs * config.algo.num_steps_per_env < config.algo.num_mini_batches:
        raise ValueError("rollout must have at least one sample per mini batch")
    if (
        not 0 < config.algo.clip_ratio < 1
        or not 0 <= config.algo.gamma <= 1
        or not 0 <= config.algo.lam <= 1
    ):
        raise ValueError("invalid PPO clip_ratio, gamma or lam")
    if config.algo.learning_rate <= 0 or config.algo.encoder_learning_rate <= 0:
        raise ValueError("PPO and encoder learning rates must be positive")
    if config.algo.schedule not in {"adaptive", "fixed"}:
        raise ValueError("unsupported learning-rate schedule")
    if not config.network.encoder_output_detach:
        raise ValueError("pe05_v1 uses a separately supervised, detached encoder")
    for field in ("episode_length_s", "fail_to_terminal_time_s"):
        if config.env[field] <= 0:
            raise ValueError(f"env.{field} must be positive")
    if not 0 <= config.commands.zero_probability <= 1:
        raise ValueError("zero_probability must be in [0, 1]")
    for owner, fields in (
        (
            config.domain_rand,
            (
                "friction_range",
                "added_mass_range",
                "inertia_range",
                "kp_range",
                "kd_range",
                "torque_scale_range",
                "default_joint_offset_range",
                "delay_ms_range",
                "imu_offset_deg_range",
            ),
        ),
        (config.gait, ("frequencies", "offsets", "durations", "swing_height")),
        (config.env, ("joint_reset_range", "base_velocity_reset_range")),
        (config.commands.ranges, ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading")),
    ):
        for field in fields:
            values = list(owner[field])
            if (
                len(values) != 2
                or not all(math.isfinite(v) for v in values)
                or values[0] > values[1]
            ):
                raise ValueError(f"invalid range: {field}")
    if not 0 < config.gait.durations[0] <= config.gait.durations[1] < 1:
        raise ValueError("gait durations must be strictly between zero and one")
    if config.domain_rand.delay_ms_range[0] < 0 or config.play.delay_ms < 0:
        raise ValueError("action delays cannot be negative")
    if config.domain_rand.restitution_mapping is not None:
        raise ValueError("PhysX restitution is not a MuJoCo scalar; no mapping has been calibrated")
    revision = config.env.get("task_revision")
    if revision not in (None, "weighted_v2"):
        raise ValueError("unsupported PE05 task revision")
    if revision == "weighted_v2":
        _validate_weighted(config)


def _validate_weighted(config: DictConfig) -> None:
    env, reward, course = config.env, config.reward, config.domain_rand.curriculum
    commands = config.commands
    strategy = commands.get("curriculum_strategy")
    if strategy not in (None, "velocity_bins"):
        raise ValueError("unsupported command curriculum strategy")
    if strategy == "velocity_bins":
        if commands.curriculum and commands.heading_command:
            raise ValueError(
                "velocity_bins requires direct yaw-rate commands (heading_command=false)"
            )
        for key in ("initial_low", "initial_high", "bin_width", "thresholds"):
            values = commands[key]
            if len(values) != (4 if key == "thresholds" else 3) or not all(
                math.isfinite(v) for v in values
            ):
                raise ValueError(f"invalid commands.{key}")
        for i, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            low, high = commands.ranges[name]
            width = commands.bin_width[i]
            if width <= 0 or not low <= commands.initial_low[i] < commands.initial_high[i] <= high:
                raise ValueError("invalid command curriculum bounds")
            cells = (high - low) / width
            if not math.isclose(cells, round(cells), abs_tol=1e-8) or cells < 1:
                raise ValueError("command curriculum range must contain whole bins")
            for bound in (commands.initial_low[i], commands.initial_high[i]):
                offset = (bound - low) / width
                if not math.isclose(offset, round(offset), abs_tol=1e-8):
                    raise ValueError("initial command bounds must align with bin edges")
        if not 0 < commands.weight_increment <= 1 or any(
            not 0 < v <= 1 for v in commands.thresholds
        ):
            raise ValueError("invalid command curriculum increment or thresholds")
        for key in (
            "resampling_time",
            "score_tracking_sigma",
            "score_yaw_sigma",
            "score_force_weight_fraction",
            "score_contact_velocity_sigma",
            "score_contact_kappa",
        ):
            if not math.isfinite(commands[key]) or commands[key] <= 0:
                raise ValueError(f"commands.{key} must be positive and finite")
    if reward.aggregation != "weighted_sum":
        raise ValueError("weighted_v2 requires a signed weighted_sum")
    if not isinstance(reward.get("zero_command_stance", True), bool):
        raise ValueError("reward.zero_command_stance must be boolean")
    for key in (
        "base_height_std",
        "foot_clearance_std",
        "slip_velocity_std",
        "orientation_std",
        "joint_limit_std",
        "feet_distance_std",
        "tracking_velocity_std",
        "tracking_yaw_std",
        "contact_force_threshold",
    ):
        if not math.isfinite(reward[key]) or reward[key] <= 0:
            raise ValueError(f"reward.{key} must be positive and finite")
    supported = {
        "tracking_lin_vel",
        "tracking_ang_vel",
        "base_height",
        "foot_clearance",
        "contact_schedule",
        "collision",
        "termination",
        "feet_slip",
        "orientation",
        "lin_vel_z",
        "ang_vel_xy",
        "torques",
        "dof_acc",
        "action_rate",
        "action_smooth",
        "dof_pos_limits",
        "feet_distance",
    }
    version = reward.get("gait_reward_version")
    if version not in (None, "pe01_shaped_v1"):
        raise ValueError("unsupported reward.gait_reward_version")
    if version == "pe01_shaped_v1":
        supported -= {"contact_schedule", "feet_slip"}
        supported |= {
            "tracking_contacts_shaped_force",
            "tracking_contacts_shaped_vel",
            "feet_regulation",
            "foot_landing_vel",
        }
        if reward.get("zero_command_stance", True):
            raise ValueError("pe01_shaped_v1 requires alternating zero-command gait")
        for key in (
            "gait_kappa",
            "gait_force_sigma",
            "gait_vel_sigma",
            "feet_regulation_height_scale",
            "about_landing_threshold",
            "landing_force_threshold",
        ):
            value = reward.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"reward.{key} must be positive and finite")
    if set(reward.scales) != supported:
        raise ValueError("weighted_v2 requires its explicit reward scale set")
    positive_terms = {"tracking_lin_vel", "tracking_ang_vel"}
    for key, weight in reward.scales.items():
        if (
            not math.isfinite(weight)
            or (key in positive_terms and weight < 0)
            or (key not in positive_terms and weight > 0)
        ):
            raise ValueError(f"invalid signed reward scale: {key}")
    bodies = set(env.body_names) - set(env.foot_names)
    if set(env.penalized_bodies) != bodies or len(env.penalized_bodies) != len(bodies):
        raise ValueError("weighted_v2 must monitor every non-foot body")
    if set(reward.collision_body_weights) != bodies or any(
        not math.isfinite(v) or v <= 0 for v in reward.collision_body_weights.values()
    ):
        raise ValueError("collision_body_weights must cover all non-foot bodies")
    if not set(env.persistent_ground_bodies) or not set(env.persistent_ground_bodies) <= bodies:
        raise ValueError("invalid persistent ground bodies")
    if env.contact_hz != config.control.physics_hz:
        raise ValueError("weighted_v2 requires contact samples at every physics step")
    for key in ("ground_contact_time_s", "failure_height", "failure_tilt_deg", "reset_clearance"):
        if not math.isfinite(env[key]) or env[key] <= 0:
            raise ValueError(f"env.{key} must be positive and finite")
    if env.reset_max_attempts < 1 or not 0 < reward.soft_joint_limit <= 1:
        raise ValueError("invalid reset attempts or soft joint limits")
    if len(env.reference_points) != 2 or any(
        len(p) != 3 or not all(math.isfinite(v) for v in p) for p in env.reference_points
    ):
        raise ValueError("reference_points requires two finite sole points")
    if not 0 <= course.initial_level <= course.max_level <= 1 or not 0 < course.increment <= 1:
        raise ValueError("invalid curriculum levels")
    if not 0 < course.min_episode_fraction <= 1:
        raise ValueError("invalid minimum episode fraction")
    if (
        not 0 <= config.commands.initial_zero_probability <= 1
        or not 0 < config.commands.initial_range_scale <= 1
    ):
        raise ValueError("invalid initial command curriculum")
    for owner in (course, config.evaluation):
        for key in (
            "max_nonfoot_fraction",
            "max_height_error",
            "max_tracking_error",
            "max_yaw_error",
        ):
            if not math.isfinite(owner[key]) or owner[key] < 0:
                raise ValueError(f"invalid quality threshold: {key}")
    if not config.evaluation.commands or config.evaluation.delay_ms < 0:
        raise ValueError("weighted evaluation requires commands and nonnegative delay")
    for command in config.evaluation.commands.values():
        if len(command) != 3 or not all(math.isfinite(v) for v in command):
            raise ValueError("evaluation commands must be finite 3D velocities")
