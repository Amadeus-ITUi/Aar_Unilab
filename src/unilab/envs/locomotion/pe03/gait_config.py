"""Validation for the independent PE03 v4 gait contract."""

import math

import numpy as np
from omegaconf import DictConfig


def validate_robustness(config: DictConfig) -> None:
    def bounded(owner, key, *, positive=False):
        values = np.asarray(owner[key], dtype=float)
        if (
            values.shape != (2,)
            or not np.isfinite(values).all()
            or values[0] > values[1]
            or (positive and values[0] <= 0)
        ):
            raise ValueError(f"invalid robustness range: {key}")

    dr = config.domain_rand
    curriculum = dr.get("curriculum", {})
    if not isinstance(curriculum.get("enabled", False), bool):
        raise ValueError("randomization curriculum enabled must be boolean")
    if curriculum.get("enabled", False):
        strategy = curriculum.get("strategy", "window_mean")
        if strategy not in {"window_mean", "episode_fraction", "per_env_mean"}:
            raise ValueError("invalid randomization curriculum strategy")
        threshold = (
            "min_episode_length" if strategy == "episode_fraction" else "mean_episode_length"
        )
        for name in ("initial_level", "increment", "max_level", threshold):
            if not math.isfinite(curriculum[name]):
                raise ValueError(f"invalid randomization curriculum {name}")
        if not 0 <= curriculum.initial_level <= curriculum.max_level <= 1:
            raise ValueError("randomization curriculum levels must be ordered within [0,1]")
        if not 0 < curriculum.increment <= 1 or curriculum[threshold] <= 0:
            raise ValueError("randomization curriculum increment and threshold must be positive")
        counts = {
            "episode_fraction": ("recent_episodes",),
            "window_mean": ("window_episodes", "consecutive_windows"),
            "per_env_mean": (),
        }[strategy]
        for name in counts:
            if type(curriculum[name]) is not int or curriculum[name] < 1:
                raise ValueError(f"randomization curriculum {name} must be a positive integer")
        if strategy == "episode_fraction" and (
            not math.isfinite(curriculum.success_fraction)
            or not 0 < curriculum.success_fraction <= 1
        ):
            raise ValueError("randomization curriculum success_fraction must be in (0,1]")
    for owner, key in ((dr, "enabled"), (dr, "push_enabled"), (config.noise, "enabled")):
        if not isinstance(owner[key], bool):
            raise ValueError(f"{key} must be boolean")
    bounded(dr, "delay_ms_range")
    if (
        dr.delay_ms_range[0] < 0
        or not math.isfinite(config.play.delay_ms)
        or config.play.delay_ms < 0
    ):
        raise ValueError("action delays must be finite and nonnegative")
    if dr.enabled:
        for name in (
            "friction_range",
            "inertia_range",
            "kp_range",
            "kd_range",
            "torque_scale_range",
        ):
            bounded(dr, name, positive=True)
        for name in ("added_mass_range", "default_joint_offset_range", "imu_offset_deg_range"):
            bounded(dr, name)
        com = np.asarray(dr.base_com_range)
        if com.shape != (3,) or not np.isfinite(com).all() or (com < 0).any():
            raise ValueError("base_com_range requires three nonnegative finite bounds")
        if not isinstance(dr.friction_buckets, int) or dr.friction_buckets < 1:
            raise ValueError("friction_buckets must be a positive integer")
    if dr.get("restitution_mapping") is not None:
        raise ValueError("no calibrated MuJoCo restitution mapping")
    if dr.push_enabled:
        if not dr.enabled:
            raise ValueError("push_enabled requires domain_rand.enabled")
        if (
            not math.isfinite(dr.push_interval_s)
            or dr.push_interval_s < 1 / config.control.physics_hz
        ):
            raise ValueError("push interval must span at least one physics step")
        if not math.isfinite(dr.max_push_vel) or dr.max_push_vel < 0:
            raise ValueError("max_push_vel must be finite and nonnegative")
    if config.noise.enabled:
        for name in ("level", "ang_vel", "gravity", "dof_pos", "dof_vel"):
            if not math.isfinite(config.noise[name]) or config.noise[name] < 0:
                raise ValueError(f"noise.{name} must be finite and nonnegative")
    reset = config.env.get("reset_randomization", {})
    if not isinstance(reset.get("enabled", False), bool):
        raise ValueError("reset_randomization.enabled must be boolean")
    if reset.get("enabled", False):
        for name in ("joint_range", "linear_velocity_range", "angular_velocity_range"):
            bounded(reset, name)
        if not reset.joint_range[0] <= 0 <= reset.joint_range[1]:
            raise ValueError("reset joint range must contain home")
        if not math.isfinite(reset.sole_clearance) or reset.sole_clearance <= 0:
            raise ValueError("reset sole_clearance must be positive and finite")
        if not isinstance(reset.max_attempts, int) or reset.max_attempts < 1:
            raise ValueError("reset max_attempts must be a positive integer")
    if config.training.get("evaluation_conditions", "nominal") not in {"nominal", "randomized"}:
        raise ValueError("evaluation_conditions must be nominal or randomized")


def validate_gait_config(config: DictConfig) -> None:
    expected = dict(
        robot="pe03",
        task_id="pe03_gait_flat",
        observation="pe03_v4",
        policy="pe03_history_velocity_mlp",
        algorithm="pe03_custom_ppo",
        simulator="mujoco",
    )
    for name, value in expected.items():
        if config[name] != value:
            raise ValueError(f"PE03 gait requires {name}={value}")
    if (
        config.env.frame_size,
        config.env.history_length,
        config.network.command_size,
        config.network.latent_dim,
    ) != (38, 30, 6, 3):
        raise ValueError("pe03_v4 requires 38D frames, 30 frames, 6 commands, 3D velocity")
    if not config.network.encoder_output_detach:
        raise ValueError("velocity supervision must remain detached from PPO")
    if config.gait.stage not in {"fixed", "variable"}:
        raise ValueError("gait stage must be fixed or variable")
    validate_robustness(config)
    if config.control.type != "P" or (
        config.control.physics_hz,
        config.control.motor_hz,
        config.control.policy_hz,
    ) != (400, 400, 50):
        raise ValueError("v4 control contract is 400 Hz physics/PD and 50 Hz policy")
    if "actuator_model" in config.control:
        from unilab.base.backend.mujoco.actuator_parameters import actuator_arrays

        actuator_arrays(config.control.actuator_model, 6)
        if config.env.dof_vel_use_pos_diff:
            raise ValueError("identified PE03 requires encoder velocity, not position difference")
    for field in ("kp", "kd", "torque_limits"):
        values = np.asarray(config.control[field])
        if values.shape != (6,) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"invalid control.{field}")
    for field in ("action_clip", "action_scale", "user_torque_limit"):
        if not math.isfinite(config.control[field]) or config.control[field] <= 0:
            raise ValueError(f"invalid control.{field}")
    if not isinstance(config.control.get("clip_joint_targets", False), bool):
        raise ValueError("control.clip_joint_targets must be a boolean")
    if len(config.env.joint_order) != 6 or len(config.env.foot_names) != 2:
        raise ValueError("v4 requires six joints and two feet")
    if len(config.play.command) != 3 or len(config.play.gait) != 3:
        raise ValueError("play requires 3 velocity and 3 gait commands")
    for gait in (config.gait.fixed, config.play.gait):
        if (
            len(gait) != 3
            or not all(math.isfinite(x) for x in gait)
            or not (gait[0] > 0 and 0.5 <= gait[1] <= 0.7 and gait[2] >= 0)
        ):
            raise ValueError("invalid gait frequency/support fraction/clearance")
    for owner in (config.commands, config.gait):
        low, high, width = map(lambda key: np.asarray(owner[key]), ("low", "high", "bin_width"))
        if (
            any(x.shape != (3,) for x in (low, high, width))
            or not np.isfinite([low, high, width]).all()
            or (high <= low).any()
            or (width <= 0).any()
        ):
            raise ValueError("invalid curriculum ranges")
    if not (
        config.gait.low[0] > 0
        and 0.5 <= config.gait.low[1] <= config.gait.high[1] < 1
        and config.gait.low[2] >= 0
    ):
        raise ValueError(
            "gait curriculum requires positive frequency, 0.5 <= support < 1, nonnegative height"
        )
    for low, high in (
        (config.commands.initial_low, config.commands.initial_high),
        (config.gait.fixed, config.gait.fixed),
    ):
        if (
            len(low) != 3
            or len(high) != 3
            or not np.isfinite([low, high]).all()
            or (np.asarray(low) > high).any()
        ):
            raise ValueError("invalid curriculum initial domain")
    if (np.asarray(config.commands.initial_low) < config.commands.low).any() or (
        np.asarray(config.commands.initial_high) > config.commands.high
    ).any():
        raise ValueError("initial velocity range exceeds curriculum limits")
    if (np.asarray(config.gait.fixed) < config.gait.low).any() or (
        np.asarray(config.gait.fixed) > config.gait.high
    ).any():
        raise ValueError("fixed gait lies outside the variable curriculum")
    if (
        not 0 <= config.commands.zero_probability <= 1
        or not 0 <= config.gait.fixed_probability <= 1
    ):
        raise ValueError("sampling probabilities must be in [0,1]")
    for field in (
        "num_envs",
        "num_steps_per_env",
        "max_iterations",
        "save_interval",
        "num_learning_epochs",
        "num_mini_batches",
    ):
        if config.algo[field] < 1:
            raise ValueError(f"invalid algo.{field}")
    if config.algo.num_envs * config.algo.num_steps_per_env < config.algo.num_mini_batches:
        raise ValueError("rollout smaller than mini batch count")
    if config.training.resume and config.training.stage_from:
        raise ValueError("resume and stage_from are mutually exclusive")
    if config.training.stage_from and config.gait.stage != "variable":
        raise ValueError("stage_from is only valid for the variable stage")
    for field in ("cpu_threads", "mujoco_threads", "evaluation_episodes"):
        if config.training[field] < 1:
            raise ValueError(f"training.{field} must be positive")
    if config.training.acceptance.consecutive != 3:
        raise ValueError("v4 acceptance requires three consecutive evaluations")
    if not all(math.isfinite(value) for value in config.reward.scales.values()):
        raise ValueError("reward weights must be finite")
    # Checkpoints predating this block retain their original exponential reward.
    collision = config.reward.get("collision", {})
    if collision.get("aggregation", "exponential") not in {"additive", "exponential"}:
        raise ValueError("collision aggregation must be additive or exponential")
    threshold = collision.get("force_threshold", 1.0)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("collision force_threshold must be finite and nonnegative")
    bodies = list(config.env.penalized_bodies)
    if len(bodies) != len(set(bodies)) or set(bodies) != (
        set(config.env.body_names) - set(config.env.foot_names)
    ):
        raise ValueError("collision requires each nonfoot body exactly once")
    multipliers = collision.get("body_multipliers", {})
    if set(multipliers) - set(bodies) or any(
        not math.isfinite(v) or v < 0 for v in multipliers.values()
    ):
        raise ValueError(
            "collision body multipliers require known nonfoot bodies and finite nonnegative values"
        )
    if config.reward.scales.get("collision", 0) > 0:
        raise ValueError("collision scale must not reward contact")
    for value in (
        config.env.episode_length_s,
        config.env.fail_to_terminal_time_s,
        config.commands.resampling_time,
        config.gait.transition_s,
        config.gait.kappa,
        config.reward.sigma_negative,
        config.reward.gait_vel_sigma,
        config.reward.force_sigma_weight_fraction,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("gait timing and reward scales must be positive and finite")
    if (
        not 0 < config.algo.clip_ratio < 1
        or not 0 <= config.algo.gamma <= 1
        or not 0 <= config.algo.lam <= 1
    ):
        raise ValueError("invalid PPO parameters")
    if (
        config.algo.schedule not in {"adaptive", "fixed"}
        or config.algo.learning_rate <= 0
        or config.algo.encoder_learning_rate <= 0
    ):
        raise ValueError("invalid optimizer configuration")
