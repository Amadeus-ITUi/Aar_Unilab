"""Independent WTW-inspired biped task with measured sole contacts and full history."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from unilab.base.backend.mujoco.batched_robot import BatchedRobotSimulation
from unilab.base.backend.mujoco.foot_workspace import FootWorkspace
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.pe03.config import ROOT, validate_config
from unilab.envs.locomotion.pe03.gait import CommandCurriculum, phase_targets, placement_targets
from unilab.envs.locomotion.pe03.gait_randomization import (
    configure_curriculum,
    configure_randomization,
    push_forces,
    reset_backend,
)
from unilab.envs.locomotion.pe03.math import inverse_rotate, multiply, rotate
from unilab.envs.locomotion.pe03.robustness_curriculum import RobustnessCurriculum


class PE03GaitEnv:
    robustness_curriculum: RobustnessCurriculum

    def __init__(
        self,
        config,
        *,
        evaluation=False,
        num_envs=None,
        visual=False,
        model_path=None,
        auto_reset=True,
        evaluation_reset_noise=False,
        robustness=None,
        randomization_level=None,
    ):
        validate_config(config)
        self.config = OmegaConf.merge(config)
        self.cfg = OmegaConf.to_container(config, resolve=True)
        self.evaluation, self.auto_reset = evaluation, auto_reset
        self.robustness = not evaluation if robustness is None else bool(robustness)
        self.num_envs = n = int(num_envs or config.algo.num_envs)
        self.rng = np.random.default_rng(config.training.seed + (100000 if evaluation else 0))
        self.dt = 1 / config.control.policy_hz
        self.substeps = config.control.physics_hz // config.control.policy_hz
        self.history_length = 30
        self.max_episode_steps = round(config.env.episode_length_s / self.dt)
        self.command_interval = round(config.commands.resampling_time / self.dt)
        self.failure_limit = round(config.env.fail_to_terminal_time_s / self.dt)
        scene = Path(model_path) if model_path is not None else ROOT / config.env.model_path
        self.reference_points = np.array(config.env.reference_points)
        workspace = (
            scene.parent / "gait_workspace.npz"
            if model_path is not None
            else ROOT / config.env.workspace_path
        )
        self.workspace = FootWorkspace(workspace, scene, self.reference_points)
        self.backend = b = BatchedRobotSimulation(
            scene,
            num_envs=n,
            joint_order=tuple(config.env.joint_order),
            body_names=tuple(config.env.body_names),
            foot_names=tuple(config.env.foot_names),
            physics_hz=config.control.physics_hz,
            nthread=config.training.mujoco_threads,
            keyframe=config.env.reset_keyframe,
            visual=visual,
            visual_style=scene.parent / "play_visual.xml" if visual else None,
            native_pd=bool(config.training.native_pd) and not evaluation,
            foot_contact_geoms=tuple(config.env.foot_contact_geoms),
            track_gait_kinematics=True,
        )
        self.home = b.home.copy()
        self.height_target = float(self.home[2])
        self.penalized = [config.env.body_names.index(name) for name in config.env.penalized_bodies]
        collision = self.cfg["reward"].get("collision", {})
        self.collision_aggregation = collision.get("aggregation", "exponential")
        self.collision_force_threshold = collision.get("force_threshold", 1.0)
        self.collision_body_multipliers = np.array(
            [
                collision.get("body_multipliers", {}).get(name, 1.0)
                for name in config.env.penalized_bodies
            ]
        )
        self.termination_bodies = [
            config.env.body_names.index(name) for name in config.env.termination_bodies
        ]
        configure_randomization(self)
        configure_curriculum(self, randomization_level)
        bounds = b.joint_range
        self.clip_joint_targets = bool(config.control.get("clip_joint_targets", False))
        if self.clip_joint_targets and not (
            np.isfinite(bounds).all()
            and (bounds[:, 0] < bounds[:, 1]).all()
            and ((self.home[7:] >= bounds[:, 0]) & (self.home[7:] <= bounds[:, 1])).all()
        ):
            raise ValueError("joint target bounds must be finite, ordered and contain home")
        margin = config.reward.soft_limit_fraction * (bounds[:, 1] - bounds[:, 0])
        self.soft_limits = np.column_stack(
            (
                bounds[:, 0] + np.minimum(margin, (self.home[7:] - bounds[:, 0]) / 2),
                bounds[:, 1] - np.minimum(margin, (bounds[:, 1] - self.home[7:]) / 2),
            )
        )
        c, g = config.commands, config.gait
        self.velocity_curriculum = CommandCurriculum(
            c.low,
            c.high,
            c.bin_width,
            c.initial_low,
            c.initial_high,
            config.curriculum.weight_increment,
        )
        self.gait_curriculum = CommandCurriculum(
            g.low,
            g.high,
            g.bin_width,
            g.fixed,
            g.fixed,
            config.curriculum.weight_increment,
            centers=True,
        )
        self.history = np.zeros((n, 30, 38), np.float32)
        self.actions = np.zeros((n, 6))
        self.last_actions = np.zeros((n, 6))
        self.old_actions = np.zeros((n, 6))
        self.previous_velocity = np.zeros((n, 6))
        self.phase = np.zeros(n)
        self.commands = np.zeros((n, 3))
        self.gaits = np.tile(g.fixed, (n, 1)).astype(float)
        self.gait_start = self.gaits.copy()
        self.gait_target = self.gaits.copy()
        self.transition_elapsed = np.ones(n) * g.transition_s
        self.standing_command = np.zeros(n, bool)
        self.velocity_bins = np.zeros(n, int)
        self.gait_bins = np.full(n, -1, int)
        self.window_scores = np.zeros((n, 4))
        self.window_correct = np.zeros(n)
        self.window_contact_samples = np.zeros(n)
        self.window_peak_error = np.zeros(n)
        self.window_peaks = np.zeros(n)
        self.window_ticks = np.zeros(n, int)
        self.episode_steps = np.zeros(n, int)
        self.failure_steps = np.zeros(n, int)
        self.episode_returns = np.zeros(n)
        self.previous_contact = np.zeros((n, 2), bool)
        self.cycle_contact = np.zeros((n, 2))
        self.cycle_ticks = np.zeros((n, 2))
        self.cycle_beta = np.zeros((n, 2))
        self.cycle_peak = np.zeros((n, 2))
        self.cycle_height_target = np.zeros((n, 2))
        self.cycle_valid = np.zeros((n, 2), bool)
        self.last_touch = np.full((n, 2), -1, int)
        self.base_velocity = np.zeros((n, 3))
        self.measurement_noise = np.zeros((n, 18))
        self.push_count = np.zeros(n, dtype=np.int64)
        self.last_push_impulse = np.zeros((n, 3))
        self.reset_height_correction = np.zeros(n)
        self.reset_attempts = np.zeros(n, dtype=np.int64)
        self.reset_fallback = np.zeros(n, dtype=bool)
        self.reset()

    def _scaled_commands(self):
        s = self.cfg["normalization"]
        return np.concatenate((self.commands, self.gaits), axis=1) * np.array(
            [
                s["lin_vel"],
                s["lin_vel"],
                s["ang_vel"],
                s["frequency"],
                s["support_fraction"],
                s["clearance"],
            ]
        )

    def _gravity(self):
        return inverse_rotate(
            self.backend.qpos[:, 3:7], np.tile([0.0, 0.0, -1.0], (self.num_envs, 1))
        )

    def _frame(self):
        b, s = self.backend, self.cfg["normalization"]
        gravity, gyro = self._gravity(), b.qvel[:, 3:6]
        if self.robustness and self.cfg["domain_rand"]["enabled"]:
            biased = multiply(self.imu_offset, b.qpos[:, 3:7])
            gravity = inverse_rotate(biased, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
            gyro = inverse_rotate(biased, rotate(b.qpos[:, 3:7], gyro))
        frame = np.concatenate(
            (
                gyro * s["ang_vel"],
                gravity,
                (b.qpos[:, 7:] - self.home[7:]) * s["dof_pos"],
                b.joint_velocity * s["dof_vel"],
                self.actions,
                self.last_actions,
                self._scaled_commands(),
                np.sin(2 * np.pi * self.phase)[:, None],
                np.cos(2 * np.pi * self.phase)[:, None],
            ),
            axis=1,
        )
        frame[:, :18] += self.measurement_noise
        return np.clip(frame, -s["clip_observations"], s["clip_observations"]).astype(np.float32)

    def _sample_measurement(self, ids):
        if np.any(self.noise_amplitude) and np.any(self.randomization_levels[ids]):
            self.measurement_noise[ids] = (
                self.rng.uniform(-1, 1, (len(ids), 18))
                * self.noise_amplitude
                * self.randomization_levels[ids, None]
            )
        else:
            self.measurement_noise[ids] = 0

    def _observe(self, *, advance=False, feet=None):
        frame = self._frame()
        if advance:
            self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = frame
        b = self.backend
        self.base_velocity[:] = inverse_rotate(b.qpos[:, 3:7], b.qvel[:, :3])
        if feet is None:
            feet = b.gait_foot_state(self.reference_points)
        contact = feet["ground_force"][..., 2] > self.cfg["reward"]["contact_force_threshold"]
        force = (
            inverse_rotate(b.qpos[:, None, 3:7], feet["ground_force"]) / self.weight[:, None, None]
        )
        privileged = np.concatenate(
            (
                self.base_velocity,
                b.qpos[:, 2:3],
                feet["clearance"],
                contact,
                force.reshape(self.num_envs, 6),
            ),
            axis=1,
        )
        return dict(
            actor=self.history.reshape(self.num_envs, -1).copy(),
            frame=frame,
            command=self._scaled_commands().astype(np.float32),
            critic=privileged.astype(np.float32),
        )

    def _sample_commands(self, ids, *, initial=False):
        if not len(ids):
            return
        if self.evaluation:
            self.commands[ids] = self.cfg["play"]["command"]
            target = np.tile(self.cfg["play"]["gait"], (len(ids), 1))
        else:
            self.commands[ids], self.velocity_bins[ids] = self.velocity_curriculum.sample(
                self.rng, len(ids)
            )
            self.standing_command[ids] = (
                self.rng.random(len(ids)) < self.cfg["commands"]["zero_probability"]
            )
            self.commands[ids[self.standing_command[ids]]] = 0
            target = np.tile(self.cfg["gait"]["fixed"], (len(ids), 1)).astype(float)
            self.gait_bins[ids] = -1
            if self.cfg["gait"]["stage"] == "variable":
                varied = self.rng.random(len(ids)) >= self.cfg["gait"]["fixed_probability"]
                target[varied], self.gait_bins[ids[varied]] = self.gait_curriculum.sample(
                    self.rng, int(varied.sum())
                )
        self.gait_start[ids] = self.gaits[ids]
        self.gait_target[ids] = target
        self.transition_elapsed[ids] = 0
        if initial:
            # Both stages reset from the fixed pose/gait, then transition continuously.
            self.gaits[ids] = self.cfg["gait"]["fixed"]
            self.gait_start[ids] = self.gaits[ids]

    def set_command(self, command, gait=None):
        self.commands[:] = np.asarray(command)
        if gait is not None and not np.allclose(gait, self.gait_target):
            self.gait_start[:] = self.gaits
            self.gait_target[:] = gait
            self.transition_elapsed[:] = 0
        return self._scaled_commands().astype(np.float32)

    def _reset_ids(self, ids):
        reset_backend(self, ids)
        for name in (
            "actions",
            "last_actions",
            "old_actions",
            "previous_velocity",
            "episode_steps",
            "failure_steps",
            "episode_returns",
            "previous_contact",
            "cycle_contact",
            "cycle_ticks",
            "cycle_beta",
            "cycle_peak",
            "cycle_height_target",
            "cycle_valid",
            "window_scores",
            "window_correct",
            "window_contact_samples",
            "window_peak_error",
            "window_peaks",
            "window_ticks",
            "push_count",
            "last_push_impulse",
        ):
            getattr(self, name)[ids] = 0
        self.last_touch[ids] = -1
        self.phase[ids] = self.cfg["gait"]["reset_phase"]
        self._sample_commands(ids, initial=True)
        self._sample_measurement(ids)
        self.history[ids] = self._frame()[ids, None]

    def reset(self):
        self._reset_ids(np.arange(self.num_envs))
        self.state = NpEnvState(
            self._observe(),
            np.zeros(self.num_envs, np.float32),
            np.zeros(self.num_envs, bool),
            np.zeros(self.num_envs, bool),
            {},
        )
        return self.state

    def _rewards(self, feet):
        b, r = self.backend, self.cfg["reward"]
        gravity = self._gravity()
        desired, height, travel, stance = phase_targets(
            self.phase, self.gaits, self.cfg["gait"]["kappa"]
        )
        xy = placement_targets(self.commands, self.gaits, travel, self.workspace.nominal)
        target_xy, target_height, projection = self.workspace.project(xy, height)
        q = b.qpos[:, 3:7]
        yaw = np.arctan2(
            2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        )
        relative = feet["reference_position"][..., :2] - b.qpos[:, None, :2]
        actual_xy = np.stack(
            (
                np.cos(yaw)[:, None] * relative[..., 0] + np.sin(yaw)[:, None] * relative[..., 1],
                -np.sin(yaw)[:, None] * relative[..., 0] + np.cos(yaw)[:, None] * relative[..., 1],
            ),
            axis=-1,
        )
        contact = feet["ground_force"][..., 2] > r["contact_force_threshold"]
        filtered = contact | self.previous_contact
        velocity = feet["contact_velocity"]
        force_norm = np.linalg.norm(feet["ground_force"], axis=-1)
        nonfoot = (
            np.linalg.norm(b.contact_forces[:, self.penalized], axis=-1)
            > self.collision_force_threshold
        )
        scale = self.cfg["control"]["action_scale"]
        first = self.episode_steps > 1
        second = self.episode_steps > 2
        values = dict(
            tracking_lin_vel=np.exp(
                -np.square(self.commands[:, :2] - self.base_velocity[:, :2]).sum(1)
                / r["tracking_sigma"]
            ),
            tracking_ang_vel=np.exp(
                -np.square(self.commands[:, 2] - b.qvel[:, 5]) / r["yaw_tracking_sigma"]
            ),
            contact_force=(
                (1 - desired)
                * (
                    1
                    - np.exp(
                        -(force_norm**2)
                        / (r["force_sigma_weight_fraction"] * self.weight[:, None]) ** 2
                    )
                )
            ).mean(1),
            contact_velocity=(
                desired * (1 - np.exp(-np.square(velocity).sum(2) / r["gait_vel_sigma"]))
            ).mean(1),
            foot_clearance=((1 - desired) * np.square(feet["clearance"] - target_height)).sum(1),
            foot_placement=np.square(actual_xy - target_xy).sum((1, 2)),
            base_height=np.square(b.qpos[:, 2] - self.height_target),
            orientation=np.square(gravity[:, :2]).sum(1),
            lin_vel_z=self.base_velocity[:, 2] ** 2,
            ang_vel_xy=np.square(b.qvel[:, 3:5]).sum(1),
            torques=np.square(b.torque).sum(1),
            dof_vel=np.square(b.joint_velocity).sum(1),
            dof_acc=np.square((b.joint_velocity - self.previous_velocity) / self.dt).sum(1),
            action_rate=np.square(self.actions - self.last_actions).sum(1),
            target_smoothness_1=np.square((self.actions - self.last_actions) * scale).sum(1)
            * first,
            target_smoothness_2=np.square(
                (self.actions - 2 * self.last_actions + self.old_actions) * scale
            ).sum(1)
            * second,
            feet_slip=(filtered * np.square(velocity[..., :2]).sum(2)).sum(1),
            dof_pos_limits=(
                np.maximum(self.soft_limits[:, 0] - b.qpos[:, 7:], 0)
                + np.maximum(b.qpos[:, 7:] - self.soft_limits[:, 1], 0)
            ).sum(1),
            collision=nonfoot.sum(1).astype(float),
        )
        terms = {key: values[key] * weight * self.dt for key, weight in r["scales"].items()}
        collision_by_body = (
            nonfoot * self.collision_body_multipliers * r["scales"].get("collision", 0.0) * self.dt
        )
        # Keep collision as one aggregate log term; per-body diagnostics are not
        # added a second time. Retain integer raw counts when multipliers differ.
        if "collision" in r:
            terms["collision"] = collision_by_body.sum(1)
        positive = sum(np.maximum(value, 0) for value in terms.values())
        negative = sum(np.minimum(value, 0) for value in terms.values())
        exponential_negative = sum(
            np.minimum(value, 0)
            for key, value in terms.items()
            if key != "collision" or self.collision_aggregation == "exponential"
        )
        additive = (
            terms["collision"]
            if self.collision_aggregation == "additive"
            else np.zeros(self.num_envs)
        )
        attenuation = np.exp(exponential_negative / r["sigma_negative"])
        reward = positive * attenuation + additive
        confident = (desired < 0.1) | (desired > 0.9)
        correct = (contact == (desired > 0.5)) & confident
        info = dict(
            desired_contact=desired,
            contact=contact,
            stance=stance,
            clearance=feet["clearance"],
            requested_clearance=height,
            contact_correct=correct.sum(1),
            contact_samples=confident.sum(1),
            base_height=b.qpos[:, 2].copy(),
            base_height_error=np.abs(b.qpos[:, 2] - self.height_target),
            base_tilt_deg=np.rad2deg(np.arccos(np.clip(-gravity[:, 2], -1, 1))),
            base_speed=np.linalg.norm(self.base_velocity[:, :2], axis=1),
            nonfoot_contact=nonfoot.any(1),
            collision_contacts=nonfoot,
            collision_by_body=collision_by_body,
            slip_speed=(np.linalg.norm(velocity[..., :2], axis=-1) * contact).sum(1)
            / np.maximum(contact.sum(1), 1),
            joint_limit=values["dof_pos_limits"] > 0,
            crossing=(actual_xy[:, 0, 1] <= actual_xy[:, 1, 1]),
            torque_saturation=(np.abs(b.torque) >= b.torque_limits * 0.99).mean(1),
            projection_distance=projection.mean(1),
            tracking_scores=np.stack(
                (
                    values["tracking_lin_vel"],
                    values["tracking_ang_vel"],
                    1 - values["contact_force"],
                    1 - values["contact_velocity"],
                ),
                axis=1,
            ),
            raw_rewards=values,
            positive_reward=positive,
            negative_reward=negative,
            exponential_negative_reward=exponential_negative,
            additive_reward=additive,
            attenuation=attenuation,
        )
        return reward.astype(np.float32), terms, info

    def _cycle_metrics(self, info, previous_phase):
        contact = info["contact"]
        old_phase = (previous_phase[:, None] + [0, 0.5]) % 1
        new_phase = (self.phase[:, None] + [0, 0.5]) % 1
        wrapped = new_phase < old_phase
        self.cycle_contact += contact
        self.cycle_ticks += 1
        self.cycle_beta += self.gaits[:, 1:2]
        self.cycle_peak = np.maximum(self.cycle_peak, info["clearance"])
        self.cycle_height_target = np.maximum(self.cycle_height_target, self.gaits[:, 2:3])
        event = wrapped & self.cycle_valid
        info["cycle_event"] = event
        info["peak_error"] = np.abs(self.cycle_peak - self.cycle_height_target)
        info["duty_error"] = np.abs(
            (self.cycle_contact - self.cycle_beta) / np.maximum(1, self.cycle_ticks)
        )
        info["actual_duty"] = self.cycle_contact / np.maximum(1, self.cycle_ticks)
        touchdown = contact & ~self.previous_contact
        info["touch_event"] = touchdown & (self.last_touch >= 0)
        info["actual_frequency"] = 1 / np.maximum(
            (self.episode_steps[:, None] - self.last_touch) * self.dt, self.dt
        )
        self.last_touch[touchdown] = np.broadcast_to(self.episode_steps[:, None], contact.shape)[
            touchdown
        ]
        for array in (
            self.cycle_contact,
            self.cycle_ticks,
            self.cycle_beta,
            self.cycle_peak,
            self.cycle_height_target,
        ):
            array[wrapped] = 0
        self.cycle_valid[wrapped] = True
        self.previous_contact[:] = contact

    def _update_curriculum(self, ids, failed):
        if self.evaluation or not len(ids):
            return
        success = (
            self.window_scores[ids] / self.command_interval >= self.cfg["curriculum"]["thresholds"]
        ).all(1)
        success &= ~failed[ids]
        if self.cfg["gait"]["stage"] == "variable":
            success &= (
                self.window_correct[ids] / np.maximum(1, self.window_contact_samples[ids])
                >= self.cfg["curriculum"]["contact_accuracy"]
            )
            success &= (self.window_peaks[ids] > 0) & (
                self.window_peak_error[ids] / np.maximum(1, self.window_peaks[ids])
                <= self.cfg["curriculum"]["clearance_tolerance"]
            )
            self.gait_curriculum.update(self.gait_bins[ids], success)
        self.velocity_curriculum.update(
            self.velocity_bins[ids], success & ~self.standing_command[ids]
        )
        for name in (
            "window_scores",
            "window_correct",
            "window_contact_samples",
            "window_peak_error",
            "window_peaks",
            "window_ticks",
        ):
            getattr(self, name)[ids] = 0

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.num_envs, 6) or not np.isfinite(action).all():
            raise ValueError("expected finite six-joint actions")
        b, c = self.backend, self.cfg["control"]
        self.old_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        offset = np.clip(action, -c["action_clip"], c["action_clip"]) * c["action_scale"]
        center = b.qpos[:, 7:] - b.default_position + b.kd.mean() * b.joint_velocity / b.kp.mean()
        offset = np.clip(
            offset,
            center - c["user_torque_limit"] / b.kp.mean(),
            center + c["user_torque_limit"] / b.kp.mean(),
        )
        target_clip_distance = np.zeros_like(offset)
        if self.clip_joint_targets:
            target = b.default_position + offset
            limited = np.clip(target, b.joint_range[:, 0], b.joint_range[:, 1])
            target_clip_distance = np.abs(target - limited)
            offset = limited - b.default_position
        self.actions[:] = offset / c["action_scale"]
        forces = push_forces(self)
        if forces is None:
            b.step(offset, self.substeps)
        else:
            b.step(offset, self.substeps, push_forces=forces)
        impulse = self.last_push_impulse.copy()
        self.episode_steps += 1
        self.base_velocity[:] = inverse_rotate(b.qpos[:, 3:7], b.qvel[:, :3])
        previous_phase = self.phase.copy()
        self.phase[:] = (self.phase + self.dt * self.gaits[:, 0]) % 1
        feet = b.gait_foot_state(self.reference_points)
        reward, terms, info = self._rewards(feet)
        info["target_clip_distance"] = target_clip_distance
        info["push_impulse"] = impulse
        info["hard_limit_excess"] = np.maximum(
            np.maximum(b.joint_range[:, 0] - b.qpos[:, 7:], b.qpos[:, 7:] - b.joint_range[:, 1]),
            0,
        )
        self._cycle_metrics(info, previous_phase)
        reasons = np.column_stack(
            (
                (
                    np.linalg.norm(feet["body_ground_force"][:, self.termination_bodies], axis=-1)
                    > self.cfg["env"]["failure_contact_force"]
                ).any(1),
                info["base_tilt_deg"] > self.cfg["env"]["failure_tilt_deg"],
                b.qpos[:, 2] < self.cfg["env"]["failure_height"],
            )
        )
        failure = reasons.any(1)
        self.failure_steps[:] = np.where(failure, self.failure_steps + 1, 0)
        terminated = self.failure_steps >= self.failure_limit
        truncated = (self.episode_steps >= self.max_episode_steps) & ~terminated
        done = terminated | truncated
        natural_done = done.copy()
        self.episode_returns += reward
        info.update(
            reward_terms={k: float(v.mean()) for k, v in terms.items()},
            episode_returns=self.episode_returns[done].copy(),
            episode_lengths=self.episode_steps[done].copy(),
            failure_reasons=reasons,
            commands=self.commands.copy(),
            gait=self.gaits.copy(),
        )
        self.window_scores += info["tracking_scores"]
        self.window_correct += info["contact_correct"]
        self.window_contact_samples += info["contact_samples"]
        self.window_peak_error += (info["peak_error"] * info["cycle_event"]).sum(1)
        self.window_peaks += info["cycle_event"].sum(1)
        self.window_ticks += 1
        # Capture terminal observations before resampling/resetting commands.
        self._sample_measurement(np.arange(self.num_envs))
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = self._frame()
        # Decide using natural ends only, after settling the OLD-level transition.
        # A promotion cuts every surviving old-level episode for PPO, but those
        # administrative cuts are not episode results in either course or logs.
        promotion_truncated = np.zeros(self.num_envs, bool)
        if natural_done.any() and self.auto_reset:
            ids = np.flatnonzero(natural_done)
            course = self.robustness_curriculum
            promoted = course.observe(self.episode_steps[ids], self.randomization_levels[ids], ids)
            if promoted and course.strategy == "per_env_mean":
                promotion_truncated = ~natural_done
                truncated |= promotion_truncated
                done |= promotion_truncated
                course.promotion_truncations += int(promotion_truncated.sum())
        info["promotion_truncated"] = promotion_truncated
        # Only materialize the pre-reset history when an episode actually ends.
        # Foot sensors are unchanged until a reset; reuse the reward's snapshot.
        final = self._observe(feet=feet) if done.any() else None
        # Partial command windows cut by promotion are discarded, not scored.
        boundary = (self.window_ticks >= self.command_interval) | natural_done
        self._update_curriculum(np.flatnonzero(boundary), terminated)
        self._sample_commands(np.flatnonzero(boundary & ~done)) if not self.evaluation else None
        self.transition_elapsed[:] = np.minimum(
            self.transition_elapsed + self.dt, self.cfg["gait"]["transition_s"]
        )
        alpha = (self.transition_elapsed / self.cfg["gait"]["transition_s"])[:, None]
        self.gaits[:] = self.gait_start + alpha * (self.gait_target - self.gait_start)
        self.previous_velocity[:] = b.joint_velocity
        if done.any() and self.auto_reset:
            ids = np.flatnonzero(done)
            self._reset_ids(ids)
            feet = None
        obs = self._observe(feet=feet)
        info["diagnostics"] = self._diagnostics(info)
        self.state = NpEnvState(obs, reward, terminated.copy(), truncated.copy(), info, final)
        return self.state

    def _diagnostics(self, info):
        impulse = info.get("push_impulse", np.zeros((self.num_envs, 3)))
        values = {f"raw_reward/{k}": float(v.mean()) for k, v in info["raw_rewards"].items()}
        if "target_clip_distance" in info:
            values["control/target_clipped_fraction"] = float(
                (info["target_clip_distance"] > 1e-12).mean()
            )
            values["control/target_clip_distance"] = float(info["target_clip_distance"].mean())
            values["control/hard_limit_excess"] = float(info["hard_limit_excess"].max())
        for key in (
            "positive_reward",
            "negative_reward",
            "exponential_negative_reward",
            "additive_reward",
            "attenuation",
            "base_height_error",
            "slip_speed",
            "torque_saturation",
            "projection_distance",
            "crossing",
        ):
            values[f"gait/{key}"] = float(info[key].mean())
        for index, body in enumerate(self.cfg["env"]["penalized_bodies"]):
            values[f"collision/{body}/contact_fraction"] = float(
                info["collision_contacts"][:, index].mean()
            )
            values[f"collision/{body}/weighted_penalty"] = float(
                info["collision_by_body"][:, index].mean()
            )
        for foot, side in enumerate(("left", "right")):
            values[f"gait/{side}_clearance"] = float(info["clearance"][:, foot].mean())
            for key in ("peak_error", "duty_error", "actual_duty"):
                selected = info[key][:, foot][info["cycle_event"][:, foot]]
                if len(selected):
                    values[f"gait/{side}_{key}"] = float(selected.mean())
            selected = info["actual_frequency"][:, foot][info["touch_event"][:, foot]]
            if len(selected):
                values[f"gait/{side}_actual_frequency"] = float(selected.mean())
        for idx, reason in enumerate(("body_contact", "tilt", "height")):
            values[f"failure/{reason}"] = float(info["failure_reasons"][:, idx].mean())
        values["curriculum/velocity_active_fraction"] = float(
            (self.velocity_curriculum.weights > 0).mean()
        )
        values["curriculum/gait_active_fraction"] = float((self.gait_curriculum.weights > 0).mean())
        values.update(
            {
                "randomization/enabled": float(
                    self.robustness and self.cfg["domain_rand"]["enabled"]
                ),
                "randomization/noise_level": float(self.cfg["noise"].get("level", 0))
                * float(self.randomization_levels.mean())
                if self.robustness and self.cfg["noise"]["enabled"]
                else 0.0,
                "randomization/push_fraction": float((np.linalg.norm(impulse, axis=1) > 0).mean()),
                "randomization/push_impulse": float(np.linalg.norm(impulse, axis=1).mean()),
                "reset/height_correction": float(self.reset_height_correction.mean()),
                "reset/fallback_fraction": float(self.reset_fallback.mean()),
                "reset/attempts": float(self.reset_attempts.mean()),
            }
        )
        values.update(self.robustness_curriculum.metrics())
        for name, value in (
            ("min", self.randomization_levels.min()),
            ("mean", self.randomization_levels.mean()),
            ("max", self.randomization_levels.max()),
        ):
            values[f"randomization/applied_level_{name}"] = float(value)
        for name, array in {
            "friction": self.friction,
            "total_mass": self.weight / self.backend.gravity_acceleration,
            "delay_ms": self.backend.delay_steps * self.backend.dt * 1000,
            "kp_factor": self.backend.kp / self.cfg["control"]["kp"],
            "kd_factor": self.backend.kd / self.cfg["control"]["kd"],
            "torque_factor": self.backend.torque_scale,
        }.items():
            values[f"randomization/{name}_mean"] = float(array.mean())
            values[f"randomization/{name}_min"] = float(array.min())
            values[f"randomization/{name}_max"] = float(array.max())
        return values

    def snapshot(self) -> dict[str, Any]:
        return dict(
            arrays={k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)},
            rng=copy.deepcopy(self.rng.bit_generator.state),
            backend=self.backend.snapshot(),
            observation=copy.deepcopy(self.state.obs),
            velocity_weights=self.velocity_curriculum.snapshot(),
            gait_weights=self.gait_curriculum.snapshot(),
            robustness_curriculum=self.robustness_curriculum.snapshot(),
            randomization_nominal=copy.deepcopy(self.randomization_nominal),
            randomization_targets=copy.deepcopy(self.randomization_targets),
        )

    def restore(self, snapshot):
        if "robustness_curriculum" in snapshot:
            self.robustness_curriculum.restore(snapshot["robustness_curriculum"])
            self.randomization_nominal = copy.deepcopy(snapshot["randomization_nominal"])
            self.randomization_targets = copy.deepcopy(snapshot["randomization_targets"])
        elif self.robustness_curriculum.enabled:
            raise ValueError("curriculum resume requires saved robustness state")
        for key, value in snapshot["arrays"].items():
            setattr(self, key, value.copy())
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.backend.restore(snapshot["backend"])
        self.velocity_curriculum.restore(snapshot["velocity_weights"])
        self.gait_curriculum.restore(snapshot["gait_weights"])
        self.state = NpEnvState(
            copy.deepcopy(snapshot["observation"]),
            np.zeros(self.num_envs, np.float32),
            np.zeros(self.num_envs, bool),
            np.zeros(self.num_envs, bool),
            {},
        )
        return self.state

    def close(self):
        self.backend.close()
