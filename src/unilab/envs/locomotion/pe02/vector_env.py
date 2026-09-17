# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# See LICENSE.pe01 for terms; PE02 migration changes are maintained independently.
"""PE02-owned migration of the original PE01 flat locomotion task semantics.

Provenance and deliberate simulator/bug-fix differences are recorded in
docs/PE02_TRAINING_MIGRATION.md. No PE01 or references module is imported.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.special import ndtr

from unilab.base.backend.mujoco.batched_robot import BatchedRobotSimulation
from unilab.base.np_env import NpEnvState
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe02.config import ROOT, load_config, validate_config
from unilab.envs.locomotion.pe02.math import inverse_rotate, multiply, rotate, wrap


class PE02VectorEnv:
    def __init__(
        self,
        config: DictConfig | None = None,
        *,
        evaluation: bool = False,
        num_envs: int | None = None,
        visual: bool = False,
        model_path: Path | None = None,
        auto_reset: bool = True,
        evaluation_reset_noise: bool = False,
    ) -> None:
        self.config = OmegaConf.merge(config if config is not None else load_config())
        validate_config(self.config)
        if self.config.observation != "pe02_v2":
            raise ValueError("PE02VectorEnv requires pe02_v2")
        self.cfg = OmegaConf.to_container(self.config, resolve=True)
        self.evaluation = evaluation
        self.auto_reset = auto_reset
        self.evaluation_reset_noise = evaluation_reset_noise
        self.num_envs = int(num_envs if num_envs is not None else self.cfg["algo"]["num_envs"])
        self.rng = np.random.default_rng(
            self.cfg["training"]["seed"] + (100000 if evaluation else 0)
        )
        env, control = self.cfg["env"], self.cfg["control"]
        self.dt = 1.0 / control["policy_hz"]
        self.substeps = control["physics_hz"] // control["policy_hz"]
        self.history_length = env["history_length"]
        self.backend = BatchedRobotSimulation(
            model_path if model_path is not None else repository_path(env["model_path"], ROOT),
            num_envs=self.num_envs,
            joint_order=tuple(env["joint_order"]),
            body_names=tuple(env["body_names"]),
            foot_names=tuple(env["foot_names"]),
            physics_hz=control["physics_hz"],
            nthread=self.cfg["training"]["mujoco_threads"],
            keyframe=env["reset_keyframe"],
            visual=visual,
            visual_style=ROOT / "src/unilab/assets/robots/pe02/play_visual.xml" if visual else None,
            native_pd=bool(self.cfg["training"].get("native_pd", True)) and not evaluation,
            foot_contact_geoms=(
                tuple(env["foot_contact_geoms"])
                if env.get("foot_contact_geoms") is not None
                else None
            ),
        )
        self.home = self.backend.home.copy()
        if env["initial_height"] is not None:
            self.home[2] = env["initial_height"]
        self.height_target = self.cfg["reward"]["base_height_target"]
        if self.height_target is None:
            self.height_target = self.home[2]
        self.penalized = [env["body_names"].index(name) for name in env["penalized_bodies"]]
        self.termination_bodies = [
            env["body_names"].index(name) for name in env["termination_bodies"]
        ]
        self.max_episode_steps = int(np.ceil(env["episode_length_s"] / self.dt))
        self.command_interval = max(1, round(self.cfg["commands"]["resampling_time"] / self.dt))
        self.gait_interval = max(1, round(self.cfg["gait"]["resampling_time"] / self.dt))
        n = self.num_envs
        self.actions = np.zeros((n, 6))
        self.last_actions = np.zeros((n, 2, 6))
        self.history = np.zeros((n, self.history_length, 30), dtype=np.float32)
        self.commands = np.zeros((n, 3))
        self.headings = np.zeros(n)
        self.gaits = np.zeros((n, 4))
        self.phase = np.zeros(n)
        self.episode_steps = np.zeros(n, dtype=np.int64)
        self.failure_steps = np.zeros(n, dtype=np.int64)
        self.episode_returns = np.zeros(n)
        self.previous_base = np.zeros((n, 3))
        self.previous_feet = np.zeros((n, 2, 3))
        self.previous_velocity = np.zeros((n, 6))
        self.base_velocity = np.zeros((n, 3))
        self.foot_velocity = np.zeros((n, 2, 3))
        self.imu_offset = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        self._configure_randomization()
        self.reset()

    def _uniform(self, bounds, shape):
        return self.rng.uniform(bounds[0], bounds[1], size=shape)

    def _configure_randomization(self) -> None:
        n, backend = self.num_envs, self.backend
        dr, control = self.cfg["domain_rand"], self.cfg["control"]
        kp = np.tile(control["kp"], (n, 1))
        kd = np.tile(control["kd"], (n, 1))
        torque_scale = np.ones((n, 6))
        default = np.tile(self.home[7:], (n, 1))
        delay = (
            np.full(n, round(self.cfg["play"]["delay_ms"] / 1000 / backend.dt))
            if self.evaluation
            else np.zeros(n)
        )
        if dr["enabled"] and not self.evaluation:
            # Source physical/gain randomization is sampled once per environment.
            kp *= self._uniform(dr["kp_range"], (n, 6))
            kd *= self._uniform(dr["kd_range"], (n, 6))
            torque_scale *= self._uniform(dr["torque_scale_range"], (n, 6))
            default += self._uniform(dr["default_joint_offset_range"], (n, 6))
            delay = np.rint(self._uniform(dr["delay_ms_range"], n) / 1000 / backend.dt)
            body_mass = np.tile(backend.base_mass, (n, 1))
            body_inertia = np.tile(backend.base_inertia, (n, 1, 1))
            body_ipos = np.tile(backend.base_ipos, (n, 1, 1))
            root = backend.body_ids[0]
            body_mass[:, root] += self._uniform(dr["added_mass_range"], n)
            scale = self._uniform(dr["inertia_range"], (n, len(backend.body_ids)))
            body_inertia[:, backend.body_ids] *= scale[..., None]
            body_mass[:, backend.body_ids[1:]] *= scale[:, 1:]
            body_ipos[:, root] += self.rng.uniform(
                -np.array(dr["base_com_range"]), dr["base_com_range"], (n, 3)
            )
            friction = np.tile(backend.base_friction, (n, 1, 1))
            buckets = self._uniform(dr["friction_range"], dr["friction_buckets"])
            sampled = buckets[self.rng.integers(len(buckets), size=n)]
            friction[:, backend.robot_geoms, 0] = sampled[:, None]
            backend.randomization = {
                "body_mass": body_mass,
                "body_inertia": body_inertia,
                "body_ipos": body_ipos,
                "geom_friction": friction,
            }
            angles = np.deg2rad(self._uniform(dr["imu_offset_deg_range"], (n, 2)))
            roll = np.column_stack(
                (np.cos(angles[:, 0] / 2), np.sin(angles[:, 0] / 2), np.zeros((n, 2)))
            )
            pitch = np.column_stack(
                (np.cos(angles[:, 1] / 2), np.zeros(n), np.sin(angles[:, 1] / 2), np.zeros(n))
            )
            self.imu_offset[:] = multiply(pitch, roll)
        if np.any(
            backend.randomization.get("body_mass", backend.base_mass)[..., backend.body_ids] <= 0
        ):
            raise ValueError("mass randomization produced a non-positive robot mass")
        backend.configure_pd(
            kp=kp,
            kd=kd,
            torque_scale=torque_scale,
            torque_limits=np.array(control["torque_limits"]),
            default_position=default,
            delay_steps=delay,
            position_difference=self.cfg["env"]["dof_vel_use_pos_diff"],
        )
        bounds = backend.joint_range
        middle, half = bounds.mean(axis=1), np.diff(bounds, axis=1)[:, 0] / 2
        half *= self.cfg["reward"]["soft_joint_limit"]
        self.soft_limits = np.column_stack((middle - half, middle + half))

    def _resample_commands(self, ids: np.ndarray) -> None:
        if self.evaluation:
            self.commands[ids] = self.cfg["play"]["command"]
            return
        cfg = self.cfg["commands"]
        for i, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            self.commands[ids, i] = self._uniform(cfg["ranges"][key], len(ids))
        self.headings[ids] = self._uniform(cfg["ranges"]["heading"], len(ids))
        zero = ids[self.rng.random(len(ids)) < cfg["zero_probability"]]
        self.commands[zero] = 0
        forward = rotate(self.backend.qpos[zero, 3:7], np.tile([1.0, 0.0, 0.0], (len(zero), 1)))
        self.headings[zero] = np.arctan2(forward[:, 1], forward[:, 0])

    def _resample_gaits(self, ids: np.ndarray) -> None:
        if self.evaluation:
            self.gaits[ids] = self.cfg["play"]["gait"]
        else:
            for index, key in enumerate(("frequencies", "offsets", "durations", "swing_height")):
                self.gaits[ids, index] = self._uniform(self.cfg["gait"][key], len(ids))

    def _gravity(self) -> np.ndarray:
        return inverse_rotate(
            self.backend.qpos[:, 3:7], np.tile([0.0, 0.0, -1.0], (self.num_envs, 1))
        )

    def _frame(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.backend.qpos[ids, 3:7]
        scale = self.cfg["normalization"]
        gyro = self.backend.qvel[ids, 3:6]
        gravity = self._gravity()[ids]
        frame = np.concatenate(
            (
                gyro * scale["ang_vel"],
                gravity,
                (self.backend.qpos[ids, 7:] - self.backend.default_position[ids])
                * scale["dof_pos"],
                self.backend.joint_velocity[ids] * scale["dof_vel"],
                self.actions[ids],
                np.sin(2 * np.pi * self.phase[ids])[:, None],
                np.cos(2 * np.pi * self.phase[ids])[:, None],
                self.gaits[ids],
            ),
            axis=-1,
        )
        clean = frame.copy()
        if not self.evaluation:
            biased = multiply(self.imu_offset[ids], q)
            frame[:, :3] = inverse_rotate(biased, rotate(q, gyro)) * scale["ang_vel"]
            frame[:, 3:6] = inverse_rotate(biased, np.tile([0.0, 0.0, -1.0], (len(ids), 1)))
            noise = self.cfg["noise"]
            if noise["enabled"]:
                amplitude = (
                    np.r_[
                        np.full(3, noise["ang_vel"] * scale["ang_vel"]),
                        np.full(3, noise["gravity"]),
                        np.full(6, noise["dof_pos"] * scale["dof_pos"]),
                        np.full(6, noise["dof_vel"] * scale["dof_vel"]),
                        np.zeros(12),
                    ]
                    * noise["level"]
                )
                frame += self.rng.uniform(-1, 1, frame.shape) * amplitude
        clip = scale["clip_observations"]
        return np.clip(frame, -clip, clip).astype(np.float32), clean.astype(np.float32)

    def _observation(self, *, advance_history: bool) -> dict[str, np.ndarray]:
        ids = np.arange(self.num_envs)
        frame, clean = self._frame(ids)
        if advance_history:
            self.history[:, :-1] = self.history[:, 1:]
            self.history[:, -1] = frame
        scale = self.cfg["normalization"]
        return {
            "actor": self.history.reshape(self.num_envs, -1).copy(),
            "frame": frame,
            "critic": np.concatenate(
                (self.base_velocity * scale["lin_vel"], clean), axis=-1
            ).astype(np.float32),
            "command": (
                self.commands * [scale["lin_vel"], scale["lin_vel"], scale["ang_vel"]]
            ).astype(np.float32),
        }

    def _reset_ids(self, ids: np.ndarray) -> None:
        if not len(ids):
            return
        n = len(ids)
        qpos = np.tile(self.home, (n, 1))
        qvel = np.zeros((n, 12))
        if self.evaluation and self.evaluation_reset_noise:
            joint_noise = self.cfg["training"].get("evaluation_joint_noise", 0.01)
            velocity_noise = self.cfg["training"].get("evaluation_velocity_noise", 0.02)
            qpos[:, 7:] += self.rng.uniform(-joint_noise, joint_noise, (n, 6))
            qpos[:, 7:] = np.clip(
                qpos[:, 7:], self.backend.joint_range[:, 0], self.backend.joint_range[:, 1]
            )
            qvel[:, :6] = self.rng.uniform(-velocity_noise, velocity_noise, (n, 6))
        if not self.evaluation:
            qpos[:, 7:] = self.backend.default_position[ids] + self._uniform(
                self.cfg["env"]["joint_reset_range"], (n, 6)
            )
            # PE02 home is close to mechanical stops. Keep random resets inside its real limits.
            qpos[:, 7:] = np.clip(
                qpos[:, 7:], self.backend.joint_range[:, 0], self.backend.joint_range[:, 1]
            )
            qvel[:, :6] = self._uniform(self.cfg["env"]["base_velocity_reset_range"], (n, 6))
        self.backend.reset(ids, qpos, qvel)
        self.phase[ids], self.episode_steps[ids], self.failure_steps[ids] = 0, 0, 0
        self.actions[ids], self.last_actions[ids], self.episode_returns[ids] = 0, 0, 0
        self.previous_velocity[ids] = 0
        self.previous_base[ids] = qpos[:, :3]
        self.previous_feet[ids] = self.backend.body_positions[ids][:, self.backend.foot_indices]
        self.base_velocity[ids] = inverse_rotate(qpos[:, 3:7], qvel[:, :3])
        self.foot_velocity[ids] = 0
        self._resample_commands(ids)
        self._resample_gaits(ids)
        frame, _ = self._frame(ids)
        self.history[ids] = frame[:, None, :]

    def reset(self) -> NpEnvState:
        self._reset_ids(np.arange(self.num_envs))
        obs = self._observation(advance_history=False)
        # Use the exact stored reset frame; no independent noise draw for actor/frame.
        obs["frame"] = self.history[:, -1].copy()
        self.state = NpEnvState(
            obs,
            np.zeros(self.num_envs, np.float32),
            np.zeros(self.num_envs, bool),
            np.zeros(self.num_envs, bool),
            {},
        )
        return self.state

    def _desired_contacts(self) -> np.ndarray:
        foot_phase = (
            self.phase[:, None] + np.column_stack((np.zeros(self.num_envs), self.gaits[:, 1]))
        ) % 1
        duration = self.gaits[:, 2:3]
        mapped = np.where(
            foot_phase < duration,
            foot_phase * 0.5 / duration,
            0.5 + (foot_phase - duration) * 0.5 / (1 - duration),
        )
        kappa = self.cfg["reward"]["gait_kappa"]
        return ndtr(mapped / kappa) * (1 - ndtr((mapped - 0.5) / kappa)) + ndtr(
            (mapped - 1) / kappa
        ) * (1 - ndtr((mapped - 1.5) / kappa))

    def _rewards(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        r, b = self.cfg["reward"], self.backend
        gyro, gravity = b.qvel[:, 3:6], self._gravity()
        forces = np.linalg.norm(b.contact_forces, axis=-1)
        foot_force = forces[:, b.foot_indices]
        foot_speed = np.linalg.norm(self.foot_velocity, axis=-1)
        height = b.foot_heights()
        desired = self._desired_contacts()
        rel = b.body_positions[:, b.foot_indices] - b.qpos[:, None, :3]
        body_feet = inverse_rotate(b.qpos[:, None, 3:7], rel)
        distance = np.abs(body_feet[:, 0, 1] - body_feet[:, 1, 1])
        landing = (
            (height < r["about_landing_threshold"])
            & (foot_force <= 0.1)
            & (self.foot_velocity[:, :, 2] < 0)
        )
        values = {
            "keep_balance": np.ones(self.num_envs),
            "tracking_lin_vel": np.exp(
                -np.square(self.commands[:, :2] - self.base_velocity[:, :2]).sum(1)
                / r["tracking_sigma"]
            ),
            "tracking_ang_vel": np.exp(
                -np.square(self.commands[:, 2] - gyro[:, 2]) / r["ang_tracking_sigma"]
            ),
            "base_height": np.square(b.qpos[:, 2] - self.height_target),
            "lin_vel_z": np.square(self.base_velocity[:, 2]),
            "ang_vel_xy": np.square(gyro[:, :2]).sum(1),
            "torques": np.square(b.torque).sum(1),
            "dof_acc": np.square((self.previous_velocity - b.joint_velocity) / self.dt).sum(1),
            "action_rate": np.square(self.actions - self.last_actions[:, 0]).sum(1),
            "action_smooth": np.square(
                self.actions - 2 * self.last_actions[:, 0] + self.last_actions[:, 1]
            ).sum(1),
            "dof_pos_limits": (
                np.maximum(self.soft_limits[:, 0] - b.qpos[:, 7:], 0)
                + np.maximum(b.qpos[:, 7:] - self.soft_limits[:, 1], 0)
            ).sum(1),
            "collision": (forces[:, self.penalized] > 1).sum(1),
            "orientation": np.square(gravity[:, :2]).sum(1),
            "feet_distance": np.clip(r["min_feet_distance"] - distance, 0, 1),
            "feet_regulation": (
                np.exp(-height / (self.height_target * 0.001))
                * np.square(self.foot_velocity[:, :, :2]).sum(2)
            ).sum(1),
            "foot_landing_vel": np.square(np.where(landing, self.foot_velocity[:, :, 2], 0)).sum(1),
            "tracking_contacts_shaped_force": (
                (1 - desired) * (1 - np.exp(-np.square(foot_force) / r["gait_force_sigma"]))
            ).mean(1),
            "tracking_contacts_shaped_vel": (
                desired * (1 - np.exp(-np.square(foot_speed) / r["gait_vel_sigma"]))
            ).mean(1),
        }
        scaled = {
            key: np.clip(values[key] * weight * self.dt, -r["clip_single"], r["clip_single"])
            for key, weight in r["scales"].items()
        }
        total = sum(scaled.values())
        if r["only_positive"]:
            total = np.maximum(total, 0)
        return np.clip(total, -r["clip_total"], r["clip_total"]).astype(np.float32), scaled

    def step(self, action: np.ndarray) -> NpEnvState:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_envs, 6) or not np.isfinite(action).all():
            raise ValueError(f"expected finite actions of shape ({self.num_envs}, 6)")
        b, c = self.backend, self.cfg["control"]
        offset = np.clip(action, -c["action_clip"], c["action_clip"]) * c["action_scale"]
        center = b.qpos[:, 7:] - b.default_position + b.kd.mean() * b.joint_velocity / b.kp.mean()
        offset = np.clip(
            offset,
            center - c["user_torque_limit"] / b.kp.mean(),
            center + c["user_torque_limit"] / b.kp.mean(),
        )
        self.actions[:] = offset / c["action_scale"]
        forces = None
        dr = self.cfg["domain_rand"]
        if not self.evaluation and dr["enabled"] and dr["push_enabled"]:
            interval = round(dr["push_interval_s"] / b.dt)
            trigger = (b.steps[:, None] + np.arange(1, self.substeps + 1)) % interval == 0
            if trigger.any():
                mass = b.randomization.get("body_mass", np.tile(b.base_mass, (self.num_envs, 1)))[
                    :, b.body_ids[0]
                ].mean()
                random_force = (
                    self.rng.uniform(-1, 1, (self.num_envs, self.substeps, 3))
                    * mass
                    * dr["max_push_vel"]
                    / b.dt
                )
                forces = rotate(b.qpos[:, None, 3:7], random_force) * trigger[..., None]
                forces[:, :, 2] *= 0.5
        b.step(offset, self.substeps, push_forces=forces)
        self.episode_steps += 1
        self.base_velocity[:] = inverse_rotate(
            b.qpos[:, 3:7], (b.qpos[:, :3] - self.previous_base) / self.dt
        )
        feet = b.body_positions[:, b.foot_indices]
        self.foot_velocity[:] = (feet - self.previous_feet) / self.dt
        self._resample_commands(np.flatnonzero(self.episode_steps % self.command_interval == 0))
        self._resample_gaits(np.flatnonzero(self.episode_steps % self.gait_interval == 0))
        self.phase[:] = (self.phase + self.dt * self.gaits[:, 0]) % 1
        if self.cfg["commands"]["heading_command"] and not self.evaluation:
            forward = rotate(b.qpos[:, 3:7], np.tile([1.0, 0.0, 0.0], (self.num_envs, 1)))
            self.commands[:, 2] = self.cfg["commands"]["heading_gain"] * wrap(
                self.headings - np.arctan2(forward[:, 1], forward[:, 0])
            )
        cfg = self.cfg["env"]
        failure = (
            np.linalg.norm(b.contact_forces[:, self.termination_bodies], axis=-1)
            > cfg["failure_contact_force"]
        ).any(1)
        failure |= self._gravity()[:, 2] > cfg["failure_gravity_z"]
        if cfg["consecutive_failure"]:
            self.failure_steps[~failure] = 0
        self.failure_steps += failure
        terminated = self.failure_steps > cfg["fail_to_terminal_time_s"] / self.dt
        truncated = (self.episode_steps > self.max_episode_steps) & ~terminated
        reward, terms = self._rewards()
        self.episode_returns += reward
        obs = self._observation(advance_history=True)
        done = terminated | truncated
        info = {
            "reward_terms": {key: float(value.mean()) for key, value in terms.items()},
            "episode_returns": self.episode_returns[done].copy(),
            "episode_lengths": self.episode_steps[done].copy(),
            "base_height": b.qpos[:, 2].copy(),
            "base_tilt_deg": np.rad2deg(np.arccos(np.clip(-self._gravity()[:, 2], -1, 1))),
            "base_speed": np.linalg.norm(self.base_velocity[:, :2], axis=-1),
            "nonfoot_contact": (
                np.linalg.norm(b.contact_forces[:, self.penalized], axis=-1) > 1
            ).any(1),
        }
        final = {key: value.copy() for key, value in obs.items()} if done.any() else None
        self.previous_base[:] = b.qpos[:, :3]
        self.previous_feet[:] = feet
        self.previous_velocity[:] = b.joint_velocity
        self.last_actions[:, 1] = self.last_actions[:, 0]
        self.last_actions[:, 0] = self.actions
        if done.any() and self.auto_reset:
            ids = np.flatnonzero(done)
            self._reset_ids(ids)
            reset_obs = self._observation(advance_history=False)
            reset_obs["frame"] = self.history[:, -1].copy()
            for key in obs:
                obs[key][ids] = reset_obs[key][ids]
        self.state = NpEnvState(obs, reward, terminated.copy(), truncated.copy(), info, final)
        return self.state

    def snapshot(self) -> dict[str, Any]:
        arrays = {
            name: value.copy()
            for name, value in vars(self).items()
            if isinstance(value, np.ndarray)
        }
        return {
            "arrays": arrays,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "backend": self.backend.snapshot(),
            "observation": copy.deepcopy(self.state.obs),
        }

    def restore(self, snapshot: dict[str, Any]) -> NpEnvState:
        for name, value in snapshot["arrays"].items():
            setattr(self, name, value.copy())
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.backend.restore(snapshot["backend"])
        self.state = NpEnvState(
            copy.deepcopy(snapshot["observation"]),
            np.zeros(self.num_envs, np.float32),
            np.zeros(self.num_envs, bool),
            np.zeros(self.num_envs, bool),
            {},
        )
        return self.state

    def close(self) -> None:
        self.backend.close()
