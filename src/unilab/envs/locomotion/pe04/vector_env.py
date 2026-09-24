"""Independent PE04 TRON1-style flat training environment over generic MuJoCo APIs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from unilab.base.backend.mujoco.training_robot import TrainingRobotSimulation
from unilab.base.np_env import NpEnvState
from unilab.catalog.registry import repository_path
from unilab.envs.locomotion.pe04.config import ROOT, load_config, validate_config
from unilab.envs.locomotion.pe04.math import inverse_rotate, rotate
from unilab.envs.locomotion.pe04.observations import actor_frame, critic_layout, critic_size
from unilab.envs.locomotion.pe04.randomization import configure_randomization, sample_wrench
from unilab.envs.locomotion.pe04.rewards import compute_rewards


class PE04VectorEnv:
    def __init__(
        self,
        config: DictConfig | None = None,
        *,
        evaluation=False,
        num_envs=None,
        visual=False,
        model_path: Path | None = None,
        auto_reset=True,
    ):
        self.config = OmegaConf.merge(config if config is not None else load_config())
        validate_config(self.config)
        self.cfg = OmegaConf.to_container(self.config, resolve=True)
        self.evaluation, self.auto_reset = evaluation, auto_reset
        self.num_envs = int(num_envs if num_envs is not None else self.cfg["algo"]["num_envs"])
        self.rng = np.random.default_rng(
            self.cfg["training"]["seed"] + (100000 if evaluation else 0)
        )
        e, c = self.cfg["env"], self.cfg["control"]
        self.dt = 1 / c["policy_hz"]
        self.substeps = c["physics_hz"] // c["policy_hz"]
        self.history_length = e["history_length"]
        self.max_episode_steps = int(np.ceil(e["episode_length_s"] / self.dt))
        scene = model_path if model_path is not None else repository_path(e["model_path"], ROOT)
        self.backend = TrainingRobotSimulation(
            scene,
            num_envs=self.num_envs,
            joint_order=tuple(e["joint_order"]),
            body_names=tuple(e["body_names"]),
            foot_names=tuple(e["foot_names"]),
            physics_hz=c["physics_hz"],
            nthread=self.cfg["training"]["mujoco_threads"],
            keyframe=e["reset_keyframe"],
            visual=visual,
            visual_style=scene.parent / "play_visual.xml" if visual else None,
            native_pd=self.cfg["training"]["native_pd"],
            foot_contact_geoms=tuple(e["foot_contact_geoms"]),
            contact_hz=e["contact_hz"],
            contact_history_length=e["contact_history_length"],
        )
        self.home = self.backend.home.copy()
        configured_height = self.cfg["reward"]["base_height_target"]
        self.height_target = self.home[2] if configured_height is None else configured_height
        self.penalized = [e["body_names"].index(name) for name in e["penalized_bodies"]]
        self.termination_bodies = [e["body_names"].index(name) for name in e["termination_bodies"]]
        self.observation_layout = critic_layout(len(e["body_names"]))
        self.soft_limits = self.backend.nominal_joint_range.copy()
        mid = self.soft_limits.mean(1)
        half = (
            np.diff(self.soft_limits, axis=1)[:, 0] * 0.5 * self.cfg["reward"]["soft_joint_limit"]
        )
        self.soft_limits[:] = np.column_stack((mid - half, mid + half))
        n = self.num_envs
        self.history = np.zeros((n, 10, 30), np.float32)
        self.actions = np.zeros((n, 6))
        self.previous_actions = np.zeros((n, 6))
        self.older_actions = np.zeros((n, 6))
        self.commands = np.zeros((n, 3))
        self.command_ticks = np.zeros(n, np.int64)
        self.gaits = np.zeros((n, 4))
        self.gait_ticks = np.zeros(n, np.int64)
        self.phase = np.zeros(n)
        self.episode_steps = np.zeros(n, np.int64)
        self.episode_returns = np.zeros(n)
        self.base_velocity = np.zeros((n, 3))
        self.target_clip_fraction = np.zeros(n)
        configure_randomization(self)
        self.reset()

    def gravity(self, ids=slice(None)):
        q = self.backend.qpos[ids, 3:7]
        return inverse_rotate(q, np.broadcast_to([0.0, 0.0, -1.0], (len(q), 3)))

    def _clean_frame(self, ids=slice(None)):
        b = self.backend
        return np.concatenate(
            (
                b.qvel[ids, 3:6],
                self.gravity(ids),
                b.qpos[ids, 7:] - self.home[7:],
                b.joint_velocity[ids],
                self.actions[ids],
                np.sin(2 * np.pi * self.phase[ids])[:, None],
                np.cos(2 * np.pi * self.phase[ids])[:, None],
                self.gaits[ids],
            ),
            axis=1,
        )

    def _observe(self, *, advance=False, ids=None):
        b = self.backend
        rows = slice(None) if ids is None else ids
        n = self.num_envs if ids is None else len(ids)
        self.base_velocity[rows] = inverse_rotate(b.qpos[rows, 3:7], b.qvel[rows, :3])
        clean = self._clean_frame(rows)
        noisy = not self.evaluation and self.cfg["noise"]["enabled"]
        # Preserve the old full-batch RNG stream on sparse reset. Only the
        # selected environments need transforms, history copies or critic terms.
        samples = (
            self.rng.normal(size=(self.num_envs, 18))[ids] if ids is not None and noisy else None
        )
        frame = actor_frame(clean, self.rng, self.cfg, noise=noisy, noise_samples=samples)
        if advance:
            if ids is not None:
                raise ValueError("partial history advance is not supported")
            self.history[:, :-1] = self.history[:, 1:].copy()
            self.history[:, -1] = actor_frame(clean, self.rng, self.cfg, noise=noisy)
        world_velocity = np.concatenate(
            (b.qvel[rows, :3], rotate(b.qpos[rows, 3:7], b.qvel[rows, 3:6])), axis=1
        )
        critic = np.concatenate(
            (
                self.base_velocity[rows],
                clean,
                b.torque[rows],
                b.joint_acceleration[rows],
                b.contact_history[rows].reshape(n, -1),
                np.broadcast_to(b.nominal_mass, (n, len(b.nominal_mass))),
                np.broadcast_to(b.nominal_inertia.reshape(-1), (n, b.nominal_inertia.size)),
                np.broadcast_to(self.cfg["control"]["kp"], (n, 6)),
                np.broadcast_to(self.cfg["control"]["kd"], (n, 6)),
                b.qpos[rows, :3],
                world_velocity,
                b.qpos[rows, :3],
            ),
            axis=1,
            dtype=np.float32,
            casting="unsafe",
        )
        if critic.shape[1] != critic_size(len(self.cfg["env"]["body_names"])):
            raise RuntimeError("PE04 critic layout disagrees with observations")
        return dict(
            actor=self.history[rows].reshape(n, -1).copy(),
            frame=frame,
            command=self.commands[rows].astype(np.float32).copy(),
            critic=critic,
        )

    def _sample_commands(self, ids):
        if not len(ids):
            return
        c = self.cfg["commands"]
        if self.evaluation:
            self.commands[ids] = self.cfg["play"]["command"]
        else:
            for i, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
                self.commands[ids, i] = self.rng.uniform(*c["ranges"][key], len(ids))
            self.commands[ids[self.rng.random(len(ids)) < c["zero_probability"]]] = 0
        self.command_ticks[ids] = np.maximum(
            1, np.ceil(self.rng.uniform(*c["resampling_time_range"], len(ids)) / self.dt)
        ).astype(int)

    def _sample_gaits(self, ids):
        if not len(ids):
            return
        if self.evaluation:
            self.gaits[ids] = self.cfg["play"]["gait"]
        else:
            for i, key in enumerate(("frequencies", "offsets", "durations", "swing_height")):
                self.gaits[ids, i] = self.rng.uniform(*self.cfg["gait"][key], len(ids))
        self.gait_ticks[ids] = max(1, round(self.cfg["gait"]["resampling_time"] / self.dt))

    def set_command(self, command, gait=None):
        self.commands[:] = command
        self.cfg["play"]["command"] = np.asarray(command).tolist()
        if gait is not None:
            self.gaits[:] = gait
            self.cfg["play"]["gait"] = list(gait)
            self.phase[:] = (self.episode_steps * self.dt * self.gaits[:, 0]) % 1
        return self.commands.astype(np.float32)

    def _reset_ids(self, ids):
        if not len(ids):
            return
        n, e, b = len(ids), self.cfg["env"], self.backend
        qpos, qvel = np.tile(self.home, (n, 1)), np.zeros((n, 12))
        if not self.evaluation:
            qpos[:, 7:] += self.rng.uniform(*e["joint_reset_range"], (n, 6))
            qpos[:, 7:] = np.clip(
                qpos[:, 7:], b.nominal_joint_range[:, 0], b.nominal_joint_range[:, 1]
            )
            qpos[:, :2] += self.rng.uniform(*e["root_xy_reset_range"], (n, 2))
            yaw = self.rng.uniform(*e["root_yaw_reset_range"], n)
            qpos[:, 3:7] = np.column_stack((np.cos(yaw / 2), np.zeros((n, 2)), np.sin(yaw / 2)))
            qvel[:, :6] = self.rng.uniform(*e["base_velocity_reset_range"], (n, 6))
            b.align_reset_soles(qpos, e["reset_clearance"])
        b.reset(ids, qpos, qvel)
        for name in (
            "actions",
            "previous_actions",
            "older_actions",
            "episode_steps",
            "episode_returns",
            "phase",
            "target_clip_fraction",
        ):
            getattr(self, name)[ids] = 0
        self._sample_commands(ids)
        self._sample_gaits(ids)
        self.base_velocity[ids] = inverse_rotate(qpos[:, 3:7], qvel[:, :3])
        self.history[ids] = actor_frame(
            self._clean_frame(ids),
            self.rng,
            self.cfg,
            noise=not self.evaluation and self.cfg["noise"]["enabled"],
        )[:, None]

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

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.num_envs, 6) or not np.isfinite(action).all():
            raise ValueError("PE04 requires finite six-joint actions")
        b, c = self.backend, self.cfg["control"]
        self.older_actions[:] = self.previous_actions
        self.previous_actions[:] = self.actions
        self.actions[:] = action
        offset = np.clip(action, -c["action_clip"], c["action_clip"]) * c["action_scale"]
        center = b.qpos[:, 7:] - b.default_position + b.kd.mean() * b.joint_velocity / b.kp.mean()
        offset = np.clip(
            offset,
            center - c["user_torque_limit"] / b.kp.mean(),
            center + c["user_torque_limit"] / b.kp.mean(),
        )
        target = b.default_position + offset
        limited = np.clip(target, b.nominal_joint_range[:, 0], b.nominal_joint_range[:, 1])
        self.target_clip_fraction[:] = (target != limited).mean(1)
        external = b.external_wrench
        wrench = sample_wrench(self)
        if wrench is not None:
            b.external_wrench = wrench if external is None else external + wrench
        try:
            b.step(limited - b.default_position, self.substeps)
        finally:
            b.external_wrench = external
        self.episode_steps += 1
        self.base_velocity[:] = inverse_rotate(b.qpos[:, 3:7], b.qvel[:, :3])
        self.phase[:] = (self.episode_steps * self.dt * self.gaits[:, 0]) % 1
        contact_norm = np.linalg.norm(b.contact_history, axis=-1)
        reward, terms = compute_rewards(self, contact_norm)
        self.episode_returns += reward
        forces = contact_norm.max(axis=1)
        terminated = (
            forces[:, self.termination_bodies] > self.cfg["env"]["termination_force"]
        ).any(1)
        truncated = (self.episode_steps >= self.max_episode_steps) & ~terminated
        done = terminated | truncated
        obs = self._observe(advance=True)
        final = copy.deepcopy(obs) if done.any() else None
        info = dict(
            reward_terms={k: float(v.mean()) for k, v in terms.items()},
            episode_returns=self.episode_returns[done].copy(),
            episode_lengths=self.episode_steps[done].copy(),
            diagnostics={
                "control/target_clip_fraction": float(self.target_clip_fraction.mean()),
                "termination/base_contact": float(terminated.mean()),
                "termination/timeout": float(truncated.mean()),
                **{
                    f"velocity/true_{axis}": float(self.base_velocity[:, i].mean())
                    for i, axis in enumerate("xyz")
                },
            },
            base_height=b.qpos[:, 2].copy(),
            base_speed=np.linalg.norm(self.base_velocity[:, :2], axis=1),
        )
        self.command_ticks -= 1
        self.gait_ticks -= 1
        active = ~done if self.auto_reset else np.ones(self.num_envs, bool)
        changed_gait = np.flatnonzero((self.gait_ticks <= 0) & active)
        self._sample_commands(np.flatnonzero((self.command_ticks <= 0) & active))
        self._sample_gaits(changed_gait)
        self.phase[:] = (self.episode_steps * self.dt * self.gaits[:, 0]) % 1
        if len(changed_gait):
            # Commands are computed after reward, before the next observation in TRON1.
            clean = self._clean_frame()
            obs["frame"][changed_gait, 24:] = clean[changed_gait, 24:]
            self.history[changed_gait, -1, 24:] = clean[changed_gait, 24:]
            obs["actor"] = self.history.reshape(self.num_envs, -1).copy()
            obs["critic"][changed_gait, 27:33] = clean[changed_gait, 24:]
        obs["command"] = self.commands.astype(np.float32).copy()
        if done.any() and self.auto_reset:
            ids = np.flatnonzero(done)
            self._reset_ids(ids)
            reset_obs = self._observe(ids=ids)
            for key in obs:
                obs[key][ids] = reset_obs[key]
        self.state = NpEnvState(obs, reward, terminated.copy(), truncated.copy(), info, final)
        return self.state

    def snapshot(self) -> dict[str, Any]:
        return dict(
            arrays={k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)},
            backend=self.backend.snapshot(),
            rng=copy.deepcopy(self.rng.bit_generator.state),
            observation=copy.deepcopy(self.state.obs),
        )

    def restore(self, snapshot):
        for key, value in snapshot["arrays"].items():
            setattr(self, key, value.copy())
        self.backend.restore(snapshot["backend"])
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
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
