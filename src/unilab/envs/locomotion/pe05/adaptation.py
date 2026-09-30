"""PE05 task adaptation: additive rewards, contact failures and episode curricula.

Only declared backend arrays/interfaces are used; no model or asset inspection.
All additional state is training state, never an extra policy input.
"""

import copy

import numpy as np

from unilab.envs.locomotion.pe05.command_curriculum import VelocityCommandCurriculum
from unilab.envs.locomotion.pe05.gait_rewards import desired_contacts, foot_costs
from unilab.envs.locomotion.pe05.math import inverse_rotate, rotate


class WeightedTask:
    def __init__(self, env):
        self.env = env
        n, b = env.num_envs, env.backend
        c = env.cfg["domain_rand"]["curriculum"]
        self.level = np.full(n, c["initial_level"] if c["enabled"] else c["max_level"], dtype=float)
        if env.evaluation or not env.cfg["domain_rand"]["enabled"]:
            self.level[:] = 0
        self.ground_ticks = np.zeros((n, len(env.penalized)), dtype=np.int64)
        self.episode_sums = np.zeros((n, 4))
        self.reset_attempts = np.zeros(n, dtype=np.int64)
        self.reset_fallback = np.zeros(n, dtype=bool)
        self.standing_command = np.zeros(n, dtype=bool)
        commands = env.cfg["commands"]
        self.command_course = (
            VelocityCommandCurriculum(commands, n, env.command_interval)
            if commands["curriculum"]
            and commands.get("curriculum_strategy") == "velocity_bins"
            and not env.evaluation
            else None
        )
        self.persistent = np.array(
            [
                env.cfg["env"]["body_names"][index] in env.cfg["env"]["persistent_ground_bodies"]
                for index in env.penalized
            ]
        )
        self.nominal = {
            "kp": np.tile(env.cfg["control"]["kp"], (n, 1)),
            "kd": np.tile(env.cfg["control"]["kd"], (n, 1)),
            "torque_scale": np.ones((n, 6)),
            "default_position": np.tile(env.home[7:], (n, 1)),
        }
        self.targets = {key: getattr(b, key).copy() for key in self.nominal}
        self.physics_targets = copy.deepcopy(b.randomization)
        self.imu_target = env.imu_offset.copy()
        self.latest = {}
        self.diagnostics = {}

    def apply_level(self, ids):
        e, b = self.env, self.env.backend
        level = self.level[ids, None]
        for name, nominal in self.nominal.items():
            getattr(b, name)[ids] = nominal[ids] + level * (self.targets[name][ids] - nominal[ids])
        bases = dict(
            body_mass=b.base_mass,
            body_inertia=b.base_inertia,
            body_ipos=b.base_ipos,
            geom_friction=b.base_friction,
        )
        for name, target in self.physics_targets.items():
            nominal = bases[name]
            scale = self.level[ids].reshape((-1,) + (1,) * nominal.ndim)
            b.randomization[name][ids] = nominal + scale * (target[ids] - nominal)
        # Small roll/pitch offsets interpolate from identity; normalize the quaternion.
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        quat = identity + level * (self.imu_target[ids] - identity)
        e.imu_offset[ids] = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    def reset(self, ids):
        e, b, cfg = self.env, self.env.backend, self.env.cfg["env"]
        self.apply_level(ids)
        qpos = np.tile(e.home, (len(ids), 1))
        qvel = np.zeros((len(ids), 12))
        self.reset_attempts[ids] = 0
        self.reset_fallback[ids] = False
        if e.evaluation and not e.evaluation_reset_noise:
            b.reset(ids, qpos, qvel)
        else:
            if e.evaluation:
                amount = e.cfg["training"]["evaluation_joint_noise"]
                low, high = -np.full((len(ids), 1), amount), np.full((len(ids), 1), amount)
                speed = e.cfg["training"]["evaluation_velocity_noise"]
                qvel[:, :6] = e.rng.uniform(-speed, speed, (len(ids), 6))
            else:
                low = cfg["joint_reset_range"][0] * self.level[ids, None]
                high = cfg["joint_reset_range"][1] * self.level[ids, None]
                qvel[:, :6] = (
                    e._uniform(cfg["base_velocity_reset_range"], (len(ids), 6))
                    * self.level[ids, None]
                )
            low = np.maximum(e.home[7:] + low, b.joint_range[:, 0])
            high = np.minimum(e.home[7:] + high, b.joint_range[:, 1])
            pending = np.arange(len(ids))
            for attempt in range(cfg["reset_max_attempts"] + 1):
                fallback = attempt == cfg["reset_max_attempts"]
                sample = qpos[pending].copy()
                sample[:, 2] = e.home[2]
                sample[:, 7:] = (
                    e.home[7:] if fallback else e.rng.uniform(low[pending], high[pending])
                )
                b.align_reset_soles(sample, cfg["reset_clearance"])
                qpos[pending] = sample
                b.reset(ids[pending], sample, qvel[pending])
                self.reset_attempts[ids[pending]] += 1
                self.reset_fallback[ids[pending]] = fallback
                bad = np.linalg.norm(b.contact_forces[ids[pending]], axis=-1).max(1) > 1e-6
                if fallback and bad.any():
                    raise RuntimeError("PE05 home reset has unresolved contacts")
                pending = pending[bad]
                if not len(pending):
                    break
        self.ground_ticks[ids] = 0
        self.episode_sums[ids] = 0
        return qpos, qvel

    def sample_commands(self, ids):
        e, cfg = self.env, self.env.cfg["commands"]
        if self.command_course is not None:
            e.commands[ids] = self.command_course.sample(e.rng, ids)
            self.standing_command[ids] = self.command_course.zero[ids]
            return
        level = self.level[ids] if cfg["curriculum"] else np.ones(len(ids))
        scale = cfg["initial_range_scale"] + (1 - cfg["initial_range_scale"]) * level
        probability = (
            cfg["initial_zero_probability"]
            + (cfg["zero_probability"] - cfg["initial_zero_probability"]) * level
        )
        for index, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            e.commands[ids, index] = e._uniform(cfg["ranges"][name], len(ids)) * scale
        self.standing_command[ids] = e.rng.random(len(ids)) < probability
        zero = ids[self.standing_command[ids]]
        e.commands[zero] = 0
        direction = rotate(e.backend.qpos[ids, 3:7], np.tile([1.0, 0.0, 0.0], (len(ids), 1)))
        e.headings[ids] = (
            np.arctan2(direction[:, 1], direction[:, 0])
            + e._uniform(cfg["ranges"]["heading"], len(ids)) * scale
        )
        forward = rotate(e.backend.qpos[zero, 3:7], np.tile([1.0, 0.0, 0.0], (len(zero), 1)))
        e.headings[zero] = np.arctan2(forward[:, 1], forward[:, 0])

    def rewards(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        e, b, r = self.env, self.env.backend, self.env.cfg["reward"]
        feet = b.gait_foot_state(np.array(e.cfg["env"]["reference_points"]))
        ground = b.ground_contact_history
        self_force = b.contact_history - ground
        threshold = r["contact_force_threshold"]
        ground_mask = np.linalg.norm(ground[:, :, e.penalized], axis=-1) > threshold
        self_mask = np.linalg.norm(self_force[:, :, e.penalized], axis=-1) > threshold
        ground_any, self_any = ground_mask.any(1), self_mask.any(1)
        collision = ground_any | self_any
        contact = feet["ground_force"][..., 2] > threshold
        moving = (np.linalg.norm(e.commands[:, :2], axis=1) > r["moving_lin_vel_threshold"]) | (
            np.abs(e.commands[:, 2]) > r["moving_ang_vel_threshold"]
        )
        phase = (e.phase[:, None] + np.column_stack((np.zeros(e.num_envs), e.gaits[:, 1]))) % 1
        support = e.gaits[:, 2:3]
        swing = phase >= support
        # Older weighted checkpoints retain their saved zero-command stance behavior.
        if r.get("zero_command_stance", True):
            swing &= moving[:, None]
        travel = np.clip((phase - support) / (1 - support), 0, 1)
        target_height = e.gaits[:, 3:4] * np.sin(np.pi * travel) ** 2
        desired = ~swing
        height_error = np.abs(b.qpos[:, 2] - e.height_target)
        tracking_error = np.linalg.norm(e.commands[:, :2] - e.base_velocity[:, :2], axis=1)
        yaw_error = np.abs(e.commands[:, 2] - b.qvel[:, 5])
        multipliers = np.array(
            [r["collision_body_weights"][e.cfg["env"]["body_names"][i]] for i in e.penalized]
        )
        relative = feet["reference_position"] - b.qpos[:, None, :3]
        body_feet = inverse_rotate(b.qpos[:, None, 3:7], relative)
        distance = np.abs(body_feet[:, 0, 1] - body_feet[:, 1, 1])
        values = dict(
            tracking_lin_vel=1 - (tracking_error / r["tracking_velocity_std"]) ** 2,
            tracking_ang_vel=1 - (yaw_error / r["tracking_yaw_std"]) ** 2,
            base_height=(height_error / r["base_height_std"]) ** 2,
            foot_clearance=(
                swing * ((feet["clearance"] - target_height) / r["foot_clearance_std"]) ** 2
            ).sum(1),
            contact_schedule=((contact.astype(float) - desired) ** 2).mean(1),
            collision=(collision * multipliers).sum(1),
            termination=np.zeros(e.num_envs),
            feet_slip=(
                contact
                * (
                    np.square(feet["contact_velocity"][..., :2]).sum(2)
                    / r["slip_velocity_std"] ** 2
                )
            ).sum(1),
            orientation=np.square(e._gravity()[:, :2] / r["orientation_std"]).sum(1),
            lin_vel_z=e.base_velocity[:, 2] ** 2,
            ang_vel_xy=np.square(b.qvel[:, 3:5]).sum(1),
            torques=np.square(b.torque).sum(1),
            dof_acc=np.square((e.previous_velocity - b.joint_velocity) / e.dt).sum(1),
            action_rate=np.square(e.actions - e.last_actions[:, 0]).sum(1),
            action_smooth=np.square(
                e.actions - 2 * e.last_actions[:, 0] + e.last_actions[:, 1]
            ).sum(1),
            dof_pos_limits=(
                (
                    np.maximum(e.soft_limits[:, 0] - b.qpos[:, 7:], 0)
                    + np.maximum(b.qpos[:, 7:] - e.soft_limits[:, 1], 0)
                )
                / r["joint_limit_std"]
            ).sum(1),
            feet_distance=(
                np.maximum(r["min_feet_distance"] - distance, 0) / r["feet_distance_std"]
            )
            ** 2,
        )
        per_foot = {}
        if r.get("gait_reward_version") == "pe01_shaped_v1":
            desired = desired_contacts(e.phase, e.gaits, r["gait_kappa"])
            per_foot = foot_costs(feet, desired, r)
            del values["contact_schedule"], values["feet_slip"]
            values.update({name: cost.sum(1) for name, cost in per_foot.items()})
            per_foot["foot_clearance"] = (
                swing * ((feet["clearance"] - target_height) / r["foot_clearance_std"]) ** 2
            )
        terms = {name: values[name] * weight * e.dt for name, weight in r["scales"].items()}
        positive = np.sum([np.maximum(v, 0) for v in terms.values()], axis=0)
        negative = np.sum([np.minimum(v, 0) for v in terms.values()], axis=0)
        total = positive + negative
        self.latest = dict(
            ground_samples=ground_mask,
            ground_contact=ground_any,
            self_contact=self_any,
            collision=collision,
            height_error=height_error,
            tracking_error=tracking_error,
            yaw_error=yaw_error,
            foot_clearance=feet["clearance"],
            target_clearance=target_height,
            swing=swing,
            foot_contact=contact,
            desired_contact=desired,
            foot_reference_velocity=feet["reference_velocity"],
            reward_terms=terms,
            foot_reward_terms={
                name: cost * r["scales"][name] * e.dt for name, cost in per_foot.items()
            },
            slip_speed=np.sqrt(np.square(feet["contact_velocity"][..., :2]).sum(2)) * contact,
        )
        if self.command_course is not None:
            masses = b.randomization.get("body_mass", np.tile(b.base_mass, (e.num_envs, 1)))
            weight = masses[:, b.body_ids].sum(1) * 9.81
            self.latest["command_scores"] = self.command_course.tracking_scores(
                e.commands, e.base_velocity, b.qvel[:, 5], e.phase, e.gaits, feet, weight
            )
        self.diagnostics = {f"raw_reward/{k}": float(v.mean()) for k, v in values.items()}
        self.diagnostics.update(
            {
                "reward/positive_sum": float(positive.mean()),
                "reward/negative_sum": float(negative.mean()),
                "reward/total": float(total.mean()),
                "behavior/height_error_m": float(height_error.mean()),
                "behavior/tracking_error_mps": float(tracking_error.mean()),
                "behavior/yaw_error_radps": float(yaw_error.mean()),
                "behavior/clearance_error_m": float(
                    (swing * np.abs(feet["clearance"] - target_height)).sum() / max(1, swing.sum())
                ),
                "behavior/slip_speed_mps": float(self.latest["slip_speed"].mean()),
                "curriculum/mean_level": float(self.level.mean()),
                "curriculum/min_level": float(self.level.min()),
                "curriculum/max_level": float(self.level.max()),
                "commands/standing_fraction": float((~moving).mean()),
                "reset/fallback_fraction": float(self.reset_fallback.mean()),
            }
        )
        if self.command_course is not None:
            for i, name in enumerate(
                ("tracking_linear", "tracking_yaw", "contact_force", "contact_velocity")
            ):
                self.diagnostics[f"command_curriculum/score_{name}"] = float(
                    self.latest["command_scores"][:, i].mean()
                )
        for i, body in enumerate(e.cfg["env"]["penalized_bodies"]):
            self.diagnostics[f"contact/{body}/ground_fraction"] = float(ground_any[:, i].mean())
            self.diagnostics[f"contact/{body}/self_fraction"] = float(self_any[:, i].mean())
            self.diagnostics[f"contact/{body}/weighted_penalty"] = float(
                (collision[:, i] * multipliers[i] * r["scales"]["collision"] * e.dt).mean()
            )
        return total.astype(np.float32), terms

    def termination(self):
        e, b, cfg = self.env, self.env.backend, self.env.cfg["env"]
        limit = int(np.ceil(cfg["ground_contact_time_s"] / (b.dt * b.contact_stride)))
        persistent = np.zeros(e.num_envs, bool)
        # Backend samples are newest-first; duration must accumulate chronologically.
        for sample in self.latest["ground_samples"][:, ::-1].transpose(1, 0, 2):
            self.ground_ticks[:] = np.where(sample, self.ground_ticks + 1, 0)
            persistent |= (self.ground_ticks[:, self.persistent] >= limit).any(1)
        tilt = np.rad2deg(np.arccos(np.clip(-e._gravity()[:, 2], -1, 1)))
        low = b.qpos[:, 2] < cfg["failure_height"]
        tilted = tilt > cfg["failure_tilt_deg"]
        e.failure_steps[:] = np.where(low | tilted, e.failure_steps + 1, 0)
        failure = persistent | (
            e.failure_steps >= int(np.ceil(cfg["fail_to_terminal_time_s"] / e.dt))
        )
        self.episode_sums += np.column_stack(
            (
                self.latest["collision"].any(1),
                self.latest["height_error"],
                self.latest["tracking_error"],
                self.latest["yaw_error"],
            )
        )
        self.diagnostics["failure/persistent_ground"] = float(persistent.mean())
        self.diagnostics["failure/low_height"] = float(low.mean())
        self.diagnostics["failure/tilt"] = float(tilted.mean())
        for i, body in enumerate(e.cfg["env"]["penalized_bodies"]):
            self.diagnostics[f"contact/{body}/continuous_ground_seconds"] = float(
                self.ground_ticks[:, i].max() * b.dt * b.contact_stride
            )
        return failure

    def apply_failure_cost(self, reward, terms, terminated):
        # Once per actual failure; time limits are not failures. A finite terminal
        # cost counteracts escaping ongoing penalties by deliberately ending early.
        term = terminated * self.env.cfg["reward"]["scales"]["termination"] * self.env.dt
        terms["termination"] = term
        reward = reward + term
        self.diagnostics["raw_reward/termination"] = float(terminated.mean())
        self.diagnostics["reward/negative_sum"] += float(term.mean())
        self.diagnostics["reward/total"] = float(reward.mean())
        return reward.astype(np.float32), terms

    def finish_episodes(self, ids, terminated):
        e = self.env
        c = e.cfg["domain_rand"]["curriculum"]
        if e.evaluation or not e.cfg["domain_rand"]["enabled"] or not c["enabled"] or not len(ids):
            return
        means = self.episode_sums[ids] / np.maximum(e.episode_steps[ids, None], 1)
        success = (~terminated[ids]) & (
            e.episode_steps[ids] >= c["min_episode_fraction"] * e.max_episode_steps
        )
        success &= (
            means
            <= [
                c["max_nonfoot_fraction"],
                c["max_height_error"],
                c["max_tracking_error"],
                c["max_yaw_error"],
            ]
        ).all(1)
        self.level[ids[success]] = np.minimum(
            c["max_level"], self.level[ids[success]] + c["increment"]
        )

    def snapshot(self):
        return copy.deepcopy(
            {
                **{
                    name: value
                    for name, value in vars(self).items()
                    if name not in {"env", "latest", "diagnostics", "command_course"}
                },
                **(
                    {"command_course": self.command_course.snapshot()}
                    if self.command_course is not None
                    else {}
                ),
            }
        )

    def restore(self, state):
        for name, value in copy.deepcopy(state).items():
            if name == "command_course":
                if self.command_course is None:
                    raise ValueError("checkpoint requires command curriculum")
                self.command_course.restore(value)
            else:
                setattr(self, name, value)
