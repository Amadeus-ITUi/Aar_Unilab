"""Optional batched training telemetry and physical randomization interfaces.

No task/reward decisions live here. Contact samples are newest-first. Joint
acceleration and contacts are evaluated under actual control and external forces.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from unilab.base.backend.mujoco.batched_robot import BatchedRobotSimulation


def rotate_vectors(q, v):
    xyz = q[..., 1:]
    t = 2 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


class TrainingRobotSimulation(BatchedRobotSimulation):
    def __init__(
        self,
        *args,
        contact_hz=200,
        contact_history_length=4,
        ground_contact_history=False,
        **kwargs,
    ):
        native_pd = kwargs.pop("native_pd", True)
        super().__init__(*args, native_pd=True, track_gait_kinematics=True, **kwargs)
        if not hasattr(self.pool._pool, "has_joint_pd_telemetry"):
            self.close()
            raise RuntimeError(
                "Training telemetry requires rebuilding the supported native extension"
            )
        self.native_pd = native_pd
        self.fused_telemetry = bool(getattr(self.pool._pool, "has_joint_pd_telemetry", False))
        self.telemetry_chunk_size = None
        self._contact_addresses = np.ascontiguousarray(self.contact_adr, dtype=np.int32)
        self._joint_acceleration = np.zeros((self.num_envs, self.action_size))
        model = self.model
        self.contact_stride = round(1 / (contact_hz * self.dt))
        if self.contact_stride < 1 or not np.isclose(self.contact_stride * self.dt * contact_hz, 1):
            raise ValueError("contact frequency must divide physics frequency")
        self.contact_history = np.zeros(
            (self.num_envs, contact_history_length, len(self.body_ids), 3)
        )
        self.record_ground_history = ground_contact_history
        if ground_contact_history:
            self.ground_contact_history = np.zeros_like(self.contact_history)
            self._ground_addresses = np.ascontiguousarray(
                self.gait_sensor_adr["ground"], dtype=np.int32
            )
            self._contact_addresses = np.r_[self._contact_addresses, self._ground_addresses]
        bodies = list(self.body_ids)
        inertia = []
        for body in bodies:
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, model.body_iquat[body])
            rotation = matrix.reshape(3, 3)
            inertia.append(rotation @ np.diag(model.body_inertia[body]) @ rotation.T)
        self.nominal_inertia = np.array(inertia)
        self.nominal_mass = model.body_mass[self.body_ids].copy()
        self.nominal_joint_range = self.joint_range.copy()
        self.root_mass = np.full(self.num_envs, self.nominal_mass[0])
        self.root_inertia = np.tile(self.nominal_inertia[0], (self.num_envs, 1, 1))

    @property
    def joint_acceleration(self):
        return self._joint_acceleration

    @property
    def foot_linear_velocity(self):
        return self.sensors[:, self.gait_sensor_adr["framelinvel"][:, None] + np.arange(3)]

    @property
    def foot_link_positions(self):
        return self.sensors[:, self.sole_pos_adr[:, None] + np.arange(3)]

    def configure_physics(self, *, added_mass, link_scale, mass_scale, com_offset, friction):
        """Apply already-sampled startup parameters; store actual dynamics for impulses."""
        n = self.num_envs
        masses = np.tile(self.base_mass, (n, 1))
        inertia = np.tile(self.base_inertia, (n, 1, 1))
        masses[:, self.body_ids[0]] += added_mass
        masses[:, self.body_ids[1:]] *= link_scale
        # Match the reference's sequential mass events; only its final event
        # jointly scales inertia (the earlier mass-only events leave it unchanged).
        masses[:, self.body_ids] *= mass_scale
        inertia[:, self.body_ids] *= mass_scale[..., None]
        if np.any(masses[:, self.body_ids] <= 0):
            raise ValueError("mass randomization produced non-positive mass")
        com = np.tile(self.base_ipos, (n, 1, 1))
        com[:, self.body_ids] += com_offset
        coefficients = np.tile(self.base_friction, (n, 1, 1))
        coefficients[:, np.r_[self.robot_geoms, self.ground_geoms], 0] = friction[:, None]
        self.randomization = dict(
            body_mass=masses, body_inertia=inertia, body_ipos=com, geom_friction=coefficients
        )
        self.root_mass[:] = masses[:, self.body_ids[0]]
        self.root_inertia[:] = self.nominal_inertia[0] * mass_scale[:, 0, None, None]

    def set_root_wrench(self, force_body, torque_body):
        """Combine task impulse with the interactive player's world-frame wrench."""
        wrench = np.zeros((self.num_envs, self.body_count, 6))
        wrench[:, self.body_ids[0], :3] = rotate_vectors(self.qpos[:, 3:7], force_body)
        wrench[:, self.body_ids[0], 3:] = rotate_vectors(self.qpos[:, 3:7], torque_body)
        return wrench

    def reset(self, ids, qpos, qvel):
        super().reset(ids, qpos, qvel)
        self.contact_history[ids] = 0
        if self.record_ground_history:
            self.ground_contact_history[ids] = 0
        sensors, acceleration = self._controlled_forward(ids=ids)
        self.sensors[ids] = sensors
        self._joint_acceleration[ids] = acceleration[:, 6:]

    def _controlled_forward(self, push=None, ids=None):
        # Upstream post-step sensors clear controls/forces. This opt-in pass
        # evaluates the same final state under the actual applied wrench.
        wrench = np.zeros((self.num_envs, self.body_count, 6))
        if self.external_wrench is not None:
            wrench += self.external_wrench
        if push is not None:
            wrench[:, self.body_ids[0], :3] += push[:, -1]
        return self.pool._pool.forward_controlled(
            np.ascontiguousarray(self.state),
            np.ascontiguousarray(self.control),
            np.ascontiguousarray(wrench.reshape(self.num_envs, -1)),
            ids=None if ids is None else np.ascontiguousarray(ids, dtype=np.int32),
        )

    def step(self, actions, substeps, *, push_forces=None):
        if substeps % self.contact_stride:
            raise ValueError("policy substeps must divide contact sampling cadence")
        if self.native_pd and self.fused_telemetry:
            self._step_with_telemetry(actions, substeps, push_forces)
            return
        for start in range(0, substeps, self.contact_stride):
            push = (
                None if push_forces is None else push_forces[:, start : start + self.contact_stride]
            )
            super().step(actions, self.contact_stride, push_forces=push)
            sensors, acceleration = self._controlled_forward(push)
            self.sensors[:] = sensors
            self._joint_acceleration[:] = acceleration[:, 6:]
            self.contact_history[:, 1:] = self.contact_history[:, :-1].copy()
            # MuJoCo contact sensor with body1 reports the opposite force sign.
            self.contact_history[:, 0] = -self.contact_forces
            if self.record_ground_history:
                self.ground_contact_history[:, 1:] = self.ground_contact_history[:, :-1].copy()
                self.ground_contact_history[:, 0] = -self.sensors[
                    :, self._ground_addresses[:, None] + np.arange(3)
                ]

    def _step_with_telemetry(self, actions, substeps, push_forces):
        wrench = None
        if push_forces is not None or self.external_wrench is not None:
            wrench = np.zeros((self.num_envs, substeps, self.body_count, 6))
            if push_forces is not None:
                wrench[:, :, self.body_ids[0], :3] = push_forces
            if self.external_wrench is not None:
                wrench += self.external_wrench[:, None]
        state, sensors, torque, velocity, fifo, contacts, acceleration = (
            self.pool._pool.step_joint_position_pd(
                nstep=substeps,
                state0=self.state,
                actions=np.ascontiguousarray(actions, dtype=np.float64),
                kp=self.kp,
                kd=self.kd,
                torque_scale=self.torque_scale,
                torque_limits=self.torque_limits,
                gear=self.gear,
                nominal=self.default_position,
                initial_velocity=self.joint_velocity,
                qpos_adr=self.qpos_adr,
                qvel_adr=self.qvel_adr,
                delay_steps=np.ascontiguousarray(self.delay_steps, dtype=np.int32),
                initial_fifo=self.delay_buffer,
                position_difference=self.position_difference,
                wrench=wrench,
                telemetry_stride=self.contact_stride,
                contact_adr=self._contact_addresses,
                chunk_size=self.telemetry_chunk_size,
            )
        )
        self.state[:], self.sensors[:] = state, sensors
        self.torque[:], self.joint_velocity[:] = torque, velocity
        self.control[:] = torque / self.gear
        self.delay_buffer[:] = fifo
        self._joint_acceleration[:] = acceleration[:, 6:]
        count = min(contacts.shape[1], self.contact_history.shape[1])
        self.contact_history[:, count:] = self.contact_history[:, :-count].copy()
        self.contact_history[:, :count] = -contacts[:, :count, : len(self.body_ids)]
        if self.record_ground_history:
            self.ground_contact_history[:, count:] = self.ground_contact_history[:, :-count].copy()
            self.ground_contact_history[:, :count] = -contacts[:, :count, len(self.body_ids) :]
        self.steps += substeps
        if not np.isfinite(self.state).all():
            raise FloatingPointError("non-finite MuJoCo state")

    def snapshot(self):
        return {
            **super().snapshot(),
            "contact_history": self.contact_history.copy(),
            "root_mass": self.root_mass.copy(),
            "root_inertia": self.root_inertia.copy(),
            "sensors": self.sensors.copy(),
            "_joint_acceleration": self._joint_acceleration.copy(),
            **(
                {"ground_contact_history": self.ground_contact_history.copy()}
                if self.record_ground_history
                else {}
            ),
        }

    def restore(self, snapshot):
        super().restore(snapshot)
        # Preserve the checkpoint's post-step acceleration/contact sample exactly.
        self.sensors[:] = copy.deepcopy(snapshot["sensors"])
