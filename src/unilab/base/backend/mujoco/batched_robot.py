"""Independent robot simulations with parallel MuJoCo stepping and cold asset metadata.

Only this backend knows MuJoCo layouts. Environment owners supply PD targets,
randomization samples and task rules. The reusable robot XML is never rewritten.
"""

from __future__ import annotations

import copy
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco.batch_env import BatchEnvPool

from unilab.base.backend.mujoco.native_batch import (
    NativeMixedPdBatchEnvPool,
    native_joint_position_pd_available,
)


def _apply_visual_style(root: ET.Element, path: Path) -> None:
    """Apply a playback-only appearance fragment without changing physical geoms."""
    style = ET.parse(path).getroot()
    for element in style:
        if element.tag in {"visual", "asset"}:
            target = root.find(element.tag)
            if target is None:
                root.append(copy.deepcopy(element))
            else:
                target.extend(copy.deepcopy(list(element)))
        elif element.tag == "worldbody":
            geoms = {geom.get("name"): geom for geom in root.findall(".//geom")}
            for override in element:
                if override.tag != "geom" or set(override.attrib) - {"name", "material", "rgba"}:
                    raise ValueError("playback geom overrides support only name, material and rgba")
                name = override.get("name")
                if name is None or name not in geoms:
                    raise ValueError(f"unknown playback geom: {name}")
                geoms[name].attrib.update(override.attrib)
        else:
            raise ValueError(f"unsupported playback appearance element: {element.tag}")


def compile_robot_scene(
    path: Path,
    bodies: tuple[str, ...],
    *,
    visual: bool,
    visual_style: Path | None = None,
) -> mujoco.MjModel:
    scene = ET.parse(path).getroot()
    include = scene.find("include")
    if include is None:
        root = scene
        asset_root = path.parent
    else:
        robot_path = path.parent / include.attrib["file"]
        root = ET.parse(robot_path).getroot()
        asset_root = robot_path.parent
        for element in scene:
            if element.tag == "worldbody":
                worldbody = root.find("worldbody")
                if worldbody is None:
                    raise ValueError("robot XML is missing worldbody")
                worldbody.extend(copy.deepcopy(list(element)))
            elif element.tag != "include":
                root.append(copy.deepcopy(element))
    if visual and visual_style is not None:
        _apply_visual_style(root, visual_style)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((asset_root / compiler.get("meshdir", "")).resolve()))
    if not visual:
        for body in root.findall(".//body"):
            for geom in list(body.findall("geom")):
                if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                    body.remove(geom)
        used = {geom.get("mesh") for geom in root.findall(".//geom")}
        asset = root.find("asset")
        if asset is not None:
            for mesh in list(asset):
                if mesh.tag == "mesh" and mesh.get("name") not in used:
                    asset.remove(mesh)
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for body_name in bodies:
        ET.SubElement(
            sensor, "framepos", name=f"batch_pos_{body_name}", objtype="body", objname=body_name
        )
        ET.SubElement(
            sensor, "framequat", name=f"batch_quat_{body_name}", objtype="body", objname=body_name
        )
        ET.SubElement(
            sensor,
            "contact",
            name=f"batch_contact_{body_name}",
            body1=body_name,
            data="force",
            reduce="netforce",
            num="1",
        )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class BatchedRobotSimulation:
    def __init__(
        self,
        path: Path,
        *,
        num_envs: int,
        joint_order: tuple[str, ...],
        body_names: tuple[str, ...],
        foot_names: tuple[str, ...],
        physics_hz: int,
        nthread: int,
        keyframe: str,
        visual: bool = False,
        visual_style: Path | None = None,
        native_pd: bool = False,
        foot_contact_geoms: tuple[str, ...] | None = None,
    ) -> None:
        self.model = compile_robot_scene(path, body_names, visual=visual, visual_style=visual_style)
        self.model.opt.timestep = 1.0 / physics_hz
        self.dt = 1.0 / physics_hz
        self.num_envs = num_envs
        self.action_size = len(joint_order)
        model = self.model
        self.body_count = model.nbody
        if (model.nq, model.nv, model.nu) != (
            7 + len(joint_order),
            6 + len(joint_order),
            len(joint_order),
        ):
            raise ValueError("batch robot requires a free root followed by one hinge per actuator")
        actual = tuple(model.joint(int(index)).name for index in model.actuator_trnid[:, 0])
        if actual != joint_order:
            raise ValueError(f"actuator order mismatch: {actual}")
        np.testing.assert_array_equal(model.jnt_qposadr[1:], np.arange(7, model.nq))
        np.testing.assert_array_equal(model.jnt_dofadr[1:], np.arange(6, model.nv))
        self.home = model.key(keyframe).qpos.copy()
        self.joint_range = model.jnt_range[1:].copy()
        self.gear = model.actuator_gear[:, 0].copy()
        self.qpos_adr = np.ascontiguousarray(model.jnt_qposadr[1:], dtype=np.int32)
        self.qvel_adr = np.ascontiguousarray(model.jnt_dofadr[1:], dtype=np.int32)
        if np.any(self.gear <= 0):
            raise ValueError("batch PD requires positive actuator gear")
        self.body_names = body_names
        self.body_ids = np.array([model.body(name).id for name in body_names])
        self.foot_indices = np.array([body_names.index(name) for name in foot_names])
        self.pos_adr = np.array([model.sensor(f"batch_pos_{name}").adr[0] for name in body_names])
        self.quat_adr = np.array([model.sensor(f"batch_quat_{name}").adr[0] for name in body_names])
        self.contact_adr = np.array(
            [model.sensor(f"batch_contact_{name}").adr[0] for name in body_names]
        )
        self.base_mass = model.body_mass.copy()
        self.base_inertia = model.body_inertia.copy()
        self.base_ipos = model.body_ipos.copy()
        self.base_friction = model.geom_friction.copy()
        self.robot_geoms = np.flatnonzero(
            (model.geom_bodyid != 0) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
        if foot_contact_geoms is not None:
            self._validate_foot_contact_geoms(foot_names, foot_contact_geoms)
        self.foot_vertices = []
        self.foot_spheres = []
        for name in foot_names:
            body_id = model.body(name).id
            vertices = []
            spheres = []
            for geom in self.robot_geoms[model.geom_bodyid[self.robot_geoms] == body_id]:
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    spheres.append(np.r_[model.geom_pos[geom], model.geom_size[geom, 0]])
                    continue
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError("foot clearance requires convex meshes or spheres")
                mesh = model.geom_dataid[geom]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                vertices.append(
                    model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
                    + model.geom_pos[geom]
                )
            if not vertices and not spheres:
                raise ValueError(f"foot {name!r} has no supported collision geometry")
            self.foot_vertices.append(np.concatenate(vertices) if vertices else np.empty((0, 3)))
            self.foot_spheres.append(np.array(spheres).reshape(-1, 4))
        # Pool clones models. Fail before allocating, rather than silently changing num_envs.
        meminfo = Path("/proc/meminfo").read_text()
        available = next(
            int(line.split()[1]) * 1024
            for line in meminfo.splitlines()
            if line.startswith("MemAvailable:")
        )
        estimated = model.nbuffer * num_envs + 512 * 1024**2
        if estimated > available * 0.8:
            raise MemoryError(
                f"{num_envs} MuJoCo models need at least {estimated / 1024**3:.1f} GiB; "
                f"available {available / 1024**3:.1f} GiB. Set algo.num_envs=256 "
                "or choose an explicit count fitting this host."
            )
        self.native_pd = native_pd and native_joint_position_pd_available()
        if native_pd and not self.native_pd:
            warnings.warn(
                "Native joint-position PD is unavailable; using the equivalent Python loop. "
                "Build third_party/mujoco_uni_mixed_pd/build_extension.sh for faster training.",
                RuntimeWarning,
                stacklevel=2,
            )
        pool_type = NativeMixedPdBatchEnvPool if self.native_pd else BatchEnvPool
        self.pool = pool_type(model, nbatch=num_envs, nthread=min(num_envs, nthread))
        self.state = np.zeros((num_envs, self.pool.nstate))
        self.sensors = np.zeros((num_envs, model.nsensordata))
        self.joint_velocity = np.zeros((num_envs, model.nu))
        self.torque = np.zeros_like(self.joint_velocity)
        self.control = np.zeros_like(self.joint_velocity)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.delay_buffer = np.empty((num_envs, 1, model.nu))
        self.randomization: dict[str, np.ndarray] = {}
        self.external_wrench: np.ndarray | None = None

    def _validate_foot_contact_geoms(
        self, foot_names: tuple[str, ...], geom_names: tuple[str, ...]
    ) -> None:
        """Ensure body contact sensors represent exactly the requested sole geoms.

        This cold-path contract avoids extra per-geom sensors in every physics step.
        A foot with additional collision geometry needs explicit contact classification
        before it can use this mode; it must not silently become exempt from penalties.
        """
        if len(geom_names) != len(foot_names) or len(set(geom_names)) != len(geom_names):
            raise ValueError("foot_contact_geoms requires one unique geom per foot")
        for body_name, geom_name in zip(foot_names, geom_names, strict=True):
            body_id = self.model.body(body_name).id
            geoms = self.robot_geoms[self.model.geom_bodyid[self.robot_geoms] == body_id]
            actual = tuple(self.model.geom(int(geom)).name for geom in geoms)
            if actual != (geom_name,) or np.any(self.model.body_parentid[1:] == body_id):
                raise ValueError(
                    f"foot_contact_geoms: {body_name!r} must be a leaf body with only "
                    f"collision geom {geom_name!r}; found {actual}. "
                    "Additional foot geometry requires explicit contact classification."
                )

    @property
    def qpos(self) -> np.ndarray:
        return self.state[:, 1 : 1 + self.model.nq]

    @property
    def qvel(self) -> np.ndarray:
        return self.state[:, 1 + self.model.nq : 1 + self.model.nq + self.model.nv]

    @property
    def body_positions(self) -> np.ndarray:
        return self.sensors[:, self.pos_adr[:, None] + np.arange(3)]

    @property
    def body_quaternions(self) -> np.ndarray:
        return self.sensors[:, self.quat_adr[:, None] + np.arange(4)]

    @property
    def contact_forces(self) -> np.ndarray:
        return self.sensors[:, self.contact_adr[:, None] + np.arange(3)]

    def foot_heights(self) -> np.ndarray:
        positions = self.body_positions[:, self.foot_indices]
        quats = self.body_quaternions[:, self.foot_indices]
        heights = np.empty((self.num_envs, len(self.foot_indices)))
        for index, vertices in enumerate(self.foot_vertices):
            w, x, y, z = quats[:, index].T
            zaxis = np.column_stack(
                (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
            )
            supports = []
            if len(vertices):
                supports.append((zaxis @ vertices.T).min(axis=1))
            spheres = self.foot_spheres[index]
            if len(spheres):
                supports.append((zaxis @ spheres[:, :3].T - spheres[:, 3]).min(axis=1))
            heights[:, index] = positions[:, index, 2] + np.minimum.reduce(supports)
        return np.clip(heights, 0, 1)

    def configure_pd(
        self,
        *,
        kp: np.ndarray,
        kd: np.ndarray,
        torque_scale: np.ndarray,
        torque_limits: np.ndarray,
        default_position: np.ndarray,
        delay_steps: np.ndarray,
        position_difference: bool,
    ) -> None:
        self.kp, self.kd, self.torque_scale = kp.copy(), kd.copy(), torque_scale.copy()
        self.torque_limits = torque_limits.copy()
        self.default_position = default_position.copy()
        self.delay_steps = delay_steps.astype(np.int64).copy()
        self.position_difference = position_difference
        self.delay_buffer = np.zeros(
            (self.num_envs, int(self.delay_steps.max()) + 1, self.action_size)
        )

    def reset(self, ids: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        if not len(ids):
            return
        initial = np.zeros((len(ids), self.pool.nstate))
        initial[:, 1 : 1 + self.model.nq] = qpos
        initial[:, 1 + self.model.nq : 1 + self.model.nq + self.model.nv] = qvel
        state, sensors = self.pool.reset(
            ids,
            initial,
            randomization={key: value[ids] for key, value in self.randomization.items()} or None,
        )
        self.state[ids], self.sensors[ids] = state, sensors
        self.joint_velocity[ids] = qvel[:, 6:]
        self.torque[ids], self.control[ids] = 0, 0
        self.steps[ids], self.delay_buffer[ids] = 0, 0

    def step(
        self, actions: np.ndarray, substeps: int, *, push_forces: np.ndarray | None = None
    ) -> None:
        if self.native_pd:
            self._step_native(actions, substeps, push_forces=push_forces)
            return
        rows = np.arange(self.num_envs)
        for index in range(substeps):
            self.delay_buffer[:, 1:] = self.delay_buffer[:, :-1]
            self.delay_buffer[:, 0] = actions
            delayed = self.delay_buffer[rows, self.delay_steps]
            desired = self.default_position + delayed
            self.torque[:] = np.clip(
                (self.kp * (desired - self.qpos[:, 7:]) - self.kd * self.joint_velocity)
                * self.torque_scale,
                -self.torque_limits,
                self.torque_limits,
            )
            self.control[:] = self.torque / self.gear
            control = self.control[:, None, :]
            spec = int(mujoco.mjtState.mjSTATE_CTRL)
            if push_forces is not None or self.external_wrench is not None:
                spec |= int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
                wrench = np.zeros((self.num_envs, self.body_count, 6))
                if push_forces is not None:
                    wrench[:, self.body_ids[0], :3] = push_forces[:, index]
                if self.external_wrench is not None:
                    wrench += self.external_wrench
                control = np.concatenate((control, wrench.reshape(self.num_envs, 1, -1)), axis=-1)
            previous = self.qpos[:, 7:].copy()
            final = index == substeps - 1
            result = self.pool.step(
                self.state,
                nstep=1,
                control_spec=spec,
                control=control,
                return_sensor=final,
                post_step_forward_sensor=final,
            )
            if final:
                self.state[:], self.sensors[:] = result
            else:
                self.state[:] = result
            self.joint_velocity[:] = (
                ((self.qpos[:, 7:] - previous + np.pi) % (2 * np.pi) - np.pi) / self.dt
                if self.position_difference
                else self.qvel[:, 6:]
            )
            self.steps += 1
        if not np.isfinite(self.state).all():
            raise FloatingPointError("non-finite MuJoCo state")

    def _step_native(
        self, actions: np.ndarray, substeps: int, *, push_forces: np.ndarray | None
    ) -> None:
        wrench = None
        if push_forces is not None or self.external_wrench is not None:
            wrench = np.zeros((self.num_envs, substeps, self.body_count, 6))
            if push_forces is not None:
                wrench[:, :, self.body_ids[0], :3] = push_forces
            if self.external_wrench is not None:
                wrench += self.external_wrench[:, None]
        state, sensors, torque, velocity, fifo = self.pool._pool.step_joint_position_pd(
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
        )
        self.state[:], self.sensors[:] = state, sensors
        self.torque[:], self.joint_velocity[:] = torque, velocity
        self.control[:] = torque / self.gear
        self.delay_buffer[:] = fifo
        self.steps += substeps
        if not np.isfinite(self.state).all():
            raise FloatingPointError("non-finite MuJoCo state")

    def snapshot(self) -> dict[str, Any]:
        names = (
            "state",
            "joint_velocity",
            "torque",
            "control",
            "steps",
            "delay_buffer",
            "kp",
            "kd",
            "torque_scale",
            "torque_limits",
            "default_position",
            "delay_steps",
        )
        return {
            **{name: getattr(self, name).copy() for name in names},
            "randomization": {key: value.copy() for key, value in self.randomization.items()},
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        for name, value in snapshot.items():
            setattr(self, name, copy.deepcopy(value))
        state, sensors = self.pool.reset(
            np.arange(self.num_envs), self.state, randomization=self.randomization or None
        )
        self.state[:], self.sensors[:] = state, sensors

    def sync_visual_data(self, data: mujoco.MjData) -> None:
        data.qpos[:] = self.qpos[0]
        data.qvel[:] = self.qvel[0]
        data.ctrl[:] = self.control[0]
        data.time = self.state[0, 0]
        mujoco.mj_forward(self.model, data)

    def create_visual_data(self) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        self.sync_visual_data(data)
        return data

    def stage_visual_forces(self, data: mujoco.MjData) -> None:
        self.external_wrench = (
            np.asarray(data.xfrc_applied)[None].copy() if np.any(data.xfrc_applied) else None
        )

    def close(self) -> None:
        self.pool.close()
