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
    native_identified_pd_available,
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


def robot_xml_path(scene_path: Path) -> Path:
    """Resolve the robot of a supported single-include scene on a cold path."""
    include = ET.parse(scene_path).getroot().find("include")
    return scene_path if include is None else scene_path.parent / include.attrib["file"]


def compile_robot_scene(
    path: Path,
    bodies: tuple[str, ...],
    *,
    visual: bool,
    visual_style: Path | None = None,
    ground_contact_bodies: tuple[str, ...] = (),
    foot_clearance_bodies: tuple[str, ...] = (),
    foot_motion_bodies: tuple[str, ...] = (),
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
    for body_name in ground_contact_bodies:
        # World-body contact excludes foot/foot and foot/robot self contacts.
        ET.SubElement(
            sensor,
            "contact",
            name=f"batch_ground_contact_{body_name}",
            body1=body_name,
            body2="world",
            data="force",
            reduce="netforce",
            num="1",
        )
    for body_name in foot_clearance_bodies:
        # Collision vertices use the link frame; objtype="body" instead reports its COM frame.
        ET.SubElement(
            sensor,
            "framepos",
            name=f"batch_sole_pos_{body_name}",
            objtype="xbody",
            objname=body_name,
        )
        ET.SubElement(
            sensor,
            "framequat",
            name=f"batch_sole_quat_{body_name}",
            objtype="xbody",
            objname=body_name,
        )
    for body_name in foot_motion_bodies:
        for tag in ("framelinvel", "frameangvel"):
            ET.SubElement(
                sensor, tag, name=f"batch_{tag}_{body_name}", objtype="xbody", objname=body_name
            )
        ET.SubElement(
            sensor,
            "contact",
            name=f"batch_contact_point_{body_name}",
            body1=body_name,
            body2="world",
            data="found force pos",
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
        track_foot_ground_contact: bool = False,
        track_foot_clearance: bool = False,
        track_gait_kinematics: bool = False,
        actuator_model: dict[str, Any] | None = None,
    ) -> None:
        self.model = compile_robot_scene(
            path,
            body_names,
            visual=visual,
            visual_style=visual_style,
            ground_contact_bodies=(
                body_names
                if track_gait_kinematics
                else foot_names
                if track_foot_ground_contact
                else ()
            ),
            foot_clearance_bodies=foot_names
            if track_foot_clearance or track_gait_kinematics
            else (),
            foot_motion_bodies=foot_names if track_gait_kinematics else (),
        )
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
        self.sum_abs_pd = actuator_model is not None
        if actuator_model is not None:
            from unilab.base.backend.mujoco.actuator_parameters import apply_actuator_parameters

            apply_actuator_parameters(model, joint_order, actuator_model)
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
        self.foot_ground_contact_adr = np.array(
            [model.sensor(f"batch_ground_contact_{name}").adr[0] for name in foot_names]
            if track_foot_ground_contact or track_gait_kinematics
            else [],
            dtype=np.int64,
        )
        self.sole_pos_adr = np.array(
            [model.sensor(f"batch_sole_pos_{name}").adr[0] for name in foot_names]
            if track_foot_clearance or track_gait_kinematics
            else [],
            dtype=np.int64,
        )
        self.sole_quat_adr = np.array(
            [model.sensor(f"batch_sole_quat_{name}").adr[0] for name in foot_names]
            if track_foot_clearance or track_gait_kinematics
            else [],
            dtype=np.int64,
        )
        self.base_mass = model.body_mass.copy()
        self.total_mass = float(model.body_mass.sum())
        self.gravity_acceleration = float(np.linalg.norm(model.opt.gravity))
        self.gait_sensor_adr = {}
        if track_gait_kinematics:
            for key in ("framelinvel", "frameangvel", "contact_point"):
                self.gait_sensor_adr[key] = np.array(
                    [model.sensor(f"batch_{key}_{name}").adr[0] for name in foot_names]
                )
            self.gait_sensor_adr["ground"] = np.array(
                [model.sensor(f"batch_ground_contact_{name}").adr[0] for name in body_names]
            )
        self.base_inertia = model.body_inertia.copy()
        self.base_ipos = model.body_ipos.copy()
        self.base_friction = model.geom_friction.copy()
        self.ground_geoms = np.flatnonzero(
            (model.geom_bodyid == 0) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
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
        self.reset_geometry = None
        if track_gait_kinematics:
            from unilab.base.backend.mujoco.reset_geometry import ResetSoleGeometry

            self.reset_geometry = ResetSoleGeometry(
                model, foot_names, self.foot_vertices, self.foot_spheres
            )
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
        native_available = (
            native_identified_pd_available
            if self.sum_abs_pd
            else native_joint_position_pd_available
        )
        self.native_pd = native_pd and native_available()
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

    @property
    def foot_ground_forces(self) -> np.ndarray:
        """Net sole/world forces in configured foot order, excluding self contacts."""
        return self.sensors[:, self.foot_ground_contact_adr[:, None] + np.arange(3)]

    def foot_heights(self) -> np.ndarray:
        """Legacy height convention retained for existing rewards and checkpoints."""
        positions = self.body_positions[:, self.foot_indices]
        quats = self.body_quaternions[:, self.foot_indices]
        return self._foot_support_heights(positions, quats)

    def foot_clearances(self) -> np.ndarray:
        """Lowest collision point above the z=0 plane, using actual link-frame poses."""
        if not len(self.sole_pos_adr):
            raise RuntimeError("foot clearances require track_foot_clearance=True")
        positions = self.sensors[:, self.sole_pos_adr[:, None] + np.arange(3)]
        quats = self.sensors[:, self.sole_quat_adr[:, None] + np.arange(4)]
        return self._foot_support_heights(positions, quats)

    def gait_foot_state(self, reference_points: np.ndarray) -> dict[str, np.ndarray]:
        """World-frame sole reference/contact motion and upward ground reaction.

        Netforce contact sensors report the force exerted by body1 on body2.
        Negating that force gives the ground reaction acting on the robot.
        Contact position is a force-weighted point; evaluate rigid-body velocity
        there rather than differencing a moving contact location on a curved sole.
        """
        if not self.gait_sensor_adr:
            raise RuntimeError("gait foot state requires track_gait_kinematics=True")
        origin = self.sensors[:, self.sole_pos_adr[:, None] + np.arange(3)]
        quat = self.sensors[:, self.sole_quat_adr[:, None] + np.arange(4)]
        local = np.broadcast_to(reference_points, origin.shape)
        cross = 2 * np.cross(quat[..., 1:], local)
        reference = origin + local + quat[..., :1] * cross + np.cross(quat[..., 1:], cross)
        linear = self.sensors[:, self.gait_sensor_adr["framelinvel"][:, None] + np.arange(3)]
        angular = self.sensors[:, self.gait_sensor_adr["frameangvel"][:, None] + np.arange(3)]
        contact = self.sensors[:, self.gait_sensor_adr["contact_point"][:, None] + np.arange(7)]
        point = np.where((contact[..., 0] > 0)[..., None], contact[..., 4:7], reference)
        return {
            "reference_position": reference,
            "reference_velocity": linear + np.cross(angular, reference - origin),
            "contact_velocity": linear + np.cross(angular, point - origin),
            "ground_force": -contact[..., 1:4],
            "body_ground_force": -self.sensors[
                :, self.gait_sensor_adr["ground"][:, None] + np.arange(3)
            ],
            "clearance": self.foot_clearances(),
        }

    def _foot_support_heights(self, positions: np.ndarray, quats: np.ndarray) -> np.ndarray:
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
        if delay_steps.shape not in {(self.num_envs,), (self.num_envs, self.action_size)}:
            raise ValueError("delay_steps must be per environment or per environment/joint")
        if not np.isfinite(delay_steps).all() or (delay_steps < 0).any():
            raise ValueError("delay_steps must be finite and nonnegative")
        self.delay_steps = np.rint(delay_steps).astype(np.int64)
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

    def align_reset_soles(self, qpos: np.ndarray, clearance: float) -> np.ndarray:
        """Align the lowest collision sole before the single normal pool reset."""
        if self.reset_geometry is None:
            raise RuntimeError("sole alignment requires gait kinematics")
        correction = clearance - self.reset_geometry.heights(qpos).min(axis=1)
        qpos[:, 2] += correction
        return correction

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
            delayed = (
                self.delay_buffer[rows, self.delay_steps]
                if self.delay_steps.ndim == 1
                else self.delay_buffer[rows[:, None], self.delay_steps, np.arange(self.action_size)]
            )
            desired = self.default_position + delayed
            p = self.kp * (desired - self.qpos[:, 7:])
            d = -self.kd * self.joint_velocity
            gain_scale = (
                np.minimum(1.0, self.torque_limits / np.maximum(np.abs(p) + np.abs(d), 1e-300))
                if self.sum_abs_pd
                else 1.0
            )
            self.torque[:] = np.clip(
                (p + d) * gain_scale * self.torque_scale,
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
        options = {"sum_abs_pd": True} if self.sum_abs_pd else {}
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
            **options,
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
