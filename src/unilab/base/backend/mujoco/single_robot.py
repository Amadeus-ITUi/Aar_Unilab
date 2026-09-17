"""Cold model loading and stepping interface for single floating-base robots."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


class SingleRobotSimulation:
    def __init__(
        self,
        model_path: Path,
        *,
        physics_hz: int,
        joint_order: tuple[str, ...],
        reset_keyframe: str | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        count = len(joint_order)
        if (self.model.nq, self.model.nv, self.model.nu) != (7 + count, 6 + count, count):
            raise ValueError(
                "model must have one free root and one hinge/actuator per policy joint"
            )
        actual = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
            for joint_id in self.model.actuator_trnid[:, 0]
        )
        if actual != joint_order:
            raise ValueError(f"model actuator joint order {actual} differs from {joint_order}")
        if not np.array_equal(self.model.jnt_qposadr[1:], np.arange(7, 7 + count)):
            raise ValueError("policy joints must follow the free root in position order")
        if not np.array_equal(self.model.actuator_trnid[:, 0], np.arange(1, 1 + count)):
            raise ValueError("actuator and generalized joint orders must match")
        self.model.opt.timestep = 1.0 / physics_hz
        self.data = mujoco.MjData(self.model)
        self._reset_keyframe_id = -1
        if reset_keyframe is not None:
            self._reset_keyframe_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, reset_keyframe
            )
            if self._reset_keyframe_id < 0:
                raise ValueError(f"reset keyframe {reset_keyframe!r} is missing from {model_path}")

    def reset(self, base_height: float | None = None) -> None:
        if self._reset_keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._reset_keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if base_height is not None:
            self.data.qpos[2] = base_height
        mujoco.mj_forward(self.model, self.data)

    def step(self, control: np.ndarray, substeps: int) -> None:
        self.data.ctrl[:] = control
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

    @property
    def joint_positions(self) -> np.ndarray:
        return np.asarray(self.data.qpos[7:])

    @property
    def joint_velocities(self) -> np.ndarray:
        return np.asarray(self.data.qvel[6:])

    @property
    def angular_velocity(self) -> np.ndarray:
        return np.asarray(self.data.qvel[3:6])

    @property
    def linear_velocity(self) -> np.ndarray:
        return np.asarray(self.data.qvel[:3])

    @property
    def control(self) -> np.ndarray:
        return np.asarray(self.data.ctrl)

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[2])


def configure_release_model(
    path: Path,
    *,
    scene_path: Path,
    physics_hz: int,
    base_height: float | None,
    reset_keyframe: str | None,
) -> None:
    """Bake runtime timing/reset overrides into an already copied release model."""
    tree = ET.parse(path)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(1.0 / physics_hz))
    body = root.find("worldbody/body")
    if body is None or body.find("freejoint") is None:
        raise ValueError("release model must have a floating root body")
    if base_height is not None:
        position = body.get("pos", "0 0 0").split()
        position[2] = str(base_height)
        body.set("pos", " ".join(position))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode")
    if reset_keyframe is not None:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, reset_keyframe)
        if key_id < 0:
            raise ValueError(f"release reset keyframe {reset_keyframe!r} is missing")
        if base_height is not None:
            scene = ET.parse(scene_path)
            key = next(
                (key for key in scene.findall("keyframe/key") if key.get("name") == reset_keyframe),
                None,
            )
            if key is None:
                raise ValueError("the overridden reset keyframe must be owned by the task scene")
            qpos = model.key_qpos[key_id].copy()
            qpos[2] = base_height
            key.set("qpos", " ".join(format(value, ".17g") for value in qpos))
            ET.indent(scene, space="  ")
            scene.write(scene_path, encoding="unicode")
