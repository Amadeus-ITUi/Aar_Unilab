"""Shared helpers for DR002 MuJoCo/PACE offline tools.

These helpers intentionally avoid Isaac Lab imports so they can run inside a
small standalone conda environment such as ``mujoco-sim2sim``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from chirp_frequency_response import SharedCommandDelayBuffer, coerce_shared_delay_steps

JOINT_NAMES = [
    "left_thigh_joint",
    "left_calf_joint",
    "left_foot_joint",
    "right_thigh_joint",
    "right_calf_joint",
    "right_foot_joint",
]
DEFAULT_IDENTIFICATION_MODEL = "src/unilab/assets/robots/dr002/we11/we11.xml"
DEFAULT_TRAINING_MODEL = "src/unilab/assets/robots/dr002/we11/we11.xml"
# Backward-compatible name used by the identification fitter.
DEFAULT_MODEL = DEFAULT_IDENTIFICATION_MODEL
DEFAULT_ANGLES = np.array(
    [0.8, -1.6, 0.0, 0.8, -1.6, 0.0],
    dtype=np.float64,
)
HAND_PLAY_KP = np.array([2.0, 7.59, 0.0, 2.0, 7.59, 0.0], dtype=np.float64)
HAND_PLAY_KD = np.array([0.080, 0.682, 0.05, 0.080, 0.682, 0.05], dtype=np.float64)
PARAM_VECTOR_ORDER = [
    "armature[6]",
    "viscous_friction[6]",
    "coulomb_friction[6]",
    "encoder_bias[6]",
    "command_delay_steps[1]",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_parent(path: str | Path) -> Path:
    p = resolve_repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def tensor_to_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be at least 1D")
    return arr


def load_chirp_data(path: str | Path) -> dict[str, np.ndarray]:
    import torch

    p = resolve_repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"chirp data not found: {p}")
    raw = torch.load(p, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a dict with time/dof_pos/des_dof_pos")

    time = tensor_to_numpy(raw["time"], "time").reshape(-1)
    target_key = "des_dof_pos" if "des_dof_pos" in raw else "target_dof_pos"
    if target_key not in raw:
        raise KeyError(f"{p} must contain des_dof_pos or target_dof_pos")
    des = tensor_to_numpy(raw[target_key], target_key)
    vel_key = None
    if "des_dof_vel" in raw:
        vel_key = "des_dof_vel"
    elif "target_dof_vel" in raw:
        vel_key = "target_dof_vel"
    des_vel = tensor_to_numpy(raw[vel_key], vel_key) if vel_key is not None else None
    dof = tensor_to_numpy(raw["dof_pos"], "dof_pos") if "dof_pos" in raw else None
    dof_vel = tensor_to_numpy(raw["dof_vel"], "dof_vel") if "dof_vel" in raw else None
    return {
        "time": time,
        "des_dof_pos": des,
        "des_dof_vel": des_vel,
        "dof_pos": dof,
        "dof_vel": dof_vel,
    }


def save_chirp_data(
    path: str | Path,
    time: np.ndarray,
    dof_pos: np.ndarray,
    des_dof_pos: np.ndarray,
    dof_vel: np.ndarray | None = None,
    des_dof_vel: np.ndarray | None = None,
) -> Path:
    import torch

    p = ensure_parent(path)
    payload = {
        # Replay samples are queried at exact controller ticks with a strict
        # previous-sample ZOH.  Keeping the clock in float64 prevents a
        # round-trip through float32 from moving nominal 200 Hz timestamps to
        # either side of those query ticks and selecting the preceding frame.
        "time": torch.as_tensor(time, dtype=torch.float64),
        "dof_pos": torch.as_tensor(dof_pos, dtype=torch.float32),
        "des_dof_pos": torch.as_tensor(des_dof_pos, dtype=torch.float32),
    }
    if dof_vel is not None:
        payload["dof_vel"] = torch.as_tensor(dof_vel, dtype=torch.float32)
    if des_dof_vel is not None:
        payload["des_dof_vel"] = torch.as_tensor(des_dof_vel, dtype=torch.float32)
    torch.save(payload, p)
    return p


def coerce_vector(value: Any, n: int, name: str, default: float | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise KeyError(f"missing parameter: {name}")
        return np.full(n, default, dtype=np.float64)
    if isinstance(value, dict):
        if n != len(JOINT_NAMES):
            raise ValueError(
                f"{name} uses joint-name keys but n={n}; expected the {len(JOINT_NAMES)}-joint WE order"
            )
        missing = [joint for joint in JOINT_NAMES if joint not in value]
        if missing:
            raise ValueError(f"{name} is missing WE joints: {missing}")
        value = [value[joint] for joint in JOINT_NAMES]
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"{name} must be scalar or length {n}, got length {arr.size}")
    return arr


def builtin_truth_params(n: int = 6) -> dict[str, Any]:
    """A deterministic physically reasonable DR002-like truth set."""
    return {
        "armature": [0.008, 0.018, 0.0012, 0.008, 0.018, 0.0012][:n],
        "viscous_friction": [0.10, 0.16, 0.025, 0.10, 0.16, 0.025][:n],
        "coulomb_friction": [0.025, 0.035, 0.008, 0.025, 0.035, 0.008][:n],
        "encoder_bias": [0.012, -0.018, 0.004, -0.010, 0.016, -0.003][:n],
        "motor_strength": [1.0] * n,
        "command_delay_steps": 4,
    }


def native_truth_params(n: int = 6) -> dict[str, Any]:
    """Near-zero overrides, useful when only testing the MuJoCo bridge."""
    return {
        "armature": [0.0] * n,
        "viscous_friction": [0.0] * n,
        "coulomb_friction": [0.0] * n,
        "encoder_bias": [0.0] * n,
        "motor_strength": [1.0] * n,
        "command_delay_steps": 0,
    }


def normalize_params(params: dict[str, Any], n: int = 6) -> dict[str, Any]:
    aliases = {
        "damping": "viscous_friction",
        "viscous_damping": "viscous_friction",
        "friction": "coulomb_friction",
        "static_friction": "coulomb_friction",
        "dynamic_friction": "coulomb_friction",
        "bias": "encoder_bias",
    }
    src = dict(params)
    for old, new in aliases.items():
        if old in src and new not in src:
            src[new] = src[old]

    delay_block = src.get("delay")
    if isinstance(delay_block, dict):
        if "shared_command_delay_steps" in delay_block and "command_delay_steps" not in src:
            src["command_delay_steps"] = delay_block["shared_command_delay_steps"]
        elif "command_delay_steps" in delay_block and "command_delay_steps" not in src:
            src["command_delay_steps"] = delay_block["command_delay_steps"]
        if "torque_delay_steps" in delay_block and "torque_delay_steps" not in src:
            src["torque_delay_steps"] = delay_block["torque_delay_steps"]

    raw_semantics = src.get("delay_semantics")
    if raw_semantics is None and isinstance(delay_block, dict):
        raw_semantics = delay_block.get("semantics")
    semantics = str(raw_semantics or "").strip().lower()
    generic_delay = None
    for key in ("time_lag", "delay_steps"):
        if key in src:
            generic_delay = src[key]
            break
    if generic_delay is None and "delay" in src and not isinstance(src["delay"], dict):
        generic_delay = src["delay"]
    if generic_delay is not None:
        if semantics in {
            "torque",
            "post_controller_torque_fifo",
            "post_controller_motor_torque_fifo",
        }:
            src.setdefault("torque_delay_steps", generic_delay)
        else:
            src.setdefault("command_delay_steps", generic_delay)

    command_delay_steps = coerce_shared_delay_steps(
        src.get("command_delay_steps", 0),
        "command_delay_steps",
    )
    torque_delay_steps = coerce_shared_delay_steps(
        src.get("torque_delay_steps", 0),
        "torque_delay_steps",
    )
    if command_delay_steps > 0 and torque_delay_steps > 0:
        order_text = " ".join(str(item).lower() for item in src.get("parameter_vector_order", []))
        legacy_torque_alias = (
            semantics
            in {
                "torque",
                "post_controller_torque_fifo",
                "post_controller_motor_torque_fifo",
            }
            and command_delay_steps == torque_delay_steps
            and "command_delay_steps" in order_text
            and "torque_delay_steps" not in order_text
        )
        if legacy_torque_alias:
            command_delay_steps = 0
        else:
            raise ValueError("command_delay_steps and torque_delay_steps cannot both be non-zero")

    out = {
        "armature": coerce_vector(src.get("armature"), n, "armature", default=0.0).tolist(),
        "viscous_friction": coerce_vector(
            src.get("viscous_friction"), n, "viscous_friction", default=0.0
        ).tolist(),
        "coulomb_friction": coerce_vector(
            src.get("coulomb_friction"), n, "coulomb_friction", default=0.0
        ).tolist(),
        "encoder_bias": coerce_vector(
            src.get("encoder_bias"), n, "encoder_bias", default=0.0
        ).tolist(),
        "motor_strength": coerce_vector(
            src.get("motor_strength"), n, "motor_strength", default=1.0
        ).tolist(),
        "command_delay_steps": command_delay_steps,
        "torque_delay_steps": torque_delay_steps,
    }
    return out


def params_to_vector(params: dict[str, Any], n: int = 6) -> np.ndarray:
    p = normalize_params(params, n)
    return np.concatenate(
        [
            np.asarray(p["armature"], dtype=np.float64),
            np.asarray(p["viscous_friction"], dtype=np.float64),
            np.asarray(p["coulomb_friction"], dtype=np.float64),
            np.asarray(p["encoder_bias"], dtype=np.float64),
            np.asarray([p["command_delay_steps"]], dtype=np.float64),
        ]
    )


def params_from_vector(
    vector: Any,
    n: int = 6,
    *,
    delay_semantics: str = "command",
) -> dict[str, Any]:
    arr = tensor_to_numpy(vector, "parameter_vector").reshape(-1)
    expected = 4 * n + 1
    if arr.size != expected:
        raise ValueError(
            f"parameter vector must have length {expected} for {n} joints, got {arr.size}"
        )
    if delay_semantics not in {"command", "torque"}:
        raise ValueError(f"unsupported vector delay semantics: {delay_semantics!r}")
    values = {
        "armature": arr[0:n],
        "viscous_friction": arr[n : 2 * n],
        "coulomb_friction": arr[2 * n : 3 * n],
        "encoder_bias": arr[3 * n : 4 * n],
        f"{delay_semantics}_delay_steps": arr[4 * n],
    }
    return normalize_params(values, n)


def _jsonify(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def extract_params(payload: Any, n: int = 6) -> dict[str, Any]:
    payload = _jsonify(payload)
    if isinstance(payload, list):
        return params_from_vector(payload, n)
    if not isinstance(payload, dict):
        raise ValueError("parameter payload must be a dict or vector")

    direct_parameter_keys = {
        "armature",
        "damping",
        "viscous_damping",
        "viscous_friction",
        "friction",
        "coulomb_friction",
        "encoder_bias",
        "bias",
    }
    if direct_parameter_keys.intersection(payload):
        return normalize_params(payload, n)

    raw_semantics = str(payload.get("delay_semantics", "")).strip().lower()
    parameter_vector_order = payload.get("parameter_vector_order", [])
    order_text = " ".join(str(item).lower() for item in parameter_vector_order)
    vector_delay_semantics = (
        "torque"
        if "torque_delay_steps" in order_text
        or raw_semantics
        in {
            "torque",
            "post_controller_torque_fifo",
            "post_controller_motor_torque_fifo",
        }
        else "command"
    )
    for key in ("truth_params", "pace_params", "best_params", "params", "mean"):
        if key in payload:
            nested = payload[key]
            if isinstance(nested, dict):
                return extract_params(nested, n)
            return params_from_vector(
                nested,
                n,
                delay_semantics=vector_delay_semantics,
            )
    for key in ("parameter_vector", "x"):
        if key in payload:
            return params_from_vector(
                payload[key],
                n,
                delay_semantics=vector_delay_semantics,
            )
    return normalize_params(payload, n)


def load_params_file(path: str | Path, n: int = 6) -> dict[str, Any]:
    p = resolve_repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"parameter file not found: {p}")
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            return extract_params(json.load(f), n)

    import torch

    return extract_params(torch.load(p, map_location="cpu"), n)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return p


def build_desired_trajectory(
    source_des: np.ndarray,
    default_angles: np.ndarray,
    excite_all: bool,
    test_motor: int,
    n: int = 6,
) -> np.ndarray:
    source_des = np.asarray(source_des, dtype=np.float64)
    if source_des.ndim == 1:
        source_des = source_des[:, None]
    if source_des.shape[1] == n:
        full = source_des.copy()
    elif source_des.shape[1] == 1:
        full = np.tile(default_angles.reshape(1, n), (source_des.shape[0], 1))
        if excite_all:
            delta = source_des[:, 0] - float(source_des[0, 0])
            full += delta[:, None]
        else:
            full[:, test_motor] = source_des[:, 0]
    else:
        raise ValueError(
            f"source des_dof_pos must have 1 or {n} columns, got {source_des.shape[1]}"
        )

    if not excite_all:
        reduced = np.tile(default_angles.reshape(1, n), (source_des.shape[0], 1))
        reduced[:, test_motor] = full[:, test_motor]
        return reduced
    return full


def _available_mujoco_joint_names(model: Any) -> list[str]:
    import mujoco

    names = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        names.append(name or f"<joint_{jid}>")
    return names


def make_meshless_urdf(path: str | Path) -> Path:
    """Create a temporary URDF copy without visual/collision mesh geometry.

    MuJoCo's URDF loader is stricter than Isaac's importer for some STL files.
    For chirp-based actuator identification we do not need render/collision
    meshes, so stripping visual/collision blocks is the most robust fallback.
    """

    src = resolve_repo_path(path)
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / "dr002_mujoco_meshless"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_{digest}.urdf"

    tree = ET.parse(src)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir", ""))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((src.parent / meshdir).resolve()))
    removed = 0
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for child in list(link.findall(tag)):
                link.remove(child)
                removed += 1
    if removed == 0:
        raise ValueError(f"no visual/collision blocks found to strip in {src}")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def make_meshless_mjcf(path: str | Path) -> Path:
    """Create a temporary MJCF copy with mesh assets/geoms removed."""

    src = make_self_contained_mjcf(path)
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / "dr002_mujoco_meshless"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_{digest}.xml"

    tree = ET.parse(src)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir", ""))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((src.parent / meshdir).resolve()))
    removed = 0

    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
            removed += 1

    for parent in root.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            # Fallback models are used for joint-space chirp identification, so
            # all visual/collision geoms can be removed. This also avoids
            # invalid generated primitives such as capsule size="0 0".
            parent.remove(child)
            removed += 1

    if removed == 0:
        raise ValueError(f"no mesh assets/geoms found to strip in {src}")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def make_self_contained_mjcf(path: str | Path) -> Path:
    """Flatten MJCF includes before moving a generated model into ``/tmp``.

    MuJoCo resolves ``<include>`` and compiler asset directories relative to
    the source XML. Identification helpers create modified temporary models;
    copying only a wrapper would therefore strand its includes. Monolithic
    MJCF files are returned unchanged.
    """

    import mujoco

    src = resolve_repo_path(path)
    if src.suffix.lower() not in (".xml", ".mjcf"):
        raise ValueError(f"include flattening only supports MJCF/XML models: {src}")
    source_tree = ET.parse(src)
    if not any(element.tag == "include" for element in source_tree.getroot().iter()):
        return src

    digest = hashlib.sha1(src.read_bytes() + b":self_contained_v1").hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / "dr002_mujoco_flattened"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_{digest}_flattened.xml"

    model = mujoco.MjModel.from_xml_path(str(src))
    mujoco.mj_saveLastXML(str(dst), model)
    flattened_tree = ET.parse(dst)
    compiler = flattened_tree.getroot().find("compiler")
    if compiler is not None:
        for attribute in ("assetdir", "meshdir", "texturedir"):
            raw = compiler.get(attribute)
            if raw and not Path(raw).is_absolute():
                compiler.set(attribute, str((src.parent / raw).resolve()))
    flattened_tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def make_fixed_base_mjcf(path: str | Path) -> Path:
    """Create a temporary MJCF copy with free joints removed.

    Keyframe ``qpos``/``qvel`` vectors are pruned at the same joint-space
    indices.  Otherwise a valid free-base model with keyframes becomes invalid
    after its free joint is removed (for example nq=13 becomes nq=6 while a
    keyframe still contains 13 qpos values).
    """

    src = make_self_contained_mjcf(path)
    if src.suffix.lower() not in (".xml", ".mjcf"):
        raise ValueError(f"fixed-base conversion only supports MJCF/XML models: {src}")

    digest = hashlib.sha1(src.read_bytes() + b":fixed_base").hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / "dr002_mujoco_fixed_base"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_{digest}_fixed_base.xml"

    tree = ET.parse(src)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir", ""))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((src.parent / meshdir).resolve()))

    # MuJoCo assigns joint state addresses in model/XML order. Record the
    # qpos/qvel spans occupied by free joints before removing the elements so
    # keyframes can be converted to the fixed-base state dimension as well.
    qpos_cursor = 0
    qvel_cursor = 0
    removed_qpos_indices: set[int] = set()
    removed_qvel_indices: set[int] = set()
    for element in root.iter():
        if element.tag == "freejoint":
            joint_type = "free"
        elif element.tag == "joint":
            joint_type = element.attrib.get("type", "hinge")
        else:
            continue
        qpos_width = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
        qvel_width = 6 if joint_type == "free" else 3 if joint_type == "ball" else 1
        if joint_type == "free":
            removed_qpos_indices.update(range(qpos_cursor, qpos_cursor + qpos_width))
            removed_qvel_indices.update(range(qvel_cursor, qvel_cursor + qvel_width))
        qpos_cursor += qpos_width
        qvel_cursor += qvel_width

    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "freejoint":
                parent.remove(child)
                removed += 1
            elif child.tag == "joint" and child.attrib.get("type") == "free":
                parent.remove(child)
                removed += 1

    if removed == 0:
        return src

    for keyframe in root.findall("keyframe"):
        for key in keyframe.findall("key"):
            for attr, expected_size, removed_indices in (
                ("qpos", qpos_cursor, removed_qpos_indices),
                ("qvel", qvel_cursor, removed_qvel_indices),
            ):
                raw = key.get(attr)
                if raw is None:
                    continue
                values = raw.split()
                if len(values) != expected_size:
                    raise ValueError(
                        f"keyframe '{key.get('name', '')}' {attr} has {len(values)} values; "
                        f"expected {expected_size} before fixed-base conversion"
                    )
                key.set(
                    attr,
                    " ".join(
                        value for index, value in enumerate(values) if index not in removed_indices
                    ),
                )

    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def load_mujoco_model_with_mesh_fallback(model_path: str | Path, fixed_base: bool = False):
    import mujoco

    resolved = resolve_repo_path(model_path)
    if fixed_base:
        resolved = make_fixed_base_mjcf(resolved)
    try:
        return mujoco.MjModel.from_xml_path(str(resolved)), resolved
    except ValueError as exc:
        msg = str(exc)
        if "mesh" not in msg.lower() and "stl" not in msg.lower():
            raise
        if resolved.suffix.lower() == ".urdf":
            meshless = make_meshless_urdf(resolved)
        elif resolved.suffix.lower() in (".xml", ".mjcf"):
            meshless = make_meshless_mjcf(resolved)
        else:
            raise
        print(
            "[WARN] MuJoCo failed to load mesh geometry from the model; "
            f"retrying with meshless temporary model: {meshless}"
        )
        return mujoco.MjModel.from_xml_path(str(meshless)), meshless


def _joint_addresses(model: Any, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    qpos_ids = []
    qvel_ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            available = ", ".join(_available_mujoco_joint_names(model))
            raise KeyError(
                f"joint '{name}' not found in MuJoCo model. Available joints: {available}"
            )
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        qvel_ids.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos_ids, dtype=np.int64), np.asarray(qvel_ids, dtype=np.int64)


def _free_root_addresses(model: Any) -> tuple[int, int] | None:
    import mujoco

    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return None


def simulate_mujoco_pd(
    model_path: str | Path,
    time: np.ndarray,
    des_dof_pos: np.ndarray,
    params: dict[str, Any],
    kp: float | list[float],
    kd: float | list[float],
    default_angles: np.ndarray,
    joint_names: list[str] | None = None,
    pin_base: bool = True,
    root_z: float = 0.35,
    disable_gravity: bool = True,
) -> dict[str, np.ndarray]:
    import mujoco

    joint_names = joint_names or JOINT_NAMES
    n = len(joint_names)
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    des_dof_pos = np.asarray(des_dof_pos, dtype=np.float64)
    default_angles = coerce_vector(default_angles, n, "default_angles")
    params = normalize_params(params, n)
    dt = float(np.median(np.diff(time))) if time.size > 1 else 0.0025

    model, loaded_model_path = load_mujoco_model_with_mesh_fallback(model_path, fixed_base=pin_base)
    model.opt.timestep = dt
    if disable_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)

    qpos_ids, qvel_ids = _joint_addresses(model, joint_names)
    free_root = _free_root_addresses(model)
    armature = np.asarray(params["armature"], dtype=np.float64)
    viscous = np.asarray(params["viscous_friction"], dtype=np.float64)
    coulomb = np.asarray(params["coulomb_friction"], dtype=np.float64)
    bias = np.asarray(params["encoder_bias"], dtype=np.float64)
    strength = np.asarray(params["motor_strength"], dtype=np.float64)
    delay_steps = coerce_shared_delay_steps(params["command_delay_steps"], "command_delay_steps")
    kp_arr = coerce_vector(kp, n, "kp")
    kd_arr = coerce_vector(kd, n, "kd")

    model.dof_armature[qvel_ids] = armature
    model.dof_damping[qvel_ids] = viscous
    model.dof_frictionloss[qvel_ids] = coulomb

    def pin_root() -> None:
        if not pin_base or free_root is None:
            return
        qadr, vadr = free_root
        data.qpos[qadr : qadr + 7] = np.array([0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0])
        data.qvel[vadr : vadr + 6] = 0.0

    mujoco.mj_resetData(model, data)
    pin_root()
    data.qpos[qpos_ids] = default_angles + bias
    data.qvel[qvel_ids] = 0.0
    mujoco.mj_forward(model, data)

    measured = np.zeros((time.shape[0], n), dtype=np.float64)
    actual = np.zeros_like(measured)
    velocity = np.zeros_like(measured)
    applied_tau = np.zeros_like(measured)
    applied_des_dof_pos = np.zeros_like(measured)
    command_delay = SharedCommandDelayBuffer(delay_steps)
    for step in range(time.shape[0]):
        q = data.qpos[qpos_ids].copy()
        qd = data.qvel[qvel_ids].copy()
        delayed_desired = command_delay.push(des_dof_pos[step, :])
        target_actual = delayed_desired + bias
        tau = strength * (kp_arr * (target_actual - q) - kd_arr * qd)
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[qvel_ids] = tau
        pin_root()
        mujoco.mj_step(model, data)
        pin_root()

        actual[step, :] = data.qpos[qpos_ids]
        velocity[step, :] = data.qvel[qvel_ids]
        measured[step, :] = actual[step, :] - bias
        applied_tau[step, :] = tau
        applied_des_dof_pos[step, :] = delayed_desired

    return {
        "time": time,
        "dof_pos": measured,
        "actual_dof_pos": actual,
        "dof_vel": velocity,
        "des_dof_pos": des_dof_pos,
        "applied_des_dof_pos": applied_des_dof_pos,
        "applied_tau": applied_tau,
        "dt": np.asarray([dt], dtype=np.float64),
    }
