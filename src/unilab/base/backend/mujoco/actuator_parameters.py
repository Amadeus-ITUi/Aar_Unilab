"""Cold-path physical actuator overrides shared by training and release export."""

import ctypes
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@lru_cache(maxsize=1)
def _constant_refresh_library() -> ctypes.CDLL:
    """Use MuJoCo's C scratch data without installing the Python default timer."""
    library = ctypes.CDLL(str(Path(mujoco.__file__).parent / f"libmujoco.so.{mujoco.__version__}"))
    library.mj_makeData.argtypes = [ctypes.c_void_p]
    library.mj_makeData.restype = ctypes.c_void_p
    library.mj_setConst.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.mj_setConst.restype = None
    library.mj_deleteData.argtypes = [ctypes.c_void_p]
    library.mj_deleteData.restype = None
    return library


def _refresh_constants(model: mujoco.MjModel) -> None:
    # Python MjData construction installs a process-wide native mjcb_time even
    # though get_mjcb_time() reports None for that native callback. In a batched
    # simulation its repeated clock reads nearly double physics stepping time.
    # Raw scratch data avoids the side effect and preserves any user callback.
    library = _constant_refresh_library()
    scratch = library.mj_makeData(model._address)
    if not scratch:
        raise MemoryError("Cannot allocate MuJoCo data to refresh actuator constants")
    try:
        library.mj_setConst(model._address, scratch)
    finally:
        library.mj_deleteData(scratch)


def actuator_arrays(settings: Mapping[str, Any], count: int) -> dict[str, np.ndarray]:
    arrays = {}
    for name in ("armature", "damping", "frictionloss", "delay_ms"):
        values = np.asarray(settings[name], dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"actuator_model.{name} requires {count} nonnegative finite values")
        arrays[name] = values
    if settings.get("saturation") != "sum_abs_pd":
        raise ValueError("identified actuator model requires sum_abs_pd saturation")
    if (arrays["delay_ms"] > 1000).any():
        raise ValueError("actuator delay exceeds one second")
    return arrays


def apply_actuator_parameters(
    model: mujoco.MjModel, joints: Sequence[str], settings: Mapping[str, Any]
) -> None:
    arrays = actuator_arrays(settings, len(joints))
    addresses = [int(model.joint(name).dofadr[0]) for name in joints]
    for name in ("armature", "damping", "frictionloss"):
        getattr(model, f"dof_{name}")[addresses] = arrays[name]
    # Recompute constraint reference inertias just as recompiling the baked XML
    # does. Updating dof_armature alone leaves dof_invweight0/body_invweight0 stale.
    _refresh_constants(model)


def bake_actuator_parameters(
    root: ET.Element, joints: Sequence[str], settings: Mapping[str, Any]
) -> None:
    arrays = actuator_arrays(settings, len(joints))
    elements = {joint.get("name"): joint for joint in root.findall(".//worldbody//joint")}
    for index, name in enumerate(joints):
        if name not in elements:
            raise ValueError(f"release model is missing actuator joint {name!r}")
        for field in ("armature", "damping", "frictionloss"):
            elements[name].set(field, format(arrays[field][index], ".17g"))
