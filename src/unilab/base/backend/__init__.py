from typing import Any, cast

from unilab.base.scene import SceneCfg

from .base import SimBackend

_MUJOCO_XML_EXPORTS = frozenset(
    {
        "add_sensor",
        "create_discardvisual_xml",
        "get_named_body_ids",
        "inject_mujoco_tracking_sensors",
        "materialize_mujoco_hfield_attached_scene",
        "materialize_scene_fragments",
        "materialize_scene_visual_override",
        "processed_xml",
    }
)


def _load_mujoco_backend() -> Any:
    from .mujoco.backend import MuJoCoBackend

    return MuJoCoBackend


def create_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    **kwargs,
) -> SimBackend:
    """Create a simulation backend.

    Args:
        backend_type: Must be ``"mujoco"``.
        scene: SceneCfg for either static or composed MuJoCo scenes.
        num_envs: Number of environments.
        sim_dt: Simulation timestep.
        **kwargs: Additional backend options such as ``position_actuator_gains``.

    Returns:
        SimBackend instance.
    """
    if scene is None:
        raise ValueError("SceneCfg must be provided")

    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    post_step_forward_sensor = kwargs.pop("post_step_forward_sensor", None)
    if backend_type == "mujoco":
        MuJoCoBackend = _load_mujoco_backend()
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        if post_step_forward_sensor is not None:
            kwargs["post_step_forward_sensor"] = post_step_forward_sensor
        return cast(SimBackend, MuJoCoBackend(scene, num_envs, sim_dt, **kwargs))
    raise ValueError(f"Unknown backend: {backend_type}")


def __getattr__(name: str):
    if name == "MuJoCoBackend":
        return _load_mujoco_backend()
    if name in _MUJOCO_XML_EXPORTS:
        from .mujoco import xml

        return getattr(xml, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimBackend",
    "MuJoCoBackend",
    "add_sensor",
    "create_discardvisual_xml",
    "create_backend",
    "get_named_body_ids",
    "inject_mujoco_tracking_sensors",
    "materialize_mujoco_hfield_attached_scene",
    "materialize_scene_fragments",
    "materialize_scene_visual_override",
    "processed_xml",
]
