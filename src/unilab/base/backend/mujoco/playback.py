"""MuJoCo-owned playback execution helpers."""

from __future__ import annotations

import os
import tempfile
import time
from os import PathLike
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from unilab.base.backend.playback_common import env_cfg_value
from unilab.base.scene import SceneCfg

ObsT = TypeVar("ObsT")


def run_mujoco_playback(
    *,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    render_spacing: float | None,
    headless: bool,
    record_video: bool,
    frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: dict[str, Any] | None,
    extra_data_getter: Callable[[], np.ndarray | None] | None = None,
) -> str | None:
    if not headless:
        return _run_mujoco_interactive_playback(
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            frame_state_getter=frame_state_getter,
        )
    if not record_video:
        raise ValueError("MuJoCo play rendering requires record_video=true.")
    if num_steps is None:
        raise ValueError("MuJoCo play rendering requires a finite num_steps value.")
    if output_video is None:
        raise ValueError("MuJoCo play rendering requires an output_video path.")
    if frame_state_getter is None:
        frame_state_getter = env.get_physics_state_snapshot
    assert frame_state_getter is not None

    obs = initialize()
    state_list = []
    marker_list: list[np.ndarray | None] = []
    for _ in range(num_steps):
        obs = step(obs)
        state_list.append(np.asarray(frame_state_getter(), dtype=np.float32).copy())
        if extra_data_getter is not None:
            marker = extra_data_getter()
            marker_list.append(
                np.asarray(marker, dtype=np.float32).copy() if marker is not None else None
            )
        else:
            marker_list.append(None)

    marker_positions_list = (
        marker_list if any(marker is not None for marker in marker_list) else None
    )

    from unilab.visualization import render_many

    cam_kw = dict(camera_kwargs or {})
    use_tracking = bool(cam_kw.pop("cam_tracking", False))
    tracking_env_idx = int(cam_kw.pop("cam_tracking_env_idx", 0))
    tracking_extra_envs = int(cam_kw.pop("cam_tracking_extra_envs", 2))
    effective_spacing = (
        float(render_spacing)
        if render_spacing is not None
        else float(env_cfg_value(env, "render_spacing", 1.0))
    )
    with tempfile.TemporaryDirectory(prefix="unilab-playback-models-") as tmp_dir:
        model_files = resolve_render_play_model_files(
            env,
            num_envs=state_list[0].shape[0],
            tmp_dir=tmp_dir,
        )

        if use_tracking:
            frames = render_many.render_states_get_frames_tracking(
                state_list,
                model_files,
                width=1280,
                height=720,
                tracking_env_idx=tracking_env_idx,
                max_extra_envs=tracking_extra_envs,
                cam_distance=cam_kw.get("cam_distance", 2.0),
                cam_elevation=cam_kw.get("cam_elevation", -20),
                cam_azimuth=cam_kw.get("cam_azimuth", 90),
                render_spacing=effective_spacing,
                marker_positions_list=marker_positions_list,
            )
        else:
            frames = render_many.render_states_get_frames(
                state_list,
                model_files,
                width=1280,
                height=720,
                camera_id=-1,
                render_spacing=effective_spacing,
                marker_positions_list=marker_positions_list,
                **cam_kw,
            )

    import mediapy as media

    ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
    media.write_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
    return str(output_video)


def _run_mujoco_interactive_playback(
    *,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    frame_state_getter: Callable[[], np.ndarray] | None,
) -> None:
    """Run one training environment in MuJoCo's native real-time viewer.

    WE11 uses the same Xbox contract as the maintained C++ Play: RB+DPadUp
    starts, LB+X stops, RB+Y resets, LY commands forward velocity, RX yaw,
    and RY integrates body-height target.  No wing or recorded-wrench channel
    is driven by this loop.
    """
    import mujoco
    import mujoco.viewer

    if frame_state_getter is None:
        frame_state_getter = env.get_physics_state_snapshot

    autostart = os.environ.get("AAR_PLAY_AUTOSTART", "0").lower() in {"1", "true", "yes"}
    control = {"started": autostart, "start": False, "stop": False, "reset": False}

    def key_callback(keycode: int) -> None:
        if keycode in (ord("1"), 257):  # 1 or Enter
            control["start"] = True
        elif keycode in (ord("P"), ord("p")):
            control["stop"] = True
        elif keycode in (ord("R"), ord("r"), ord("0")):
            control["reset"] = True

    def set_command(command: np.ndarray) -> None:
        state = getattr(env, "state", None)
        if state is None:
            state = getattr(env, "_state", None)
        info = getattr(state, "info", None)
        if not isinstance(info, dict) or "commands" not in info:
            return
        commands = np.asarray(info["commands"])
        if commands.ndim == 2 and commands.shape[0] and commands.shape[1] >= 3:
            commands[0, :3] = command

    def copy_state(model: Any, data: Any) -> None:
        snapshot = np.asarray(frame_state_getter(), dtype=np.float64)[0]
        data.time = snapshot[0]
        data.qpos[:] = snapshot[1 : 1 + model.nq]
        data.qvel[:] = snapshot[1 + model.nq : 1 + model.nq + model.nv]
        mujoco.mj_forward(model, data)

    def forward_viewer_force(data: Any) -> None:
        """Apply native viewer Ctrl-drag forces to the live batch environment."""
        forces = np.asarray(data.xfrc_applied, dtype=np.float64)
        body_ids = np.flatnonzero(np.linalg.norm(forces, axis=1) > 0.0).astype(np.int32)
        if body_ids.size == 0:
            return
        backend = getattr(env, "_backend", None)
        apply_force = getattr(backend, "apply_body_force", None)
        if callable(apply_force):
            apply_force(body_ids, forces[body_ids][None, :, :])
        data.xfrc_applied[:] = 0.0

    obs = initialize()
    ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 0.02))
    height = 0.25
    previous = {"start": False, "stop": False, "reset": False}
    completed_steps = 0

    with tempfile.TemporaryDirectory(prefix="unilab-live-play-") as tmp_dir:
        model_path = resolve_render_play_model_files(env, num_envs=1, tmp_dir=tmp_dir)
        if isinstance(model_path, list):
            model_path = model_path[0]
        model = mujoco.MjModel.from_binary_path(model_path) if str(model_path).endswith(".mjb") else mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        copy_state(model, data)

        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            print("Python Play controls: 1/Enter or RB+DPadUp start; P or LB+X stop; R or RB+Y reset")
            print("Axes: LY=vx, RX=yaw, RY=height; wing/wrench disturbance disabled")
            while viewer.is_running() and (num_steps is None or completed_steps < num_steps):
                loop_start = time.monotonic()
                axes = None
                buttons = None
                try:
                    import glfw

                    gamepad = glfw.get_gamepad_state(glfw.JOYSTICK_1)
                    if gamepad is not None:
                        axes = gamepad.axes
                        buttons = gamepad.buttons
                except (ImportError, TypeError):
                    pass

                command = np.array([0.0, 0.0, height], dtype=np.float32)
                if axes is not None and len(axes) >= 4:
                    ly = -float(axes[1])
                    rx = -float(axes[2])
                    ry = float(axes[3])
                    command[0] = 0.0 if abs(ly) < 0.12 else np.clip(ly, -1.0, 1.0)
                    command[1] = 0.0 if abs(rx) < 0.12 else np.clip(rx, -1.0, 1.0)
                    if abs(ry) >= 0.12:
                        height = float(np.clip(height + ry * 0.03 * ctrl_dt, 0.20, 0.30))
                    command[2] = height

                combos = {"start": False, "stop": False, "reset": False}
                if buttons is not None:
                    try:
                        import glfw

                        rb = bool(buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER])
                        lb = bool(buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER])
                        combos["start"] = rb and bool(buttons[glfw.GAMEPAD_BUTTON_DPAD_UP])
                        combos["stop"] = lb and bool(buttons[glfw.GAMEPAD_BUTTON_X])
                        combos["reset"] = rb and bool(buttons[glfw.GAMEPAD_BUTTON_Y])
                    except (IndexError, TypeError):
                        pass
                for event in combos:
                    if combos[event] and not previous[event]:
                        control[event] = True
                previous = combos

                if control.pop("reset", False):
                    obs = initialize()
                    control["started"] = False
                    height = 0.25
                if control.pop("stop", False):
                    control["started"] = False
                if control.pop("start", False):
                    obs = initialize()
                    control["started"] = True
                    height = 0.25

                if control["started"]:
                    forward_viewer_force(data)
                    set_command(command)
                    obs = step(obs)
                    completed_steps += 1

                copy_state(model, data)
                viewer.sync()
                remaining = ctrl_dt - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    return None


def _configured_model_file(env: Any) -> str | None:
    cfg = getattr(env, "cfg", None)
    scene = getattr(cfg, "scene", None) if cfg is not None else None
    if scene is None:
        return None
    if not isinstance(scene, SceneCfg):
        raise TypeError("env.cfg.scene must be a SceneCfg")
    return scene.model_file


def _visual_model_file(env: Any) -> str | None:
    backend = getattr(env, "_backend", None)
    backend_visual_model_file = getattr(backend, "scene_visual_model_file", None)
    if backend_visual_model_file:
        return str(backend_visual_model_file)
    return _configured_model_file(env)


def resolve_render_play_model_files(
    env: Any,
    *,
    num_envs: int,
    tmp_dir: str | Path,
) -> str | list[str]:
    """Resolve visual MuJoCo model files for offline play/video export."""
    visual_model_file = _visual_model_file(env)
    if not hasattr(env, "get_playback_model"):
        if visual_model_file is None:
            raise ValueError("MuJoCo playback requires either cfg.scene or get_playback_model().")
        return visual_model_file

    first_model = env.get_playback_model(0)
    if isinstance(first_model, (str, Path)):
        return str(first_model)

    import mujoco as _mujoco

    mujoco: Any = _mujoco

    visual_base = (
        mujoco.MjModel.from_xml_path(visual_model_file) if visual_model_file is not None else None
    )
    tmp_root = Path(tmp_dir)
    path_by_model_id: dict[int, str] = {}
    model_files: list[str] = []
    for env_idx in range(num_envs):
        playback_model = env.get_playback_model(env_idx)
        if isinstance(playback_model, (str, Path)):
            model_files.append(str(playback_model))
            continue
        key = id(playback_model)
        saved = path_by_model_id.get(key)
        if saved is None:
            output_path = tmp_root / f"model_{len(path_by_model_id)}.mjb"
            if visual_model_file is None or visual_base is None:
                mujoco.mj_saveModel(playback_model, str(output_path))
                saved = str(output_path)
            else:
                saved = materialize_visual_playback_model(
                    visual_model_file=visual_model_file,
                    visual_base_model=visual_base,
                    playback_model=playback_model,
                    output_path=output_path,
                )
            path_by_model_id[key] = saved
        model_files.append(saved)

    if len(set(model_files)) == 1:
        return model_files[0]
    return model_files


def materialize_visual_playback_model(
    *,
    visual_model_file: str,
    visual_base_model: Any,
    playback_model: Any,
    output_path: str | Path,
) -> str:
    """Compile a visual MuJoCo model using geom sizes from a playback model."""
    import mujoco as _mujoco

    mujoco: Any = _mujoco

    spec = mujoco.MjSpec.from_file(visual_model_file)
    for geom_id in range(visual_base_model.ngeom):
        geom_name = mujoco.mj_id2name(visual_base_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not geom_name:
            continue
        playback_geom_id = mujoco.mj_name2id(playback_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if playback_geom_id < 0:
            continue
        geom = spec.geom(geom_name)
        if geom is None:
            continue
        geom.size = list(np.asarray(playback_model.geom_size[playback_geom_id], dtype=np.float64))

    visual_model = spec.compile()
    output = Path(output_path)
    mujoco.mj_saveModel(visual_model, str(output))
    return str(output)
