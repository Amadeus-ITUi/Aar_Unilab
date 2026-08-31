"""Interactive viewer for an RL task's training environment.

Builds the env via the same `registry.make(...)` path that the trainer uses,
then steps it with zero actions so you can verify spawn distribution,
procedural terrain, sensor placements, asset materials, and domain
randomization look right before kicking off a real training run.

The script composes the selected task's Hydra config so reset, commands,
randomization, control, and reward settings match training.

MuJoCo stitches all `--num_envs` robot replicas into the same scene
and drives every replica's qpos/qvel each frame from `env.get_physics_state_snapshot()`.

Usage:
    python scripts/visualize_task_env.py --task DR002JoystickFlatWE11
    python scripts/visualize_task_env.py --task DR002JoystickGetupWE11 --getup-difficulty 1 --freeze-initial-pose
    python scripts/visualize_task_env.py --task DR002JoystickRoughWE11 --num_envs 16
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "conf" / "ppo"
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import BackendAdapter, ensure_registries

ensure_registries()

from unilab.base import registry

if TYPE_CHECKING:
    from unilab.base.scene import SceneCfg


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an RL task's training environment with zero actions."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="DR002JoystickRoughWE11",
        choices=[
            "DR002JoystickFlatWE11",
            "DR002JoystickGetupWE11",
            "DR002JoystickRoughWE11",
        ],
        help="WE11 task to visualize.",
    )
    parser.add_argument(
        "--start-pose",
        type=str,
        choices=["upright", "getup", "mixed"],
        default=None,
        help=(
            "Override reset_pose.mode for this viewer only. Use 'getup' to inspect "
            "the mechanical-limit fallen start pose."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mujoco"],
        default="mujoco",
        help="Physics backend to construct the env with (default: mujoco).",
    )
    parser.add_argument(
        "--getup-difficulty",
        type=float,
        default=None,
        help="Force a deterministic Getup curriculum pose in [0, 1].",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of envs to construct and visualize (default: 4).",
    )
    parser.add_argument(
        "--freeze-initial-pose",
        action="store_true",
        help=(
            "Display the exact reset state without advancing physics or applying "
            "zero policy actions."
        ),
    )
    args = parser.parse_args(argv)
    if args.getup_difficulty is not None and not 0.0 <= args.getup_difficulty <= 1.0:
        parser.error("--getup-difficulty must be in [0, 1]")
    if args.getup_difficulty is not None and args.task != "DR002JoystickGetupWE11":
        parser.error("--getup-difficulty is only valid for DR002JoystickGetupWE11")
    return args


def _stitch_replicas(parent_scene_xml: Path, robot_base_xml: Path, env_origins: np.ndarray):
    """Attach (N - 1) extra robots to the parent scene."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(parent_scene_xml))
    for i in range(1, len(env_origins)):
        child = mujoco.MjSpec.from_file(str(robot_base_xml))
        for geom in list(child.worldbody.geoms):
            if geom.name == "floor":
                child.delete(geom)
        for light in list(child.lights):
            child.delete(light)
        for tex in list(child.textures):
            if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                child.delete(tex)
        for sensor in list(child.sensors):
            child.delete(sensor)
        for kf in list(child.keys):
            child.delete(kf)
        frame = spec.worldbody.add_frame(
            pos=[float(env_origins[i, 0]), float(env_origins[i, 1]), 0.0]
        )
        spec.attach(child, prefix=f"env{i}/", frame=frame)
    return spec.compile()


def _env_scene(env) -> "SceneCfg | None":
    cfg = getattr(env, "cfg", None)
    scene = getattr(cfg, "scene", None) if cfg is not None else None
    if scene is None:
        return None
    model_file = getattr(scene, "model_file", None)
    if model_file is None:
        raise TypeError("env.cfg.scene must expose model_file")
    return cast("SceneCfg", scene)


def _mujoco_visual_xml_paths(env) -> tuple[Path, Path]:
    scene = _env_scene(env)
    static_model_file = None if scene is None else scene.model_file
    backend = getattr(env, "_backend", None)
    parent_xml = getattr(backend, "scene_visual_model_file", None)
    if parent_xml is None:
        parent_xml = static_model_file
    if parent_xml is None:
        raise ValueError(
            "MuJoCo visualization requires cfg.scene or backend scene_visual_model_file."
        )
    robot_xml = static_model_file or parent_xml
    return Path(parent_xml), Path(robot_xml)


def _run_mujoco(env, num_envs: int, *, freeze_initial_pose: bool = False) -> None:
    import mujoco
    import mujoco.viewer

    parent_xml, robot_xml = _mujoco_visual_xml_paths(env)
    env_origins = env._spawn.origins_for(np.arange(num_envs))

    if num_envs > 1 and not env_origins[:, :2].any():
        print(
            f"[visualize_task_env] NOTE: env has no terrain; all {num_envs} "
            "robots will overlap at the world origin (this matches the env's "
            "actual reset spawn)."
        )

    decoder_model = mujoco.MjModel.from_xml_path(str(parent_xml))
    decoder_data = mujoco.MjData(decoder_model)
    nq_per = int(decoder_model.nq)
    nv_per = int(decoder_model.nv)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

    if num_envs == 1:
        viz_model = decoder_model
    else:
        viz_model = _stitch_replicas(parent_xml, robot_xml, env_origins)
    viz_data = mujoco.MjData(viz_model)

    actions = np.zeros((num_envs, env.action_space.shape[0]), dtype=np.float32)
    env.init_state()
    ctrl_dt = float(env.cfg.ctrl_dt)

    print(
        f"[visualize_task_env] viz_model: {viz_model.ngeom} geoms, "
        f"{viz_model.nbody} bodies, nq={viz_model.nq} (per-env nq={nq_per})."
    )
    if freeze_initial_pose:
        print(
            "[visualize_task_env] Initial pose is frozen: physics and policy control "
            "are not being advanced."
        )
    print("[visualize_task_env] Opening MuJoCo viewer — close the window or press Esc to quit.")

    with mujoco.viewer.launch_passive(viz_model, viz_data) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            if not freeze_initial_pose:
                env.step(actions)
            phys = env.get_physics_state_snapshot()
            for i in range(num_envs):
                mujoco.mj_setState(
                    decoder_model, decoder_data, phys[i].astype(np.float64), state_spec
                )
                viz_data.qpos[i * nq_per : (i + 1) * nq_per] = decoder_data.qpos
                viz_data.qvel[i * nv_per : (i + 1) * nv_per] = decoder_data.qvel
            mujoco.mj_forward(viz_model, viz_data)
            viewer.sync()
            sleep = ctrl_dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)


def _build_env_cfg_override(
    task_name: str,
    start_pose: str | None = None,
    getup_difficulty: float | None = None,
) -> dict[str, Any]:
    """Compose the training task config, then apply viewer-only pose overrides."""
    if task_name not in registry._envs:
        raise SystemExit(
            f"Task '{task_name}' is not registered. Available: {sorted(registry._envs.keys())}"
        )
    selectors = {
        "DR002JoystickFlatWE11": "dr002_joystick_flat_we11/mujoco",
        "DR002JoystickGetupWE11": "dr002_joystick_getup_we11/mujoco",
        "DR002JoystickRoughWE11": "dr002_joystick_rough_we11/mujoco",
    }
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=[f"task={selectors[task_name]}"])
    override = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override()
    if start_pose is not None:
        override["reset_pose"] = {"mode": start_pose}
    if getup_difficulty is not None:
        override["reset_pose"] = {"mode": "getup"}
        override["getup_curriculum"] = {"forced_difficulty": getup_difficulty}
    return override


def _print_backend_scene(env) -> None:
    backend = getattr(env, "_backend", None)
    scene = _env_scene(env)
    scene_source = getattr(backend, "scene_visual_model_file", None)
    if scene_source is None:
        scene_source = getattr(backend, "scene_model_file", None)
    if scene_source is None and scene is not None:
        scene_source = scene.model_file
    if scene_source is None:
        scene_source = "<unknown>"
    print(f"[visualize_task_env] backend scene: {scene_source}")

    artifacts_dir = getattr(backend, "scene_artifacts_dir", None)
    if artifacts_dir is not None:
        print(f"[visualize_task_env] artifacts: {artifacts_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_envs < 1:
        raise SystemExit(f"--num_envs must be >= 1, got {args.num_envs}")

    print(
        f"[visualize_task_env] task={args.task} backend={args.backend} "
        f"num_envs={args.num_envs} start_pose={args.start_pose or 'task-default'} "
        f"getup_difficulty={args.getup_difficulty if args.getup_difficulty is not None else 'task-default'} "
        f"freeze_initial_pose={args.freeze_initial_pose}"
    )

    env_cfg_override = _build_env_cfg_override(args.task, args.start_pose, args.getup_difficulty)
    env = registry.make(
        args.task,
        num_envs=args.num_envs,
        sim_backend=args.backend,
        env_cfg_override=env_cfg_override,
    )

    _print_backend_scene(env)

    try:
        _run_mujoco(
            env,
            args.num_envs,
            freeze_initial_pose=args.freeze_initial_pose,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
