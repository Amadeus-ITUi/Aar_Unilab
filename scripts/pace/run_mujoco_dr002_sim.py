#!/usr/bin/env python3
"""Run the DR002 MuJoCo model with a simple joint-space PD controller."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time as walltime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_dr002_common import (  # noqa: E402
    DEFAULT_ANGLES,
    DEFAULT_TRAINING_MODEL,
    HAND_PLAY_KD,
    HAND_PLAY_KP,
    JOINT_NAMES,
    coerce_vector,
    ensure_parent,
    load_chirp_data,
    load_mujoco_model_with_mesh_fallback,
    resolve_repo_path,
)

LEG_JOINT_IDS = np.asarray([0, 1, 3, 4], dtype=np.int64)
WHEEL_JOINT_IDS = np.asarray([2, 5], dtype=np.int64)
DEFAULT_ARMATURE = np.asarray(
    [
        0.003395025986270572,
        0.019759596621712217,
        0.0008,
        0.003395025986270572,
        0.019759596621712217,
        0.0008,
    ],
    dtype=np.float64,
)
DEFAULT_EFFORT_LIMIT = np.asarray([5.5, 14.0, 5.5, 5.5, 14.0, 5.5], dtype=np.float64)
DEFAULT_JOINT_FRICTION = np.asarray(
    [
        0.00039277340996955734,
        0.18304102565728816,
        0.0,
        0.00039277340996955734,
        0.18304102565728816,
        0.0,
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DR002 in MuJoCo viewer or headless mode.")
    parser.add_argument(
        "--model",
        "--mjcf",
        "--urdf",
        default=DEFAULT_TRAINING_MODEL,
        help="MJCF/URDF path; defaults to the repository-local WE11 model.",
    )
    parser.add_argument("--mode", choices=("hold", "sine", "chirp", "replay"), default="hold")
    parser.add_argument(
        "--trajectory", default=None, help="chirp_data.pt with des_dof_pos for --mode replay."
    )
    parser.add_argument(
        "--control-mode",
        "--control_mode",
        choices=("position", "mixed"),
        default="position",
        help="position: all joints use position PD. mixed: thigh/calf use position PD, foot wheels use velocity D control.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Seconds to run. Use <=0 for viewer-until-closed.",
    )
    parser.add_argument("--dt", type=float, default=0.0025, help="MuJoCo timestep.")
    parser.add_argument(
        "--motor", type=int, default=0, help="Motor index for sine/chirp single-joint excitation."
    )
    parser.add_argument(
        "--motors", type=int, nargs="+", default=None, help="Explicit motor indices for sine/chirp."
    )
    parser.add_argument(
        "--excite_all", action="store_true", help="Excite all joints for sine/chirp."
    )
    parser.add_argument(
        "--amplitude", type=float, nargs="+", default=[0.25], help="Scalar or 6 joint amplitudes."
    )
    parser.add_argument("--frequency", type=float, default=1.0, help="Sine frequency in Hz.")
    parser.add_argument("--min_freq", type=float, default=0.1, help="Chirp start frequency in Hz.")
    parser.add_argument("--max_freq", type=float, default=3.0, help="Chirp end frequency in Hz.")
    parser.add_argument("--phase_mode", choices=("same", "spread"), default="spread")
    parser.add_argument("--default_angles", type=float, nargs=6, default=DEFAULT_ANGLES.tolist())
    parser.add_argument(
        "--lock-left-joints",
        action="store_true",
        help="Keep left_thigh/left_calf/left_foot fixed at default_angles during simulation.",
    )
    parser.add_argument(
        "--lock-joints",
        type=int,
        nargs="*",
        default=None,
        help="Extra 0-based joint indices to keep fixed at default_angles during simulation.",
    )
    parser.add_argument("--kp", type=float, nargs="+", default=HAND_PLAY_KP.tolist())
    parser.add_argument("--kd", type=float, nargs="+", default=HAND_PLAY_KD.tolist())
    parser.add_argument(
        "--armature",
        type=float,
        nargs="+",
        default=DEFAULT_ARMATURE.tolist(),
        help="MuJoCo dof armature in policy joint order.",
    )
    parser.add_argument(
        "--effort-limit",
        "--effort_limit",
        type=float,
        nargs="+",
        default=DEFAULT_EFFORT_LIMIT.tolist(),
        help="Torque limits in Gym joint order. Applied unless --no-torque-clip is set.",
    )
    parser.add_argument(
        "--joint-friction",
        "--joint_friction",
        type=float,
        nargs="+",
        default=DEFAULT_JOINT_FRICTION.tolist(),
        help="MuJoCo dof frictionloss in policy joint order.",
    )
    parser.add_argument(
        "--no-torque-clip",
        "--no_torque_clip",
        action="store_true",
        help="Disable effort-limit clipping and apply raw PD torque directly.",
    )
    parser.add_argument("--enable_gravity", action="store_true", help="Keep gravity enabled.")
    parser.add_argument(
        "--free_base", action="store_true", help="Do not pin a free root joint if present."
    )
    parser.add_argument(
        "--root_z", type=float, default=0.35, help="Pinned root z when a free root exists."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without opening MuJoCo viewer."
    )
    parser.add_argument("--log_csv", default=None, help="Optional CSV path for time/q/des/tau.")
    parser.add_argument(
        "--print_every", type=int, default=400, help="Print every N sim steps in headless mode."
    )
    parser.add_argument(
        "--no_viewer_exit_hack",
        action="store_true",
        help="Disable the hard-exit workaround for MuJoCo viewer teardown crashes.",
    )
    return parser.parse_args()


def selected_motors(args: argparse.Namespace, n: int) -> list[int]:
    if args.excite_all:
        motors = list(range(n))
    elif args.motors is not None:
        motors = args.motors
    else:
        motors = [args.motor]
    bad = [m for m in motors if m < 0 or m >= n]
    if bad:
        raise ValueError(f"motor indices out of range [0, {n - 1}]: {bad}")
    return sorted(set(motors))


def desired_at_time(
    t: float,
    args: argparse.Namespace,
    default_angles: np.ndarray,
    amplitude: np.ndarray,
    motors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    des = default_angles.copy()
    des_vel = np.zeros_like(default_angles)
    if args.mode == "hold":
        return des, des_vel

    if args.mode == "sine":
        phase = 2.0 * math.pi * args.frequency * t
        phase_rate = 2.0 * math.pi * args.frequency
    elif args.mode == "chirp":
        duration = max(args.duration, args.dt)
        phase = (
            2.0
            * math.pi
            * (args.min_freq * t + ((args.max_freq - args.min_freq) / (2.0 * duration)) * t * t)
        )
        phase_rate = (
            2.0 * math.pi * (args.min_freq + ((args.max_freq - args.min_freq) / duration) * t)
        )
    else:
        raise ValueError("desired_at_time is not used for replay mode")

    for k, motor in enumerate(motors):
        phase_offset = 0.0
        if args.phase_mode == "spread" and len(motors) > 1:
            phase_offset = 2.0 * math.pi * k / len(motors)
        signal = amplitude[motor] * math.sin(phase + phase_offset)
        if args.control_mode == "mixed" and motor in WHEEL_JOINT_IDS:
            des_vel[motor] = signal
        else:
            des[motor] = default_angles[motor] + signal
            des_vel[motor] = amplitude[motor] * phase_rate * math.cos(phase + phase_offset)
    return des, des_vel


def expand_replay_array(values: np.ndarray, defaults: np.ndarray, key: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] == 1:
        full = np.tile(defaults.reshape(1, -1), (values.shape[0], 1))
        full[:, 0] = values[:, 0]
        return full
    if values.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"trajectory {key} must have 1 or {len(JOINT_NAMES)} columns, got {values.shape[1]}"
        )
    return values


def infer_wheel_velocity_targets(
    time: np.ndarray, des_dof_pos: np.ndarray, default_angles: np.ndarray
) -> np.ndarray:
    des_vel = np.zeros_like(des_dof_pos)
    if time.shape[0] < 2:
        return des_vel
    for motor in WHEEL_JOINT_IDS:
        # Legacy wheel sweeps were sometimes stored as position chirps. For mixed
        # replay, convert only the wheel position target shape into a velocity target.
        des_vel[:, motor] = np.gradient(des_dof_pos[:, motor] - default_angles[motor], time)
    return des_vel


def load_replay_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = load_chirp_data(path)
    time = data["time"]
    des_pos = expand_replay_array(data["des_dof_pos"], DEFAULT_ANGLES, "des_dof_pos")
    des_vel = data.get("des_dof_vel")
    if des_vel is not None:
        des_vel = expand_replay_array(des_vel, np.zeros_like(DEFAULT_ANGLES), "des_dof_vel")
    return time, des_pos, des_vel


def compute_tau(
    control_mode: str,
    q: np.ndarray,
    qd: np.ndarray,
    des_pos: np.ndarray,
    des_vel: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    if control_mode == "position":
        return kp * (des_pos - q) - kd * qd

    tau = np.zeros_like(q)
    tau[LEG_JOINT_IDS] = (
        kp[LEG_JOINT_IDS] * (des_pos[LEG_JOINT_IDS] - q[LEG_JOINT_IDS])
        - kd[LEG_JOINT_IDS] * qd[LEG_JOINT_IDS]
    )
    tau[WHEEL_JOINT_IDS] = kd[WHEEL_JOINT_IDS] * (des_vel[WHEEL_JOINT_IDS] - qd[WHEEL_JOINT_IDS])
    return tau


def joint_addresses(model, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    qpos_ids = []
    qvel_ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint {name!r} not found in MuJoCo model")
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        qvel_ids.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos_ids, dtype=np.int64), np.asarray(qvel_ids, dtype=np.int64)


def free_root_addresses(model) -> tuple[int, int] | None:
    import mujoco

    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return None


def main() -> None:
    args = parse_args()
    n = len(JOINT_NAMES)
    if args.dt <= 0:
        raise ValueError("--dt must be positive")
    if args.duration <= 0 and args.headless and args.mode != "replay":
        raise ValueError("--duration must be positive in --headless mode")
    if args.mode == "replay" and not args.trajectory:
        raise ValueError("--mode replay requires --trajectory")

    default_angles = np.asarray(args.default_angles, dtype=np.float64)
    amplitude = coerce_vector(args.amplitude, n, "amplitude")
    kp = coerce_vector(args.kp, n, "kp")
    kd = coerce_vector(args.kd, n, "kd")
    armature = coerce_vector(args.armature, n, "armature")
    effort_limit = coerce_vector(args.effort_limit, n, "effort_limit")
    joint_friction = coerce_vector(args.joint_friction, n, "joint_friction")
    motors = selected_motors(args, n)

    import mujoco

    model, loaded_path = load_mujoco_model_with_mesh_fallback(
        args.model, fixed_base=not args.free_base
    )
    model.opt.timestep = args.dt
    if not args.enable_gravity:
        model.opt.gravity[:] = 0.0

    data = mujoco.MjData(model)
    qpos_ids, qvel_ids = joint_addresses(model, JOINT_NAMES)
    free_root = free_root_addresses(model)
    locked_joint_ids = set(args.lock_joints or [])
    if args.lock_left_joints:
        locked_joint_ids.update([0, 1, 2])
    bad_locked = sorted(idx for idx in locked_joint_ids if idx < 0 or idx >= n)
    if bad_locked:
        raise ValueError(f"--lock-joints indices out of range [0, {n - 1}]: {bad_locked}")
    locked_joint_ids = sorted(locked_joint_ids)
    model.dof_armature[qvel_ids] = armature
    model.dof_frictionloss[qvel_ids] = joint_friction

    replay_time = None
    replay_des = None
    replay_des_vel = None
    if args.mode == "replay":
        replay_time, replay_des, replay_des_vel = load_replay_trajectory(args.trajectory)
        if replay_des_vel is None and args.control_mode == "mixed":
            replay_des_vel = infer_wheel_velocity_targets(replay_time, replay_des, default_angles)
        args.duration = (
            float(replay_time[-1])
            if args.duration <= 0
            else min(args.duration, float(replay_time[-1]))
        )

    def pin_root() -> None:
        if args.free_base or free_root is None:
            return
        qadr, vadr = free_root
        data.qpos[qadr : qadr + 7] = np.array([0.0, 0.0, args.root_z, 1.0, 0.0, 0.0, 0.0])
        data.qvel[vadr : vadr + 6] = 0.0

    def lock_joints() -> None:
        if not locked_joint_ids:
            return
        data.qpos[qpos_ids[locked_joint_ids]] = default_angles[locked_joint_ids]
        data.qvel[qvel_ids[locked_joint_ids]] = 0.0

    def desired(step: int, t: float) -> tuple[np.ndarray, np.ndarray]:
        if replay_des is not None:
            idx = min(step, replay_des.shape[0] - 1)
            des_vel = (
                np.zeros(n, dtype=np.float64)
                if replay_des_vel is None
                else replay_des_vel[idx].copy()
            )
            return replay_des[idx].copy(), des_vel
        return desired_at_time(t, args, default_angles, amplitude, motors)

    mujoco.mj_resetData(model, data)
    pin_root()
    data.qpos[qpos_ids] = default_angles
    data.qvel[qvel_ids] = 0.0
    lock_joints()
    mujoco.mj_forward(model, data)

    log_rows: list[list[float]] = []
    total_steps = math.inf if args.duration <= 0 else int(round(args.duration / args.dt))
    max_abs_raw_tau = 0.0
    num_effort_clips = 0

    def step_once(step: int) -> tuple[float, np.ndarray, np.ndarray]:
        nonlocal max_abs_raw_tau, num_effort_clips
        t = step * args.dt
        des, des_vel = desired(step, t)
        q = data.qpos[qpos_ids].copy()
        qd = data.qvel[qvel_ids].copy()
        raw_tau = compute_tau(args.control_mode, q, qd, des, des_vel, kp, kd)
        tau = raw_tau if args.no_torque_clip else np.clip(raw_tau, -effort_limit, effort_limit)
        max_abs_raw_tau = max(max_abs_raw_tau, float(np.max(np.abs(raw_tau))))
        if not args.no_torque_clip:
            num_effort_clips += int(np.count_nonzero(np.abs(raw_tau) > effort_limit))
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[qvel_ids] = tau
        mujoco.mj_step(model, data)
        pin_root()
        lock_joints()
        if locked_joint_ids:
            mujoco.mj_forward(model, data)
        if args.log_csv:
            log_rows.append(
                [
                    t,
                    *q.tolist(),
                    *des.tolist(),
                    *tau.tolist(),
                    *raw_tau.tolist(),
                    *qd.tolist(),
                    *des_vel.tolist(),
                ]
            )
        return t, q, tau

    print(f"[INFO] model: {resolve_repo_path(args.model)}")
    if Path(loaded_path) != resolve_repo_path(args.model):
        print(f"[INFO] loaded fallback model: {loaded_path}")
    print(
        f"[INFO] mode={args.mode}, control_mode={args.control_mode}, "
        f"dt={args.dt}, duration={args.duration}, joints={JOINT_NAMES}"
    )
    print(f"[INFO] kp={kp.tolist()}, kd={kd.tolist()}")
    if locked_joint_ids:
        print(f"[INFO] locked_joints={[(idx, JOINT_NAMES[idx]) for idx in locked_joint_ids]}")
    print(
        f"[INFO] armature={armature.tolist()}, effort_limit={effort_limit.tolist()}, "
        f"joint_friction={joint_friction.tolist()}, torque_clip={not args.no_torque_clip}"
    )

    if args.headless:
        step = 0
        while step < total_steps:
            t, q, tau = step_once(step)
            if args.print_every > 0 and step % args.print_every == 0:
                print(
                    f"[SIM] step={step:06d} t={t:8.3f} q={np.round(q, 4).tolist()} tau={np.round(tau, 4).tolist()}"
                )
            step += 1
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            while viewer.is_running() and step < total_steps:
                tick = walltime.time()
                step_once(step)
                viewer.sync()
                elapsed = walltime.time() - tick
                if elapsed < args.dt:
                    walltime.sleep(args.dt - elapsed)
                step += 1

    if args.log_csv:
        out = ensure_parent(args.log_csv)
        header = (
            ["time"]
            + [f"q_{name}" for name in JOINT_NAMES]
            + [f"des_{name}" for name in JOINT_NAMES]
            + [f"tau_{name}" for name in JOINT_NAMES]
            + [f"raw_tau_{name}" for name in JOINT_NAMES]
            + [f"qd_{name}" for name in JOINT_NAMES]
            + [f"desvel_{name}" for name in JOINT_NAMES]
        )
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(log_rows)
        print(f"[DONE] wrote log: {out}")

    print(f"[SUMMARY] max_abs_raw_tau={max_abs_raw_tau:.6g}, num_effort_clips={num_effort_clips}")
    print("[DONE] MuJoCo simulation finished")
    sys.stdout.flush()
    sys.stderr.flush()
    if not args.headless and not args.no_viewer_exit_hack:
        os._exit(0)


if __name__ == "__main__":
    main()
