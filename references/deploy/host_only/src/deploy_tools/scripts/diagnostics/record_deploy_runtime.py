#!/usr/bin/env python3
import argparse
import csv
import os
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray


def deployment_root():
    configured_root = os.environ.get("WALKING_EAGLE_DEPLOY_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    source_path = Path(__file__).resolve()
    for candidate in source_path.parents:
        if (candidate / "start_robot.sh").is_file() and (candidate / "src").is_dir():
            return candidate
    return Path.cwd().resolve()


def run_cmd(args, timeout=3.0):
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        ).stdout
    except Exception as exc:
        return f"[command failed] {' '.join(args)}: {type(exc).__name__}: {exc}\n"


def make_writer(path, header):
    fp = open(path, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(header)
    fp.flush()
    return fp, writer


def vector_get(values, index, default=0.0):
    return values[index] if index < len(values) else default


def parse_can_summary(text):
    state = "UNKNOWN"
    berr_tx = "?"
    berr_rx = "?"
    rx_packets = "?"
    tx_packets = "?"

    m = re.search(r"can state ([A-Z-]+) \(berr-counter tx (\d+) rx (\d+)\)", text)
    if m:
        state, berr_tx, berr_rx = m.group(1), m.group(2), m.group(3)

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("RX:") and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if len(parts) >= 2:
                rx_packets = parts[1]
        if line.strip().startswith("TX:") and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if len(parts) >= 2:
                tx_packets = parts[1]

    return f"state={state} berr_tx={berr_tx} berr_rx={berr_rx} rx_pkts={rx_packets} tx_pkts={tx_packets}"


def start_candump(out_dir, ifaces, enabled=True):
    if not enabled:
        return None, None
    if shutil.which("candump") is None:
        print("[WARN] candump not found; CAN frame capture disabled.")
        return None, None

    path = out_dir / "candump.log"
    fp = open(path, "w")
    cmd = ["candump", "-ta", "-e"] + list(ifaces)
    try:
        proc = subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        print(f"[WARN] failed to start candump: {type(exc).__name__}: {exc}")
        fp.close()
        return None, None
    return proc, fp


def stop_process(proc, fp):
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=2.0)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if fp is not None:
        fp.flush()
        fp.close()


def summarize_candump(candump_path, out_path):
    frame_re = re.compile(r"\)\s+(\S+)\s+([0-9A-Fa-f]{3,8})\s+\[\d+\]")
    total = 0
    by_iface = {}
    by_comm = {}
    by_iface_comm = {}
    by_iface_motor_low = {}
    by_iface_motor_mid = {}
    first_t = None
    last_t = None
    time_re = re.compile(r"\(([-+0-9.]+)\)")

    if not candump_path.exists():
        return

    with open(candump_path, "r", errors="replace") as fp:
        for line in fp:
            m = frame_re.search(line)
            if not m:
                continue
            tm = time_re.search(line)
            if tm:
                try:
                    t = float(tm.group(1))
                    first_t = t if first_t is None else min(first_t, t)
                    last_t = t if last_t is None else max(last_t, t)
                except ValueError:
                    pass
            iface, can_id_text = m.group(1), m.group(2)
            try:
                can_id = int(can_id_text, 16)
            except ValueError:
                continue
            total += 1
            comm = (can_id >> 24) & 0x1F
            low = can_id & 0xFF
            mid = (can_id >> 8) & 0xFF
            by_iface[iface] = by_iface.get(iface, 0) + 1
            by_comm[comm] = by_comm.get(comm, 0) + 1
            by_iface_comm[(iface, comm)] = by_iface_comm.get((iface, comm), 0) + 1
            by_iface_motor_low[(iface, low)] = by_iface_motor_low.get((iface, low), 0) + 1
            by_iface_motor_mid[(iface, mid)] = by_iface_motor_mid.get((iface, mid), 0) + 1

    with open(out_path, "w") as fp:
        fp.write("=== candump summary ===\n")
        fp.write(f"frames: {total}\n")
        if first_t is not None and last_t is not None:
            elapsed = max(last_t - first_t, 1e-9)
            fp.write(f"time_span: {elapsed:.6f}s\n")
            fp.write(f"avg_frame_rate: {total / elapsed:.2f} fps\n")
        fp.write("\nby interface:\n")
        for iface, count in sorted(by_iface.items()):
            fp.write(f"  {iface}: {count}\n")
        fp.write("\nby comm_type:\n")
        for comm, count in sorted(by_comm.items()):
            fp.write(f"  0x{comm:02X}: {count}\n")
        fp.write("\nby interface + comm_type:\n")
        for (iface, comm), count in sorted(by_iface_comm.items()):
            fp.write(f"  {iface} comm=0x{comm:02X}: {count}\n")
        fp.write("\nby interface + low-byte motor id:\n")
        for (iface, motor_id), count in sorted(by_iface_motor_low.items()):
            if motor_id != 0:
                fp.write(f"  {iface} low_id={motor_id}: {count}\n")
        fp.write("\nby interface + mid-byte motor id:\n")
        for (iface, motor_id), count in sorted(by_iface_motor_mid.items()):
            if motor_id != 0:
                fp.write(f"  {iface} mid_id={motor_id}: {count}\n")


def topic_qos(reliable=False):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Record deploy runtime topics and CAN state without sending any control command."
    )
    parser.add_argument("--duration", type=float, default=60.0, help="record seconds; 0 means until Ctrl+C")
    parser.add_argument("--out-dir", default="", help="output directory; default logs/deploy_record_TIMESTAMP")
    parser.add_argument("--can-interval", type=float, default=1.0, help="seconds between CAN snapshots")
    parser.add_argument("--summary-interval", type=float, default=1.0, help="seconds between console summaries")
    parser.add_argument("--no-candump", action="store_true", help="disable passive candump frame capture")
    parser.add_argument("--commands-topic", default="/policy/commands")
    parser.add_argument("--joint-states-topic", default="/policy/joint_states")
    parser.add_argument("--imu-topic", default="/IMU_data")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--can-ifaces", nargs="*", default=["can0"])
    parser.add_argument(
        "--motor-names",
        nargs="*",
        default=[f"motor_{i}" for i in range(1, 7)],
        help="joint names to write as fixed columns",
    )
    args = parser.parse_args()

    root_dir = deployment_root()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = root_dir / "logs" / f"deploy_record_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / "startup_snapshot.txt"
    with open(snapshot_path, "w") as fp:
        fp.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fp.write(f"cwd: {os.getcwd()}\n\n")
        fp.write("=== git rev-parse HEAD ===\n")
        fp.write(run_cmd(["git", "-C", str(root_dir), "rev-parse", "HEAD"]))
        fp.write("\n=== git status --short ===\n")
        fp.write(run_cmd(["git", "-C", str(root_dir), "status", "--short"]))
        fp.write("\n=== ros2 node list ===\n")
        fp.write(run_cmd(["ros2", "node", "list"], timeout=5.0))
        fp.write("\n=== ros2 topic list ===\n")
        fp.write(run_cmd(["ros2", "topic", "list"], timeout=5.0))
        fp.write("\n=== CAN initial state ===\n")
        for iface in args.can_ifaces:
            fp.write(f"\n--- {iface} ---\n")
            fp.write(run_cmd(["ip", "-details", "-statistics", "link", "show", iface]))

    cmd_fp, cmd_writer = make_writer(
        out_dir / "policy_commands.csv",
        ["t_rel", "t_wall"] + [f"cmd_{i}" for i in range(6)],
    )
    joint_header = ["t_rel", "t_wall"]
    for name in args.motor_names:
        joint_header += [f"{name}_pos", f"{name}_vel", f"{name}_effort"]
    joint_fp, joint_writer = make_writer(out_dir / "policy_joint_states.csv", joint_header)
    imu_fp, imu_writer = make_writer(
        out_dir / "imu_data.csv",
        [
            "t_rel",
            "t_wall",
            "qx",
            "qy",
            "qz",
            "qw",
            "wx",
            "wy",
            "wz",
            "gravity_x",
            "gravity_y",
            "gravity_z",
        ],
    )
    vel_fp, vel_writer = make_writer(
        out_dir / "cmd_vel.csv",
        ["t_rel", "t_wall", "lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"],
    )
    can_fp = open(out_dir / "can_stats.txt", "w")
    summary_fp = open(out_dir / "summary.txt", "w")
    candump_proc, candump_fp = start_candump(out_dir, args.can_ifaces, enabled=not args.no_candump)

    rclpy.init()
    node = rclpy.create_node("record_deploy_runtime")
    t0 = time.time()
    counts = {"commands": 0, "joint": 0, "imu": 0, "cmd_vel": 0}
    last = {"commands": None, "joint_pos": None, "imu_g": None, "cmd_vel": None}

    def rel_now():
        wall = time.time()
        return wall - t0, wall

    def commands_cb(msg):
        t_rel, wall = rel_now()
        data = list(msg.data)
        cmd_writer.writerow([f"{t_rel:.6f}", f"{wall:.6f}"] + [f"{vector_get(data, i):.8f}" for i in range(6)])
        counts["commands"] += 1
        last["commands"] = [vector_get(data, i) for i in range(6)]

    def joint_cb(msg):
        t_rel, wall = rel_now()
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity))
        eff = dict(zip(msg.name, msg.effort))
        row = [f"{t_rel:.6f}", f"{wall:.6f}"]
        for name in args.motor_names:
            row += [
                f"{pos.get(name, float('nan')):.8f}",
                f"{vel.get(name, float('nan')):.8f}",
                f"{eff.get(name, float('nan')):.8f}",
            ]
        joint_writer.writerow(row)
        counts["joint"] += 1
        last["joint_pos"] = [pos.get(name, float("nan")) for name in args.motor_names]

    def imu_cb(msg):
        t_rel, wall = rel_now()
        q = msg.orientation
        w = msg.angular_velocity
        g = msg.angular_velocity_covariance
        imu_writer.writerow(
            [
                f"{t_rel:.6f}",
                f"{wall:.6f}",
                f"{q.x:.8f}",
                f"{q.y:.8f}",
                f"{q.z:.8f}",
                f"{q.w:.8f}",
                f"{w.x:.8f}",
                f"{w.y:.8f}",
                f"{w.z:.8f}",
                f"{g[0]:.8f}",
                f"{g[1]:.8f}",
                f"{g[2]:.8f}",
            ]
        )
        counts["imu"] += 1
        last["imu_g"] = [g[0], g[1], g[2]]

    def cmd_vel_cb(msg):
        t_rel, wall = rel_now()
        vel_writer.writerow(
            [
                f"{t_rel:.6f}",
                f"{wall:.6f}",
                f"{msg.linear.x:.8f}",
                f"{msg.linear.y:.8f}",
                f"{msg.linear.z:.8f}",
                f"{msg.angular.x:.8f}",
                f"{msg.angular.y:.8f}",
                f"{msg.angular.z:.8f}",
            ]
        )
        counts["cmd_vel"] += 1
        last["cmd_vel"] = [msg.linear.x, msg.linear.y, msg.angular.z]

    node.create_subscription(Float32MultiArray, args.commands_topic, commands_cb, topic_qos(reliable=True))
    node.create_subscription(JointState, args.joint_states_topic, joint_cb, topic_qos(reliable=False))
    node.create_subscription(Imu, args.imu_topic, imu_cb, topic_qos(reliable=False))
    node.create_subscription(Twist, args.cmd_vel_topic, cmd_vel_cb, topic_qos(reliable=False))

    duration_text = "until Ctrl+C" if args.duration <= 0.0 else f"{args.duration:.1f}s"
    print(f"Recording deploy runtime for {duration_text}")
    print(f"Output directory: {out_dir}")
    print("不发送控制帧、不调用服务，只订阅话题并读取 ip link 状态。")
    if candump_proc is not None:
        print("candump passive capture: enabled -> candump.log")
    else:
        print("candump passive capture: disabled")

    prev_counts = counts.copy()
    next_can = time.time()
    next_summary = time.time() + args.summary_interval
    deadline = None if args.duration <= 0.0 else time.time() + args.duration

    try:
        while rclpy.ok() and (deadline is None or time.time() < deadline):
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.time()

            if now >= next_can:
                can_fp.write(f"\n===== t={now - t0:.3f}s wall={now:.6f} =====\n")
                for iface in args.can_ifaces:
                    text = run_cmd(["ip", "-details", "-statistics", "link", "show", iface])
                    can_fp.write(f"\n--- {iface} {parse_can_summary(text)} ---\n")
                    can_fp.write(text)
                    if not text.endswith("\n"):
                        can_fp.write("\n")
                can_fp.flush()
                next_can += args.can_interval

            if now >= next_summary:
                dt = max(args.summary_interval, 1e-6)
                rates = {key: (counts[key] - prev_counts[key]) / dt for key in counts}
                line = (
                    f"t={now - t0:7.2f}s "
                    f"hz cmd={rates['commands']:6.1f} joint={rates['joint']:6.1f} "
                    f"imu={rates['imu']:6.1f} cmd_vel={rates['cmd_vel']:5.1f} "
                    f"last_cmd={last['commands']} imu_g={last['imu_g']}"
                )
                print(line, flush=True)
                summary_fp.write(line + "\n")
                summary_fp.flush()
                prev_counts = counts.copy()
                next_summary += args.summary_interval
    except KeyboardInterrupt:
        print("\nInterrupted, closing record files...")
    finally:
        stop_process(candump_proc, candump_fp)
        for fp in [cmd_fp, joint_fp, imu_fp, vel_fp, can_fp, summary_fp]:
            fp.flush()
            fp.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        summarize_candump(out_dir / "candump.log", out_dir / "can_frame_summary.txt")

    elapsed = max(time.time() - t0, 1e-6)
    print("\n=== record summary ===")
    for key in ["commands", "joint", "imu", "cmd_vel"]:
        print(f"{key}: {counts[key]} samples, avg {counts[key] / elapsed:.2f} Hz")
    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
