#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import rclpy
from esd_link_msgs.msg import LinkStatus, LowerState, PolicyCommand
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


def topic_qos(reliable=False):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Record deploy runtime topics and ESD-Link state without sending a control command."
    )
    parser.add_argument("--duration", type=float, default=60.0, help="record seconds; 0 means until Ctrl+C")
    parser.add_argument("--out-dir", default="", help="output directory; default logs/deploy_record_TIMESTAMP")
    parser.add_argument("--summary-interval", type=float, default=1.0, help="seconds between console summaries")
    parser.add_argument("--commands-topic", default="/policy/commands")
    parser.add_argument("--joint-states-topic", default="/policy/joint_states")
    parser.add_argument("--imu-topic", default="/IMU_data")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
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
        fp.write("\n=== ESD-Link device ===\n")
        fp.write(run_cmd(["readlink", "-f", "/dev/serial/by-id/usb-Sirin_Systems___OmniX_Robotics_ESD-SLAVE_314137523333-if00"]))

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
    stamped_fp, stamped_writer = make_writer(
        out_dir / "policy_stamped_commands.csv",
        ["t_rel", "t_wall", "source_state_seq", "inference_start_ns", "inference_end_ns"]
        + [f"action_{i}" for i in range(6)],
    )
    lower_fp, lower_writer = make_writer(
        out_dir / "lower_state.csv",
        ["t_rel", "t_wall", "host_monotonic_ns", "parse_done_monotonic_ns",
         "session_id", "schema_id", "device_sample_time_us", "state_sample_seq",
         "last_applied_command_seq", "command_status_flags", "control_state",
         "imu_valid_mask", "fault_flags", "active_port_mask", "offline_port_mask"]
        + [item for port in range(1, 9) for item in
           (f"p{port}_valid", f"p{port}_q", f"p{port}_dq", f"p{port}_tau")],
    )
    link_fp, link_writer = make_writer(
        out_dir / "esd_link_status.csv",
        ["t_rel", "t_wall", "link_state", "control_mode", "device_control_state",
         "session_valid", "session_id", "schema_id", "config_fingerprint",
         "active_port_mask", "offline_port_mask", "fault_flags", "last_reject_code",
         "state_rate_hz", "latest_state_age_ms", "maximum_interarrival_ms",
         "observation_to_command_ms", "command_to_applied_ms", "valid_frames",
         "cobs_errors", "crc_errors", "sequence_gaps", "rejected_commands",
         "watchdog_events", "control_fault_events", "reconnects",
         "device_valid_commands", "device_invalid_frames",
         "device_rejected_commands", "device_command_age_ms", "last_command_sequence",
         "last_applied_command_sequence"],
    )
    summary_fp = open(out_dir / "summary.txt", "w")

    rclpy.init()
    node = rclpy.create_node("record_deploy_runtime")
    t0 = time.time()
    counts = {"commands": 0, "stamped": 0, "lower": 0, "link": 0,
              "joint": 0, "imu": 0, "cmd_vel": 0}
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

    def stamped_cb(msg):
        t_rel, wall = rel_now()
        stamped_writer.writerow(
            [f"{t_rel:.6f}", f"{wall:.6f}", msg.source_state_sample_seq,
             msg.inference_start_monotonic_ns, msg.inference_end_monotonic_ns]
            + [f"{value:.8f}" for value in msg.action]
        )
        counts["stamped"] += 1

    def lower_cb(msg):
        t_rel, wall = rel_now()
        row = [f"{t_rel:.6f}", f"{wall:.6f}", msg.host_monotonic_ns,
               msg.parse_done_monotonic_ns, msg.session_id, msg.schema_id,
               msg.device_sample_time_us, msg.state_sample_seq,
               msg.last_applied_command_seq, msg.command_status_flags,
               msg.control_state, msg.imu_valid_mask, msg.fault_flags,
               msg.active_port_mask, msg.offline_port_mask]
        for index in range(8):
            row.extend([msg.valid_mask[index], f"{msg.position_rad[index]:.8f}",
                        f"{msg.velocity_rad_s[index]:.8f}", f"{msg.effort_nm[index]:.8f}"])
        lower_writer.writerow(row)
        counts["lower"] += 1

    def link_cb(msg):
        t_rel, wall = rel_now()
        link_writer.writerow([
            f"{t_rel:.6f}", f"{wall:.6f}", msg.link_state, msg.control_mode,
            msg.device_control_state, int(msg.session_valid), msg.session_id,
            msg.schema_id, msg.config_fingerprint, msg.active_port_mask,
            msg.offline_port_mask, msg.fault_flags, msg.last_reject_code,
            f"{msg.state_rate_hz:.4f}", f"{msg.latest_state_age_ms:.4f}",
            f"{msg.maximum_interarrival_ms:.4f}",
            f"{msg.observation_to_command_ms:.4f}", f"{msg.command_to_applied_ms:.4f}",
            msg.valid_frames, msg.cobs_errors, msg.crc_errors, msg.sequence_gaps,
            msg.rejected_commands, msg.watchdog_events, msg.control_fault_events,
            msg.reconnects,
            msg.device_valid_commands, msg.device_invalid_frames,
            msg.device_rejected_commands, msg.device_command_age_ms,
            msg.last_command_sequence, msg.last_applied_command_sequence,
        ])
        counts["link"] += 1

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
    node.create_subscription(PolicyCommand, "/policy/stamped_commands", stamped_cb, topic_qos(reliable=True))
    node.create_subscription(LowerState, "/lower/state", lower_cb, topic_qos(reliable=False))
    node.create_subscription(LinkStatus, "/lower/link_status", link_cb, topic_qos(reliable=True))
    node.create_subscription(JointState, args.joint_states_topic, joint_cb, topic_qos(reliable=False))
    node.create_subscription(Imu, args.imu_topic, imu_cb, topic_qos(reliable=False))
    node.create_subscription(Twist, args.cmd_vel_topic, cmd_vel_cb, topic_qos(reliable=False))

    duration_text = "until Ctrl+C" if args.duration <= 0.0 else f"{args.duration:.1f}s"
    print(f"Recording deploy runtime for {duration_text}")
    print(f"Output directory: {out_dir}")
    print("不发送控制帧、不调用服务；记录 LowerState、PolicyCommand 与 ESD-Link 统计。")

    prev_counts = counts.copy()
    next_summary = time.time() + args.summary_interval
    deadline = None if args.duration <= 0.0 else time.time() + args.duration

    try:
        while rclpy.ok() and (deadline is None or time.time() < deadline):
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.time()

            if now >= next_summary:
                dt = max(args.summary_interval, 1e-6)
                rates = {key: (counts[key] - prev_counts[key]) / dt for key in counts}
                line = (
                    f"t={now - t0:7.2f}s "
                    f"hz cmd={rates['commands']:6.1f} joint={rates['joint']:6.1f} "
                    f"stamped={rates['stamped']:5.1f} lower={rates['lower']:6.1f} "
                    f"link={rates['link']:3.1f} imu={rates['imu']:6.1f} cmd_vel={rates['cmd_vel']:5.1f} "
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
        for fp in [cmd_fp, stamped_fp, lower_fp, link_fp, joint_fp, imu_fp, vel_fp, summary_fp]:
            fp.flush()
            fp.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    elapsed = max(time.time() - t0, 1e-6)
    print("\n=== record summary ===")
    for key in ["commands", "stamped", "lower", "link", "joint", "imu", "cmd_vel"]:
        print(f"{key}: {counts[key]} samples, avg {counts[key] / elapsed:.2f} Hz")
    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
