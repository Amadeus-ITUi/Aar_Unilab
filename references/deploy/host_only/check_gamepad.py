#!/usr/bin/env python3
"""交互式 ROS 2 手柄映射与断触检测（使用隔离话题，不输出控制命令）。"""

import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy


TESTS = [
    ("左摇杆-上", "axis"), ("左摇杆-下", "axis"),
    ("左摇杆-左", "axis"), ("左摇杆-右", "axis"),
    ("右摇杆-上", "axis"), ("右摇杆-下", "axis"),
    ("右摇杆-左", "axis"), ("右摇杆-右", "axis"),
    ("方向键-上", "either"), ("方向键-下", "either"),
    ("方向键-左", "either"), ("方向键-右", "either"),
    ("A", "button"), ("B", "button"), ("X", "button"), ("Y", "button"),
    ("LB", "button"), ("RB", "button"), ("LT", "either"), ("RT", "either"),
    ("Back", "button"), ("Start", "button"), ("Guide", "button"),
    ("左摇杆按压", "button"), ("右摇杆按压", "button"),
]


# ROS 2 joy_node 对 Linux xpad 轴值取反后的标准 Xbox 360 映射。
# WE11 生命周期启动器直接读取 /dev/input/js0，因此轴符号与此表相反。
EXPECTED_MAPPING = {
    "左摇杆-上": "axes[1] +", "左摇杆-下": "axes[1] -",
    "左摇杆-左": "axes[0] +", "左摇杆-右": "axes[0] -",
    "右摇杆-上": "axes[4] +", "右摇杆-下": "axes[4] -",
    "右摇杆-左": "axes[3] +", "右摇杆-右": "axes[3] -",
    "方向键-上": "axes[7] +", "方向键-下": "axes[7] -",
    "方向键-左": "axes[6] +", "方向键-右": "axes[6] -",
    "A": "buttons[0]", "B": "buttons[1]", "X": "buttons[2]", "Y": "buttons[3]",
    "LB": "buttons[4]", "RB": "buttons[5]",
    "LT": "axes[2] -", "RT": "axes[5] -",
    "Back": "buttons[6]", "Start": "buttons[7]", "Guide": "buttons[8]",
    "左摇杆按压": "buttons[9]", "右摇杆按压": "buttons[10]",
}


@dataclass
class Result:
    name: str
    source: str
    samples: int
    active: int
    dropouts: int
    longest_gap_ms: float
    rate: float
    passed: bool


class JoyMonitor(Node):
    def __init__(self, topic):
        super().__init__("gamepad_mapping_checker")
        self.lock = threading.Lock()
        self.latest = None
        self.received_at = 0.0
        self.samples = []
        self.create_subscription(Joy, topic, self.callback, qos_profile_sensor_data)

    def callback(self, msg):
        now = time.monotonic()
        axes = tuple(float(v) for v in msg.axes)
        buttons = tuple(int(v) for v in msg.buttons)
        with self.lock:
            self.latest = (axes, buttons)
            self.received_at = now
            self.samples.append((now, axes, buttons))
            if len(self.samples) > 2000:
                del self.samples[:1000]

    def snapshot(self):
        with self.lock:
            return self.latest, self.received_at

    def since(self, timestamp):
        with self.lock:
            return [s for s in self.samples if s[0] >= timestamp]


def ensure_joystick_device(path):
    if os.path.exists(path):
        return
    print(f"未发现手柄设备 {path}，尝试加载 Xbox 兼容驱动 xpad……")
    attempts = [(["modprobe", "xpad"], False)]
    if os.geteuid() != 0:
        attempts.append((["sudo", "-n", "modprobe", "xpad"], False))
        if sys.stdin.isatty():
            attempts.append((["sudo", "modprobe", "xpad"], True))
    loaded = False
    for command, interactive in attempts:
        try:
            quiet = {} if interactive else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            if subprocess.run(command, check=False, **quiet).returncode == 0:
                loaded = True
                break
        except FileNotFoundError:
            continue
    if loaded:
        for _ in range(20):
            if os.path.exists(path):
                return
            time.sleep(0.1)
    raise RuntimeError(
        f"找不到手柄设备 {path}。请执行 sudo modprobe xpad 后重新插拔手柄，"
        "或用 --dev 指定正确设备"
    )


def difference_candidates(current, baseline, allowed):
    axes, buttons = current
    base_axes, base_buttons = baseline
    found = []
    if allowed in ("axis", "either"):
        for i in range(min(len(axes), len(base_axes))):
            delta = axes[i] - base_axes[i]
            if abs(delta) >= 0.45:
                found.append((abs(delta), "axis", i, 1 if delta > 0 else -1))
    if allowed in ("button", "either"):
        for i in range(min(len(buttons), len(base_buttons))):
            if buttons[i] != base_buttons[i]:
                found.append((1.0, "button", i, 1 if buttons[i] else -1))
    return sorted(found, reverse=True)


def is_active(sample, baseline, kind, index, direction):
    axes, buttons = sample
    base_axes, base_buttons = baseline
    if kind == "button":
        return index < len(buttons) and index < len(base_buttons) and buttons[index] != base_buttons[index]
    return (index < len(axes) and index < len(base_axes)
            and direction * (axes[index] - base_axes[index]) >= 0.30)


def wait_for_fresh(monitor, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        latest, stamp = monitor.snapshot()
        if latest is not None and time.monotonic() - stamp < 0.5:
            return latest
        time.sleep(0.02)
    return None


def spin_monitor(monitor):
    try:
        rclpy.spin(monitor)
    except ExternalShutdownException:
        pass


def run_test(monitor, name, allowed, hold_s, timeout_s):
    print(f"\n[{name}] 请先保持回正/松开；随后操作并持续保持至少 {hold_s:.1f}s。")
    time.sleep(0.35)
    baseline = wait_for_fresh(monitor, 1.0)
    if baseline is None:
        return Result(name, "无数据", 0, 0, 0, 0.0, 1.0, False)

    deadline = time.monotonic() + timeout_s
    chosen = None
    start = 0.0
    while time.monotonic() < deadline:
        current, _ = monitor.snapshot()
        candidates = difference_candidates(current, baseline, allowed) if current else []
        if candidates:
            _, kind, index, direction = candidates[0]
            chosen = (kind, index, direction)
            start = time.monotonic()
            break
        time.sleep(0.005)
    if chosen is None:
        print("  超时：未检测到该操作。")
        return Result(name, "未识别", 0, 0, 0, 0.0, 1.0, False)

    kind, index, direction = chosen
    source = f"{'axes' if kind == 'axis' else 'buttons'}[{index}]"
    if kind == "axis":
        source += " +" if direction > 0 else " -"
    print(f"  识别为 {source}，请继续保持……")
    end = start + hold_s
    while time.monotonic() < end:
        time.sleep(0.005)
    samples = monitor.since(start)
    samples = [s for s in samples if s[0] <= end]
    states = [(t, is_active((a, b), baseline, kind, index, direction)) for t, a, b in samples]
    active = sum(state for _, state in states)
    dropouts = 0
    longest_gap = 0.0
    gap_start = None
    previously_active = True
    for stamp, state in states:
        if not state and previously_active:
            dropouts += 1
            gap_start = stamp
        elif state and not previously_active and gap_start is not None:
            longest_gap = max(longest_gap, stamp - gap_start)
            gap_start = None
        previously_active = state
    if gap_start is not None:
        longest_gap = max(longest_gap, end - gap_start)
    rate = 1.0 - active / len(states) if states else 1.0
    passed = bool(states) and rate <= 0.02 and longest_gap <= 0.03
    print(f"  {'通过' if passed else '异常'}：断触率 {rate:.1%}，断触段 {dropouts}，最长 {longest_gap * 1000:.0f} ms")
    print("  请松开并回正，等待 1.0s……")
    time.sleep(1.0)
    return Result(name, source, len(states), active, dropouts, longest_gap * 1000, rate, passed)


def main():
    parser = argparse.ArgumentParser(description="完整检测手柄映射和保持期间的断触率")
    parser.add_argument("--dev", default="/dev/input/js0", help="手柄设备（默认 /dev/input/js0）")
    parser.add_argument("--topic", default="/gamepad_check/joy", help="隔离的 Joy 话题")
    parser.add_argument("--hold", type=float, default=0.5, help="每项保持检测秒数")
    parser.add_argument("--timeout", type=float, default=15.0, help="等待每项操作的超时秒数")
    parser.add_argument("--no-start-joy", action="store_true", help="不启动 joy_node，直接订阅 --topic")
    args = parser.parse_args()
    if args.hold <= 0 or args.timeout <= 0:
        parser.error("--hold 和 --timeout 必须大于 0")
    if not args.no_start_joy:
        try:
            ensure_joystick_device(args.dev)
        except RuntimeError as exc:
            parser.error(str(exc))

    joy_process = None
    rclpy.init()
    monitor = JoyMonitor(args.topic)
    executor_thread = threading.Thread(target=spin_monitor, args=(monitor,), daemon=True)
    executor_thread.start()
    try:
        if not args.no_start_joy:
            cmd = ["ros2", "run", "joy", "joy_node", "--ros-args",
                   "-r", f"joy:={args.topic}", "-r", "__node:=gamepad_check_joy_node",
                   "-p", f"dev:={args.dev}", "-p", "deadzone:=0.0",
                   "-p", "autorepeat_rate:=100.0"]
            joy_process = subprocess.Popen(cmd, start_new_session=True)
        first = wait_for_fresh(monitor, 5.0)
        if first is None:
            raise RuntimeError(f"5 秒内未收到 {args.topic}；请检查设备、权限和 joy 包")
        print(f"收到 Joy：{len(first[0])} 个 axes，{len(first[1])} 个 buttons。")
        print("检测时机器人控制话题不会被发布。每项完成后固定留出 1 秒回正时间。")
        results = [run_test(monitor, name, allowed, args.hold, args.timeout)
                   for name, allowed in TESTS]
        print("\n================ 检测汇总 ================")
        print(f"{'控件':<12} {'映射':<14} {'断触率':>8} {'断触段':>8} {'最长间断':>10} 结果")
        for item in results:
            print(f"{item.name:<12} {item.source:<14} {item.rate:>7.1%} {item.dropouts:>8} "
                  f"{item.longest_gap_ms:>8.0f}ms {'通过' if item.passed else '异常'}")
        observed = {item.name: item.source for item in results}
        mismatches = {
            name: (EXPECTED_MAPPING[name], observed.get(name, "未检测"))
            for name in EXPECTED_MAPPING
            if observed.get(name) != EXPECTED_MAPPING[name]
        }
        print("\n================ Xbox 映射契约 ================")
        if mismatches:
            print("映射不符合 Linux xpad 标准：")
            for name, (expected, actual) in mismatches.items():
                print(f"  {name}: 预期 {expected}，实际 {actual}")
        else:
            print("全部 25 项映射符合 Linux xpad 标准。")

        lifecycle_ok = (
            observed.get("方向键-上") == "axes[7] +"
            and observed.get("X") == "buttons[2]"
            and observed.get("A") == "buttons[0]"
        )
        print(
            "WE11 生命周期组合键："
            + (
                "通过（ROS axes[7] + 对应原始 js0 轴值为负；上+X 启动，上+A 停止）"
                if lifecycle_ok else "失败，禁止启动机器人"
            )
        )
        collisions = {}
        for item in results:
            if item.source not in ("未识别", "无数据"):
                collision_key = item.source.rstrip(" +-") if item.source.startswith("axes[") else item.source
                collisions.setdefault(collision_key, []).append(item.name)
        suspicious = {key: names for key, names in collisions.items() if len(names) > 1}
        if suspicious:
            print("\n复用的轴（正反方向共用同一 axis 属正常）：")
            for source, names in suspicious.items():
                print(f"  {source}: {', '.join(names)}")
        return 0 if all(r.passed for r in results) and not mismatches and lifecycle_ok else 2
    except (KeyboardInterrupt, RuntimeError) as exc:
        print(f"\n检测中止：{exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if joy_process is not None and joy_process.poll() is None:
            os.killpg(joy_process.pid, signal.SIGINT)
            try:
                joy_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(joy_process.pid, signal.SIGTERM)


if __name__ == "__main__":
    sys.exit(main())
