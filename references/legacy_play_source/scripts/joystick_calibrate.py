#!/usr/bin/env python3
"""Xbox 手柄轴/按钮校准脚本，配合 Play 里 rl_sim_mujoco 的 joystick 层使用。

按提示做一个动作，脚本会记录期间变化最大的那根轴（或按下的那个按钮）以及
它的极性方向，最后打印一份 config.yaml 的映射建议。

用法:
    python3 scripts/joystick_calibrate.py                # 默认 /dev/input/js0
    JS_DEVICE=/dev/input/js1 python3 scripts/joystick_calibrate.py
"""

import os
import struct
import sys
import time
from dataclasses import dataclass


JS_EVENT_FMT = "IhBB"  # timestamp (u32), value (i16), type (u8), number (u8)
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FMT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80  # ored with type on init events; ignore those
AXIS_ACTIVATE_THRESHOLD = 16000   # ~50% of full range (int16 max)
BUTTON_ACTIVATE_VALUE = 1


@dataclass
class Prompt:
    key: str            # slot name to store in results
    kind: str           # "axis" or "button"
    description: str    # human-readable ask
    # For axis prompts we also want the sign the operator gets when they follow
    # the description. e.g. "push LY up" -> expected sign +1; some sticks are
    # inverted so we record what actually happens.
    expected_sign: int = 1


PROMPTS = [
    Prompt("ly_forward", "axis", "把左摇杆推到最上（对应前进）"),
    Prompt("rx_left",    "axis", "把右摇杆推到最左（对应左转）"),
    Prompt("ry_up",      "axis", "把右摇杆推到最上（对应升高机身）"),
    Prompt("button_a",   "button", "按一次 A 键"),
    Prompt("button_b",   "button", "按一次 B 键"),
    Prompt("button_x",   "button", "按一次 X 键"),
    Prompt("button_y",   "button", "按一次 Y 键"),
    Prompt("button_lb",  "button", "按一次 LB (左上肩键)"),
    Prompt("button_rb",  "button", "按一次 RB (右上肩键)"),
]


def open_device(path: str):
    try:
        return os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        print(f"error: joystick device {path} 不存在。用 JS_DEVICE=... 指定其他设备。",
              file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"error: 打不开 {path}，检查权限（一般 input 组即可）。",
              file=sys.stderr)
        sys.exit(1)


AXIS_REST_THRESHOLD = 4000     # ~12% of full range; below this we treat as centered
AXIS_REST_DWELL_S = 0.6        # need this long below the rest threshold
AXIS_REST_TIMEOUT_S = 6.0      # give up waiting after this


def flush_pending(fd, seconds: float = 0.2):
    """吃掉初始化事件和残留事件。"""
    end = time.monotonic() + seconds
    os.set_blocking(fd, False)
    try:
        while time.monotonic() < end:
            try:
                data = os.read(fd, JS_EVENT_SIZE)
                if not data:
                    break
            except BlockingIOError:
                time.sleep(0.01)
            except OSError:
                break
    finally:
        os.set_blocking(fd, True)


def wait_for_axes_to_center(fd, timeout: float = AXIS_REST_TIMEOUT_S) -> bool:
    """把摇杆放松、等所有轴回到零附近后返回 True。

    脚本会持续读事件更新每根轴的当前值；只要连续 AXIS_REST_DWELL_S 秒内所有已
    活动过的轴都低于 AXIS_REST_THRESHOLD，就认为已回正。超时返回 False。
    """
    current = {}   # axis_number -> latest absolute value
    stable_since = None
    deadline = time.monotonic() + timeout
    os.set_blocking(fd, False)
    try:
        while time.monotonic() < deadline:
            try:
                data = os.read(fd, JS_EVENT_SIZE)
            except BlockingIOError:
                data = None
            if data and len(data) == JS_EVENT_SIZE:
                _ts, value, etype, number = struct.unpack(JS_EVENT_FMT, data)
                if not (etype & JS_EVENT_INIT) and (etype & 0x7F) == JS_EVENT_AXIS:
                    current[number] = abs(value)
            now = time.monotonic()
            all_rested = all(v <= AXIS_REST_THRESHOLD for v in current.values())
            if all_rested:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= AXIS_REST_DWELL_S:
                    return True
            else:
                stable_since = None
            time.sleep(0.01)
    finally:
        os.set_blocking(fd, True)
    return False


def countdown(seconds: float):
    """在同一行倒计时，给操作者时间松手/回正。"""
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        sys.stdout.write(f"\r  下一步倒计时 {remaining:.1f}s ")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


def wait_for_event(fd, kind: str, timeout: float = 15.0):
    """等一个有意义的按下 / 摇杆越过阈值的事件，返回 (number, value)。"""
    deadline = time.monotonic() + timeout
    max_seen_axis = {}   # axis_number -> (abs_value, signed_value)
    while time.monotonic() < deadline:
        data = os.read(fd, JS_EVENT_SIZE)
        if len(data) < JS_EVENT_SIZE:
            continue
        _ts, value, etype, number = struct.unpack(JS_EVENT_FMT, data)
        if etype & JS_EVENT_INIT:
            continue
        raw_type = etype & 0x7F
        if kind == "button":
            if raw_type == JS_EVENT_BUTTON and value == BUTTON_ACTIVATE_VALUE:
                return number, value
        elif kind == "axis":
            if raw_type != JS_EVENT_AXIS:
                continue
            prev_abs, _prev_val = max_seen_axis.get(number, (0, 0))
            if abs(value) > prev_abs:
                max_seen_axis[number] = (abs(value), value)
            if abs(value) >= AXIS_ACTIVATE_THRESHOLD:
                return number, value
    if kind == "axis" and max_seen_axis:
        # Nothing crossed the threshold; return the largest deviation.
        number, (_abs, value) = max(max_seen_axis.items(),
                                    key=lambda kv: kv[1][0])
        return number, value
    return None, None


def format_summary(results):
    lines = []
    lines.append("")
    lines.append("=========== 校准结果 ===========")
    for prompt in PROMPTS:
        entry = results.get(prompt.key)
        if entry is None:
            lines.append(f"{prompt.description}: (超时未检测到)")
            continue
        number, value = entry
        if prompt.kind == "axis":
            sign = "+" if value > 0 else "-"
            lines.append(
                f"{prompt.description}: axis[{number}] 触发时值 = {value:>+6d} "
                f"(方向 {sign})")
        else:
            lines.append(f"{prompt.description}: button[{number}]")
    lines.append("")

    ly = results.get("ly_forward")
    rx = results.get("rx_left")
    ry = results.get("ry_up")

    lines.append("---- 建议的映射（可以直接对照到源码/配置） ----")
    if ly is not None:
        axis, value = ly
        # 训练/部署侧约定：LY 前进产生 control.x > 0。
        # rl_sim_mujoco.cpp 里当前写的是 `ly = -axis[1] / max_value`，
        # 期望"摇杆向上时 raw < 0"（joy 设备的标准），此时 ly > 0。
        need_invert = value > 0  # value > 0 表示 raw 值正，跟标准约定相反
        lines.append(
            f"  左摇杆前后 (control.x): axis[{axis}], "
            f"invert = {'true' if need_invert else 'false'}")
    if rx is not None:
        axis, value = rx
        # 训练侧："左转" = control.yaw > 0。
        # rl_sim_mujoco.cpp 使用 `rx = -axis[N] / max_value`，也就是要求
        # "摇杆向左时 raw < 0"。
        # 但很多 xbox 手柄向左时 raw < 0，向右时 raw > 0 —— 这里"向左时 value < 0"
        # 是符合默认约定的。若你按提示向左推得到 value > 0，说明这根轴反了。
        need_invert = value > 0
        lines.append(
            f"  右摇杆左右 (control.yaw): axis[{axis}], "
            f"invert = {'true' if need_invert else 'false'}")
    if ry is not None:
        axis, value = ry
        # Play/config 里 height_velocity_command.joystick_invert_sign 控制。
        # rl_sim_mujoco 现在的代码是 `stick = -axis[N] / max_value`，
        # 期望"向上 raw < 0"。
        need_invert = value > 0
        lines.append(
            f"  右摇杆上下 (机身高度速度): axis[{axis}], "
            f"joystick_invert_sign = {'true' if need_invert else 'false'}")
        lines.append(
            f"    → 写到 policy/dr002/we11/config.yaml 的 height_velocity_command 块:")
        lines.append(f"        joystick_axis: {axis}")
        lines.append(
            f"        joystick_invert_sign: "
            f"{'true' if need_invert else 'false'}")
    lines.append("")
    lb = results.get("button_lb")
    rb = results.get("button_rb")
    if lb is not None and rb is not None:
        lines.append("---- 翅膀按钮配置 ----")
        lines.append(f"  decrease_button: {lb[0]}  # LB")
        lines.append(f"  increase_button: {rb[0]}  # RB")
        lines.append("")
    lines.append("---- 按钮索引 (rl_sim_mujoco.cpp GetSysJoystick 里 sys_js_button[N]) ----")
    for key, label in [
        ("button_a", "A"),
        ("button_b", "B"),
        ("button_x", "X"),
        ("button_y", "Y"),
        ("button_lb", "LB"),
        ("button_rb", "RB"),
    ]:
        entry = results.get(key)
        if entry:
            number, _ = entry
            lines.append(f"  {label}: button[{number}]")
    lines.append("")
    lines.append("代码中的默认约定（供对照）:")
    lines.append("  A=0, B=1, X=2, Y=3, LB=4, RB=5, LStick=9, RStick=10")
    lines.append("  axes: LX=0, LY=1, LT=2, RX=3, RY=4, RT=5, DPadX=6, DPadY=7")
    lines.append("如果实测和默认不一致，请把差异告诉我，我改代码。")
    return "\n".join(lines)


def main():
    device = os.environ.get("JS_DEVICE", "/dev/input/js0")
    print(f"打开手柄设备: {device}")
    fd = open_device(device)
    flush_pending(fd)

    results = {}
    print("")
    print("接下来按提示做动作。摇杆推到底、按钮按到底再松开。")
    print("轴事件的阈值是全量程一半左右；如果 15 秒内没检测到，脚本会跳到下一步。")
    print("")

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {prompt.description}")
        flush_pending(fd, 0.15)
        number, value = wait_for_event(fd, prompt.kind, timeout=15.0)
        if number is None:
            print("  ! 超时，跳过")
        else:
            if prompt.kind == "axis":
                print(f"  -> axis[{number}] = {value:+d}")
            else:
                print(f"  -> button[{number}] pressed")
            results[prompt.key] = (number, value)
        # 给操作者松手 / 回正的时间。摇杆题额外等其"物理回中"到静息阈值，
        # 按钮题只需要一个短倒计时。
        if prompt.kind == "axis":
            print("  松手让摇杆回中……")
            time.sleep(0.3)
            rested = wait_for_axes_to_center(fd)
            if not rested:
                print("  (等待摇杆回中超时，继续下一步)")
            countdown(1.5)
        else:
            countdown(1.0)

    os.close(fd)
    print(format_summary(results))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断。")
        sys.exit(1)
