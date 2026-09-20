#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
BRIDGE_LOG="${LOG_DIR}/set_zeros_1_6_bridge_${STAMP}.log"
JOY_LOG="${LOG_DIR}/set_zeros_1_6_joy_${STAMP}.log"
BRIDGE_PID=""
JOY_PID=""
BUTTON_INDEX="${ZERO_BUTTON_INDEX:-0}" # Xbox A
BUTTON_NAME="${ZERO_BUTTON_NAME:-A}"
TIMEOUT_S="${ZERO_BUTTON_TIMEOUT:-120}"
JOY_DEV="${ZERO_JOY_DEV:-/dev/input/js0}"

# shellcheck source=scripts/lib/gamepad.sh
source "${ROOT_DIR}/scripts/lib/gamepad.sh"

cleanup() {
    local status=$?
    if [[ -n "$JOY_PID" ]] && kill -0 "$JOY_PID" 2>/dev/null; then
        kill -TERM -- "-$JOY_PID" 2>/dev/null || kill -TERM "$JOY_PID" 2>/dev/null || true
    fi
    if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -TERM -- "-$BRIDGE_PID" 2>/dev/null || kill -TERM "$BRIDGE_PID" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$BRIDGE_PID" 2>/dev/null || kill -KILL "$BRIDGE_PID" 2>/dev/null || true
    fi
    wait "$JOY_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u
mkdir -p "$LOG_DIR"

if pgrep -f '(^|/)(esd_link_bridge_node|motors_node|joy_node|xbox_vel_publisher)( |$)' >/dev/null 2>&1; then
    echo "检测到已有 bridge/motors/joy 节点，请先停止现有部署链路。"
    exit 1
fi

echo "ESD-Link 置零仅修改 P1~P6；P7/P8 不置零，但冻结协议要求 P1~P8 全部在线。"
echo "全过程保持 DISABLED，不发送使能；请先摆好 P1~P6 机械零点并确认硬件断电手段。"

ensure_gamepad_device "$JOY_DEV"

setsid bash -lc "
    source /opt/ros/humble/setup.bash
    source '${ROOT_DIR}/install/setup.bash'
    exec ros2 launch xbox xbox.launch.py joy_dev:='$JOY_DEV'
" >"$JOY_LOG" 2>&1 &
JOY_PID=$!

setsid bash -lc "
    source /opt/ros/humble/setup.bash
    source '${ROOT_DIR}/install/setup.bash'
    exec ros2 launch esd_link_bridge esd_link_bridge.launch.py
" >"$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

echo "正在建立 ESD-Link 只读会话并验证 P1~P8、IMU、DISABLED，请勿按键..."
bridge_ready=0
for second in $(seq 1 20); do
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        echo "esd_link_bridge_node 已退出，日志：$BRIDGE_LOG"
        tail -n 80 "$BRIDGE_LOG" 2>/dev/null || true
        exit 1
    fi
    if timeout 2s python3 - <<'PY'
import rclpy
from esd_link_msgs.msg import LowerState
from rclpy.qos import qos_profile_sensor_data

rclpy.init()
node = rclpy.create_node("wait_esd_zero_prerequisites")
ready = False

def state_callback(msg):
    global ready
    ready = (
        msg.control_state == 2
        and msg.imu_valid_mask & 0x07 == 0x07
        and msg.offline_port_mask == 0
        and msg.fault_flags == 0
        and list(msg.port_id) == list(range(1, 9))
        and all((value & 0x03) == 0x03 for value in msg.valid_mask)
    )

node.create_subscription(LowerState, "/lower/state", state_callback, qos_profile_sensor_data)
try:
    while rclpy.ok() and not ready:
        rclpy.spin_once(node, timeout_sec=0.05)
finally:
    node.destroy_node()
    rclpy.shutdown()
raise SystemExit(0 if ready else 1)
PY
    then
        bridge_ready=1
        break
    fi
    if (( second == 1 || second % 5 == 0 )); then
        echo "等待 ESD-Link 只读状态... ${second}/20s"
    fi
done
if (( bridge_ready == 0 )); then
    echo "20 秒内未获得完整 DISABLED 状态，日志：$BRIDGE_LOG"
    tail -n 80 "$BRIDGE_LOG" 2>/dev/null || true
    exit 1
fi

echo "ESD-Link 只读检查通过，P1~P8 均在线。"
echo "电机保持失能；请再次确认 P1~P6 已位于机械零点。"
echo "等待手柄 ${BUTTON_NAME} 键释放后再次按下（${TIMEOUT_S}s 超时）..."
if ! timeout "${TIMEOUT_S}s" python3 - "$BUTTON_INDEX" "$BUTTON_NAME" <<'PY'
import sys

import rclpy
from sensor_msgs.msg import Joy

button_index = int(sys.argv[1])
button_name = sys.argv[2]
rclpy.init()
node = rclpy.create_node("wait_for_set_zeros_1_6_button")
state = {"armed": False, "pressed": False}

def callback(msg: Joy) -> None:
    pressed = len(msg.buttons) > button_index and msg.buttons[button_index] == 1
    if not state["armed"]:
        if not pressed:
            state["armed"] = True
            node.get_logger().info(f"Detected {button_name} release; waiting for a new press.")
        return
    if pressed:
        state["pressed"] = True

node.create_subscription(Joy, "/joy", callback, 10)
try:
    while rclpy.ok() and not state["pressed"]:
        rclpy.spin_once(node, timeout_sec=0.1)
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
sys.exit(0 if state["pressed"] else 1)
PY
then
    echo "未检测到手柄 ${BUTTON_NAME} 键，取消设置零点。"
    exit 1
fi

echo "检测到 ${BUTTON_NAME}，通过冻结 ESD-Link 协议设置 P1~P6 软件零点..."
output="$(timeout 20s ros2 service call /lower/set_zeros esd_link_msgs/srv/SetLowerZeros \
    '{persist: true, port_ids: [1, 2, 3, 4, 5, 6], assigned_position_rad: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}' 2>&1)" || {
    echo "$output"
    echo "设置零点调用失败，bridge 日志：$BRIDGE_LOG"
    exit 1
}
echo "$output"
if echo "$output" | grep -Eqi 'success: false|success=False'; then
    echo "1~6 号电机零点设置失败。"
    exit 1
fi

echo "1~6 号电机机械零点设置成功。"
echo "bridge 已接受新配置指纹并重建会话；当前状态："
timeout 5s ros2 topic echo --once /lower/link_status 2>/dev/null || true
echo "流程结束，正在关闭本脚本启动的节点。"
