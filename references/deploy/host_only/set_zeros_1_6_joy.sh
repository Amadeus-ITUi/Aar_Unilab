#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MOTOR_LOG="${LOG_DIR}/set_zeros_1_6_motors_${STAMP}.log"
JOY_LOG="${LOG_DIR}/set_zeros_1_6_joy_${STAMP}.log"
MOTOR_PID=""
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
    if [[ -n "$MOTOR_PID" ]] && kill -0 "$MOTOR_PID" 2>/dev/null; then
        kill -TERM -- "-$MOTOR_PID" 2>/dev/null || kill -TERM "$MOTOR_PID" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$MOTOR_PID" 2>/dev/null || kill -KILL "$MOTOR_PID" 2>/dev/null || true
    fi
    wait "$JOY_PID" 2>/dev/null || true
    wait "$MOTOR_PID" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u
mkdir -p "$LOG_DIR"

if pgrep -f '(^|/)(motors_node|joy_node|xbox_vel_publisher)( |$)' >/dev/null 2>&1; then
    echo "检测到已有 motors_node/joy 节点，请先停止现有部署链路。"
    exit 1
fi

echo "只启动 1~6 号电机零点设置流程；不会启动或设置 7/8 号电机。"
echo "请先将 1~6 号电机全部摆到机械零点，并确认急停可用。"

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
    exec ros2 run motors motors_node --ros-args --params-file '${ROOT_DIR}/src/motors/config/motors.yaml'
" >"$MOTOR_LOG" 2>&1 &
MOTOR_PID=$!

echo "正在启动 1~6 号电机并等待 CAN 初始化完成，请勿按键..."
motor_ready=0
for second in $(seq 1 60); do
    if ! kill -0 "$MOTOR_PID" 2>/dev/null; then
        echo "motors_node 已退出，日志：$MOTOR_LOG"
        tail -n 80 "$MOTOR_LOG" 2>/dev/null || true
        exit 1
    fi
    if grep -q '自动初始化流程完成' "$MOTOR_LOG" 2>/dev/null; then
        motor_ready=1
        break
    fi
    if (( second == 1 || second % 5 == 0 )); then
        echo "等待 motors_node 初始化... ${second}/60s"
    fi
    sleep 1
done
if (( motor_ready == 0 )); then
    echo "motors_node 在 60 秒内未完成初始化，日志：$MOTOR_LOG"
    tail -n 80 "$MOTOR_LOG" 2>/dev/null || true
    exit 1
fi

echo "1~6 号电机节点已启动。"
echo "电机不会自动移动；请手动将 1~6 号电机摆到机械零点。"
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

echo "检测到 ${BUTTON_NAME}，同步设置 1~6 号电机机械零点..."
output="$(timeout 60s ros2 service call /set_zeros motors/srv/SetZeros '{}' 2>&1)" || {
    echo "$output"
    echo "设置零点调用失败，电机日志：$MOTOR_LOG"
    exit 1
}
echo "$output"
if echo "$output" | grep -Eqi 'success: false|success=False'; then
    echo "1~6 号电机零点设置失败。"
    exit 1
fi

echo "1~6 号电机机械零点设置成功。"
echo "当前状态："
timeout 5s ros2 topic echo --once /policy/joint_states 2>/dev/null || true
echo "流程结束，正在关闭本脚本启动的节点。"
