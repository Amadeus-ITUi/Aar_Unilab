#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${READONLY_FOXGLOVE_LOG_DIR:-${ROOT_DIR}/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
IMU_SESSION="readonly_imu_session"
MOTOR_SESSION="readonly_motor_state_session"
FOXGLOVE_SESSION="readonly_foxglove_session"
IMU_LOG="${LOG_DIR}/${IMU_SESSION}_${STAMP}.log"
MOTOR_LOG="${LOG_DIR}/${MOTOR_SESSION}_${STAMP}.log"
FOXGLOVE_LOG="${LOG_DIR}/${FOXGLOVE_SESSION}_${STAMP}.log"
FOXGLOVE_PORT="${READONLY_FOXGLOVE_PORT:-8765}"
FOXGLOVE_ADDRESS="${READONLY_FOXGLOVE_ADDRESS:-0.0.0.0}"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export RMW_FASTRTPS_USE_QOS_FROM_XML="${RMW_FASTRTPS_USE_QOS_FROM_XML:-1}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT_DIR}/rt_fastdds_profile.xml}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

cleanup() {
    local status=$?
    trap - INT TERM EXIT
    screen -S "$FOXGLOVE_SESSION" -X quit >/dev/null 2>&1 || true
    screen -S "$MOTOR_SESSION" -X quit >/dev/null 2>&1 || true
    screen -S "$IMU_SESSION" -X quit >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup INT TERM EXIT

stop_session() {
    screen -S "$1" -X quit >/dev/null 2>&1 || true
}

# Remove only stale sessions owned by this read-only launcher.
stop_session "$FOXGLOVE_SESSION"
stop_session "$MOTOR_SESSION"
stop_session "$IMU_SESSION"

set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u
mkdir -p "$LOG_DIR"

if ! ip link show can0 >/dev/null 2>&1; then
    echo "can0 不存在，停止只读监控。" >&2
    exit 1
fi
if ! ip -details link show can0 2>/dev/null | head -n 1 | grep -q 'UP'; then
    echo "can0 未处于 UP 状态，停止只读监控。" >&2
    ip -details link show can0 2>/dev/null || true
    exit 1
fi

for node_name in motors_node wing_motor_node lab_inference_node xbox_vel_publisher; do
    if pgrep -x "$node_name" >/dev/null 2>&1; then
        echo "检测到 $node_name 正在运行；为避免控制链路冲突，请先停止它。" >&2
        exit 1
    fi
done

if ! ros2 pkg prefix foxglove_bridge >/dev/null 2>&1; then
    echo "未安装 foxglove_bridge，请执行：sudo apt install ros-humble-foxglove-bridge" >&2
    exit 1
fi
if ! ros2 pkg executables deploy_tools 2>/dev/null | grep -q '^deploy_tools read_only_motor_state_node$'; then
    echo "未找到 deploy_tools/read_only_motor_state_node，请先重新编译并 source 工作区。" >&2
    echo "命令：colcon build --packages-select deploy_tools --symlink-install" >&2
    exit 1
fi

start_screen() {
    local session="$1"
    local log_file="$2"
    local command_line="$3"
    screen -dmS "$session" bash -lc \
        "exec >'$log_file' 2>&1; export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'; export RMW_FASTRTPS_USE_QOS_FROM_XML='$RMW_FASTRTPS_USE_QOS_FROM_XML'; export FASTRTPS_DEFAULT_PROFILES_FILE='$FASTRTPS_DEFAULT_PROFILES_FILE'; export ROS_LOCALHOST_ONLY='$ROS_LOCALHOST_ONLY'; export ROS_DOMAIN_ID='$ROS_DOMAIN_ID'; exec $command_line"
}

wait_for_node() {
    local node_name="$1"
    local timeout_s="${2:-15}"
    local session_name="${3:-}"
    local deadline=$((SECONDS + timeout_s))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if ros2 node list 2>/dev/null | grep -Eq "(^|/)$node_name$"; then
            if [ -z "$session_name" ] || screen -S "$session_name" -Q select . >/dev/null 2>&1; then
                return 0
            fi
        fi
        sleep 1
    done
    return 1
}

echo "=== READ-ONLY IMU + MOTOR STATE + FOXGLOVE ==="
echo "不会启动 motors_node、wing_motor_node、lab_inference_node 或 Xbox 节点。"
echo "只读电机节点只发送 CAN 0x02 状态请求，不发送 0x01/0x03/0x04/0x06。"

start_screen "$IMU_SESSION" "$IMU_LOG" "ros2 launch hipnuc_imu imu_node.launch.py"
if ! wait_for_node imu_node 15 "$IMU_SESSION"; then
    echo "IMU 节点启动失败，日志：$IMU_LOG" >&2
    tail -n 80 "$IMU_LOG" 2>/dev/null || true
    exit 1
fi

start_screen "$MOTOR_SESSION" "$MOTOR_LOG" \
    "ros2 run deploy_tools read_only_motor_state_node --channel can0 --motor-ids 1,2,3,4,5,6 --request-rate 20 --publish-rate 50 --default-angles -0.92020,0.98338,0.0,-0.92020,0.98338,0.0"
if ! wait_for_node readonly_motor_state_node 15 "$MOTOR_SESSION"; then
    echo "只读电机状态节点启动失败，日志：$MOTOR_LOG" >&2
    tail -n 100 "$MOTOR_LOG" 2>/dev/null || true
    exit 1
fi

# Keep the bridge read-only and forward only the topics needed for debugging.
start_screen "$FOXGLOVE_SESSION" "$FOXGLOVE_LOG" \
    "ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=${FOXGLOVE_PORT} address:=${FOXGLOVE_ADDRESS} topic_whitelist:=\"['^/IMU_data$','^/policy/joint_states$','^/rosout$','^/parameter_events$']\" capabilities:=\"[connectionGraph,assets]\""
if ! wait_for_node foxglove_bridge 15 "$FOXGLOVE_SESSION"; then
    echo "Foxglove Bridge 启动失败，日志：$FOXGLOVE_LOG" >&2
    tail -n 100 "$FOXGLOVE_LOG" 2>/dev/null || true
    exit 1
fi

echo
echo "只读监控链路已启动。"
echo "Foxglove: ws://<本机IP>:${FOXGLOVE_PORT}"
echo "本机 IP："
ip -4 -br addr show wlan0 eth0 2>/dev/null || ip -4 -br addr show
echo "ROS topics: /IMU_data, /policy/joint_states"
echo "日志："
echo "  IMU:       $IMU_LOG"
echo "  Motor:     $MOTOR_LOG"
echo "  Foxglove:  $FOXGLOVE_LOG"
echo "按 Ctrl+C 停止全部只读节点。"

while screen -ls 2>/dev/null | grep -Eq "${IMU_SESSION}|${MOTOR_SESSION}|${FOXGLOVE_SESSION}"; do
    sleep 1
done

echo "只读监控会话已退出。"
