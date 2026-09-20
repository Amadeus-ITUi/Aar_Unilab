#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${READONLY_FOXGLOVE_LOG_DIR:-${ROOT_DIR}/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BRIDGE_SESSION="readonly_esd_link_session"
FOXGLOVE_SESSION="readonly_foxglove_session"
BRIDGE_LOG="${LOG_DIR}/${BRIDGE_SESSION}_${STAMP}.log"
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
    screen -S "$BRIDGE_SESSION" -X quit >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup INT TERM EXIT

stop_session() {
    screen -S "$1" -X quit >/dev/null 2>&1 || true
}

# Remove only stale sessions owned by this read-only launcher.
stop_session "$FOXGLOVE_SESSION"
stop_session "$BRIDGE_SESSION"

set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u
mkdir -p "$LOG_DIR"

for node_name in esd_link_bridge_node motors_node wing_motor_node lab_inference_node xbox_vel_publisher; do
    if pgrep -x "$node_name" >/dev/null 2>&1; then
        echo "检测到 $node_name 正在运行；为避免控制链路冲突，请先停止它。" >&2
        exit 1
    fi
done

if ! ros2 pkg prefix foxglove_bridge >/dev/null 2>&1; then
    echo "未安装 foxglove_bridge，请执行：sudo apt install ros-humble-foxglove-bridge" >&2
    exit 1
fi
if ! ros2 pkg executables esd_link_bridge 2>/dev/null | grep -q '^esd_link_bridge esd_link_bridge_node$'; then
    echo "未找到 esd_link_bridge，请先重新编译并 source 工作区。" >&2
    echo "命令：colcon build --packages-select esd_link_msgs esd_link_bridge --symlink-install" >&2
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

echo "=== READ-ONLY ESD-LINK + FOXGLOVE ==="
echo "bridge 仅建立冻结协议会话，保持 DISABLED；不启动 CAN、独立 IMU、推理或手柄节点。"

start_screen "$BRIDGE_SESSION" "$BRIDGE_LOG" "ros2 launch esd_link_bridge esd_link_bridge.launch.py"
if ! wait_for_node esd_link_bridge_node 15 "$BRIDGE_SESSION"; then
    echo "ESD-Link bridge 启动失败，日志：$BRIDGE_LOG" >&2
    tail -n 100 "$BRIDGE_LOG" 2>/dev/null || true
    exit 1
fi

# Keep the bridge read-only and forward only the topics needed for debugging.
start_screen "$FOXGLOVE_SESSION" "$FOXGLOVE_LOG" \
    "ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=${FOXGLOVE_PORT} address:=${FOXGLOVE_ADDRESS} topic_whitelist:=\"['^/lower/state$','^/lower/link_status$','^/IMU_data$','^/policy/joint_states$','^/policy/wing_angles$','^/motors/runtime_status$','^/wing/runtime_status$','^/rosout$','^/parameter_events$']\" capabilities:=\"[connectionGraph,assets]\""
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
echo "ROS topics: /lower/state, /lower/link_status 及全部兼容状态话题"
echo "日志："
echo "  ESD-Link:  $BRIDGE_LOG"
echo "  Foxglove:  $FOXGLOVE_LOG"
echo "按 Ctrl+C 停止全部只读节点。"

while screen -ls 2>/dev/null | grep -Eq "${BRIDGE_SESSION}|${FOXGLOVE_SESSION}"; do
    sleep 1
done

echo "只读监控会话已退出。"
