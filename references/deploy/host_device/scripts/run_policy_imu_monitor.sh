#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export RMW_FASTRTPS_USE_QOS_FROM_XML="${RMW_FASTRTPS_USE_QOS_FROM_XML:-1}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT_DIR}/rt_fastdds_profile.xml}"

for process_pattern in \
    '[/]esd_link_bridge_node([[:space:]]|$)' \
    '[/]lab_inference_node([[:space:]]|$)' \
    '[/]motors_node([[:space:]]|$)' \
    '[/]wing_motor_node([[:space:]]|$)'; do
    if pgrep -f "$process_pattern" >/dev/null 2>&1; then
        echo "检测到已有部署/控制进程；请先停止，避免与只读 IMU 检查并行。" >&2
        exit 1
    fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BRIDGE_LOG="${ROOT_DIR}/logs/esd_policy_imu_monitor_bridge_${STAMP}.log"
BRIDGE_PID=""

cleanup() {
    local status=$?
    trap - INT TERM EXIT
    if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" >/dev/null 2>&1; then
        kill -INT -- "-${BRIDGE_PID}" >/dev/null 2>&1 || true
        for _ in $(seq 1 30); do
            kill -0 "$BRIDGE_PID" >/dev/null 2>&1 || break
            sleep 0.1
        done
        if kill -0 "$BRIDGE_PID" >/dev/null 2>&1; then
            kill -TERM -- "-${BRIDGE_PID}" >/dev/null 2>&1 || true
        fi
        wait "$BRIDGE_PID" >/dev/null 2>&1 || true
    fi
    echo "bridge 日志：${BRIDGE_LOG}"
    exit "$status"
}
trap cleanup INT TERM EXIT

mkdir -p "${ROOT_DIR}/logs"
setsid ros2 launch esd_link_bridge esd_link_bridge.launch.py \
    >"${BRIDGE_LOG}" 2>&1 &
BRIDGE_PID=$!

ros2 run deploy_tools esd_policy_imu_monitor
