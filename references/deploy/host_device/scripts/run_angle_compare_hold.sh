#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"

if [ "$#" -ne 1 ]; then
    echo "用法：$0 {-0.1|0|+0.1}" >&2
    exit 2
fi
case "$TARGET" in
    -0.1) NORMALIZED_TARGET="-0.1" ;;
    0|0.0) NORMALIZED_TARGET="0" ;;
    +0.1|0.1) NORMALIZED_TARGET="+0.1" ;;
    *)
        echo "目标只能是 -0.1、0 或 +0.1 rad。" >&2
        exit 2
        ;;
esac

set +u
source /opt/ros/humble/setup.bash
source "${ROOT_DIR}/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export RMW_FASTRTPS_USE_QOS_FROM_XML="${RMW_FASTRTPS_USE_QOS_FROM_XML:-1}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT_DIR}/rt_fastdds_profile.xml}"

if pgrep -f '[/]esd_link_bridge_node([[:space:]]|$)' >/dev/null 2>&1; then
    echo "检测到 esd_link_bridge_node 已运行；请先停止，避免重复控制。" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BRIDGE_LOG="${ROOT_DIR}/logs/esd_angle_compare_bridge_${STAMP}.log"
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
    if [ -x "${ROOT_DIR}/install/esd_link_bridge/lib/esd_link_bridge/esd_link_emergency_disable" ]; then
        "${ROOT_DIR}/install/esd_link_bridge/lib/esd_link_bridge/esd_link_emergency_disable" || true
    fi
    echo "bridge 日志：${BRIDGE_LOG}"
    exit "$status"
}
trap cleanup INT TERM EXIT

mkdir -p "${ROOT_DIR}/logs"
setsid ros2 launch esd_link_bridge esd_link_bridge.launch.py \
    >"${BRIDGE_LOG}" 2>&1 &
BRIDGE_PID=$!

ros2 run deploy_tools esd_angle_compare_hold --policy-q "$NORMALIZED_TARGET"
