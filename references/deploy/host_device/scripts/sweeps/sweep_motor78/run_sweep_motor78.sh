#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="${SWEEP_LOG_DIR:-$SCRIPT_DIR/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOY_LOG="$LOG_DIR/xbox_${STAMP}.log"
BRIDGE_LOG="$LOG_DIR/esd_link_bridge_${STAMP}.log"
SWEEP_LOG="$LOG_DIR/sweep_motor78_${STAMP}.csv"
JOY_DEV="${SWEEP_JOY_DEV:-/dev/input/js0}"
JOY_PID=""
BRIDGE_PID=""
TEST_FLOW=0

if [[ "${1:-}" == "--test-flow" ]]; then
    TEST_FLOW=1
    shift
fi

# shellcheck source=scripts/lib/gamepad.sh
source "$ROOT_DIR/scripts/lib/gamepad.sh"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        echo "停止本次翼扫频启动的 ESD-Link bridge..."
        kill -INT -- "-$BRIDGE_PID" 2>/dev/null || kill -INT "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi
    if [[ -n "$JOY_PID" ]] && kill -0 "$JOY_PID" 2>/dev/null; then
        echo "停止本次翼扫频启动的 Xbox 手柄链路..."
        kill -TERM -- "-$JOY_PID" 2>/dev/null || kill -TERM "$JOY_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if ! kill -0 "$JOY_PID" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if kill -0 "$JOY_PID" 2>/dev/null; then
            kill -KILL -- "-$JOY_PID" 2>/dev/null || kill -KILL "$JOY_PID" 2>/dev/null || true
        fi
        wait "$JOY_PID" 2>/dev/null || true
    fi
    exit "$status"
}

joy_message_ready() {
    local timeout_s=${1:-1}
    timeout "${timeout_s}s" ros2 topic echo \
        /joy sensor_msgs/msg/Joy --once --qos-profile sensor_data \
        >/dev/null 2>&1
}

joy_publisher_count() {
    local joy_info
    if ! joy_info="$(ros2 topic info /joy 2>/dev/null)"; then
        echo 0
        return 0
    fi
    awk '/^Publisher count:/ {print $3; found=1} END {if (!found) print 0}' <<<"$joy_info"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +u
source /opt/ros/humble/setup.bash
source "$ROOT_DIR/install/setup.bash"
set -u

mkdir -p "$LOG_DIR"

if (( TEST_FLOW )) && { pgrep -f '[/]esd_link_bridge_node([[:space:]]|$)' >/dev/null \
        || ros2 node list 2>/dev/null | grep -Eq '(^|/)esd_link_bridge_node$'; }; then
    echo "测试流程要求重启 bridge 以确保使用刚构建的新二进制。" >&2
    echo "请先在原终端 Ctrl-C 停止现有 bridge，再重新运行本脚本。" >&2
    exit 1
fi

if ! ros2 node list 2>/dev/null | grep -Eq '(^|/)esd_link_bridge_node$'; then
    echo "启动 ESD-Link bridge（默认 DISABLED）..."
    setsid bash -lc "
        source /opt/ros/humble/setup.bash
        source '$ROOT_DIR/install/setup.bash'
        exec ros2 launch esd_link_bridge esd_link_bridge.launch.py
    " >"$BRIDGE_LOG" 2>&1 &
    BRIDGE_PID=$!
    for _ in $(seq 1 150); do
        if ros2 node list 2>/dev/null | grep -Eq '(^|/)esd_link_bridge_node$'; then
            break
        fi
        sleep 0.1
    done
    if ! ros2 node list 2>/dev/null | grep -Eq '(^|/)esd_link_bridge_node$'; then
        echo "ESD-Link bridge 启动失败：$BRIDGE_LOG" >&2
        tail -n 100 "$BRIDGE_LOG" 2>/dev/null || true
        exit 1
    fi
fi

existing_joy_publishers="$(joy_publisher_count)"
if [[ "$existing_joy_publishers" =~ ^[0-9]+$ ]] && (( existing_joy_publishers > 0 )); then
    if joy_message_ready 3; then
        echo "检测到已有可用 /joy，复用当前手柄链路。"
    else
        echo "检测到已有 /joy 发布者，但 3 秒内没有收到手柄数据。" >&2
        echo "请先清理失效的 joy_node，避免重复启动两个手柄节点。" >&2
        exit 1
    fi
else
    ensure_gamepad_device "$JOY_DEV" || {
        echo "可通过 SWEEP_JOY_DEV=/dev/input/jsX 指定设备。" >&2
        exit 1
    }

    echo "启动 Xbox 手柄链路: $JOY_DEV"
    setsid bash -lc "
        source /opt/ros/humble/setup.bash
        source '$ROOT_DIR/install/setup.bash'
        exec ros2 launch xbox xbox.launch.py joy_dev:='$JOY_DEV'
    " >"$JOY_LOG" 2>&1 &
    JOY_PID=$!

    echo "持续等待 /joy 数据（20s 超时）..."
    if ! joy_message_ready 20; then
        if ! kill -0 "$JOY_PID" 2>/dev/null; then
            echo "Xbox 手柄链路启动失败，日志：$JOY_LOG" >&2
        else
            echo "20 秒内没有收到 /joy 数据，停止翼扫频。" >&2
        fi
        echo "Xbox 日志：$JOY_LOG" >&2
        tail -n 80 "$JOY_LOG" 2>/dev/null || true
        exit 1
    fi
    echo "Xbox A 键已就绪。日志：$JOY_LOG"
fi

has_log_arg=0
for arg in "$@"; do
    if [[ "$arg" == "--log" || "$arg" == --log=* ]]; then
        has_log_arg=1
        break
    fi
done

echo "启动 P7/P8 镜像扫频；CSV 默认写入：$SWEEP_LOG"
if (( has_log_arg )); then
    ros2 run deploy_tools esd_wing_sweep "$@"
else
    ros2 run deploy_tools esd_wing_sweep --log "$SWEEP_LOG" "$@"
fi
