#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${SWEEP_LAUNCH_LOG_DIR:-$SCRIPT_DIR/logs/launcher}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BRIDGE_LOG="$LOG_DIR/esd_link_bridge_${STAMP}.log"
BRIDGE_PID=""
DEPLOYMENT=""
READ_ONLY=0

args=("$@")
for ((index=0; index<${#args[@]}; index++)); do
    case "${args[index]}" in
        --deployment)
            if (( index + 1 >= ${#args[@]} )); then
                echo "--deployment 需要 robot_id/contract_id" >&2
                exit 2
            fi
            DEPLOYMENT="${args[index+1]}"
            ((index+=1))
            ;;
        --list|--validate-only|--resolve-robot-profile)
            READ_ONLY=1
            ;;
    esac
done

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -INT -- "-$BRIDGE_PID" 2>/dev/null || kill -INT "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi
    exit "$status"
}

if ! [[ -r /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble 环境不存在：/opt/ros/humble/setup.bash" >&2
    exit 1
fi
if ! [[ -r "$ROOT_DIR/install/setup.bash" ]]; then
    echo "工作区尚未构建：$ROOT_DIR/install/setup.bash" >&2
    echo "请先构建 deploy_tools 和所选 robot package。" >&2
    exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source "$ROOT_DIR/install/setup.bash"
set -u

export WE11_DEPLOY_ROOT="$ROOT_DIR"
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

if (( READ_ONLY )); then
    exec ros2 run deploy_tools esd_frequency_sweep --root "$ROOT_DIR" "$@"
fi

# Validate every selected parameter before starting bridge.
ros2 run deploy_tools esd_frequency_sweep \
    --root "$ROOT_DIR" "$@" --validate-only >/dev/null

resolver=(ros2 run deploy_tools esd_frequency_sweep --root "$ROOT_DIR" --resolve-robot-profile)
if [[ -n "$DEPLOYMENT" ]]; then
    resolver+=(--deployment "$DEPLOYMENT")
fi
ROBOT_PROFILE="$("${resolver[@]}")"
if ! [[ -f "$ROBOT_PROFILE" ]]; then
    echo "无法解析已安装 Robot Profile：$ROBOT_PROFILE" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! ros2 node list 2>/dev/null | grep -Eq '(^|/)esd_link_bridge_node$'; then
    echo "启动 ESD-Link bridge：$ROBOT_PROFILE"
    setsid bash -lc "
        source /opt/ros/humble/setup.bash
        source '$ROOT_DIR/install/setup.bash'
        exec ros2 launch esd_link_bridge esd_link_bridge.launch.py \
            robot_profile_path:='$ROBOT_PROFILE'
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
else
    echo "检测到已有 ESD-Link bridge；扫频节点将核验其 Profile 和 fingerprint。"
fi

ros2 run deploy_tools esd_frequency_sweep --root "$ROOT_DIR" "$@"
