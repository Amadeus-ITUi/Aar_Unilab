#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERVISOR="${DEPLOY_ROOT}/src/deploy_tools/scripts/control/we11_supervisor.py"
BUILD_REQUESTED=0
SUPERVISOR_ARGS=()

usage() {
    cat <<'EOF'
用法: ./start_robot.sh [--build] [--preflight-only] [--no-fault-latch]

选项:
  --build            启动前编译部署所需 ROS 2 包；默认不编译
  --preflight-only   只执行自动检查，电机保持失能
  --no-fault-latch   测试用：故障后返回非零状态，不锁存等待
  -h, --help         显示此帮助
EOF
}

for arg in "$@"; do
    case "$arg" in
        --build)
            BUILD_REQUESTED=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            SUPERVISOR_ARGS+=("$arg")
            ;;
    esac
done

if (( BUILD_REQUESTED )) && [[ "${WE11_AUTOSTART:-0}" == "1" ]]; then
    echo "开机自启模式禁止 --build；请在维护终端手动运行 ./start_robot.sh --build" >&2
    exit 2
fi

if [[ "${WE11_AUTOSTART:-0}" != "1" ]] && \
   systemctl is-active --quiet we11-deploy.service 2>/dev/null; then
    echo "we11-deploy.service 正在持有 RGB 待机灯；日常请使用手柄上+X 启动。" >&2
    echo "如需直接运行，请先执行 sudo systemctl stop we11-deploy.service。" >&2
    exit 2
fi

export WE11_DEPLOY_ROOT="$DEPLOY_ROOT"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
export FASTRTPS_DEFAULT_PROFILES_FILE="${DEPLOY_ROOT}/rt_fastdds_profile.xml"
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

if [[ ! -r /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble 环境不存在：/opt/ros/humble/setup.bash" >&2
    exit 1
fi
if [[ ! -r "$SUPERVISOR" ]]; then
    echo "WE11 supervisor 不存在：$SUPERVISOR" >&2
    exit 1
fi
if [[ ! -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]]; then
    echo "FastDDS 配置不存在：$FASTRTPS_DEFAULT_PROFILES_FILE" >&2
    exit 1
fi

set +u
source /opt/ros/humble/setup.bash
set -u

if (( BUILD_REQUESTED )); then
    if ! command -v colcon >/dev/null 2>&1; then
        echo "未找到 colcon，无法执行 --build" >&2
        exit 1
    fi
    echo "编译 ROS 2 workspace（--build）..."
    (
        cd "$DEPLOY_ROOT"
        timeout "${COLCON_BUILD_TIMEOUT:-300}" \
            colcon build \
                --paths src/hipnuc_imu src/motors src/inference src/xbox src/deploy_tools \
                --symlink-install \
                --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
    )
fi

if [[ ! -r "${DEPLOY_ROOT}/install/setup.bash" ]]; then
    echo "工作区尚未构建：${DEPLOY_ROOT}/install/setup.bash；请先使用 ./start_robot.sh --build" >&2
    exit 1
fi

set +u
source "${DEPLOY_ROOT}/install/setup.bash"
set -u

exec /usr/bin/python3 "$SUPERVISOR" "${SUPERVISOR_ARGS[@]}"
