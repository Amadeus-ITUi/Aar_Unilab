#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export RMW_FASTRTPS_USE_QOS_FROM_XML="${RMW_FASTRTPS_USE_QOS_FROM_XML:-1}"
if [[ -f "$SCRIPT_DIR/rt_fastdds_profile.xml" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$SCRIPT_DIR/rt_fastdds_profile.xml}"
fi

if [[ ! -f "$SCRIPT_DIR/install/setup.bash" ]]; then
    echo "[ERROR] 未找到 install/setup.bash，请先在仓库根目录完成 colcon build。" >&2
    exit 1
fi

# Do not enable nounset here; ROS setup files may read unset variables.
source "$SCRIPT_DIR/install/setup.bash"

if ! ros2 pkg executables deploy_tools 2>/dev/null | grep -q '^deploy_tools record_deploy_runtime$'; then
    echo "[ERROR] 未找到 deploy_tools/record_deploy_runtime，请先重新编译工作区。" >&2
    exit 1
fi

export WALKING_EAGLE_DEPLOY_ROOT="${WALKING_EAGLE_DEPLOY_ROOT:-$SCRIPT_DIR}"
exec ros2 run deploy_tools record_deploy_runtime "$@"
