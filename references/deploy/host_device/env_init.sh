#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${YELLOW}$1${NC}"; }
ok() { echo -e "${GREEN}$1${NC}"; }
err() { echo -e "${RED}$1${NC}" >&2; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
ARCH="$(uname -m)"

if [[ "$ARCH" == "x86_64" ]]; then
    MNN_PLATFORM="mnn-linux-x64"
    CONVERTER_PLATFORM="x64"
elif [[ "$ARCH" == "aarch64" ]]; then
    MNN_PLATFORM="mnn-linux-aarch64"
    CONVERTER_PLATFORM=""
else
    err "不支持的 CPU 架构: $ARCH"
    exit 1
fi

run_sudo() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

source_setup_file() {
    local setup_file=$1
    set +u
    # shellcheck disable=SC1090
    source "$setup_file"
    set -u
}

install_ros_apt_source() {
    if [[ -f "$ROS_SETUP" ]]; then
        return
    fi

    info "未检测到 $ROS_SETUP，配置 ROS 2 apt 源..."
    run_sudo apt-get update
    run_sudo apt-get install -y curl gnupg lsb-release ca-certificates software-properties-common
    run_sudo add-apt-repository -y universe
    run_sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
        | run_sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
}

install_apt_deps() {
    install_ros_apt_source

    info "安装系统/ROS2 依赖..."
    run_sudo apt-get update
    run_sudo apt-get install -y \
        build-essential cmake git curl screen can-utils iproute2 usbutils ccache \
        python3-pip python3-rosdep python3-vcstool \
        python3-colcon-common-extensions python3-colcon-cmake \
        python3-catkin-pkg python3-ament-package \
        libeigen3-dev libboost-system-dev libspdlog-dev libfmt-dev \
        "ros-${ROS_DISTRO}-ros-base" \
        "ros-${ROS_DISTRO}-ament-cmake-python" \
        "ros-${ROS_DISTRO}-rmw-fastrtps-cpp" \
        "ros-${ROS_DISTRO}-joy" \
        "ros-${ROS_DISTRO}-launch-ros" \
        "ros-${ROS_DISTRO}-sensor-msgs" \
        "ros-${ROS_DISTRO}-geometry-msgs" \
        "ros-${ROS_DISTRO}-std-msgs" \
        "ros-${ROS_DISTRO}-std-srvs" \
        "ros-${ROS_DISTRO}-rosidl-default-generators" \
        "ros-${ROS_DISTRO}-rosidl-default-runtime" \
        "ros-${ROS_DISTRO}-rosidl-typesupport-fastrtps-c" \
        "ros-${ROS_DISTRO}-rosidl-typesupport-fastrtps-cpp" \
        "ros-${ROS_DISTRO}-ament-index-cpp"

    if ! rosdep --version >/dev/null 2>&1; then
        err "rosdep 安装异常"
        exit 1
    fi
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
        run_sudo rosdep init || true
    fi
    rosdep update || true
}

prepare_shell_env() {
    if [[ ! -f "$ROS_SETUP" ]]; then
        err "未找到 ROS 环境: $ROS_SETUP"
        exit 1
    fi

    # 避免 conda 的 python3 抢走 ament/colcon 使用的系统 Python 包。
    export PATH="/opt/ros/${ROS_DISTRO}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    unset PYTHONHOME || true
    unset PYTHONPATH || true
    source_setup_file "$ROS_SETUP"

    /usr/bin/python3 -c "import colcon_cmake, catkin_pkg" >/dev/null
}

ensure_mnn_runtime() {
    local mnn_root="src/inference/thirdparty/${MNN_PLATFORM}"
    local mnn_lib="${mnn_root}/lib/libMNN.so"
    if [[ -f "$mnn_lib" ]]; then
        ok "MNN runtime 已存在: $mnn_lib"
        return
    fi

    info "缺少 $mnn_lib，下载并编译 MNN runtime..."
    local work_dir="${TMPDIR:-/tmp}/MNN-${ARCH}-runtime"
    rm -rf "$work_dir"
    git clone --depth 1 https://github.com/alibaba/MNN.git "$work_dir"
    cmake -S "$work_dir" -B "$work_dir/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_SHARED_LIBS=ON \
        -DMNN_BUILD_TEST=OFF \
        -DMNN_BUILD_BENCHMARK=OFF \
        -DMNN_BUILD_DEMO=OFF \
        -DMNN_BUILD_CONVERTER=OFF \
        -DMNN_BUILD_TOOLS=OFF
    cmake --build "$work_dir/build" --target MNN -j"$(nproc)"

    mkdir -p "$(dirname "$mnn_lib")"
    cp "$work_dir/build/libMNN.so" "$mnn_lib"
    if [[ ! -f "${mnn_root}/include/MNN/Interpreter.hpp" ]]; then
        mkdir -p "${mnn_root}/include"
        cp -a "$work_dir/include/MNN" "${mnn_root}/include/"
    fi
    ok "已安装 MNN runtime: $mnn_lib"
}

ensure_x64_converter_hint() {
    if [[ "$ARCH" != "x86_64" ]]; then
        return
    fi
    local converter="src/inference/thirdparty/MNNConverter/${CONVERTER_PLATFORM}/MNNConvert.sh"
    if [[ -x "$converter" ]]; then
        ok "MNNConvert 已存在: $converter"
    else
        info "未发现 x64 MNNConvert；部署运行不依赖转换器，只影响本机 ONNX->MNN 转换。"
    fi
}

build_workspace() {
    info "安装 ROS 包依赖..."
    rosdep install --from-paths src --ignore-src -r -y || true

    info "编译 ROS2 workspace..."
    timeout "${COLCON_BUILD_TIMEOUT:-600}" \
        colcon build --paths src/motors src/esd_link_msgs src/esd_link_bridge src/inference src/xbox src/deploy_tools \
            --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
    source_setup_file install/setup.sh

    local required=(
        "install/esd_link_bridge/lib/esd_link_bridge/esd_link_bridge_node"
        "install/esd_link_bridge/lib/esd_link_bridge/esd_link_emergency_disable"
        "install/inference/lib/inference/lab_inference_node"
        "install/inference/share/inference/launch/lab_inference_node.launch.py"
        "install/inference/share/inference/config/lab_inference.yaml"
        "install/inference/share/inference/models/lab_policy.mnn"
        "install/xbox/lib/xbox/xbox_vel_publisher"
        "install/deploy_tools/lib/deploy_tools/read_motor_status"
    )
    for path in "${required[@]}"; do
        if [[ ! -e "$path" ]]; then
            err "缺少构建/安装产物: $path"
            exit 1
        fi
    done
}

main() {
    info "Walking_Eagle-Deploy 环境初始化"
    info "仓库: $REPO_DIR"
    info "架构: $ARCH"
    info "ROS_DISTRO: $ROS_DISTRO"

    install_apt_deps
    prepare_shell_env
    ensure_mnn_runtime
    ensure_x64_converter_hint
    build_workspace

    ok "环境初始化完成。启动实机链路："
    echo "cd $REPO_DIR"
    echo "./start_robot.sh"
}

main "$@"
