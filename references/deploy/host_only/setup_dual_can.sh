#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${YELLOW}$1${NC}"; }
ok() { echo -e "${GREEN}$1${NC}"; }
err() { echo -e "${RED}$1${NC}" >&2; }

# Legacy filename kept for start_robot.sh compatibility.
# Current deploy uses one CAN interface:
#   native gs_usb can0 if the adapter exposes SocketCAN directly, otherwise
#   slcand creates can0 from CANable's CDC ACM device (/dev/ttyACM0).
CAN_IFACE="${CAN_IFACE:-can0}"
BITRATE="${CAN_BITRATE:-1000000}"
TXQUEUELEN="${CAN_TXQUEUELEN:-1000}"
SLCAN_TTY="${SLCAN_TTY:-/dev/ttyACM0}"
SLCAN_SPEED="${SLCAN_SPEED:-s8}"  # slcand speed code: s8 = 1 Mbps, s6 = 500 kbps

run_sudo() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

require_cmd() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        err "找不到命令: $cmd"
        return 1
    fi
}

iface_exists() {
    ip link show "$1" >/dev/null 2>&1
}

iface_up() {
    ip link show "$1" 2>/dev/null | grep -q "<[^>]*UP"
}

configure_native_can() {
    local iface=$1
    info "配置 CAN 接口 ${iface}，bitrate=${BITRATE}"
    run_sudo ip link set "$iface" down || true
    if run_sudo ip link set "$iface" type can bitrate "$BITRATE" restart-ms 100; then
        ok "${iface} 已设置 bitrate=${BITRATE}"
    else
        info "${iface} 不支持通过 ip link 设置 bitrate；如果它由 slcand 创建，bitrate 已由 slcand ${SLCAN_SPEED} 设置。"
    fi
    run_sudo ip link set "$iface" txqueuelen "$TXQUEUELEN" || true
    run_sudo ip link set "$iface" up
}

create_slcan_iface() {
    local tty=$1
    local iface=$2

    require_cmd slcand || {
        err "请先安装 can-utils：sudo apt install can-utils"
        exit 1
    }
    if [ ! -e "$tty" ]; then
        err "找不到 CANable 串口设备 $tty"
        err "当前 ACM/USB 串口："
        ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
        exit 1
    fi

    info "通过 slcand 创建 ${iface}: ${tty} -> ${iface}, speed=${SLCAN_SPEED}"
    if iface_exists "$iface"; then
        run_sudo ip link set "$iface" down || true
        run_sudo ip link delete "$iface" || true
    fi
    run_sudo pkill slcand || true
    run_sudo slcand -o -c "-${SLCAN_SPEED}" "$tty" "$iface"

    for _ in $(seq 1 20); do
        iface_exists "$iface" && return 0
        sleep 0.1
    done
    err "slcand 未能创建 ${iface}"
    exit 1
}

require_cmd ip

info "加载 CAN 内核模块..."
run_sudo modprobe can || true
run_sudo modprobe can_raw || true
run_sudo modprobe gs_usb || true
run_sudo modprobe slcan || true

if iface_exists "$CAN_IFACE"; then
    configure_native_can "$CAN_IFACE"
else
    info "${CAN_IFACE} 尚不存在；当前 CANable2 枚举为 CDC ACM 时应使用 ${SLCAN_TTY}"
    create_slcan_iface "$SLCAN_TTY" "$CAN_IFACE"
    run_sudo ip link set "$CAN_IFACE" txqueuelen "$TXQUEUELEN" || true
    run_sudo ip link set "$CAN_IFACE" up
fi

if ! iface_up "$CAN_IFACE"; then
    err "${CAN_IFACE} 拉起失败"
    ip -details link show "$CAN_IFACE" 2>/dev/null || true
    exit 1
fi

ok "${CAN_IFACE} 已就绪"
ip -details -statistics link show "$CAN_IFACE" || true
