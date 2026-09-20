#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="/home/esd/Pheonix/Deploy"
HARDWARE_WAIT_SECONDS="${WE11_BOOT_HARDWARE_WAIT_SECONDS:-60}"

if [[ ! -r "${DEPLOY_ROOT}/install/setup.bash" ]]; then
    echo "WE11 工作区尚未构建；开机自启禁止编译，请在维护终端运行 ./start_robot.sh --build" >&2
    exit 1
fi

deadline=$((SECONDS + HARDWARE_WAIT_SECONDS))
while (( SECONDS < deadline )); do
    can_ready=0
    imu_ready=0

    if [[ -e /sys/class/net/can0 ]] &&
       /usr/bin/ip -brief link show can0 2>/dev/null | /usr/bin/grep -qw 'UP'; then
        can_ready=1
    fi
    [[ -r /dev/ttyUSB0 && -w /dev/ttyUSB0 ]] && imu_ready=1

    if (( can_ready && imu_ready )); then
        echo "WE11 基础设备已就绪：can0、/dev/ttyUSB0"
        break
    fi

    echo "等待 WE11 基础设备：can0=${can_ready} imu=${imu_ready}"
    /usr/bin/sleep 1
done

if ! (( can_ready && imu_ready )); then
    echo "等待 WE11 基础设备超时（${HARDWARE_WAIT_SECONDS}s）：需要 can0 UP、可读写 /dev/ttyUSB0" >&2
    exit 1
fi

echo "等待 Xbox 手柄 /dev/input/js0；允许稍后连接，不使用键盘或鼠标..."
joy_wait_seconds=0
until [[ -r /dev/input/js0 ]]; do
    if (( joy_wait_seconds > 0 && joy_wait_seconds % 10 == 0 )); then
        echo "仍在等待 Xbox 手柄 /dev/input/js0（${joy_wait_seconds}s）"
    fi
    /usr/bin/sleep 1
    ((joy_wait_seconds += 1))
done

echo "WE11 开机设备已就绪：can0、/dev/ttyUSB0、/dev/input/js0"
