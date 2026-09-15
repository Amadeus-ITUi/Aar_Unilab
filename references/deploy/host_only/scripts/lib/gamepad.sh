#!/usr/bin/env bash

# Ensure that an Xbox-compatible controller has a readable Linux joystick node.
# Usage: ensure_gamepad_device [/dev/input/js0]
ensure_gamepad_device() {
    local joy_dev="${1:-/dev/input/js0}"

    if [[ ! -e "$joy_dev" ]]; then
        echo "未发现手柄设备 $joy_dev，尝试加载 Xbox 兼容驱动 xpad..."
        if ! command -v modprobe >/dev/null 2>&1 || ! modinfo xpad >/dev/null 2>&1; then
            echo "错误：系统未安装 xpad 内核驱动；USB 手柄无法生成 $joy_dev。" >&2
            return 1
        fi

        if modprobe xpad 2>/dev/null; then
            :
        elif command -v sudo >/dev/null 2>&1 && sudo -n modprobe xpad 2>/dev/null; then
            :
        elif [[ -t 0 ]] && command -v sudo >/dev/null 2>&1; then
            echo "加载 xpad 需要管理员权限。"
            sudo modprobe xpad || return 1
        else
            echo "错误：无法自动加载 xpad。请先执行：sudo modprobe xpad" >&2
            return 1
        fi

        command -v udevadm >/dev/null 2>&1 && udevadm settle --timeout=3 2>/dev/null || true
        local attempt
        for attempt in {1..20}; do
            [[ -e "$joy_dev" ]] && break
            sleep 0.1
        done
    fi

    if [[ ! -e "$joy_dev" ]]; then
        echo "错误：加载 xpad 后仍未出现 $joy_dev。请重新插拔手柄，或指定正确的 js 设备。" >&2
        ls -l /dev/input/js* 2>/dev/null || true
        return 1
    fi
    if [[ ! -r "$joy_dev" ]]; then
        echo "错误：当前用户无权读取 $joy_dev。" >&2
        echo "请执行 sudo usermod -aG input \"$USER\"，然后注销并重新登录。" >&2
        return 1
    fi

    echo "手柄设备已就绪：$joy_dev"
}
