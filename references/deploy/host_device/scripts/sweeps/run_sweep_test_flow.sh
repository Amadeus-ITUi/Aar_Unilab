#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    14|25|36|78)
        pair="$1"
        shift
        ;;
    *)
        echo "用法: $0 {14|25|36|78}" >&2
        echo "14/25/36: 当前位接管 -> 正式姿态 -> 8 秒低幅扫频 -> A 键确认 -> 正式扫频 -> DISABLED" >&2
        echo "78: 自动启动 bridge/手柄 -> 三次 A 键确认 -> P7/P8 镜像扫频 -> DISABLED" >&2
        exit 2
        ;;
esac

exec "$SCRIPT_DIR/sweep_motor${pair}/run_sweep_motor${pair}.sh" --test-flow "$@"
