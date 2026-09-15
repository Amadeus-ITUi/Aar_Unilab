#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
用法: $0 {14-kp4-kd0p2|25-kp4-kd0p2|36-kp0-kd0p05|36-kp0-kd0p2|all}

仅补当前 ESD-Link 数据集中尚缺的四组 Kp/Kd；不改变基础 YAML、激励或 CSV 字段：
  14-kp4-kd0p2   P1/P4 位置 chirp，Kp/Kd=4/0.2
  25-kp4-kd0p2   P2/P5 位置 chirp，Kp/Kd=4/0.2（保持当前 ±0.15 rad）
  36-kp0-kd0p05  P3/P6 速度 chirp，Kp/Kd=0/0.05
  36-kp0-kd0p2   P3/P6 速度 chirp，Kp/Kd=0/0.2
  all             按上述顺序连续执行；每组仍需完成各自的手柄确认
EOF
}

run_one() {
    local selection="$1"
    local pair kp kd gain_tag
    case "$selection" in
        14-kp4-kd0p2)
            pair=14; kp=4.0; kd=0.2; gain_tag=kp4_kd0p2
            ;;
        25-kp4-kd0p2)
            pair=25; kp=4.0; kd=0.2; gain_tag=kp4_kd0p2
            ;;
        36-kp0-kd0p05)
            pair=36; kp=0.0; kd=0.05; gain_tag=kp0_kd0p05
            ;;
        36-kp0-kd0p2)
            pair=36; kp=0.0; kd=0.2; gain_tag=kp0_kd0p2
            ;;
        *)
            usage >&2
            return 2
            ;;
    esac

    echo
    echo "============================================================"
    echo "准备执行 $selection"
    echo "基础轨迹保持不变；被扫端口 Kp/Kd=$kp/$kd"
    echo "输出文件将根据电机组和实际 Kp/Kd 自动命名（$gain_tag）"
    echo "============================================================"
    "$SCRIPT_DIR/run_sweep_test_flow.sh" "$pair" \
        --sweep-kp "$kp" \
        --sweep-kd "$kd"
}

selection="${1:-}"
case "$selection" in
    all)
        for item in \
            14-kp4-kd0p2 \
            25-kp4-kd0p2 \
            36-kp0-kd0p05 \
            36-kp0-kd0p2
        do
            run_one "$item"
        done
        ;;
    14-kp4-kd0p2|25-kp4-kd0p2|36-kp0-kd0p05|36-kp0-kd0p2)
        run_one "$selection"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
