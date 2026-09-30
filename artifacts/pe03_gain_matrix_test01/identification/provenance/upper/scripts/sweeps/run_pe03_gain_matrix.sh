#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="${SWEEP_RUNNER:-$SCRIPT_DIR/run_sweep.sh}"
BATCH_STAMP="${SWEEP_BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$ROOT_DIR/scripts/sweeps/logs/pe03/gain_matrix/$BATCH_STAMP"
VALIDATE_ONLY=0

usage() {
    cat <<'EOF'
usage: ./scripts/sweeps/run_pe03_gain_matrix.sh [--validate-only] [--output-dir DIR]

Sequentially collects the PE03 A/B/C gain matrix: hip, thigh and calf for
each gain set (nine independent calls to run_sweep.sh). DIR may be absolute or
relative to the Deploy root.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --validate-only)
            VALIDATE_ONLY=1
            shift
            ;;
        --output-dir)
            if (( $# < 2 )); then
                echo "--output-dir 需要目录" >&2
                exit 2
            fi
            if [[ "$2" = /* ]]; then
                OUTPUT_DIR="$2"
            else
                OUTPUT_DIR="$ROOT_DIR/$2"
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数：$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$RUNNER" ]]; then
    echo "扫频入口不可执行：$RUNNER" >&2
    exit 1
fi

run_ids=(
    A_baseline_hip A_baseline_thigh A_baseline_calf
    B_position_hip B_position_thigh B_position_calf
    C_damping_hip C_damping_thigh C_damping_calf
)
groups=(hip thigh calf hip thigh calf hip thigh calf)
kps=(4.3 4.3 4.9 5.2 5.2 5.9 4.3 4.3 4.9)
kds=(0.34 0.34 0.24 0.34 0.34 0.24 0.42 0.42 0.30)

echo "预检 PE03 三组增益、三个关节组，共 9 次扫频。"
for index in "${!run_ids[@]}"; do
    number=$((index + 1))
    printf '[validate %d/9] %-19s group=%-5s Kp=%s Kd=%s\n' \
        "$number" "${run_ids[index]}" "${groups[index]}" "${kps[index]}" "${kds[index]}"
    "$RUNNER" \
        --deployment pe03/gait_v4 \
        --group "${groups[index]}" \
        --sweep-kp "${kps[index]}" \
        --sweep-kd "${kds[index]}" \
        --validate-only
done

if (( VALIDATE_ONLY )); then
    echo "9 次扫频配置全部校验通过；未启动硬件。"
    exit 0
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "批次输出目录已存在，拒绝混写：$OUTPUT_DIR" >&2
    exit 2
fi

echo "预检完成，开始自动采集；任意一次失败都会终止剩余任务。"
for index in "${!run_ids[@]}"; do
    number=$((index + 1))
    output_file="$OUTPUT_DIR/$(printf '%02d' "$number")_${run_ids[index]}.csv"
    printf '[run %d/9] %-19s group=%-5s Kp=%s Kd=%s\n' \
        "$number" "${run_ids[index]}" "${groups[index]}" "${kps[index]}" "${kds[index]}"
    set +e
    "$RUNNER" \
        --deployment pe03/gait_v4 \
        --group "${groups[index]}" \
        --sweep-kp "${kps[index]}" \
        --sweep-kd "${kds[index]}" \
        --output-file "$output_file"
    status=$?
    set -e
    if (( status != 0 )); then
        echo "第 $number/9 次采集失败（${run_ids[index]}，exit=$status）；后续任务已停止。" >&2
        echo "已完成的数据保留在：$OUTPUT_DIR" >&2
        exit "$status"
    fi
done

echo "PE03 增益矩阵 9 次采集全部完成。"
echo "输出目录：$OUTPUT_DIR"
