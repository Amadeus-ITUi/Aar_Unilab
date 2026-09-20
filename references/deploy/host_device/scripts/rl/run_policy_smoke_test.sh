#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/logs/policy_smoke_${STAMP}"
RECORDER_PID=""

cleanup_recorder() {
    if [[ -n "$RECORDER_PID" ]] && kill -0 "$RECORDER_PID" 2>/dev/null; then
        kill -INT "$RECORDER_PID" 2>/dev/null || true
        wait "$RECORDER_PID" 2>/dev/null || true
    fi
    RECORDER_PID=""
}
trap cleanup_recorder EXIT

echo "A5 首次真实 MNN POLICY（固定 3 秒自动回 STANDBY）："
echo "  1. WAIT_STANDBY 时按一次 Y，等待 3 秒 STANDBY 完成。"
echo "  2. 按一次 X；真实策略只运行 3 秒，随后自动退回 STANDBY。"
echo "  3. 回到 STANDBY 后按一次 Y 整体失能，再 Ctrl-C 结束。"
echo "  4. 不推动摇杆；运行记录保存到 $OUT_DIR"
echo

mkdir -p "$OUT_DIR"
bash "$ROOT_DIR/record_deploy_runtime.sh" \
    --duration 0 --out-dir "$OUT_DIR" \
    >"$OUT_DIR/recorder_console.log" 2>&1 &
RECORDER_PID=$!

EXTRA_ARGS=()
if systemctl is-active --quiet we11-deploy.service 2>/dev/null; then
    echo "检测到 we11-deploy.service：禁用本次 supervisor RGB；以终端状态为准。"
    echo "测试期间不要按方向键上+X/A生命周期组合键。"
    EXTRA_ARGS+=(--no-rgb)
fi

set +e
"$ROOT_DIR/start_robot.sh" \
    --policy-smoke-seconds 3.0 "${EXTRA_ARGS[@]}" "$@"
STATUS=$?
set -e

cleanup_recorder
trap - EXIT
echo "A5 记录已保存：$OUT_DIR"
exit "$STATUS"
