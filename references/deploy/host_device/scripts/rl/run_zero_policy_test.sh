#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "零策略台架流程："
echo "  1. 青色 WAIT_STANDBY 时按 Y，完成 3 秒 STANDBY 斜坡。"
echo "  2. 白色常亮后按 X，进入 POLICY；六维下发动作仍固定为 0。"
echo "  3. 观察完成后按 X 回 STANDBY，再按 Y 整体失能。"
echo "  4. 最后在本终端 Ctrl-C 结束 supervisor。"
echo

EXTRA_ARGS=()
if systemctl is-active --quiet we11-deploy.service 2>/dev/null; then
    echo "检测到 we11-deploy.service：本次禁用 supervisor RGB，后台绿灯不表示测试状态。"
    echo "测试期间不要按方向键上+X/A生命周期组合键。"
    EXTRA_ARGS+=(--no-rgb)
fi

exec "$ROOT_DIR/start_robot.sh" --force-zero-policy "${EXTRA_ARGS[@]}" "$@"
